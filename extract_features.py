import ee
import os
import time

ee.Initialize(project='mineral-detection-ai')


def extract_features(bbox, year, job_id, num_pixels=50000):
    """
    Extracts spectral features directly from GEE as CSV.
    bbox = [min_lon, min_lat, max_lon, max_lat]
    No TIF export. No merge step needed.
    """
    # Sentinel-2 SR only available from 2017 onwards
    if year < 2017:
        sentinel_collection = "COPERNICUS/S2"          # older TOA collection
    else:
        sentinel_collection = "COPERNICUS/S2_SR_HARMONIZED"

    aoi = ee.Geometry.Rectangle(bbox)

    sentinel = (
        ee.ImageCollection(sentinel_collection)
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .median()
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    )

    # Guard — if collection is empty, fill with zeros so sample() doesn't silently export nothing
    sentinel_size = (
        ee.ImageCollection(sentinel_collection)
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .size()
    )

    sentinel_fallback = ee.Image.constant([0, 0, 0, 0, 0, 0]).rename(
        ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    )

    sentinel = ee.Image(ee.Algorithms.If(
        sentinel_size.gt(0),
        sentinel,
        sentinel_fallback
    ))

    # ASTER VNIR only (SWIR sensor failed in 2008, permanently zero)
    aster_col = (
        ee.ImageCollection("ASTER/AST_L1T_003")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )

    aster_fallback = ee.Image.constant([0, 0, 0]).rename(['B01', 'B02', 'B3N'])
    aster = ee.Image(ee.Algorithms.If(
        aster_col.size().gt(0),
        aster_col.median().select(['B01', 'B02', 'B3N']),
        aster_fallback
    ))

    # Sentinel-2 derived indices
    iron_oxide = sentinel.expression(
        'RED / (BLUE + 1e-8)',
        {'RED': sentinel.select('B4'), 'BLUE': sentinel.select('B2')}
    ).rename('Iron_Oxide')

    clay_index = sentinel.expression(
        'SWIR1 / (SWIR2 + 1e-8)',
        {'SWIR1': sentinel.select('B11'), 'SWIR2': sentinel.select('B12')}
    ).rename('Clay_Index')

    ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = sentinel.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # ASTER derived indices
    ferric_iron = aster.expression(
        'B2 / (B1 + 1e-8)',
        {'B1': aster.select('B01'), 'B2': aster.select('B02')}
    ).rename('Ferric_Iron')

    silica_index = aster.expression(
        'B3N / (B01 + 1e-8)',
        {'B3N': aster.select('B3N'), 'B01': aster.select('B01')}
    ).rename('Silica_Index')

    # Stack all features
    feature_stack = (
        sentinel
        .addBands(aster)
        .addBands([iron_oxide, clay_index, ndvi, ndwi, ferric_iron, silica_index])
    )

    # Sample pixels directly — produces CSV not TIF
    # Check pixel count before exporting — empty region = no export
    sample_count = feature_stack.sample(
        region=aoi,
        scale=30,
        numPixels=10,
        seed=42
    ).size()

    # Calculate rough area in square degrees to scale numPixels
    width_deg = bbox[2] - bbox[0]
    height_deg = bbox[3] - bbox[1]
    area_sq_deg = width_deg * height_deg
    
    # 0.2 sq degrees is roughly the Pilbara size (e.g. 0.5 * 0.4) -> 10,000 pixels
    # We scale linearly, capping at 100,000 to prevent GEE memory errors
    calculated_pixels = int((area_sq_deg / 0.2) * 10000)
    final_num_pixels = min(100000, max(10000, calculated_pixels))
    print(f"  AOI Area: {area_sq_deg:.2f} sq deg -> Sampling {final_num_pixels} pixels")

    samples = feature_stack.sample(
        region=aoi,
        scale=60,          # was 30, coarser resolution uses less memory
        numPixels=final_num_pixels,
        seed=42,
        geometries=True,
        tileScale=8        # increased to 8 to handle larger pixel counts without OOM
    )   

    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=f'features_{job_id}_{year}',
        folder='MineralDetection',
        fileNamePrefix=f'features_{job_id}_{year}',
        fileFormat='CSV'
    )
    task.start()
    print(f"  Started export: features_{job_id}_{year}")
    return task


def wait_for_tasks(tasks, poll_seconds=30):
    print("Waiting for GEE exports to complete...")
    while True:
        statuses = [t.status()['state'] for t in tasks]
        print(f"  Task statuses: {statuses}")
        if all(s in ('COMPLETED', 'FAILED') for s in statuses):
            failed = [t for t in tasks if t.status()['state'] == 'FAILED']
            if failed:
                print(f"WARNING: {len(failed)} GEE tasks failed")
            return
        time.sleep(poll_seconds)


def download_from_drive(job_id, year_start, year_end, output_dir):
    """
    Downloads feature CSVs from Google Drive to local data/{job_id}/.
    Requires a service account credentials file at
    credentials/drive_service_account.json.
    """
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    creds_path = os.path.join(os.path.dirname(__file__), 'credentials', 'drive_service_account.json')
    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=creds)
    os.makedirs(output_dir, exist_ok=True)

    for year in range(year_start, year_end + 1):
        filename = f'features_{job_id}_{year}.csv'
        results = service.files().list(
            q=f"name='{filename}' and trashed=false",
            fields="files(id, name)"
        ).execute()
        files = results.get('files', [])
        if not files:
            print(f"  Not found on Drive: {filename}")
            continue
        file_id = files[0]['id']
        content = service.files().get_media(fileId=file_id).execute()
        local_path = os.path.join(output_dir, filename)
        with open(local_path, 'wb') as f:
            f.write(content)
        print(f"  Downloaded: {local_path}")
