import pandas as pd
import numpy as np
import joblib
import json
import os

from mineral_associations import MINERAL_TO_COMMODITY


FEATURE_COLUMNS = [
    'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
    'Iron_Oxide', 'Clay_Index', 'NDVI', 'NDWI',
    'B01', 'B02', 'B3N',
    'Ferric_Iron', 'Silica_Index',
]

MIN_COVERAGE_PCT = 2.0   # a mineral must dominate at least this % of pixels to be reported

def predict(job_id, year, data_dir, generate_map=False, min_coverage_pct=MIN_COVERAGE_PCT):
    """
    Predicts mineral probabilities for all sampled pixels.

    Parameters:
        job_id (str):         Job identifier.
        year (int):           Year to predict.
        data_dir (str):       Path to data/{job_id}/.
        generate_map (bool):  If True, also produces a GeoTIFF probability map.

    Returns:
        dict with job_id, year, minerals (list), n_pixels.
    """
    model_path   = os.path.join(data_dir, 'mineral_model.pkl')
    encoder_path = os.path.join(data_dir, 'mineral_encoder.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train() first.")

    model   = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    csv_path = os.path.join(data_dir, f'features_{job_id}_{year}.csv')
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        if col in ['B2','B3','B4','B8','B11','B12'] and df[col].max() > 1.0:
            df[col] = df[col] / 10000.0

    X = pd.DataFrame(df[FEATURE_COLUMNS].values, columns=FEATURE_COLUMNS)
    probs = model.predict_proba(X)
    n_pixels = len(df)

    # Dominant predicted mineral per pixel. This sums to 100% across all
    # classes, unlike the old ">0.5 probability" threshold, which breaks
    # down once there are many classes (probability mass spreads thin,
    # so almost nothing ever crosses 0.5 even when a mineral genuinely
    # dominates a large share of the AOI).
    predicted_class_idx = np.argmax(probs, axis=1)

    all_minerals = []
    for i, mineral in enumerate(encoder.classes_):
        mineral_probs = probs[:, i]
        dominant_count = int((predicted_class_idx == i).sum())
        coverage_pct = float(dominant_count / n_pixels * 100)

        all_minerals.append({
            'mineral':      mineral,
            'confidence':   round(float(mineral_probs.max()), 3),
            'mean_prob':    round(float(mineral_probs.mean()), 3),
            'coverage_pct': round(coverage_pct, 1),
        })

    # Only report minerals that actually win the "most likely" vote for a
    # meaningful share of pixels — this drops the long tail of classes
    # that only ever showed up as a stray, weak secondary probability.
    significant = [m for m in all_minerals if m['coverage_pct'] >= min_coverage_pct]
    significant.sort(key=lambda x: x['coverage_pct'], reverse=True)

    if not significant:
        significant = sorted(all_minerals, key=lambda x: x['coverage_pct'], reverse=True)[:1]

    for m in significant:
        assoc = MINERAL_TO_COMMODITY.get(m['mineral'], {})
        m['metals'] = assoc.get('metals', [])
        m['uses']   = assoc.get('uses', [])

    result = {
        'job_id':   job_id,
        'year':     year,
        'minerals': significant,
        'n_pixels': n_pixels,
    }

    json_path = os.path.join(data_dir, f'result_{year}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON summary: {json_path}")
    for m in significant:
        print(f"  {m['mineral']}: confidence={m['confidence']}, coverage={m['coverage_pct']}%")

    if generate_map:
        _export_probability_tif(job_id, year, data_dir, df, probs, encoder)

    return result


def _export_probability_tif(job_id, year, data_dir, df, probs, encoder):
    """
    Generates a uint8 + LZW compressed multi-band GeoTIFF where each band
    is one mineral's probability map (0–255, where 255 = 100% probability).
    Only called when generate_map=True.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from scipy.interpolate import griddata
    except ImportError:
        print("WARNING: rasterio or scipy not installed. Skipping TIF export.")
        return

    # Extract coordinates from GEE's .geo column (Point geometry as JSON string)
    # GEE exports geometries as: {"type":"Point","coordinates":[lon, lat]}
    if '.geo' not in df.columns:
        print("WARNING: No .geo column found. Cannot generate spatial TIF.")
        return

    try:
        coords = df['.geo'].apply(
            lambda g: json.loads(g)['coordinates'] if isinstance(g, str) else [0, 0]
        )
        lons = np.array([c[0] for c in coords])
        lats = np.array([c[1] for c in coords])
    except Exception as e:
        print(f"WARNING: Could not parse coordinates: {e}")
        return

    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()

    # ~30m resolution in decimal degrees at equator
    resolution = 0.0003
    # predict_map.py — replace the width/height lines
    width  = min(500, max(1, int((max_lon - min_lon) / resolution)))
    height = min(500, max(1, int((max_lat - min_lat) / resolution)))

    # Build regular grid for interpolation
    grid_lons = np.linspace(min_lon, max_lon, width)
    grid_lats = np.linspace(max_lat, min_lat, height)  # top to bottom
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    n_classes = probs.shape[1]

    output_path = os.path.join(data_dir, f'mineral_probability_map_{year}.tif')

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=n_classes,
        dtype='uint8',          # 75% smaller than float32
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'          # additional 20-30% compression, lossless
    ) as dst:
        for i, mineral in enumerate(encoder.classes_):
            prob_values = probs[:, i]

            # Interpolate scattered sample points to regular grid
            grid_probs = griddata(
                points=np.column_stack([lons, lats]),
                values=prob_values,
                xi=(grid_lon_2d, grid_lat_2d),
                method='linear',
                fill_value=0.0
            )

            band_uint8 = (np.clip(grid_probs, 0, 1) * 255).astype(np.uint8)
            dst.write(band_uint8, i + 1)
            dst.set_band_description(i + 1, mineral)

    print(f"Saved pixel map ({n_classes} bands, uint8+LZW): {output_path}")
    print("Tip: to read probabilities back, divide pixel values by 255.")


def predict_nearest_point(lat, lng, job_id, year, data_dir):
    """
    Finds the nearest sampled pixel to (lat, lng) from the existing
    features CSV and returns its mineral probabilities.
    No new GEE calls — uses data already extracted for this job.
    """
    csv_path = os.path.join(data_dir, f'features_{job_id}_{year}.csv')
    if not os.path.exists(csv_path):
        return {'error': 'No data for this job/year'}

    df = pd.read_csv(csv_path)
    if '.geo' not in df.columns:
        return {'error': 'No coordinate data in samples'}

    coords = df['.geo'].apply(
        lambda g: json.loads(g)['coordinates'] if isinstance(g, str) else [0, 0]
    )
    lons = np.array([c[0] for c in coords])
    lats = np.array([c[1] for c in coords])

    # Nearest neighbor by Euclidean distance (good enough at small scale)
    distances = np.sqrt((lons - lng)**2 + (lats - lat)**2)
    nearest_idx = np.argmin(distances)
    nearest_dist_km = distances[nearest_idx] * 111  # rough deg-to-km

    model_path   = os.path.join(data_dir, 'mineral_model.pkl')
    encoder_path = os.path.join(data_dir, 'mineral_encoder.pkl')
    if not os.path.exists(model_path):
        model_path   = os.path.join(os.path.dirname(__file__), 'global_mineral_model.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'global_mineral_encoder.pkl')

    model   = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    row = df.iloc[[nearest_idx]].copy()
    for col in FEATURE_COLUMNS:
        if col not in row.columns:
            row[col] = 0.0
        if col in ['B2','B3','B4','B8','B11','B12'] and row[col].iloc[0] > 1.0:
            row[col] = row[col] / 10000.0

    # ── Terrain classification using raw spectral indices ──
    # These checks run BEFORE the model to catch non-mineral terrain
    ndvi_val       = float(row['NDVI'].iloc[0])       if 'NDVI'       in row.columns else 0.0
    ndwi_val       = float(row['NDWI'].iloc[0])       if 'NDWI'       in row.columns else 0.0
    iron_oxide_val = float(row['Iron_Oxide'].iloc[0]) if 'Iron_Oxide' in row.columns else 0.0
    clay_val       = float(row['Clay_Index'].iloc[0]) if 'Clay_Index' in row.columns else 0.0
    ferric_val     = float(row['Ferric_Iron'].iloc[0])if 'Ferric_Iron'in row.columns else 0.0

    water_detected = ndwi_val > 0.3

    # Determine terrain type from spectral indices
    # Thresholds calibrated from actual mineralized terrain statistics:
    #   Iron_Oxide p25=1.55, Clay_Index p25=1.08, Ferric_Iron p25=0.93
    # Only flag as unmineralized when ALL indices are well below typical mineral terrain
    mineralized = True
    terrain_type = 'mineral'

    if water_detected:
        mineralized = False
        terrain_type = 'water'
    elif ndvi_val > 0.45:
        # Very dense vegetation — clearly no exposed rock
        mineralized = False
        terrain_type = 'vegetation'
    elif iron_oxide_val < 0.8 and clay_val < 0.85 and ferric_val < 0.6 and ndvi_val > 0.25:
        # Low mineral indices combined with moderate vegetation
        mineralized = False
        terrain_type = 'vegetation'

    # ── Run model prediction ──
    X = row[FEATURE_COLUMNS].fillna(0)
    probs = model.predict_proba(X)[0]

    minerals = []
    MIN_PROB_PCT = 3.0
    for i, mineral in enumerate(encoder.classes_):
        prob_pct = round(float(probs[i]) * 100, 1) if mineralized else 0.0
        if prob_pct < MIN_PROB_PCT:
            continue
        assoc = MINERAL_TO_COMMODITY.get(mineral, {})
        minerals.append({
            'mineral':     mineral,
            'probability': prob_pct,
            'metals':      assoc.get('metals', []),
            'uses':        assoc.get('uses', [])
        })
    minerals.sort(key=lambda x: x['probability'], reverse=True)

    if not minerals:
        i = int(np.argmax(probs))
        mineral = encoder.classes_[i]
        assoc = MINERAL_TO_COMMODITY.get(mineral, {})
        minerals = [{
            'mineral': mineral,
            'probability': round(float(probs[i]) * 100, 1) if mineralized else 0.0,
            'metals': assoc.get('metals', []),
            'uses': assoc.get('uses', [])
        }]

    return {
        'lat': lat, 'lng': lng,
        'mineralized': mineralized,
        'terrain_type': terrain_type,
        'water_present': bool(water_detected),
        'ndwi': round(float(ndwi_val), 3),
        'ndvi': round(float(ndvi_val), 3),
        'nearest_sample_distance_km': round(float(nearest_dist_km), 2),
        'minerals': minerals
    }

