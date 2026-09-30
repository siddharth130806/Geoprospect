import os
import json

from extract_features import extract_features, wait_for_tasks, download_from_drive
from multiclass_label_generation import generate_labels
from sample_rasters import build_training_dataset
from train_model import train
from predict_map import predict


def run_pipeline(job_id, bbox, year_start, year_end, generate_map=False):
    """
    Runs the complete GeoProspect AI pipeline for a given AOI and year range.

    Parameters:
        job_id (str):        Unique identifier for this job. Used as folder name.
        bbox (list):         [min_lon, min_lat, max_lon, max_lat]
        year_start (int):    First year to process.
        year_end (int):      Last year to process (inclusive).
        generate_map (bool): If True, also produces a pixel-level probability
                             GeoTIFF in addition to the JSON summary.
                             Default False — JSON summary only.

    Returns:
        list of result dicts, one per year.
    """
    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, 'data', job_id)
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Job: {job_id}")
    print(f"BBox: {bbox}")
    print(f"Years: {year_start} – {year_end}")
    print(f"Generate pixel map: {generate_map}")
    print(f"{'='*60}\n")

    # Step 1 — Extract features from GEE as CSV (no TIF)
    print("Step 1/5 — Extracting features from GEE...")
    tasks = []
    for year in range(year_start, year_end + 1):
        task = extract_features(bbox, year, job_id)
        tasks.append(task)
    wait_for_tasks(tasks)
    download_from_drive(job_id, year_start, year_end, data_dir)

    # After download_from_drive call
    valid_years = []
    for year in range(year_start, year_end + 1):
        csv_path = os.path.join(data_dir, f'features_{job_id}_{year}.csv')
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100:
            valid_years.append(year)
        else:
            print(f"  WARNING: No valid CSV for {year} — skipping")

    if not valid_years:
        raise RuntimeError(
            "All GEE exports failed or produced empty files. "
            "Check Earth Engine task manager for errors."
        )

    print(f"Valid years with data: {valid_years}")


    # Step 2 — Generate labels
    print("\nStep 2/5 — Generating dynamic mineral labels...")
    all_minerals = set()
    successful_years = []
    for year in valid_years:
        csv_path = os.path.join(data_dir, f'features_{job_id}_{year}.csv')
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
            print(f"  Skipping {year} — empty or missing CSV")
            continue
        try:
            minerals = generate_labels(job_id, year, data_dir)
            if minerals:
                all_minerals.update(minerals)
                successful_years.append(year)
            else:
                print(f"  Skipping {year} — no minerals found")
        except Exception as e:
            print(f"  Skipping {year} — label generation failed: {e}")
            continue

    if not successful_years:
        raise RuntimeError(
            "Label generation failed for all years. "
            "Try lowering sam_threshold in generate_labels() or "
            "check that usgs_spectral_references.csv exists."
        )
    print(f"Minerals discovered: {sorted(all_minerals)}")
    print(f"Successful years: {successful_years}")

    # Step 3 — Build training dataset
    print("\nStep 3/5 — Building training dataset...")
    build_training_dataset(job_id, successful_years[0], successful_years[-1], data_dir)

    # Step 4 — Train model
    print("\nStep 4/5 — Training XGBoost model...")
    global_model   = os.path.join(project_root, 'global_mineral_model.pkl')
    global_encoder = os.path.join(project_root, 'global_mineral_encoder.pkl')

    if os.path.exists(global_model):
        print("Using global model — skipping retraining")
        import shutil
        shutil.copy(global_model, os.path.join(data_dir, 'mineral_model.pkl'))
        shutil.copy(global_encoder, os.path.join(data_dir, 'mineral_encoder.pkl'))
        global_metadata_path = os.path.join(os.path.dirname(__file__), 'global_model_metadata.json')
        if os.path.exists(global_metadata_path):
            shutil.copy(global_metadata_path, os.path.join(data_dir, 'model_metadata.json'))
    else:
        print("No global model found — training from AOI data")
        try:
            train(data_dir)
        except ValueError as e:
            if "Only one mineral class" in str(e):
                print(f"  WARNING: {e}")
                print("  Single mineral AOI — using nearest-neighbor classifier instead")
                _save_single_class_model(data_dir)
            else:
                raise

    # Step 5 — Predict
    print("\nStep 5/5 — Predicting minerals...")
    results = []
    for year in successful_years:
        try:
            result = predict(
                job_id=job_id,
                year=year,  
                data_dir=data_dir,
                generate_map=generate_map
            )
            results.append(result)
            print(f"  {year}: {[m['mineral'] for m in result['minerals']]}")
        except Exception as e:
            print(f"  Skipping prediction for {year}: {e}")
            continue

    if not results:
        raise RuntimeError("Prediction failed for all years.")

    # Save job summary
    summary = {
        'job_id':     job_id,
        'bbox':       bbox,
        'year_start': year_start,
        'year_end':   year_end,
        'results':    results
    }
    summary_path = os.path.join(data_dir, 'job_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nJob {job_id} complete. Summary saved to {summary_path}")
    return results

def _save_single_class_model(data_dir):
    import joblib
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.dummy import DummyClassifier

    df = pd.read_csv(os.path.join(data_dir, 'training_dataset.csv'))
    mineral_encoder = LabelEncoder()
    mineral_encoder.fit(df['Mineral_Type'].astype(str))

    # DummyClassifier always predicts the one class with 100% confidence
    model = DummyClassifier(strategy='most_frequent')
    y = mineral_encoder.transform(df['Mineral_Type'].astype(str))
    feature_columns = ['B2','B3','B4','B8','B11','B12','Iron_Oxide','Clay_Index',
                       'NDVI','NDWI','B01','B02','B3N','Ferric_Iron','Silica_Index']
    feature_columns = [f for f in feature_columns if f in df.columns]
    X = df[feature_columns].fillna(0).values
    model.fit(X, y)

    joblib.dump(model,           os.path.join(data_dir, 'mineral_model.pkl'))
    joblib.dump(mineral_encoder, os.path.join(data_dir, 'mineral_encoder.pkl'))
    print(f"  Saved single-class model for: {mineral_encoder.classes_}")


"""if __name__ == '__main__':
    # Local test — replace with any AOI and year range
    run_pipeline(
        job_id='test_001',
        bbox=[118.5, -22.5, 118.8, -22.2],
        year_start=2022,
        year_end=2023,
        generate_map=False
    )"""
if __name__ == '__main__':
    run_pipeline(
        job_id='58f65475',
        bbox=[11.0, 46.8, 16.0, 48.2],
        year_start=2020,
        year_end=2023,
        generate_map=False
    )