import pandas as pd
import os

def build_training_dataset(job_id, year_start, year_end, data_dir):
    all_dfs = []
    
    # Scan whatever label CSVs actually exist — don't assume all years succeeded
    for year in range(year_start, year_end + 1):
        path = os.path.join(data_dir, f'labels_{job_id}_{year}.csv')
        if not os.path.exists(path):
            print(f"  Skipping {year} — labels CSV not found")
            continue
        if os.path.getsize(path) < 100:
            print(f"  Skipping {year} — labels CSV is empty")
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"  Skipping {year} — could not parse CSV")
            continue
        df['Year'] = year
        all_dfs.append(df)
        print(f"  {year}: {len(df)} labelled pixels loaded")

    if not all_dfs:
        raise ValueError(
            f"No labelled CSVs found in {data_dir}. "
            "Check that generate_labels() completed successfully."
        )

    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset = dataset[
        ~dataset['Mineral_Type'].isin(['noise', 'unclassified'])
    ].copy()

    if len(dataset) == 0:
        raise ValueError(
            "Training dataset empty after filtering. "
            "Try lowering sam_threshold in generate_labels()."
        )

    out_path = os.path.join(data_dir, 'training_dataset.csv')
    dataset.to_csv(out_path, index=False)
    print(f"\nTraining dataset: {len(dataset)} samples, "
          f"{dataset['Mineral_Type'].nunique()} classes")
    for mineral, count in dataset['Mineral_Type'].value_counts().items():
        print(f"  {mineral}: {count}")
    print(f"Saved: {out_path}")
    return out_path