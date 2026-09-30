# build_global_dataset.py
import pandas as pd
import glob
import os

all_csvs = glob.glob('data/*/training_dataset.csv')
dfs = []
for csv in all_csvs:
    job_id = csv.split('/')[1]
    df = pd.read_csv(csv)
    df['source_job'] = job_id
    dfs.append(df)
    print(f"  {job_id}: {len(df)} samples, minerals: {df['Mineral_Type'].unique().tolist()}")

global_df = pd.concat(dfs, ignore_index=True)
global_df.to_csv('global_training_dataset.csv', index=False)
print(f"\nTotal: {len(global_df)} samples, {global_df['Mineral_Type'].nunique()} mineral classes")
print(global_df['Mineral_Type'].value_counts())