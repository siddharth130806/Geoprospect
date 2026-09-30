import numpy as np
import pandas as pd
import os

try:
    import hdbscan
except ImportError:
    raise ImportError("Install hdbscan: pip install hdbscan")


def spectral_angle(a, b):
    """
    Computes the spectral angle in radians between two vectors.
    Smaller angle = more similar spectrum = better mineral match.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    cos_a = np.dot(a, b) / denom
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


DEFAULT_SPECTRAL_PATH = os.path.join(os.path.dirname(__file__), 'usgs_spectral_references.csv')


def generate_labels(job_id, year, data_dir,
                    spectral_library_path=None,
                    sam_threshold=0.5):   # was 0.3, raised to 0.8

    # Default to project root regardless of where the script is called from
    if spectral_library_path is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        spectral_library_path = os.path.join(project_root, 'usgs_spectral_references.csv')

    """
    Reads the feature CSV for a given job and year, clusters pixels
    using HDBSCAN, matches each cluster to the USGS spectral library
    using SAM, and saves a labelled CSV back to data_dir.

    Parameters:
        job_id (str):                  Job identifier.
        year_start (int):              Start year.
        year_end (int):                End year.
        data_dir (str):                Path to data/{job_id}/.
        spectral_library_path (str):   Path to usgs_spectral_references.csv.
        sam_threshold (float):         Max SAM angle (radians) for a valid
                                       mineral assignment. ~0.3 rad ≈ 17°.
                                       Pixels with no match below this
                                       threshold are labelled 'unclassified'.

    Returns:
        list of mineral names found (excludes noise and unclassified).
    """
    csv_path = os.path.join(data_dir, f'features_{job_id}_{year}.csv')
    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"Feature CSV is empty: {csv_path} — check GEE task completed successfully")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"No data in {csv_path} — GEE export may have failed silently")

    sentinel_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    for band in sentinel_bands:
        if band in df.columns:
            df[band] = df[band] / 10000.0

    # Recompute indices after normalizing bands so they match library scale
    df['Iron_Oxide']  = df['B4']  / (df['B2']  + 1e-8)
    df['Clay_Index']  = df['B11'] / (df['B12'] + 1e-8)
    df['NDVI']        = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 1e-8)
    df['NDWI']        = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + 1e-8)

    # ASTER bands also come scaled — normalize
    aster_bands = ['ASTER_B01', 'ASTER_B02', 'ASTER_B3N', 'B01', 'B02', 'B3N']
    for band in aster_bands:
        if band in df.columns and df[band].max() > 1.0:
            df[band] = df[band] / 10000.0

    if 'ASTER_B01' in df.columns:
        df['Ferric_Iron']  = df['ASTER_B02'] / (df['ASTER_B01'] + 1e-8)
        df['Silica_Index'] = df['ASTER_B3N'] / (df['ASTER_B01'] + 1e-8)

    # Clean infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['Iron_Oxide', 'Clay_Index', 'NDVI', 'NDWI',
                'Ferric_Iron', 'Silica_Index', 'B4', 'B8', 'B11', 'B12']
    )
    print(f"After cleaning: {len(df)} valid pixels")

    # Feature columns used for clustering
    all_possible_features = [
        'Iron_Oxide', 'Clay_Index', 'NDVI', 'NDWI',
        'Ferric_Iron', 'Silica_Index',
        'B4', 'B8', 'B11', 'B12'
    ]
    cluster_features = [f for f in all_possible_features if f in df.columns]
    print(f"  Using {len(cluster_features)} features for clustering: {cluster_features}")
    X = df[cluster_features].values

    # Normalize features for clustering
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # HDBSCAN clustering — discovers clusters without specifying count
    # -1 = noise pixels (cloud shadow, water, urban) — discarded later
    print("Running HDBSCAN clustering...")
    n_pixels = len(df)
    min_cluster_size = max(5, n_pixels // 200)   # was // 50
    min_samples = max(2, n_pixels // 500)         # was // 200

    print(f"  Pixels: {n_pixels}, min_cluster_size: {min_cluster_size}, min_samples: {min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_epsilon=0.5,
        cluster_selection_method='leaf'
    )
    df['cluster_id'] = clusterer.fit_predict(X_scaled)

    # If still no clusters, retry with more aggressive settings
    n_clusters = len([c for c in df['cluster_id'].unique() if c >= 0])
    if n_clusters == 0:
        print("  Retrying with relaxed parameters...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=2,
            metric='euclidean',
            cluster_selection_epsilon=1.0,
            cluster_selection_method='eom'
        )
        df['cluster_id'] = clusterer.fit_predict(X_scaled)
        n_clusters = len([c for c in df['cluster_id'].unique() if c >= 0])

    # If still nothing, force KMeans as last resort
    if n_clusters == 0:
        print("  HDBSCAN failed — falling back to KMeans with k=3...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster_id'] = kmeans.fit_predict(X_scaled)

    unique_clusters = sorted([c for c in df['cluster_id'].unique() if c >= 0])
    n_noise = (df['cluster_id'] == -1).sum()
    print(f"Found {len(unique_clusters)} clusters, {n_noise} noise pixels discarded")

    if len(unique_clusters) == 0:
        print("WARNING: No clusters found. Try reducing min_cluster_size.")
        df['Mineral_Type'] = 'unclassified'
        df['label'] = 0
        out_path = os.path.join(data_dir, f'labels_{job_id}_{year}.csv')
        df.to_csv(out_path, index=False)
        return []

    # Load USGS spectral reference library
    if not os.path.exists(spectral_library_path):
        raise FileNotFoundError(
            f"Spectral library not found: {spectral_library_path}\n"
            "Run build_spectral_library.py first."
        )
    library = pd.read_csv(spectral_library_path)

    # Only use reference columns that exist in the feature set
    ref_cols = [c for c in cluster_features if c in library.columns]
    print(f"  SAM reference columns: {ref_cols}")
    ref_matrix = library[ref_cols].values
    mineral_names = library['mineral_name'].tolist()

    # SAM match: for each cluster, compare mean spectrum to all references
    cluster_to_mineral = {}
    print("SAM matching clusters to spectral library...")
    for cid in unique_clusters:
        cluster_pixels = df[df['cluster_id'] == cid][ref_cols].values
        mean_spectrum = cluster_pixels.mean(axis=0)

        angles = [spectral_angle(mean_spectrum, ref) for ref in ref_matrix]
        best_idx = int(np.argmin(angles))
        best_angle = angles[best_idx]

        # ADD THIS — print actual angles to see what values you're getting
        print(f"  Cluster {cid}: best match = {mineral_names[best_idx]}, "
            f"angle = {best_angle:.4f} rad ({np.degrees(best_angle):.2f}°), "
            f"threshold = {sam_threshold} rad ({np.degrees(sam_threshold):.2f}°)")

        if best_angle < sam_threshold:
            mineral = mineral_names[best_idx]
        else:
            mineral = 'unclassified'

        cluster_to_mineral[cid] = mineral
        print(f"  Cluster {cid} ({(df['cluster_id']==cid).sum()} px) "
              f"→ {mineral} (angle={best_angle:.3f} rad)")

    # Assign mineral labels to every pixel
    df['Mineral_Type'] = df['cluster_id'].map(cluster_to_mineral)
    df.loc[df['cluster_id'] == -1, 'Mineral_Type'] = 'noise'
    df['Mineral_Type'] = df['Mineral_Type'].fillna('unclassified')
    df['label'] = (~df['Mineral_Type'].isin(['noise', 'unclassified'])).astype(int)

    # Save labelled CSV
    out_path = os.path.join(data_dir, f'labels_{job_id}_{year}.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved labelled CSV: {out_path}")

    minerals_found = [
        m for m in df['Mineral_Type'].unique()
        if m not in ('noise', 'unclassified')
    ]
    print(f"Minerals found in {year}: {minerals_found}")
    return minerals_found
