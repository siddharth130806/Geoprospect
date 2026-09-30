import numpy as np
import pandas as pd
import os
import glob

SENTINEL2_DIR = 'splib07/ASCIIdata/ASCIIdata_splib07b_rsSentinel2/ChapterM_Minerals'
ASTER_DIR     = 'splib07/ASCIIdata/ASCIIdata_splib07b_rsASTER/ChapterM_Minerals'

# Sentinel-2 band order in splib07b files: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
S2_BAND_IDX = {'B2': 1, 'B3': 2, 'B4': 3, 'B8': 7, 'B11': 11, 'B12': 12}

# ASTER band order in splib07b files: B1,B2,B3N,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14
ASTER_BAND_IDX = {'ASTER_B01': 0, 'ASTER_B02': 1, 'ASTER_B3N': 2}

TARGET_MINERALS = [
    'Hematite', 'Goethite', 'Magnetite',
    'Kaolinite', 'Montmorillonite', 'Illite', 'Muscovite', 'Chlorite',
    'Calcite', 'Dolomite', 'Magnesite', 'Siderite',
    'Alunite', 'Gypsum', 'Jarosite',
    'Quartz', 'Albite', 'Orthoclase',
    'Olivine', 'Serpentine', 'Enstatite', 'Augite', 'Chromite',
    'Epidote', 'Actinolite', 'Talc', 'Pyrophyllite',
    'Pyrite', 'Chalcopyrite', 'Galena', 'Sphalerite',
]


def read_splib_file(filepath):
    try:
        values = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # First line is header, skip it
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                val = float(line)
                if val < -1e30:
                    val = np.nan
                values.append(val)
            except ValueError:
                continue
        return np.array(values) if values else None
    except Exception:
        return None

def get_mineral_files(mineral_name, directory):
    """
    Returns all files matching a mineral name in the directory.
    Uses case-insensitive partial match on filename.
    Prefers AREF (absolute reflectance) over RREF.
    """
    pattern = os.path.join(directory, f'*{mineral_name}*AREF.txt')
    files = glob.glob(pattern, recursive=False)
    if not files:
        pattern = os.path.join(directory, f'*{mineral_name}*.txt')
        files = glob.glob(pattern, recursive=False)
    return files


def build_reference_library(output_path='usgs_spectral_references.csv'):
    records = []

    for mineral in TARGET_MINERALS:
        s2_files    = get_mineral_files(mineral, SENTINEL2_DIR)
        aster_files = get_mineral_files(mineral, ASTER_DIR)

        if not s2_files:
            print(f"  Skipping {mineral} — no Sentinel-2 file found")
            continue

        # Average across multiple samples of same mineral
        s2_spectra = []
        for f in s2_files:
            vals = read_splib_file(f)
            if vals is not None and len(vals) >= 13:
                s2_spectra.append(vals[:13])

        aster_spectra = []
        for f in aster_files:
            vals = read_splib_file(f)
            if vals is not None and len(vals) >= 3:
                aster_spectra.append(vals[:14])

        if not s2_spectra:
            print(f"  Skipping {mineral} — could not parse Sentinel-2 data")
            continue

        s2_mean    = np.nanmean(s2_spectra, axis=0)
        aster_mean = np.nanmean(aster_spectra, axis=0) if aster_spectra else np.zeros(14)

        row = {'mineral_name': mineral}

        for band, idx in S2_BAND_IDX.items():
            row[band] = float(s2_mean[idx]) if idx < len(s2_mean) else 0.0

        for band, idx in ASTER_BAND_IDX.items():
            row[band] = float(aster_mean[idx]) if idx < len(aster_mean) else 0.0

        # Compute derived indices
        b2  = row['B2']  + 1e-8
        b3  = row['B3']
        b4  = row['B4']
        b8  = row['B8']
        b11 = row['B11']
        b12 = row['B12'] + 1e-8
        a01 = row['ASTER_B01'] + 1e-8
        a02 = row['ASTER_B02']
        a3n = row['ASTER_B3N']

        row['Iron_Oxide']   = b4  / b2
        row['Clay_Index']   = b11 / b12
        row['NDVI']         = (b8 - b4) / (b8 + b4 + 1e-8)
        row['NDWI']         = (b3 - b8) / (b3 + b8 + 1e-8)
        row['Ferric_Iron']  = a02 / a01
        row['Silica_Index'] = a3n / a01

        records.append(row)
        print(f"  Built: {mineral} ({len(s2_files)} Sentinel-2 samples, "
              f"{len(aster_files)} ASTER samples)")

    if not records:
        raise RuntimeError(
            "No minerals loaded. Check SENTINEL2_DIR and ASTER_DIR paths."
        )

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} mineral references → {output_path}")
    return df


if __name__ == '__main__':
    build_reference_library()