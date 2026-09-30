import pandas as pd
import numpy as np

TARGET_MINERALS = [
    'Hematite', 'Goethite', 'Magnetite', 'Limonite',
    'Kaolinite', 'Montmorillonite', 'Illite',
    'Muscovite', 'Chlorite', 'Smectite',
    'Calcite', 'Dolomite', 'Magnesite', 'Siderite',
    'Alunite', 'Gypsum', 'Jarosite', 'Anhydrite',
    'Quartz', 'Feldspar', 'Albite', 'Orthoclase',
    'Olivine', 'Serpentine', 'Pyroxene', 'Enstatite',
    'Augite', 'Chromite',
    'Epidote', 'Actinolite', 'Prehnite', 'Talc',
    'Pyrophyllite', 'Diaspore',
    'Apatite', 'Monazite',
    'Pyrite', 'Chalcopyrite',
    'Dry_Grass', 'Dry_Soil', 'Sand',
]

np.random.seed(42)
records = []
for mineral in TARGET_MINERALS:
    b2 = np.random.uniform(0.05, 0.4)
    b3 = np.random.uniform(0.1, 0.5)
    b4 = np.random.uniform(0.1, 0.6)
    b8 = np.random.uniform(0.2, 0.8)
    b11 = np.random.uniform(0.1, 0.7)
    b12 = np.random.uniform(0.1, 0.6)
    
    a01 = b3 
    a02 = b4 
    a3n = b8 
    
    row = {
        'mineral_name': mineral,
        'B2': b2, 'B3': b3, 'B4': b4, 'B8': b8, 'B11': b11, 'B12': b12,
        'ASTER_B01': a01, 'ASTER_B02': a02, 'ASTER_B3N': a3n,
    }
    
    row['Iron_Oxide']   = b4  / (b2  + 1e-8)
    row['Clay_Index']   = b11 / (b12 + 1e-8)
    row['NDVI']         = (b8 - b4) / (b8 + b4 + 1e-8)
    row['NDWI']         = (b3 - b8) / (b3 + b8 + 1e-8)
    row['Ferric_Iron']  = a02 / (a01 + 1e-8)
    row['Silica_Index'] = a3n / (a01 + 1e-8)
    
    records.append(row)

df = pd.DataFrame(records)
df.to_csv('usgs_spectral_references.csv', index=False)
print("Mock usgs_spectral_references.csv generated successfully.")
