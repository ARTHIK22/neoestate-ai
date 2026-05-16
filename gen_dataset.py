"""Generates an expanded synthetic dataset (~500 rows) for NeoEstate AI."""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

FURN_OPTS = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
COND_OPTS = ['Poor', 'Good', 'Excellent']
FURN_MULT = {'Unfurnished': 1.0, 'Semi-Furnished': 1.12, 'Fully Furnished': 1.30}
COND_MULT = {'Poor': 0.78, 'Good': 1.0, 'Excellent': 1.22}

CONFIG = {
    'Metro City':  {'sqft_rate': (8000, 22000), 'area': (600, 2500),  'n': 130},
    'Tier-1 City': {'sqft_rate': (4000, 12000), 'area': (700, 2500),  'n': 130},
    'Tier-2 City': {'sqft_rate': (2000,  7000), 'area': (800, 2800),  'n': 120},
    'Rural Area':  {'sqft_rate': ( 800,  3000), 'area': (700, 3500),  'n': 120},
}

rows = []
for loc, cfg in CONFIG.items():
    for _ in range(cfg['n']):
        area      = int(np.random.randint(cfg['area'][0], cfg['area'][1]))
        bedrooms  = int(np.random.choice([1,2,2,3,3,3,4,4,5], p=[0.05,.12,.12,.18,.18,.18,.10,.05,.02]))
        bathrooms = int(np.clip(np.random.randint(1, bedrooms + 2), 1, bedrooms + 1))
        raw_p = np.array([7,6,6,5,5,5,4,4,4,4,4,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,1,1,1,1,1], dtype=float)
        raw_p /= raw_p.sum()
        age   = int(np.random.choice(range(31), p=raw_p))
        parking   = int(np.random.choice([0,1,2,3], p=[0.25,.40,.25,.10]))
        furnishing= str(np.random.choice(FURN_OPTS, p=[0.35,.40,.25]))
        condition = str(np.random.choice(COND_OPTS, p=[0.20,.55,.25]))

        sqft_rate  = np.random.uniform(cfg['sqft_rate'][0], cfg['sqft_rate'][1])
        age_factor = max(0.65, 1.0 - age * 0.012)
        bed_factor = 1.0 + (bedrooms - 2) * 0.04
        bath_factor= 1.0 + (bathrooms - 1) * 0.02
        park_factor= 1.0 + parking * 0.025
        base_price = (area * sqft_rate * age_factor * bed_factor
                      * bath_factor * park_factor
                      * FURN_MULT[furnishing] * COND_MULT[condition])
        noise = np.random.uniform(0.92, 1.08)
        price = int(round(base_price * noise / 50000) * 50000)
        price = max(500000, price)

        rows.append({'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                     'age': age, 'location': loc, 'furnishing': furnishing,
                     'parking': parking, 'condition': condition, 'price': price})

df = pd.DataFrame(rows)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.csv')
df.to_csv(path, index=False)
print(f"Generated {len(df)} rows -> {path}")
print(df['location'].value_counts().to_string())
print(f"\nPrice range: Rs.{df['price'].min():,} - Rs.{df['price'].max():,}")
