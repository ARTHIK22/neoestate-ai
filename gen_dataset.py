"""
Downloads Bengaluru House Price dataset (13,320 rows) from GitHub,
cleans it, and maps it to the NeoEstate AI schema:
  area, bedrooms, bathrooms, age, location, furnishing, parking, condition, price
"""
import urllib.request, pandas as pd, numpy as np, io, ssl, os, re

np.random.seed(42)

# ── 1. Download ────────────────────────────────────────────────────────────────
print("Downloading Bengaluru House Data...")
ctx = ssl.create_default_context()
ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
url = "https://raw.githubusercontent.com/Lokeshrathi/Bangalore-House-Prices/master/Bengaluru_House_Data.csv"
raw = urllib.request.urlopen(url, context=ctx).read()
df  = pd.read_csv(io.BytesIO(raw))
print(f"  Downloaded: {len(df)} rows")

# ── 2. Parse total_sqft ───────────────────────────────────────────────────────
def parse_sqft(val):
    val = str(val).strip()
    if re.match(r'^\d+\.?\d*$', val):
        return float(val)
    # Range like "1000-1500" → midpoint
    m = re.match(r'^([\d.]+)\s*-\s*([\d.]+)$', val)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    # "10 Sq. Meter" type — skip
    return np.nan

df['area'] = df['total_sqft'].apply(parse_sqft)

# ── 3. Parse bedrooms from size ───────────────────────────────────────────────
def parse_beds(val):
    if pd.isnull(val):
        return np.nan
    m = re.search(r'(\d+)', str(val))
    return int(m.group(1)) if m else np.nan

df['bedrooms'] = df['size'].apply(parse_beds)

# ── 4. Bathrooms ──────────────────────────────────────────────────────────────
df['bathrooms'] = df['bath'].fillna(df['bedrooms'])

# ── 5. Price: dataset is in Lakhs → convert to INR ───────────────────────────
# price column = Lakhs (e.g. 72 means Rs 72,00,000)
df['price_inr'] = df['price'] * 1e5

# ── 6. Location tier (map Bangalore neighbourhood → Metro City) ───────────────
# All rows are Bangalore → Metro City
df['location'] = 'Metro City'

# ── 7. Synthetic but realistic ancillary columns ──────────────────────────────
n = len(df)

# age: newer buildings are more common in Bangalore (tech boom 2000-2020)
df['age'] = np.random.choice(range(0, 26),
                              size=n,
                              p=np.array([6,6,5,5,5,5,4,4,4,4,4,3,3,3,3,
                                           3,3,2,2,2,2,2,2,1,1,1], dtype=float) /
                                np.array([6,6,5,5,5,5,4,4,4,4,4,3,3,3,3,
                                           3,3,2,2,2,2,2,2,1,1,1], dtype=float).sum())

# furnishing: realistic distribution for Bangalore
furn_probs   = [0.30, 0.45, 0.25]
df['furnishing'] = np.random.choice(['Unfurnished','Semi-Furnished','Fully Furnished'],
                                     size=n, p=furn_probs)

# parking: city apartments often have 1 slot; expensive ones have 2
df['parking'] = np.random.choice([0,1,2,3], size=n, p=[0.20, 0.50, 0.25, 0.05])

# condition: most are Good/Excellent in Bangalore's newer stock
df['condition'] = np.random.choice(['Poor','Good','Excellent'], size=n, p=[0.12, 0.60, 0.28])

# ── 8. Clean and filter ───────────────────────────────────────────────────────
df = df[['area','bedrooms','bathrooms','age','location','furnishing','parking','condition','price_inr']]
df = df.rename(columns={'price_inr': 'price'})

# Drop nulls
df = df.dropna(subset=['area','bedrooms','bathrooms','price'])

# Sanity filters
df = df[
    (df['area']      >= 300)  & (df['area']      <= 10000) &
    (df['bedrooms']  >= 1)    & (df['bedrooms']  <= 10)    &
    (df['bathrooms'] >= 1)    & (df['bathrooms'] <= 10)    &
    (df['price']     >= 5e5)  & (df['price']     <= 5e8)
]

df['bedrooms']  = df['bedrooms'].astype(int)
df['bathrooms'] = df['bathrooms'].astype(int)
df['age']       = df['age'].astype(int)
df['parking']   = df['parking'].astype(int)
df['price']     = df['price'].astype(int)
# Round price to nearest 50k for cleanliness
df['price'] = (df['price'] / 50000).round() * 50000
df['price'] = df['price'].clip(lower=500000).astype(int)

print(f"  After cleaning: {len(df)} rows")
print(f"  Price range: Rs.{df['price'].min():,} - Rs.{df['price'].max():,}")

# ── 9. Augment with other city tiers ─────────────────────────────────────────
# Tier-1 Cities (Pune, Hyderabad) ≈ 60-75% of Bangalore prices
# Tier-2 Cities (Jaipur, Lucknow) ≈ 30-45% of Bangalore prices
# Rural Areas                      ≈ 10-18% of Bangalore prices
TIER_CONFIG = {
    'Tier-1 City': {'scale': (0.58, 0.78), 'n': 1500},
    'Tier-2 City': {'scale': (0.28, 0.45), 'n': 1200},
    'Rural Area':  {'scale': (0.08, 0.18), 'n':  800},
}

extra_frames = [df]  # start with real Bangalore data

for tier, cfg in TIER_CONFIG.items():
    sample = df.sample(n=cfg['n'], replace=True, random_state=hash(tier) % (2**31))
    sample = sample.copy()
    sample['location'] = tier
    scale = np.random.uniform(cfg['scale'][0], cfg['scale'][1], size=len(sample))
    # Add noise so prices aren't just scaled copies
    noise = np.random.uniform(0.90, 1.10, size=len(sample))
    sample['price'] = (sample['price'] * scale * noise / 50000).round() * 50000
    sample['price'] = sample['price'].clip(lower=500000).astype(int)
    # Tier-2 / Rural tend to be less furnished
    if tier in ('Tier-2 City', 'Rural Area'):
        sample['furnishing'] = np.random.choice(
            ['Unfurnished','Semi-Furnished','Fully Furnished'],
            size=len(sample), p=[0.50, 0.35, 0.15])
    extra_frames.append(sample)

final = pd.concat(extra_frames, ignore_index=True)
final = final.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"\n  Location breakdown:")
print(final['location'].value_counts().to_string())
print(f"  Total rows: {len(final)}")
print(f"  Price range: Rs.{final['price'].min():,} - Rs.{final['price'].max():,}")

# ── 10. Save ──────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.csv')
final.to_csv(out, index=False)
print(f"\nSaved {len(final)} rows -> {out}")

