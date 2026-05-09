# House Price Prediction — Model Training & Evaluation
# Run this script standalone to see model metrics and charts.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
FURNISHING_ORDER = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
CONDITION_ORDER  = ['Poor', 'Good', 'Excellent']

FEATURES = ['area', 'bedrooms', 'bathrooms', 'age',
            'location_encoded', 'furnishing_encoded', 'parking', 'condition_encoded']

FEATURE_LABELS = ['Area', 'Bedrooms', 'Bathrooms', 'House Age',
                  'Location', 'Furnishing', 'Parking', 'Condition']

# ──────────────────────────────────────────────
# Load Dataset
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))

print(f"\n{'='*60}")
print(f"  Dataset: {len(data)} rows × {len(data.columns)} columns")
print(f"{'='*60}")
print(data.head(5).to_string(index=False))
print()

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
le_loc = LabelEncoder()
data['location_encoded']   = le_loc.fit_transform(data['location'])
data['furnishing_encoded'] = data['furnishing'].map({v: i for i, v in enumerate(FURNISHING_ORDER)})
data['condition_encoded']  = data['condition'].map({v: i for i, v in enumerate(CONDITION_ORDER)})

print("Location encoding :", dict(zip(le_loc.classes_, le_loc.transform(le_loc.classes_))))
print("Furnishing encoding:", {v: i for i, v in enumerate(FURNISHING_ORDER)})
print("Condition encoding :", {v: i for i, v in enumerate(CONDITION_ORDER)})
print()

X = data[FEATURES]
y = data['price']

# ──────────────────────────────────────────────
# Train / Test Split
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}\n")

# ──────────────────────────────────────────────
# Model Comparison
# ──────────────────────────────────────────────
models = {
    "Linear Regression":  LinearRegression(),
    "Random Forest":      RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = []
print(f"{'='*65}")
print(f"{'Model':<25} {'MAE':>14} {'RMSE':>16} {'R²':>8}")
print(f"{'='*65}")

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"{name:<25} ₹{mae:>12,.0f}  ₹{rmse:>12,.0f}  {r2:>7.4f}")
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

print(f"{'='*65}\n")

# ──────────────────────────────────────────────
# Best Model — Random Forest
# ──────────────────────────────────────────────
best_model = models["Random Forest"]
y_pred_rf  = best_model.predict(X_test)

# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("House Price Prediction — Model Analysis", fontsize=14, fontweight='bold')

# 1. Actual vs Predicted
axes[0].scatter(y_test / 1e6, y_pred_rf / 1e6,
                color='steelblue', alpha=0.75, edgecolors='k', linewidths=0.4, s=60)
mn = min(y_test.min(), y_pred_rf.min()) / 1e6
mx = max(y_test.max(), y_pred_rf.max()) / 1e6
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
axes[0].set_xlabel("Actual Price (₹ Millions)")
axes[0].set_ylabel("Predicted Price (₹ Millions)")
axes[0].set_title("Actual vs Predicted (Random Forest)")
axes[0].legend()

# 2. Feature Importance
importances = best_model.feature_importances_
sorted_idx  = np.argsort(importances)
colors_bar  = ['#a78bfa' if i == sorted_idx[-1] else '#667eea' for i in range(len(importances))]
axes[1].barh([FEATURE_LABELS[i] for i in sorted_idx],
             importances[sorted_idx],
             color=[colors_bar[i] for i in sorted_idx])
axes[1].set_xlabel("Importance Score")
axes[1].set_title("Feature Importance (Random Forest)")

# 3. Model Comparison
r2_scores   = [r["R2"] for r in results]
model_names = [r["Model"] for r in results]
bar_colors  = ['#4c72b0', '#a78bfa', '#55a868']
bars = axes[2].bar(model_names, r2_scores, color=bar_colors, width=0.5)
axes[2].set_ylim(0, 1.1)
axes[2].set_ylabel("R² Score")
axes[2].set_title("Model Comparison (R² Score)")
for bar, v in zip(bars, r2_scores):
    axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                 f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
out_path = os.path.join(BASE_DIR, "model_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Chart saved → {out_path}\n")

# ──────────────────────────────────────────────
# Interactive Prediction
# ──────────────────────────────────────────────
print(f"{'='*60}")
print("  CUSTOM PREDICTION")
print(f"{'='*60}")
print(f"  Locations  : {list(le_loc.classes_)}")
print(f"  Furnishing : {FURNISHING_ORDER}")
print(f"  Condition  : {CONDITION_ORDER}")
print(f"{'='*60}\n")

area      = float(input("Enter area (sq ft)            : "))
bedrooms  = int(input("Enter number of bedrooms      : "))
bathrooms = int(input("Enter number of bathrooms     : "))
age       = int(input("Enter age of house (years)    : "))
location  = input(f"Enter location                : ").strip()
furnishing= input(f"Enter furnishing              : ").strip()
parking   = int(input("Enter parking spots (0-3)     : "))
condition = input(f"Enter condition               : ").strip()

# Safe encoding
try:
    loc_enc  = le_loc.transform([location])[0]
except ValueError:
    print(f"⚠️  Unknown location. Defaulting to 'Rural Area'.")
    loc_enc  = le_loc.transform(["Rural Area"])[0]

furn_enc  = FURNISHING_ORDER.index(furnishing) if furnishing in FURNISHING_ORDER else 1
cond_enc  = CONDITION_ORDER.index(condition)   if condition  in CONDITION_ORDER  else 1

input_vec = [[area, bedrooms, bathrooms, age, loc_enc, furn_enc, parking, cond_enc]]
predicted = best_model.predict(input_vec)[0]
margin    = predicted * 0.10

print(f"\n{'─'*45}")
print(f"  💰 Predicted Price  : ₹ {int(predicted):>15,}")
print(f"  📉 Low Estimate     : ₹ {int(predicted - margin):>15,}")
print(f"  📈 High Estimate    : ₹ {int(predicted + margin):>15,}")
print(f"  📐 Price per sq ft  : ₹ {int(predicted / area):>15,}")
print(f"{'─'*45}\n")
