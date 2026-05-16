# pyrefly: ignore [missing-import]
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

FURNISHING_ORDER = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
CONDITION_ORDER  = ['Poor', 'Good', 'Excellent']

# City registry: tier -> [{name, rate (INR/sqft)}]
CITIES = {
    "Metro City": [
        {"name": "Mumbai",    "rate": 18500},
        {"name": "Delhi",     "rate": 15000},
        {"name": "Bangalore", "rate": 12500},
        {"name": "Chennai",   "rate": 10000},
        {"name": "Kolkata",   "rate": 8500},
    ],
    "Tier-1 City": [
        {"name": "Pune",        "rate": 8000},
        {"name": "Hyderabad",   "rate": 7500},
        {"name": "Ahmedabad",   "rate": 6500},
        {"name": "Chandigarh",  "rate": 7200},
        {"name": "Kochi",       "rate": 6800},
        {"name": "Surat",       "rate": 5800},
    ],
    "Tier-2 City": [
        {"name": "Jaipur",     "rate": 5000},
        {"name": "Lucknow",    "rate": 4500},
        {"name": "Nagpur",     "rate": 4200},
        {"name": "Bhopal",     "rate": 4000},
        {"name": "Indore",     "rate": 4800},
        {"name": "Coimbatore", "rate": 4500},
        {"name": "Vadodara",   "rate": 4200},
        {"name": "Patna",      "rate": 3800},
    ],
    "Rural Area": [
        {"name": "Rural - UP",            "rate": 1500},
        {"name": "Rural - Rajasthan",     "rate": 1200},
        {"name": "Rural - Maharashtra",   "rate": 1800},
        {"name": "Rural - Punjab",        "rate": 2000},
        {"name": "Rural - MP",            "rate": 1300},
    ],
}


def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(base_dir, "dataset.csv"))

    le_loc = LabelEncoder()
    data['location_encoded']   = le_loc.fit_transform(data['location'])
    data['furnishing_encoded'] = data['furnishing'].map(
        {v: i for i, v in enumerate(FURNISHING_ORDER)})
    data['condition_encoded']  = data['condition'].map(
        {v: i for i, v in enumerate(CONDITION_ORDER)})

    features = ['area', 'bedrooms', 'bathrooms', 'age',
                'location_encoded', 'furnishing_encoded', 'parking', 'condition_encoded']
    X = data[features]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    estimators = {
        'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'Linear Regression': LinearRegression(),
    }

    results = {}
    for name, mdl in estimators.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        results[name] = {
            'model': mdl,
            'r2':    round(r2_score(y_test, y_pred), 4),
            'mae':   int(mean_absolute_error(y_test, y_pred)),
        }

    primary = results['Random Forest']
    return estimators, results, le_loc, primary['r2'], primary['mae'], len(X_train)


print("Training models…")
MODELS, RESULTS, le_loc, R2, MAE, N_TRAIN = load_models()
print(f"  Done. R²={R2:.4f}  MAE=Rs.{MAE:,}  n_train={N_TRAIN}")


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/stats')
def stats():
    return jsonify({'r2': R2, 'mae': MAE, 'n_train': N_TRAIN})


@app.route('/api/cities')
def cities():
    return jsonify(CITIES)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json(force=True)
        if not d:
            return jsonify({'error': 'Empty request body'}), 400

        # ── Validate ──────────────────────────────────────────────────────
        try:
            area      = float(d['area'])
            bedrooms  = int(d['bedrooms'])
            bathrooms = int(d['bathrooms'])
            age       = int(d['age'])
            parking   = int(d['parking'])
        except (KeyError, ValueError, TypeError) as exc:
            return jsonify({'error': f'Invalid numeric field: {exc}'}), 400

        if area <= 0:
            return jsonify({'error': 'Area must be greater than 0'}), 400
        if bedrooms < 0 or bathrooms < 0:
            return jsonify({'error': 'Bedrooms/bathrooms cannot be negative'}), 400
        if not (0 <= age <= 100):
            return jsonify({'error': 'Age must be between 0 and 100'}), 400
        if not (0 <= parking <= 3):
            return jsonify({'error': 'Parking must be 0–3'}), 400

        location  = d.get('location', '')
        furnishing = d.get('furnishing', '')
        condition  = d.get('condition', '')

        if location not in le_loc.classes_:
            return jsonify({'error': f'Unknown location: {location}'}), 400
        if furnishing not in FURNISHING_ORDER:
            return jsonify({'error': f'Invalid furnishing: {furnishing}'}), 400
        if condition not in CONDITION_ORDER:
            return jsonify({'error': f'Invalid condition: {condition}'}), 400

        # ── Encode ────────────────────────────────────────────────────────
        loc_enc  = int(le_loc.transform([location])[0])
        furn_enc = FURNISHING_ORDER.index(furnishing)
        cond_enc = CONDITION_ORDER.index(condition)
        inp = [[area, bedrooms, bathrooms, age, loc_enc, furn_enc, parking, cond_enc]]

        # ── All model predictions ─────────────────────────────────────────
        model_preds = {}
        for name, mdl in MODELS.items():
            p = float(mdl.predict(inp)[0])
            model_preds[name] = {
                'price': max(0, int(p)),
                'r2':    RESULTS[name]['r2'],
                'mae':   RESULTS[name]['mae'],
            }

        price = model_preds['Random Forest']['price']

        # ── Furnishing × Condition matrix (RF) ───────────────────────────
        matrix = {}
        for f in FURNISHING_ORDER:
            matrix[f] = {}
            for c in CONDITION_ORDER:
                p = float(MODELS['Random Forest'].predict([[
                    area, bedrooms, bathrooms, age,
                    loc_enc, FURNISHING_ORDER.index(f),
                    parking, CONDITION_ORDER.index(c)
                ]])[0])
                matrix[f][c] = max(0, int(p))

        # ── Feature importance (RF) ───────────────────────────────────────
        labels = ['Area', 'Bedrooms', 'Bathrooms', 'House Age',
                  'Location', 'Furnishing', 'Parking', 'Condition']
        importance = [
            {'feature': l, 'value': round(v * 100, 1)}
            for l, v in zip(labels, MODELS['Random Forest'].feature_importances_)
        ]
        importance.sort(key=lambda x: x['value'], reverse=True)

        return jsonify({
            'price':      price,
            'price_low':  max(0, int(price * 0.90)),
            'price_high': int(price * 1.10),
            'price_sqft': int(price / area) if area > 0 else 0,
            'matrix':     matrix,
            'importance': importance,
            'models':     model_preds,
        })

    except Exception as exc:
        return jsonify({'error': f'Server error: {exc}'}), 500


if __name__ == '__main__':
    print("\n  NeoEstate AI  --  http://localhost:5000\n")
    app.run(debug=True, port=5000)
