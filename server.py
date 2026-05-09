# pyrefly: ignore [missing-import]
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

FURNISHING_ORDER = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
CONDITION_ORDER  = ['Poor', 'Good', 'Excellent']

# ── Load & train model at startup ──────────────────────────────────────────
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(base_dir, "dataset.csv"))

    le_loc = LabelEncoder()
    data['location_encoded']   = le_loc.fit_transform(data['location'])
    data['furnishing_encoded'] = data['furnishing'].map({v: i for i, v in enumerate(FURNISHING_ORDER)})
    data['condition_encoded']  = data['condition'].map({v: i for i, v in enumerate(CONDITION_ORDER)})

    features = ['area','bedrooms','bathrooms','age',
                'location_encoded','furnishing_encoded','parking','condition_encoded']
    X = data[features]; y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestRegressor(n_estimators=300, random_state=42)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    return mdl, le_loc, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

model, le_loc, r2, mae = load_model()

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/stats')
def stats():
    return jsonify({'r2': round(r2, 4), 'mae': int(mae)})

@app.route('/api/predict', methods=['POST'])
def predict():
    d = request.json
    area      = float(d['area'])
    bedrooms  = int(d['bedrooms'])
    bathrooms = int(d['bathrooms'])
    age       = int(d['age'])
    parking   = int(d['parking'])
    location  = d['location']
    furnishing= d['furnishing']
    condition = d['condition']

    loc_enc  = int(le_loc.transform([location])[0])
    furn_enc = FURNISHING_ORDER.index(furnishing)
    cond_enc = CONDITION_ORDER.index(condition)

    price = float(model.predict([[area, bedrooms, bathrooms, age,
                                   loc_enc, furn_enc, parking, cond_enc]])[0])

    # Comparison matrix
    matrix = {}
    for f in FURNISHING_ORDER:
        matrix[f] = {}
        for c in CONDITION_ORDER:
            p = float(model.predict([[area, bedrooms, bathrooms, age,
                                       loc_enc, FURNISHING_ORDER.index(f),
                                       parking, CONDITION_ORDER.index(c)]])[0])
            matrix[f][c] = int(p)

    # Feature importance
    labels = ['Area','Bedrooms','Bathrooms','House Age','Location','Furnishing','Parking','Condition']
    importance = [{'feature': l, 'value': round(v*100, 1)}
                  for l, v in zip(labels, model.feature_importances_)]
    importance.sort(key=lambda x: x['value'], reverse=True)

    return jsonify({
        'price':      int(price),
        'price_low':  int(price * 0.90),
        'price_high': int(price * 1.10),
        'price_sqft': int(price / area) if area > 0 else 0,
        'matrix':     matrix,
        'importance': importance,
    })

if __name__ == '__main__':
    print("\n  NeoEstate AI -- running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
