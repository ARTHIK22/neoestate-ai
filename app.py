import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="NeoEstate AI", page_icon="🏠", layout="centered")

# ── FUTURISTIC CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
:root {
    --bg: #edf3f8;
    --panel: #f4f8fb;
    --panel-strong: #eaf1f6;
    --text: #1f2a37;
    --muted: #62758a;
    --primary: #7bb3c9;
    --primary-deep: #5a9ab0;
    --secondary: #a6d7c7;
    --warm: #f4c7b6;
    --success: #70c7a1;
    --danger: #ee8d8d;
    --shadow-dark: rgba(141, 160, 179, 0.18);
    --shadow-light: rgba(255, 255, 255, 0.95);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }

.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background:
        radial-gradient(circle at top, rgba(122, 178, 207, 0.20), transparent 40%),
        radial-gradient(circle at bottom right, rgba(166, 215, 199, 0.18), transparent 30%),
        radial-gradient(circle at bottom left, rgba(244, 199, 182, 0.18), transparent 30%);
    pointer-events: none; z-index: 0;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #edf1f5; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--primary), var(--secondary));
    border-radius: 999px;
}

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem; letter-spacing: 0.35em;
    color: var(--primary-deep); text-transform: uppercase;
    margin-bottom: 0.8rem;
    animation: pulse-text 2.5s ease-in-out infinite;
}
@keyframes pulse-text {
    0%,100% { opacity: 1; } 50% { opacity: 0.62; }
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem; font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg, var(--primary-deep), var(--secondary), var(--warm));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    text-shadow: 0 8px 25px rgba(122, 178, 207, 0.18);
}
.hero-sub {
    font-size: 0.95rem; color: var(--muted); font-weight: 400;
    letter-spacing: 0.04em;
}

/* ── Glass Card ── */
.glass-card {
    background: linear-gradient(145deg, #f8fbff, #edf3f7);
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    margin: 0.7rem 0;
    box-shadow: 10px 10px 24px var(--shadow-dark), -8px -8px 18px var(--shadow-light), inset 1px 1px 0 rgba(255,255,255,0.8);
    transition: all 0.3s ease;
}
.glass-card:hover {
    box-shadow: 12px 12px 26px var(--shadow-dark), -10px -10px 20px var(--shadow-light);
    transform: translateY(-1px);
}

/* ── Section Header ── */
.section-head {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 1rem;
}
.section-num {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; font-weight: 700;
    color: var(--primary-deep);
    background: rgba(123, 179, 201, 0.12);
    border: 1px solid rgba(123, 179, 201, 0.4);
    border-radius: 8px; padding: 0.15rem 0.45rem;
    letter-spacing: 0.1em;
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem; font-weight: 700;
    color: #4c667d; letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Stat Cards ── */
.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem;
    margin: 1rem 0;
}
.stat-card {
    background: linear-gradient(145deg, #f6f9fc, #edf4f8);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 18px; padding: 0.9rem 0.8rem;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 8px 8px 18px rgba(165, 180, 196, 0.14), -8px -8px 18px rgba(255,255,255,0.9);
    transition: all 0.3s ease;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 10px 10px 22px rgba(155, 171, 187, 0.18), -10px -10px 22px rgba(255,255,255,0.92);
}
.stat-label {
    font-size: 0.65rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.4rem;
    font-family: 'Orbitron', monospace;
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem; font-weight: 700; color: var(--primary-deep);
}
.stat-value.purple { color: #8aa5c5; }
.stat-value.pink   { color: #d89c8d; }
.stat-value.green  { color: var(--success); }

/* ── Price Result Box ── */
.price-result {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, rgba(123, 179, 201, 0.10), rgba(166, 215, 199, 0.12), rgba(244, 199, 182, 0.10));
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 22px; padding: 2.2rem 2rem;
    text-align: center; margin: 1.5rem 0;
    box-shadow: 14px 14px 26px rgba(148, 162, 176, 0.18), -12px -12px 22px rgba(255,255,255,0.86);
}
.price-result::after {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.30), transparent 35%, rgba(255,255,255,0.18));
}
.price-eyebrow {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; letter-spacing: 0.3em;
    color: var(--primary-deep); text-transform: uppercase;
    margin-bottom: 0.8rem; position: relative; z-index: 1;
}
.price-main {
    font-family: 'Orbitron', monospace;
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(135deg, var(--primary-deep), var(--secondary), var(--warm));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative; z-index: 1;
    margin: 0.3rem 0;
}
.price-range {
    font-size: 0.82rem; color: var(--muted);
    position: relative; z-index: 1; margin-top: 0.3rem;
}
.price-sqft {
    font-family: 'Orbitron', monospace;
    font-size: 0.78rem; color: var(--primary-deep);
    position: relative; z-index: 1;
    margin-top: 0.4rem; letter-spacing: 0.08em;
}

/* ── Tag Pills ── */
.tag-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.8rem 0; }
.tag-pill {
    font-family: 'Orbitron', monospace;
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 20px; padding: 0.2rem 0.65rem;
    font-size: 0.7rem; color: #c4b5fd;
    letter-spacing: 0.05em;
}

/* ── Info caption ── */
.neo-caption {
    font-size: 0.78rem; color: #6666aa;
    border-left: 2px solid #7c3aed44;
    padding-left: 0.6rem; margin: 0.4rem 0;
    font-style: italic;
}

/* ── Divider ── */
.neo-divider {
    height: 1px; margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.5), rgba(0,245,212,0.3), transparent);
}

/* ── Override Streamlit widget styles ── */
.stSelectbox label, .stRadio label, .stNumberInput label,
.stSlider label, .stCheckbox label {
    color: var(--muted) !important; font-size: 0.85rem !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: linear-gradient(145deg, #f8fbff, #edf3f8) !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    border-radius: 12px !important; color: var(--text) !important;
    box-shadow: inset 3px 3px 8px rgba(178, 191, 204, 0.12), inset -3px -3px 8px rgba(255,255,255,0.82) !important;
}
.stButton > button {
    width: 100%; padding: 0.9rem;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.9rem !important; font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    border-radius: 16px !important; color: #fff !important;
    box-shadow: 8px 8px 16px rgba(127, 153, 170, 0.18), -8px -8px 16px rgba(255,255,255,0.88) !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--primary-deep), var(--secondary)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 10px 10px 18px rgba(127, 153, 170, 0.22), -10px -10px 18px rgba(255,255,255,0.9) !important;
}
.stInfo { background: rgba(112, 199, 161, 0.10) !important; border-left-color: var(--success) !important; color: #355d52 !important; }
.stSuccess { background: rgba(114, 184, 183, 0.10) !important; border-left-color: var(--primary) !important; color: #355d52 !important; }
.stError { background: rgba(238, 141, 141, 0.10) !important; border-left-color: var(--danger) !important; color: #7d4f4e !important; }

div[data-testid="stExpander"] {
    background: linear-gradient(145deg, #f6f9fc, #ebf2f8) !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    border-radius: 16px !important;
    box-shadow: 8px 8px 18px rgba(155, 170, 186, 0.12), -8px -8px 18px rgba(255,255,255,0.9) !important;
}
</style>
""", unsafe_allow_html=True)

# ── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◈ AI-Powered Real Estate Intelligence ◈</div>
    <div class="hero-title">NeoEstate AI</div>
    <div class="hero-sub">Predict property value with machine learning precision</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD & CACHE MODEL ────────────────────────────────────────────────────────
FURNISHING_ORDER = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
CONDITION_ORDER  = ['Poor', 'Good', 'Excellent']

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(base_dir, "dataset.csv"))
    le_loc = LabelEncoder()
    data['location_encoded']   = le_loc.fit_transform(data['location'])
    data['furnishing_encoded'] = data['furnishing'].map({v: i for i, v in enumerate(FURNISHING_ORDER)})
    data['condition_encoded']  = data['condition'].map({v: i for i, v in enumerate(CONDITION_ORDER)})
    features = ['area','bedrooms','bathrooms','age','location_encoded','furnishing_encoded','parking','condition_encoded']
    X = data[features]; y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestRegressor(n_estimators=300, random_state=42)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    return mdl, le_loc, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

model, le_loc, r2, mae = load_model()

# ── MODEL STATS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-label">Engine</div>
        <div class="stat-value purple" style="font-size:0.75rem;">Random Forest</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Training Data</div>
        <div class="stat-value green">82</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">R² Score</div>
        <div class="stat-value">{r2:.2f}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Avg Error</div>
        <div class="stat-value pink" style="font-size:0.85rem;">₹{int(mae/1e5)/10}L</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)

# ── ① LOCATION ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <span class="section-num">01</span>
    <span class="section-title">📍 Location Matrix</span>
</div>""", unsafe_allow_html=True)

location = st.selectbox("Select Location Type", ("Metro City","Tier-1 City","Tier-2 City","Rural Area"), label_visibility="collapsed")

loc_desc = {
    "Metro City":  "🌆 Major metropolitan — premium demand & pricing",
    "Tier-1 City": "🏙️ Large city with strong infrastructure",
    "Tier-2 City": "🏘️ Mid-sized city with growing demand",
    "Rural Area":  "🌾 Outskirts / village — base land rates",
}
st.markdown(f'<div class="neo-caption">{loc_desc[location]}</div>', unsafe_allow_html=True)
st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)

# ── ② PROPERTY TYPE ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <span class="section-num">02</span>
    <span class="section-title">🏗️ Property Type</span>
</div>""", unsafe_allow_html=True)

property_type = st.radio("Type", ("🏞️ Vacant Land","🏠 Built House"), horizontal=True, label_visibility="collapsed")
if property_type == "🏞️ Vacant Land":
    st.markdown('<div class="neo-caption">ℹ️ Bedrooms / bathrooms / furnishing default to 0 for vacant land.</div>', unsafe_allow_html=True)

st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)

# ── ③ SIZE ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <span class="section-num">03</span>
    <span class="section-title">📐 Plot Dimensions</span>
</div>""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1: length = st.number_input("Length (ft)", min_value=1, value=35, step=1)
with c2: width  = st.number_input("Width  (ft)", min_value=1, value=30, step=1)
area = length * width
st.info(f"◈  Computed Area: **{area} sq ft**  ·  {area*0.0929:.1f} m²  ·  {area/272.25:.2f} Marla")

st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)

# ── ④–⑦ HOUSE DETAILS ────────────────────────────────────────────────────────
bedrooms = bathrooms = age = parking = 0
furnishing = "Unfurnished"
condition  = "Good"

if property_type == "🏠 Built House":
    st.markdown("""
    <div class="section-head">
        <span class="section-num">04</span>
        <span class="section-title">🏡 Interior Details</span>
    </div>""", unsafe_allow_html=True)

    c3, c4, c5 = st.columns(3)
    with c3: bedrooms  = st.number_input("🛏 Bedrooms",       min_value=1, max_value=10, value=3)
    with c4: bathrooms = st.number_input("🚿 Bathrooms",      min_value=1, max_value=8,  value=2)
    with c5: age       = st.number_input("📅 Age (yrs)",      min_value=0, max_value=50, value=5)

    st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-head">
        <span class="section-num">05</span>
        <span class="section-title">🛋️ Furnishing Status</span>
    </div>""", unsafe_allow_html=True)

    furnishing = st.select_slider("Furnishing", FURNISHING_ORDER, value="Semi-Furnished", label_visibility="collapsed")
    furn_captions = {
        "Unfurnished":     "🪑 Bare shell — no furniture or fittings",
        "Semi-Furnished":  "🛏️ Basic furniture, wardrobes, kitchen fittings",
        "Fully Furnished": "✨ Premium interiors, ACs, modular kitchen, appliances",
    }
    st.markdown(f'<div class="neo-caption">{furn_captions[furnishing]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-head">
        <span class="section-num">06</span>
        <span class="section-title">🚗 Parking Bays</span>
    </div>""", unsafe_allow_html=True)

    parking = st.radio("Parking", [0,1,2,3],
                       format_func=lambda x: ["❌ None","● 1 Bay","●● 2 Bays","●●● 3+ Bays"][x],
                       horizontal=True, label_visibility="collapsed")

    st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-head">
    <span class="section-num">07</span>
    <span class="section-title">🔧 Property Condition</span>
</div>""", unsafe_allow_html=True)

condition = st.select_slider("Condition", CONDITION_ORDER, value="Good", label_visibility="collapsed")
cond_captions = {
    "Poor":      "🔴 Requires major renovation or structural repairs",
    "Good":      "🟡 Well maintained and ready to move in",
    "Excellent": "🟢 Recently renovated — premium finishes, new build quality",
}
st.markdown(f'<div class="neo-caption">{cond_captions[condition]}</div>', unsafe_allow_html=True)

st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)

# ── PREDICT ───────────────────────────────────────────────────────────────────
if st.button("◈  INITIATE VALUATION SCAN  ◈", use_container_width=True):

    if area <= 0:
        st.error("❌ Invalid dimensions — area must be greater than zero.")
        st.stop()

    try:
        loc_enc  = le_loc.transform([location])[0]
    except Exception:
        st.error("❌ Unknown location selected.")
        st.stop()

    furn_enc = FURNISHING_ORDER.index(furnishing)
    cond_enc = CONDITION_ORDER.index(condition)

    pred       = model.predict([[area, bedrooms, bathrooms, age, loc_enc, furn_enc, parking, cond_enc]])[0]
    margin     = pred * 0.10
    price_low  = int(pred - margin)
    price_high = int(pred + margin)
    price_sqft = int(pred / area)

    # Helper for display
    def fmt(v):
        if v >= 1e7: return f"₹{v/1e7:.2f} Cr"
        elif v >= 1e5: return f"₹{v/1e5:.1f} L"
        else: return f"₹{int(v):,}"

    st.markdown(f"""
    <div class="price-result">
        <div class="price-eyebrow">◈ AI Valuation Complete ◈</div>
        <div class="price-main">{fmt(pred)}</div>
        <div class="price-range">Range: {fmt(price_low)} – {fmt(price_high)}</div>
        <div class="price-sqft">◈ {fmt(price_sqft)} per sq ft ◈</div>
    </div>
    """, unsafe_allow_html=True)

    # Tag pills
    tags = [f"📍 {location}", f"📐 {area} sqft", f"🛋️ {furnishing}",
            f"🔧 {condition}", f"🚗 {parking} parking"]
    if property_type == "🏠 Built House":
        tags += [f"🛏 {bedrooms}bed", f"🚿 {bathrooms}bath", f"📅 {age}yr"]
    pills = "".join(f'<span class="tag-pill">{t}</span>' for t in tags)
    st.markdown(f'<div class="tag-row">{pills}</div>', unsafe_allow_html=True)

    # Feature Importance
    with st.expander("⚡ Feature Importance Analysis"):
        labels = ['Area','Bedrooms','Bathrooms','House Age','Location','Furnishing','Parking','Condition']
        df_imp = pd.DataFrame({
            "Feature": labels,
            "Importance %": [round(v*100,1) for v in model.feature_importances_]
        }).sort_values("Importance %", ascending=False).reset_index(drop=True)
        st.dataframe(df_imp, use_container_width=True, hide_index=True)

    # Comparison table
    with st.expander("🔄 Furnishing × Condition Price Matrix"):
        rows = []
        for f in FURNISHING_ORDER:
            for c in CONDITION_ORDER:
                p = model.predict([[area, bedrooms, bathrooms, age,
                                    loc_enc, FURNISHING_ORDER.index(f),
                                    parking, CONDITION_ORDER.index(c)]])[0]
                rows.append({"Furnishing": f, "Condition": c, "Price": fmt(p)})
        pivot = pd.DataFrame(rows).pivot(index="Furnishing", columns="Condition", values="Price")
        st.dataframe(pivot[["Poor","Good","Excellent"]], use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#444466; font-size:0.72rem; font-family:'Orbitron',monospace; letter-spacing:0.1em;">
    NEOESTATE AI · POWERED BY RANDOM FOREST · ESTIMATES ARE INDICATIVE ONLY
</div>
""", unsafe_allow_html=True)
