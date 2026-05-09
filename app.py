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
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #05050f;
    color: #e0e0ff;
}
.stApp { background: #05050f; }

/* ── Animated cyber background ── */
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, #1a0533 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, #001a33 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a1f; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 4px; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem; letter-spacing: 0.35em;
    color: #00f5d4; text-transform: uppercase;
    margin-bottom: 0.8rem;
    animation: pulse-text 2s ease-in-out infinite;
}
@keyframes pulse-text {
    0%,100% { opacity: 1; } 50% { opacity: 0.6; }
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem; font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg, #00f5d4 0%, #7c3aed 50%, #f0abfc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 30px #7c3aed88);
}
.hero-sub {
    font-size: 0.95rem; color: #8888bb; font-weight: 300;
    letter-spacing: 0.04em;
}

/* ── Glass Card ── */
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(124,58,237,0.06) 100%);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 24px rgba(124,58,237,0.1), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    border-color: rgba(124,58,237,0.6);
    box-shadow: 0 4px 40px rgba(124,58,237,0.25), inset 0 1px 0 rgba(255,255,255,0.08);
}

/* ── Section Header ── */
.section-head {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 1rem;
}
.section-num {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; font-weight: 700;
    color: #00f5d4;
    background: rgba(0,245,212,0.08);
    border: 1px solid rgba(0,245,212,0.3);
    border-radius: 6px; padding: 0.15rem 0.4rem;
    letter-spacing: 0.1em;
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem; font-weight: 700;
    color: #c4b5fd; letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Stat Cards ── */
.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem;
    margin: 1rem 0;
}
.stat-card {
    background: linear-gradient(135deg, rgba(0,245,212,0.04), rgba(124,58,237,0.08));
    border: 1px solid rgba(0,245,212,0.2);
    border-radius: 12px; padding: 0.9rem 0.8rem;
    text-align: center; position: relative; overflow: hidden;
    transition: all 0.3s;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00f5d4, transparent);
}
.stat-card:hover {
    border-color: rgba(0,245,212,0.5);
    box-shadow: 0 0 20px rgba(0,245,212,0.1);
    transform: translateY(-2px);
}
.stat-label {
    font-size: 0.65rem; color: #6666aa;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.4rem;
    font-family: 'Orbitron', monospace;
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem; font-weight: 700; color: #00f5d4;
}
.stat-value.purple { color: #c4b5fd; }
.stat-value.pink   { color: #f0abfc; }
.stat-value.green  { color: #34d399; }

/* ── Price Result Box ── */
.price-result {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, rgba(0,245,212,0.06), rgba(124,58,237,0.12), rgba(240,171,252,0.06));
    border: 1px solid rgba(124,58,237,0.5);
    border-radius: 20px; padding: 2.2rem 2rem;
    text-align: center; margin: 1.5rem 0;
    box-shadow: 0 0 60px rgba(124,58,237,0.2), 0 0 120px rgba(0,245,212,0.05);
    animation: glow-pulse 3s ease-in-out infinite;
}
@keyframes glow-pulse {
    0%,100% { box-shadow: 0 0 60px rgba(124,58,237,0.2), 0 0 120px rgba(0,245,212,0.05); }
    50%      { box-shadow: 0 0 80px rgba(124,58,237,0.35), 0 0 160px rgba(0,245,212,0.08); }
}
.price-result::before {
    content: ''; position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent 0%, rgba(124,58,237,0.03) 25%, transparent 50%);
    animation: spin 8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.price-result::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #7c3aed, #00f5d4, #7c3aed, transparent);
}
.price-eyebrow {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; letter-spacing: 0.3em;
    color: #00f5d4; text-transform: uppercase;
    margin-bottom: 0.8rem; position: relative; z-index: 1;
}
.price-main {
    font-family: 'Orbitron', monospace;
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(135deg, #00f5d4, #7c3aed, #f0abfc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative; z-index: 1;
    filter: drop-shadow(0 0 20px #7c3aed88);
    margin: 0.3rem 0;
}
.price-range {
    font-size: 0.82rem; color: #8888bb;
    position: relative; z-index: 1; margin-top: 0.3rem;
}
.price-sqft {
    font-family: 'Orbitron', monospace;
    font-size: 0.78rem; color: #00f5d4;
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
    color: #a0a0cc !important; font-size: 0.85rem !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    border-radius: 10px !important; color: #e0e0ff !important;
}
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    border-radius: 10px !important; color: #e0e0ff !important;
}
.stButton > button {
    width: 100%; padding: 0.85rem;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.9rem !important; font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: 1px solid rgba(124,58,237,0.6) !important;
    border-radius: 12px !important; color: #fff !important;
    box-shadow: 0 0 30px rgba(124,58,237,0.4) !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9333ea, #6366f1) !important;
    box-shadow: 0 0 50px rgba(124,58,237,0.7) !important;
    transform: translateY(-2px) !important;
}
.stInfo { background: rgba(0,245,212,0.06) !important; border-left-color: #00f5d4 !important; }
.stSuccess { background: rgba(52,211,153,0.08) !important; border-left-color: #34d399 !important; }
.stError   { background: rgba(239,68,68,0.08) !important; border-left-color: #ef4444 !important; }

div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 12px !important;
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
