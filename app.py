import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BreathSafe AI", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_aqi.csv")

# ---------- styling ----------
st.markdown("""
<style>
.hero {
    padding: 2.2rem 2rem; border-radius: 18px; margin-bottom: 1rem;
    background: linear-gradient(135deg, #0f766e 0%, #0e7490 50%, #1e40af 100%);
    color: white;
}
.hero h1 { font-size: 2.6rem; margin: 0; }
.hero p  { font-size: 1.1rem; opacity: .9; margin: .4rem 0 0 0; }
.card {
    padding: 1.1rem 1.2rem; border-radius: 14px;
    border: 1px solid rgba(128,128,128,.25);
    height: 100%;
}
.card h4 { margin: 0 0 .4rem 0; }
.badge {
    display: inline-block; padding: .35rem .9rem; border-radius: 999px;
    color: white; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["City", "Date", "AQI"])


AQI_LEVELS = [
    ("Good", 0, 50, "#16a34a", "Air is clean. Enjoy outdoor activities."),
    ("Satisfactory", 51, 100, "#84cc16", "Acceptable. Very sensitive people may feel mild effects."),
    ("Moderate", 101, 200, "#f59e0b", "Sensitive groups may have breathing discomfort."),
    ("Poor", 201, 300, "#ef4444", "Prolonged exposure can affect most people."),
    ("Very Poor", 301, 400, "#a855f7", "Respiratory illness likely on long exposure. Limit outdoor time."),
    ("Severe", 401, 500, "#1f2937", "Serious health impact. Avoid outdoor exertion."),
]


def level_for(aqi):
    for name, low, high, color, advice in AQI_LEVELS:
        if aqi <= high:
            return name, color, advice
    return AQI_LEVELS[-1][0], AQI_LEVELS[-1][3], AQI_LEVELS[-1][4]


# ---------- hero ----------
st.markdown("""
<div class="hero">
  <h1>BreathSafe AI</h1>
  <p>Understand the air you breathe. Live estimates, trends and short-term AQI forecasts.</p>
</div>
""", unsafe_allow_html=True)

b1, b2, _ = st.columns([1.3, 1.3, 5])
with b1:
    st.page_link("pages/1_Dashboard.py", label="Open Dashboard", icon="📊")
with b2:
    st.page_link("pages/1_Dashboard.py", label="Live AQI Check", icon="📡")

# ---------- headline stats ----------
try:
    df = load_data()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Cities covered", df["City"].nunique())
    s2.metric("Records analysed", f"{len(df):,}")
    s3.metric("Years of data", f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}")
    s4.metric("Forecast horizons", "1, 3, 7 days")
except FileNotFoundError:
    df = None
    st.warning("Cleaned dataset not found. Run src/data_cleaning.py first.")

st.divider()

# ---------- interactive section ----------
view = st.radio(
    "Explore",
    ["AQI Guide", "City Snapshot", "How it works"],
    horizontal=True,
    label_visibility="collapsed",
)

if view == "AQI Guide":
    st.subheader("What does your AQI mean?")
    value = st.slider("Slide to check an AQI value", 0, 500, 120)
    name, color, advice = level_for(value)
    c1, c2 = st.columns([1, 3])
    c1.markdown(f'<span class="badge" style="background:{color}">{name}</span>',
                unsafe_allow_html=True)
    c2.write(advice)

    cols = st.columns(len(AQI_LEVELS))
    for col, (n, lo, hi, clr, _) in zip(cols, AQI_LEVELS):
        col.markdown(
            f'<div style="background:{clr};color:white;padding:.6rem;'
            f'border-radius:10px;text-align:center;font-size:.85rem">'
            f'<b>{n}</b><br>{lo}-{hi}</div>',
            unsafe_allow_html=True)

elif view == "City Snapshot":
    st.subheader("Quick city snapshot")
    if df is None:
        st.info("Data not available.")
    else:
        city = st.selectbox("Pick a city", sorted(df["City"].unique()))
        cdf = df[df["City"] == city].sort_values("Date")
        recent = cdf.tail(1).iloc[0]
        avg = cdf["AQI"].mean()
        name, color, advice = level_for(recent["AQI"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Latest recorded AQI", int(recent["AQI"]),
                  delta=f"{recent['AQI'] - avg:+.0f} vs city average",
                  delta_color="inverse")
        m2.metric("City average", round(avg, 1))
        m3.markdown(f'<span class="badge" style="background:{color}">{name}</span>',
                    unsafe_allow_html=True)

        show_trend = st.toggle("Show last 90 days trend", value=True)
        if show_trend:
            st.line_chart(cdf.set_index("Date")["AQI"].tail(90))
        st.caption(f"Latest record date: {recent['Date'].date()} (historical dataset)")

else:
    st.subheader("How BreathSafe AI works")
    show_details = st.toggle("Show technical details", value=False)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>1. Collect</h4>'
                    'Live readings come from the nearest OpenAQ station. '
                    'Historical data covers Indian cities.</div>', unsafe_allow_html=True)
        if show_details:
            st.caption("OpenAQ v3 API, geocoding via Open-Meteo, unit conversion to ug/m3.")
    with c2:
        st.markdown('<div class="card"><h4>2. Predict</h4>'
                    'A machine learning model estimates AQI from pollutant levels.</div>',
                    unsafe_allow_html=True)
        if show_details:
            st.caption("Random Forest regression with imputation and one-hot city encoding.")
    with c3:
        st.markdown('<div class="card"><h4>3. Forecast</h4>'
                    'Recent AQI patterns are used to project the next 1, 3 and 7 days.</div>',
                    unsafe_allow_html=True)
        if show_details:
            st.caption("Lag and rolling-mean features, compared against a persistence baseline.")

st.divider()
st.caption("BreathSafe AI | Model-based estimates, not official AQI readings.")