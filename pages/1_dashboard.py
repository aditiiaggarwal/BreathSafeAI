"""BreathSafe AI - Dashboard page.
Location: <src>/pages/1_Dashboard.py  (next to app.py's folder structure)
"""
import os
import sys
import joblib
import requests
import pandas as pd
import streamlit as st

# pages/ is one level below src/, so go up one level to import your modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from prediction import predict_aqi                      # loads aqi_model.pkl
from live_data_processing import process_live_data
from openaq_api import (get_locations, get_latest_measurements,
                        GEOCODING_URL)

BASE_DIR = os.path.dirname(SRC_DIR)                     # project root
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_aqi.csv")
LIVE_PATH = os.path.join(BASE_DIR, "data", "live_measurements.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "models", "future_aqi_models.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "forecast_model_comparison.csv")

st.set_page_config(page_title="BreathSafe AI - Dashboard", layout="wide")


# ---------- cached loaders ----------
@st.cache_data
def load_history():
    df = pd.read_csv(CLEAN_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["City", "Date", "AQI"])
    return df.drop_duplicates(["City", "Date"]).sort_values(["City", "Date"])


@st.cache_resource
def load_forecast_models():
    return joblib.load(FORECAST_PATH)


def category_color(cat):
    return {"Good": "🟢", "Satisfactory": "🟡", "Moderate": "🟠",
            "Poor": "🔴", "Very Poor": "🟣", "Severe": "⚫"}.get(cat, "⚪")


# ---------- forecast: build latest feature row for a city ----------
def latest_features(city_df, city, horizon):
    s = city_df.set_index("Date")["AQI"]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    d = s.reindex(idx).ffill(limit=7)
    if len(d) < 15:
        return None, None
    last = idx[-1]
    target = last + pd.Timedelta(days=horizon)
    row = {
        "City": city,
        "AQI_today": d.iloc[-1],
        "AQI_lag_1": d.iloc[-2],
        "AQI_lag_3": d.iloc[-4],
        "AQI_lag_7": d.iloc[-8],
        "AQI_lag_14": d.iloc[-15],
        "AQI_rolling_7": d.iloc[-7:].mean(),
        "AQI_rolling_14": d.iloc[-14:].mean(),
        "TargetMonth": target.month,
        "TargetDayOfYear": target.dayofyear,
    }
    return pd.DataFrame([row]), target


# ---------- UI ----------
st.title("BreathSafe AI Dashboard")
st.caption("AQI exploration, live estimation and short-term forecasting")

tab1, tab2, tab3 = st.tabs(["📊 Historical", "📡 Live AQI", "🔮 Forecast"])

# ===== TAB 1: HISTORICAL =====
with tab1:
    df = load_history()
    city = st.selectbox("City", sorted(df["City"].unique()), key="hist_city")
    cdf = df[df["City"] == city]

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(cdf))
    c2.metric("Average AQI", round(cdf["AQI"].mean(), 1))
    c3.metric("Max AQI", int(cdf["AQI"].max()))

    st.subheader("AQI over time")
    st.line_chart(cdf.set_index("Date")["AQI"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Monthly average")
        st.bar_chart(cdf.groupby(cdf["Date"].dt.month)["AQI"].mean())
    with col_b:
        st.subheader("Category distribution")
        st.bar_chart(cdf["AQI_Bucket"].value_counts())

    st.subheader("Most polluted cities")
    st.bar_chart(df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10))

# ===== TAB 2: LIVE =====
with tab2:
    query = st.text_input("Search any city", placeholder="e.g. Jhajjar, Haryana")

    if query:
        places = []
        try:
            r = requests.get(GEOCODING_URL, params={"name": query, "count": 10,
                             "language": "en", "format": "json"}, timeout=20)
            places = r.json().get("results", [])
        except requests.RequestException as e:
            st.error(f"Geocoding failed: {e}")

        if not places:
            st.warning("City not found.")
        else:
            labels = [f"{p['name']}, {p.get('admin1', '')}, {p.get('country', '')}"
                      for p in places]
            pick = st.selectbox("Select location", range(len(places)),
                                format_func=lambda i: labels[i])
            place = places[pick]

            if st.button("Get live AQI estimate"):
                with st.spinner("Finding nearest station..."):
                    stations = get_locations(place["latitude"], place["longitude"],
                                             place["country_code"])
                    stations = [s for s in stations if s.get("sensors")
                                and s["calculated_distance"] != float("inf")]

                if not stations:
                    st.error("No usable monitoring stations nearby.")
                else:
                    station = stations[0]
                    with st.spinner("Fetching measurements..."):
                        get_latest_measurements(station, place["name"])

                    live = pd.read_csv(LIVE_PATH)
                    # Use only this station's rows (CSV accumulates all runs)
                    live = live[(live["location_id"] == station["id"]) &
                                (live["city"] == place["name"])]
                    try:
                        values = process_live_data(live)
                        result = predict_aqi(place["name"], values)

                        st.success(f"{category_color(result['Category'])} "
                                   f"Predicted AQI: **{result['AQI']}** "
                                   f"({result['Category']})")
                        st.write(f"Station: **{station['name']}** "
                                 f"({station['calculated_distance']:.1f} km away)")
                        st.dataframe(pd.DataFrame(
                            {"Pollutant": list(values.keys()),
                             "Value": [None if pd.isna(v) else round(v, 3)
                                       for v in values.values()]}),
                            hide_index=True)
                        st.map(pd.DataFrame({
                            "lat": [place["latitude"], station["coordinates"]["latitude"]],
                            "lon": [place["longitude"], station["coordinates"]["longitude"]]}))
                        st.caption("Model-based estimate, not an official AQI. Missing "
                                   "pollutants are filled with historical medians, and "
                                   "cities outside the training data reduce accuracy.")
                    except ValueError as e:
                        st.error(str(e))

# ===== TAB 3: FORECAST =====
with tab3:
    df = load_history()
    if not os.path.exists(FORECAST_PATH):
        st.warning("Run future_forecasting.py first to create the forecast models.")
    else:
        bundle = load_forecast_models()
        city = st.selectbox("City", sorted(df["City"].unique()), key="fc_city")
        cdf = df[df["City"] == city]
        rows = []
        for h in bundle["horizons"]:
            X, target = latest_features(cdf, city, h)
            if X is None:
                st.warning("Not enough history for this city.")
                break
            pred = float(bundle["models"][h].predict(X)[0])
            rows.append({"Horizon": f"{h} day", "Date": target.date(),
                         "Forecast AQI": round(max(pred, 0), 1)})
        if rows:
            st.info(f"Forecasts start from the last available record: "
                    f"**{cdf['Date'].max().date()}**")
            out = pd.DataFrame(rows)
            st.dataframe(out, hide_index=True)
            st.bar_chart(out.set_index("Horizon")["Forecast AQI"])

        if os.path.exists(REPORT_PATH):
            st.subheader("Model vs persistence baseline")
            st.dataframe(pd.read_csv(REPORT_PATH), hide_index=True)