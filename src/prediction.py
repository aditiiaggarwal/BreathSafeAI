
import pandas as pd
import joblib
import os
from datetime import date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "aqi_model.pkl")

saved_model = joblib.load(MODEL_PATH)
model = saved_model["model"]

def predict_aqi(city, pollutant_values, prediction_date=None):

    if prediction_date is None:
        prediction_date = date.today()

    prediction_date = pd.to_datetime(prediction_date)

    input_data = {
        "City": city,
        "PM2.5": pollutant_values.get("PM2.5"),
        "PM10": pollutant_values.get("PM10"),
        "NO": pollutant_values.get("NO"),
        "NO2": pollutant_values.get("NO2"),
        "NOx": pollutant_values.get("NOx"),
        "NH3": pollutant_values.get("NH3"),
        "CO": pollutant_values.get("CO"),
        "SO2": pollutant_values.get("SO2"),
        "O3": pollutant_values.get("O3"),
        "Benzene": pollutant_values.get("Benzene"),
        "Toluene": pollutant_values.get("Toluene"),
        "Month": prediction_date.month,
        "DayOfYear": prediction_date.day_of_year
    }

    input_df = pd.DataFrame([input_data])

    predicted_aqi = model.predict(input_df)[0]

    predicted_aqi = max(0, predicted_aqi)

    if predicted_aqi <= 50:
        category = "Good"
    elif predicted_aqi <= 100:
        category = "Satisfactory"
    elif predicted_aqi <= 200:
        category = "Moderate"
    elif predicted_aqi <= 300:
        category = "Poor"
    elif predicted_aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    return {
        "AQI": round(float(predicted_aqi), 2),
        "Category": category
    }


if __name__ == "__main__":

    sample = {
        "PM2.5": 45.0,
        "PM10": 80.0,
        "NO": 10.0,
        "NO2": 20.0,
        "NOx": 25.0,
        "NH3": 10.0,
        "CO": 1.0,
        "SO2": 15.0,
        "O3": 30.0,
        "Benzene": 2.0,
        "Toluene": 5.0
    }

    result = predict_aqi("Delhi", sample)

    print("BREATHSAFE AI PREDICTION")
    print("Predicted AQI:", result["AQI"])
    print("AQI Category:", result["Category"])

