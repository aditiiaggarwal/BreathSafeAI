
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_data_processing import prepare_live_data
from prediction import predict_aqi


LIVE_CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "live_measurements.csv"
)


def run_live_prediction():

    print("\nBREATHSAFE AI - LIVE AQI PREDICTION")
    print("=" * 55)

    # Read city and station information
    df = pd.read_csv(LIVE_CSV_PATH)

    if df.empty:
        print("No live measurements available.")
        return

    city = df["city"].dropna().iloc[0]

    station = df["station"].dropna().iloc[0]

    distance = df["distance_km"].dropna().iloc[0]

    print("City:", city)
    print("Monitoring Station:", station)
    print("Station Distance:", distance, "km")

    # Process live pollutant measurements
    print("\nProcessing live measurements...")

    pollutant_values = prepare_live_data()

    # Run Random Forest prediction
    print("\nRunning AI prediction...")

    result = predict_aqi(
        city,
        pollutant_values
    )

    # Display results
    print("\n" + "=" * 55)
    print("BREATHSAFE AI PREDICTION RESULT")
    print("=" * 55)

    print("City:", city)
    print("Predicted AQI:", result["AQI"])
    print("AQI Category:", result["Category"])

    print("-" * 55)

    print("Station:", station)
    print("Distance:", distance, "km")

    print("\nIMPORTANT:")
    print("This is a model-based estimate using nearby")
    print("station measurements, not an official AQI reading.")

    print("The city was not present in historical training data.")
    print("Interpret this prediction with caution.")

    print("=" * 55)


if __name__ == "__main__":

    try:
        run_live_prediction()

    except Exception as error:
        print("\nPrediction Error:", error)

