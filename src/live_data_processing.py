
import os
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIVE_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "live_measurements.csv"
)

# Molecular weights (g/mol), used for gas unit conversion
MOLECULAR_WEIGHTS = {
    "no": 30.006,
    "no2": 46.0055,
    "nox": 46.0055,
    "nh3": 17.031,
    "co": 28.01,
    "so2": 64.066,
    "o3": 48.00,
    "benzene": 78.11,
    "toluene": 92.14
}

# OpenAQ pollutant names mapped to Kaggle model feature names
POLLUTANT_MAPPING = {
    "pm2.5": "PM2.5",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no": "NO",
    "nitric oxide": "NO",
    "no2": "NO2",
    "nitrogen dioxide": "NO2",
    "nox": "NOx",
    "nitrogen oxides": "NOx",
    "nh3": "NH3",
    "ammonia": "NH3",
    "co": "CO",
    "carbon monoxide": "CO",
    "so2": "SO2",
    "sulfur dioxide": "SO2",
    "sulphur dioxide": "SO2",
    "o3": "O3",
    "ozone": "O3",
    "benzene": "Benzene",
    "toluene": "Toluene"
}

MODEL_POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene"
]


def normalize_pollutant(name):
    """Convert OpenAQ pollutant names into model feature names."""

    if pd.isna(name):
        return None

    name = str(name).strip().lower()

    return POLLUTANT_MAPPING.get(name)


def convert_unit(value, unit, pollutant):
    """
    Convert OpenAQ measurements into units compatible
    with the historical Kaggle dataset.

    PM and most gases: micrograms/m3
    CO: mg/m3
    """

    try:
        value = float(value)
    except (ValueError, TypeError):
        return np.nan

    if not np.isfinite(value) or value < 0:
        return np.nan

    unit = str(unit).strip().lower()

# Fix incorrectly encoded unit symbols
    unit = unit.replace("Â", "")
    unit = unit.replace("â", "")

# Standardize unit symbols
    unit = unit.replace("μ", "µ")
    unit = unit.replace("³", "3")
    unit = unit.replace("^", "")

    # Particulate matter is stored in µg/m3
    if pollutant in ["PM2.5", "PM10"]:

        if unit in ["µg/m3", "ug/m3", "µg/m^3", "ug/m^3"]:
            return value

        if unit in ["mg/m3", "mg/m^3"]:
            return value * 1000

        return np.nan

    # Kaggle CO values are generally represented in mg/m3
    if pollutant == "CO":

        if unit in ["mg/m3", "mg/m^3"]:
            return value

        if unit in ["µg/m3", "ug/m3", "µg/m^3", "ug/m^3"]:
            return value / 1000

        target_conversion = 1000

    else:
        # Other gaseous pollutants are represented in µg/m3
        if unit in ["µg/m3", "ug/m3", "µg/m^3", "ug/m^3"]:
            return value

        if unit in ["mg/m3", "mg/m^3"]:
            return value * 1000

        target_conversion = 1

    # Convert ppm / ppb using ideal gas approximation:
    # 25°C and 1 atmosphere.
    molecular_weight = MOLECULAR_WEIGHTS.get(
        pollutant.lower()
    )

    if molecular_weight is None:
        return np.nan

    if unit == "ppm":
        concentration = (
            value * molecular_weight / 24.45
        ) * 1000

    elif unit == "ppb":
        concentration = (
            value * molecular_weight / 24.45
        )

    else:
        return np.nan

    return concentration / target_conversion


def load_live_data(file_path=LIVE_DATA_PATH):
    """Load live measurements saved by the OpenAQ pipeline."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Live data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "pollutant",
        "value",
        "unit",
        "timestamp"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def process_live_data(df):
    """
    Clean live OpenAQ measurements and prepare pollutant
    values for the existing AQI prediction model.
    """

    if df is None or df.empty:
        raise ValueError("No live measurements available.")

    df = df.copy()

    # Map pollutant names
    df["model_pollutant"] = df["pollutant"].apply(
        normalize_pollutant
    )

    # Remove unsupported pollutant names
    df = df.dropna(subset=["model_pollutant"])

    if df.empty:
        raise ValueError(
            "No supported pollutants found in live data."
        )

    # Convert timestamps
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    # Convert measurements into model-compatible units
    df["clean_value"] = df.apply(
        lambda row: convert_unit(
            row["value"],
            row["unit"],
            row["model_pollutant"]
        ),
        axis=1
    )

    df = df.dropna(
        subset=["timestamp", "clean_value"]
    )

    # Initialize every model pollutant as missing.
    # Do NOT replace unavailable readings with zero.
    pollutant_values = {
        pollutant: np.nan
        for pollutant in MODEL_POLLUTANTS
    }

    # Use the latest available reading for each pollutant
    for pollutant in MODEL_POLLUTANTS:

        pollutant_data = df[
            df["model_pollutant"] == pollutant
        ].sort_values("timestamp")

        if not pollutant_data.empty:

            latest_reading = pollutant_data.iloc[-1]

            pollutant_values[pollutant] = float(
                latest_reading["clean_value"]
            )

    # Print processing report
    print("\nBREATHSAFE AI - LIVE DATA PROCESSING")
    print("=" * 50)

    for pollutant, value in pollutant_values.items():

        if pd.isna(value):
            print(f"{pollutant}: Not available")
        else:
            print(f"{pollutant}: {value:.4f}")

    available = sum(
        not pd.isna(value)
        for value in pollutant_values.values()
    )

    print("-" * 50)
    print(f"Available pollutants: {available}/{len(MODEL_POLLUTANTS)}")

    return pollutant_values


def prepare_live_data(file_path=LIVE_DATA_PATH):

    df = load_live_data(file_path)

    pollutant_values = process_live_data(df)

    return pollutant_values


if __name__ == "__main__":

    try:
        live_pollutants = prepare_live_data()

        print("\nLive pollutant data processed successfully.")

    except (FileNotFoundError, ValueError) as error:
        print("\nProcessing Error:", error)
