
import pandas as pd
import os

# ==========================================
# 1. DEFINE FILE PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "raw", "city_day.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "cleaned_aqi.csv"
)


# ==========================================
# 2. DATA CLEANING FUNCTION
# ==========================================

def clean_dataset():

    print("\n" + "=" * 50)
    print("       BREATHSAFE AI DATA CLEANING")
    print("=" * 50)

    # Load dataset
    df = pd.read_csv(INPUT_PATH)

    print("\nOriginal dataset shape:", df.shape)

    original_records = len(df)

    # ======================================
    # 3. HANDLE DATE AND DUPLICATES
    # ======================================

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.drop_duplicates()

    print("After removing duplicates:", len(df))

    # ======================================
    # 4. REMOVE INVALID RECORDS
    # ======================================

    df = df.dropna(
        subset=["City", "Date", "AQI"]
    )

    df["AQI"] = pd.to_numeric(
        df["AQI"],
        errors="coerce"
    )

    df = df.dropna(subset=["AQI"])

    # Keep valid AQI values between 0 and 500
    df = df[
        (df["AQI"] >= 0) &
        (df["AQI"] <= 500)
    ]

    print("After removing invalid AQI:", len(df))

    # ======================================
    # 5. HANDLE MISSING POLLUTANT VALUES
    # ======================================

    pollutant_columns = [
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
        "Toluene",
        "Xylene"
    ]

    print("\nHandling missing pollutant values...")

    for column in pollutant_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df.groupby("City")[column].transform(
            lambda values: values.fillna(values.median())
        )

    # ======================================
    # 6. HANDLE MISSING AQI CATEGORIES
    # ======================================

    df["AQI_Bucket"] = df["AQI_Bucket"].fillna(
        "Unknown"
    )

    # ======================================
    # 7. SORT DATASET
    # ======================================

    df = df.sort_values(
        ["City", "Date"]
    )

    df = df.reset_index(drop=True)

    # ======================================
    # 8. SAVE CLEANED DATASET
    # ======================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ======================================
    # 9. DATA QUALITY REPORT
    # ======================================

    print("\n" + "=" * 50)
    print("          DATA CLEANING REPORT")
    print("=" * 50)

    print("\nOriginal records:", original_records)

    print("Cleaned records:", len(df))

    print(
        "Records removed:",
        original_records - len(df)
    )

    print(
        "Duplicate records remaining:",
        df.duplicated().sum()
    )

    print("\nRemaining missing values:")

    print(df.isnull().sum())

    print("\nFinal dataset shape:", df.shape)

    print("\nSaved cleaned dataset to:")

    print(OUTPUT_PATH)

    print("\n" + "=" * 50)
    print("       DATA CLEANING COMPLETED")
    print("=" * 50)




if __name__ == "__main__":

    clean_dataset()
