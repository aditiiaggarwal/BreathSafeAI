
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/city_day.csv")

# 1. Check AQI values above 500
high_aqi = df[df["AQI"] > 500]

print("\nRECORDS WITH AQI ABOVE 500:")
print("Total:", len(high_aqi))
print("Percentage:", round(len(high_aqi) / df["AQI"].notna().sum() * 100, 2), "%")

# 2. AQI categories of these records
print("\nAQI CATEGORIES ABOVE 500:")
print(high_aqi["AQI_Bucket"].value_counts(dropna=False))

# 3. Cities with the most high AQI records
print("\nCITY-WISE HIGH AQI RECORDS:")
print(high_aqi["City"].value_counts())

# 4. Top 20 highest AQI records
print("\nTOP 20 HIGHEST AQI RECORDS:")
print(
    high_aqi[
        ["City", "Date", "AQI", "AQI_Bucket"]
    ].sort_values("AQI", ascending=False).head(20)
)

# 5. Check repeated city-date combinations
print("\nDUPLICATE CITY-DATE RECORDS:")
print(df.duplicated(subset=["City", "Date"]).sum())

# 6. Check AQI distribution percentiles
print("\nAQI PERCENTILES:")
print(df["AQI"].quantile([0.50, 0.75, 0.90, 0.95, 0.99, 1.00]))

# 7. Missing AQI records
print("\nMISSING AQI RECORDS:")
print(df["AQI"].isnull().sum())
