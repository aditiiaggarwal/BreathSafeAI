
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/city_day.csv")

# 1. Number of unique cities
print("\nTOTAL CITIES:")
print(df["City"].nunique())

print("\nCITY NAMES:")
print(sorted(df["City"].unique()))

# 2. AQI category distribution
print("\nAQI CATEGORY DISTRIBUTION:")
print(df["AQI_Bucket"].value_counts())

# 3. Date range
df["Date"] = pd.to_datetime(df["Date"])

print("\nDATE RANGE:")
print("Starting Date:", df["Date"].min())
print("Ending Date:", df["Date"].max())

# 4. Highest AQI record
print("\nHIGHEST AQI RECORD:")
max_aqi_index = df["AQI"].idxmax()
print(df.loc[max_aqi_index])

# 5. Lowest AQI record
print("\nLOWEST AQI RECORD:")
min_aqi_index = df["AQI"].idxmin()
print(df.loc[min_aqi_index])

# 6. AQI statistics
print("\nAQI STATISTICS:")
print(df["AQI"].describe())

# 7. Missing AQI values by city
print("\nMISSING AQI VALUES BY CITY:")
print(df.groupby("City")["AQI"].apply(
    lambda x: x.isnull().sum()
).sort_values(ascending=False))

# 8. Check extreme AQI values
print("\nAQI VALUES ABOVE 500:")
print(df[df["AQI"] > 500][["City", "Date", "AQI", "AQI_Bucket"]])
