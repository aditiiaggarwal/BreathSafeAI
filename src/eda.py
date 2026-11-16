
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_aqi.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

plt.figure(figsize=(10, 5))
df["AQI"].hist(bins=50)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, "aqi_distribution.png"))
plt.close()

city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
city_aqi.plot(kind="bar")
plt.title("Average AQI by City")
plt.xlabel("City")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "city_aqi.png"))
plt.close()

monthly_aqi = df.groupby(df["Date"].dt.month)["AQI"].mean()

plt.figure(figsize=(10, 5))
monthly_aqi.plot(kind="line", marker="o")
plt.title("Monthly Average AQI")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(range(1, 13))
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_aqi.png"))
plt.close()

plt.figure(figsize=(9, 5))
df["AQI_Bucket"].value_counts().plot(kind="bar")
plt.title("AQI Category Distribution")
plt.xlabel("AQI Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "aqi_categories.png"))
plt.close()

high_aqi = df[df["AQI"] > 500]

print("EDA COMPLETED")
print("\nDATASET SHAPE:", df.shape)
print("\nTOP 10 MOST POLLUTED CITIES:")
print(city_aqi.head(10))
print("\nRECORDS WITH AQI ABOVE 500:", len(high_aqi))
print("\nCITIES WITH AQI ABOVE 500:")
print(high_aqi["City"].value_counts())
print("\nEXTREME AQI RECORDS:")
print(high_aqi[["City", "Date", "AQI", "AQI_Bucket"]].sort_values("AQI", ascending=False).head(10))
print("\nSAVED GRAPHS TO:", OUTPUT_DIR)
