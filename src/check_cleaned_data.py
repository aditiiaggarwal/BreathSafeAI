
import pandas as pd

df = pd.read_csv("data/processed/cleaned_aqi.csv")

print("DATASET SHAPE:", df.shape)

print("\nMISSING VALUE PERCENTAGE:")
print((df.isnull().mean() * 100).round(2))

print("\nMISSING VALUES CITY-WISE:")
print(df.groupby("City")["Xylene"].apply(lambda x: x.isnull().sum()).sort_values(ascending=False).head(10))

print("\nAQI CATEGORY DISTRIBUTION:")
print(df["AQI_Bucket"].value_counts())

print("\nAQI STATISTICS:")
print(df["AQI"].describe())
