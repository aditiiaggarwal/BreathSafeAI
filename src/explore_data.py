
import pandas as pd

# Load the dataset
file_path = "data/raw/city_day.csv"
df = pd.read_csv(file_path)

# Display basic information
print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATASET INFORMATION:")
df.info()

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())
