
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "city_day.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.drop_duplicates()
df = df.dropna(subset=["City", "Date", "AQI"])
df = df[(df["AQI"] >= 0) & (df["AQI"] <= 500)]

print("DATASET PREPARED")
print("Records available:", len(df))

df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear

numeric_features = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Month", "DayOfYear"
]

categorical_features = ["City"]

feature_columns = categorical_features + numeric_features

X = df[feature_columns]
y = df["AQI"]

df = df.sort_values("Date")
X = X.loc[df.index]
y = y.loc[df.index]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_name = None
best_mae = float("inf")

print("\nMODEL TRAINING STARTED")

for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("\n", name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    if mae < best_mae:
        best_mae = mae
        best_model = pipeline
        best_name = name

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(REPORT_DIR, "model_comparison.csv"),
    index=False
)

joblib.dump(
    {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_columns
    },
    os.path.join(MODEL_DIR, "aqi_model.pkl")
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nBEST MODEL:", best_name)
print("\nMODEL SAVED SUCCESSFULLY")
print("Location:", os.path.join(MODEL_DIR, "aqi_model.pkl"))
