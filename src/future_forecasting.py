
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "cleaned_aqi.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR, "future_aqi_models.pkl"
)

RESULT_PATH = os.path.join(
    REPORT_DIR, "forecast_model_comparison.csv"
)

HORIZONS = [1, 3, 7]

NUMERIC_FEATURES = [
    "AQI_today",
    "AQI_lag_1",
    "AQI_lag_3",
    "AQI_lag_7",
    "AQI_lag_14",
    "AQI_rolling_7",
    "AQI_rolling_14",
    "TargetMonth",
    "TargetDayOfYear"
]

CATEGORICAL_FEATURES = ["City"]


def load_dataset():

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["AQI"] = pd.to_numeric(
        df["AQI"], errors="coerce"
    )

    df = df.dropna(
        subset=["City", "Date", "AQI"]
    )

    df = df[
        (df["AQI"] >= 0) &
        (df["AQI"] <= 500)
    ]

    df = df.drop_duplicates(
        subset=["City", "Date"],
        keep="last"
    )

    df = df.sort_values(
        ["City", "Date"]
    )

    print("\nFUTURE AQI FORECASTING")
    print("=" * 55)
    print("Historical records:", len(df))
    print("Cities:", df["City"].nunique())
    print(
        "Date range:",
        df["Date"].min().date(),
        "to",
        df["Date"].max().date()
    )

    return df


def create_forecasting_dataset(df, horizon):

    all_city_data = []

    for city, city_df in df.groupby("City"):

        city_df = city_df.set_index("Date")

        actual_aqi = city_df["AQI"]

        daily_dates = pd.date_range(
            start=actual_aqi.index.min(),
            end=actual_aqi.index.max(),
            freq="D"
        )

        daily_aqi = actual_aqi.reindex(daily_dates)

        # Fill historical input features only
        filled_aqi = daily_aqi.ffill(limit=7)

        features = pd.DataFrame(index=daily_dates)

        features["City"] = city

        features["AQI_today"] = filled_aqi

        features["AQI_lag_1"] = filled_aqi.shift(1)
        features["AQI_lag_3"] = filled_aqi.shift(3)
        features["AQI_lag_7"] = filled_aqi.shift(7)
        features["AQI_lag_14"] = filled_aqi.shift(14)

        features["AQI_rolling_7"] = (
            filled_aqi.rolling(7, min_periods=3).mean()
        )

        features["AQI_rolling_14"] = (
            filled_aqi.rolling(14, min_periods=5).mean()
        )

        target_dates = daily_dates + pd.Timedelta(
            days=horizon
        )

        features["TargetMonth"] = target_dates.month
        features["TargetDayOfYear"] = target_dates.dayofyear

        # Actual future AQI values only
        features["TargetAQI"] = (
            daily_aqi.reindex(target_dates).to_numpy()
        )

        features["TargetDate"] = target_dates

        features = features.dropna(
            subset=["TargetAQI"]
        )

        all_city_data.append(
            features.reset_index(drop=True)
        )

    if not all_city_data:
        raise ValueError(
            f"No forecasting records available for {horizon}-day horizon."
        )

    return pd.concat(
        all_city_data,
        ignore_index=True
    )


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(
        actual, predicted
    )

    return mae, rmse, r2


def train_forecasting_model(dataset, horizon):

    dataset = dataset.sort_values(
        "TargetDate"
    ).copy()

    # Both methods must be evaluated on records
    # where today's AQI is actually available.
    dataset = dataset.dropna(
        subset=["AQI_today"]
    )

    # Split using unique dates so records from
    # the same date cannot enter both sets.
    unique_dates = np.sort(
        dataset["TargetDate"].unique()
    )

    split_index = int(
        len(unique_dates) * 0.8
    )

    if split_index <= 0 or split_index >= len(unique_dates):
        raise ValueError(
            "Not enough unique dates for chronological evaluation."
        )

    cutoff_date = unique_dates[split_index]

    train_data = dataset[
        dataset["TargetDate"] < cutoff_date
    ]

    test_data = dataset[
        dataset["TargetDate"] >= cutoff_date
    ]

    feature_columns = (
        CATEGORICAL_FEATURES + NUMERIC_FEATURES
    )

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["TargetAQI"]
    y_test = test_data["TargetAQI"]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Not enough records for training and testing."
        )

    # Preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    # Random Forest model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print(f"\nTRAINING {horizon}-DAY FORECAST MODEL")
    print("-" * 55)

    model.fit(X_train, y_train)

    # Random Forest predictions
    predictions = model.predict(X_test)

    rf_mae, rf_rmse, rf_r2 = calculate_metrics(
        y_test, predictions
    )

    # Persistence baseline:
    # Future AQI = Today's AQI
    baseline_predictions = X_test["AQI_today"]

    baseline_mae, baseline_rmse, baseline_r2 = calculate_metrics(
        y_test, baseline_predictions
    )

    # Percentage improvement in MAE
    improvement = (
        (baseline_mae - rf_mae) / baseline_mae
    ) * 100 if baseline_mae != 0 else 0

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nRANDOM FOREST RESULTS")
    print("MAE:", round(rf_mae, 2))
    print("RMSE:", round(rf_rmse, 2))
    print("R2 Score:", round(rf_r2, 4))

    print("\nPERSISTENCE BASELINE RESULTS")
    print("MAE:", round(baseline_mae, 2))
    print("RMSE:", round(baseline_rmse, 2))
    print("R2 Score:", round(baseline_r2, 4))

    print("\nBASELINE COMPARISON")
    print("MAE Improvement:", round(improvement, 2), "%")

    if rf_mae < baseline_mae:
        print("Result: Random Forest performs better.")
    elif rf_mae > baseline_mae:
        print("Result: Persistence baseline performs better.")
    else:
        print("Result: Both models have equal MAE.")

    result = {
        "Horizon": f"{horizon} Day",
        "Training Records": len(X_train),
        "Testing Records": len(X_test),

        "RF MAE": round(rf_mae, 2),
        "Baseline MAE": round(baseline_mae, 2),

        "RF RMSE": round(rf_rmse, 2),
        "Baseline RMSE": round(baseline_rmse, 2),

        "RF R2": round(rf_r2, 4),
        "Baseline R2": round(baseline_r2, 4),

        "MAE Improvement (%)": round(improvement, 2)
    }

    return model, result


def main():

    df = load_dataset()

    trained_models = {}
    results = []

    for horizon in HORIZONS:

        dataset = create_forecasting_dataset(
            df, horizon
        )

        print(
            f"\n{horizon}-day dataset records:",
            len(dataset)
        )

        model, result = train_forecasting_model(
            dataset, horizon
        )

        trained_models[horizon] = model

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    # Save all trained forecasting models
    joblib.dump(
        {
            "models": trained_models,
            "horizons": HORIZONS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES
        },
        MODEL_PATH
    )

    print("\n" + "=" * 70)
    print("FINAL FORECAST MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print("\nFORECAST MODELS SAVED SUCCESSFULLY")
    print("Model:", MODEL_PATH)
    print("Report:", RESULT_PATH)


if __name__ == "__main__":

    main()
