# Decision Tree Model Training with Hyperparameter Tuning
# Includes:
# - GridSearchCV
# - Expanded feature set from LLM ETL
# - Evaluation metrics (MAE, RMSE, MAPE, Bias)
# - Permutation importance
# - Train on past, test on today

import os, io, json, logging, traceback
import numpy as np
import pandas as pd

from google.cloud import storage

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.inspection import permutation_importance

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "structured/outputs")
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")

logging.basicConfig(level="INFO")

# -------------------- IO --------------------
def _read_csv_from_gcs(client, bucket, key):
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def _write_json_to_gcs(client, bucket, key, obj):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(json.dumps(obj, indent=2))


# -------------------- CLEANING --------------------
def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


# -------------------- TRAIN --------------------
def run_once():

    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---- Time split ----
    df["scraped_at_dt"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["date"] = df["scraped_at_dt"].dt.date

    unique_dates = sorted(df["date"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "not enough time splits"}

    today = unique_dates[-1]

    train_df = df[df["date"] < today].copy()
    test_df  = df[df["date"] == today].copy()

    # ---- Target cleaning ----
    df["price_num"] = _clean_numeric(df["price"])
    train_df = train_df[train_df["price_num"].notna()]

    # ---- Feature set (UPDATED) ----
    cat_cols = [
        "make", "model",
        "fuel", "transmission",
        "color", "city", "state", "zip_code"
    ]

    num_cols = ["year", "mileage"]

    # Clean numerics
    for col in num_cols:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    features = cat_cols + num_cols

    # ---- Preprocessing ----
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    model = DecisionTreeRegressor(random_state=42)

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    # ---- Hyperparameter tuning (REQUIRED) ----
    param_grid = {
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_leaf": [5, 10, 20],
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    X_train = train_df[features]
    y_train = train_df["price_num"]

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # ---- Evaluation on today ----
    results = {
        "status": "ok",
        "best_params": grid.best_params_,
        "metrics": {},
        "feature_importance": None
    }

    if not test_df.empty:
        X_test = test_df[features]
        y_test = test_df["price_num"]

        preds = best_model.predict(X_test)

        mask = y_test.notna()
        if mask.any():
            y_true = y_test[mask]
            y_pred = preds[mask]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            bias = np.mean(y_pred - y_true)

            results["metrics"] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MAPE": float(mape),
                "Bias": float(bias)
            }

    # ---- Permutation Importance (REQUIRED) ----
    perm = permutation_importance(
        best_model,
        X_train,
        y_train,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    feature_names = best_model.named_steps["pre"].get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean
    }).sort_values(by="importance", ascending=False)

    results["feature_importance"] = importance_df.head(50).to_dict(orient="records")

    # ---- Save artifacts ----
    out_key = f"{OUTPUT_PREFIX}/training_results.json"
    _write_json_to_gcs(client, GCS_BUCKET, out_key, results)

    return results


# -------------------- HTTP ENTRY --------------------
def train_dt_http(request):
    try:
        result = run_once()
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error(traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
