import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

def mean_absolute_percentage_error_custom(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def train_hybrid_pipeline(features_csv_path="data/processed/features.csv", model_dir="model"):
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(features_csv_path):
        print(f"[ERROR] Features file missing at: {features_csv_path}")
        sys.exit(1)
        
    print(f"Loading features dataset from: {features_csv_path}")
    df = pd.read_csv(features_csv_path)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["datetime"] = df["timestamp"]
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["timestamp"] = df["datetime"]
        
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Chronological Split (Split date Oct 1, 2023 or last 3 months)
    max_date = df["timestamp"].max()
    split_date = pd.to_datetime("2023-10-01") if max_date >= pd.to_datetime("2023-10-01") else max_date - pd.DateOffset(months=3)
    
    train_df = df[df["timestamp"] < split_date].copy()
    test_df = df[df["timestamp"] >= split_date].copy()

    print("\n--- CHRONOLOGICAL SPLIT RANGES ---")
    print(f"Train Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df)} rows)")
    print(f"Test Period : {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} rows)")

    genesis_time = train_df["timestamp"].min()

    trend_features = ["time_step"]
    xgb_features = [
        "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_holiday",
        "temperature_2m", "relative_humidity_2m", "windspeed_10m",
        "heat_index", "ac_load_proxy", "precipitation", "is_raining",
        "rain_rolling_3h", "rain_rolling_6h",
        "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h"
    ]
    xgb_features = [f for f in xgb_features if f in df.columns]
    target = "load"

    # Stage 1: Base Linear Trend Model
    print("\n--- STAGE 1: FITTING BASE RIDGE LINEAR TREND MODEL ---")
    trend_model = Ridge(alpha=1.0)
    trend_model.fit(train_df[trend_features], train_df[target])

    train_trend = trend_model.predict(train_df[trend_features])
    test_trend = trend_model.predict(test_df[trend_features])
    train_residuals = train_df[target] - train_trend

    # Stage 2: TimeSeriesSplit Cross Validation on Residuals
    print("\n--- STAGE 2: TIME-SERIES CROSS-VALIDATION (5-FOLD) ON RESIDUALS ---")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes, cv_rmses, cv_mapes = [], [], []

    for fold, (t_idx, v_idx) in enumerate(tscv.split(train_df)):
        X_tr, y_tr = train_df.iloc[t_idx][xgb_features], train_residuals.iloc[t_idx]
        X_va, y_va = train_df.iloc[v_idx][xgb_features], train_df.iloc[v_idx][target]
        val_trend_slice = train_trend[v_idx]

        fold_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
        fold_model.fit(X_tr, y_tr)
        
        preds = val_trend_slice + fold_model.predict(X_va)
        mae = mean_absolute_error(y_va, preds)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        mape = mean_absolute_percentage_error_custom(y_va, preds)
        
        cv_maes.append(mae)
        cv_rmses.append(rmse)
        cv_mapes.append(mape)
        print(f"Fold {fold + 1} MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW | MAPE: {mape:.2f}%")

    print(f"Mean CV MAE: {np.mean(cv_maes):.2f} MW | Mean RMSE: {np.mean(cv_rmses):.2f} MW | Mean MAPE: {np.mean(cv_mapes):.2f}%")

    # Stage 3: Train Final Residual Model
    print("\n--- STAGE 3: TRAINING FINAL XGBOOST RESIDUAL MODEL ---")
    final_xgb = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    final_xgb.fit(train_df[xgb_features], train_residuals)

    # Holdout Test Benchmark
    test_preds = test_trend + final_xgb.predict(test_df[xgb_features])
    naive_preds = test_df["lag_24h"]

    xgb_mae = mean_absolute_error(test_df[target], test_preds)
    xgb_rmse = np.sqrt(mean_squared_error(test_df[target], test_preds))
    xgb_mape = mean_absolute_percentage_error_custom(test_df[target], test_preds)

    naive_mae = mean_absolute_error(test_df[target], naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(test_df[target], naive_preds))
    naive_mape = mean_absolute_percentage_error_custom(test_df[target], naive_preds)

    print("\n=======================================================")
    print("           MODEL BENCHMARK COMPARISON TABLE           ")
    print("=======================================================")
    print(f"{'Model':<25} | {'MAE (MW)':<10} | {'RMSE (MW)':<10} | {'MAPE (%)':<10}")
    print("-" * 63)
    print(f"{'24h Naive Baseline':<25} | {naive_mae:<10.2f} | {naive_rmse:<10.2f} | {naive_mape:<10.2f}%")
    print(f"{'Hybrid Ridge + XGBoost':<25} | {xgb_mae:<10.2f} | {xgb_rmse:<10.2f} | {xgb_mape:<10.2f}%")
    print("=======================================================")

    # Feature Importances
    importances = final_xgb.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': xgb_features,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("\n--- TOP 10 FEATURE IMPORTANCES ---")
    print(feat_imp_df.head(10))

    # Save unified artifact dictionary
    artifact_path = os.path.join(model_dir, "xgb_model.pkl")
    artifact_payload = {
        "trend_model": trend_model,
        "xgb_model": final_xgb,
        "model": final_xgb,
        "trend_features": trend_features,
        "xgb_features": xgb_features,
        "feature_cols": xgb_features,
        "genesis_time": genesis_time,
        "metrics": {
            "naive": {"mae": float(naive_mae), "rmse": float(naive_rmse), "mape": float(naive_mape)},
            "xgb": {"mae": float(xgb_mae), "rmse": float(xgb_rmse), "mape": float(xgb_mape)},
            "cv_xgb": {"mae": float(np.mean(cv_maes)), "rmse": float(np.mean(cv_rmses)), "mape": float(np.mean(cv_mapes))}
        },
        "feature_importances": feat_imp_df
    }
    joblib.dump(artifact_payload, artifact_path)
    print(f"[✓] Hybrid model artifacts exported to {artifact_path}")

if __name__ == "__main__":
    train_hybrid_pipeline()

