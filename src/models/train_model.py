import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def main():
    features_path = os.path.join("data", "processed", "features.csv")
    model_dir = os.path.join("model")
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(features_path):
        print(f"[ERROR] Features file missing at: {features_path}")
        sys.exit(1)
        
    print(f"Loading features dataset from: {features_path}")
    df = pd.read_csv(features_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Define Target and Features
    target_col = 'load'
    ignore_cols = ['datetime', 'timestamp', 'load', 'Unnamed: 0', 'compensation_method']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    print(f"Target Column: {target_col}")
    print(f"Engineered Features ({len(feature_cols)}): {feature_cols}")
    
    # Chronological Last 3-Month Holdout Split
    max_date = df['datetime'].max()
    split_date = max_date - pd.DateOffset(months=3)
    
    train_mask = df['datetime'] < split_date
    test_mask = df['datetime'] >= split_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print("\n--- CHRONOLOGICAL SPLIT RANGES ---")
    print(f"Train Period: {train_df['datetime'].min()} to {train_df['datetime'].max()} ({len(train_df)} rows)")
    print(f"Test Period : {test_df['datetime'].min()} to {test_df['datetime'].max()} ({len(test_df)} rows)")
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    # 1. 24-Hour Naive Baseline Model Evaluation
    print("\n--- EVALUATING 24-HOUR NAIVE BASELINE ---")
    # Naive baseline predicts load(t) = lag_24h
    y_pred_naive = test_df['lag_24h'].values
    
    naive_mae = mean_absolute_error(y_test, y_pred_naive)
    naive_rmse = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    naive_mape = mean_absolute_percentage_error(y_test, y_pred_naive)
    
    print(f"Naive 24h Baseline — MAE: {naive_mae:.2f} MW, RMSE: {naive_rmse:.2f} MW, MAPE: {naive_mape:.2f}%")
    
    # 2. Time-Series Cross-Validation on Training Period
    print("\n--- TIME-SERIES CROSS-VALIDATION (5-FOLD) ON TRAIN SET ---")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes, cv_rmses, cv_mapes = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        cv_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
        cv_model.fit(X_tr, y_tr)
        preds_va = cv_model.predict(X_va)
        
        cv_maes.append(mean_absolute_error(y_va, preds_va))
        cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds_va)))
        cv_mapes.append(mean_absolute_percentage_error(y_va, preds_va))
        
    print(f"Cross-Validation Mean MAE: {np.mean(cv_maes):.2f} MW | RMSE: {np.mean(cv_rmses):.2f} MW | MAPE: {np.mean(cv_mapes):.2f}%")
    
    # 3. Train Final XGBoost Model
    print("\n--- TRAINING XGBOOST MODEL ---")
    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # 4. Evaluate XGBoost on Holdout Test Set
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    xgb_mape = mean_absolute_percentage_error(y_test, y_pred_xgb)
    
    print("\n=======================================================")
    print("           MODEL BENCHMARK COMPARISON TABLE           ")
    print("=======================================================")
    print(f"{'Model':<25} | {'MAE (MW)':<10} | {'RMSE (MW)':<10} | {'MAPE (%)':<10}")
    print("-" * 63)
    print(f"{'24h Naive Baseline':<25} | {naive_mae:<10.2f} | {naive_rmse:<10.2f} | {naive_mape:<10.2f}%")
    print(f"{'XGBoost Regressor':<25} | {xgb_mae:<10.2f} | {xgb_rmse:<10.2f} | {xgb_mape:<10.2f}%")
    print("=======================================================")
    
    # 5. Feature Importances
    importances = xgb_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\n--- TOP 10 FEATURE IMPORTANCES ---")
    print(feat_imp_df.head(10))
    
    # 6. Save Model Artifacts
    model_payload = {
        'model': xgb_model,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'metrics': {
            'naive': {'mae': naive_mae, 'rmse': naive_rmse, 'mape': naive_mape},
            'xgb': {'mae': xgb_mae, 'rmse': xgb_rmse, 'mape': xgb_mape},
            'cv_xgb': {'mae': float(np.mean(cv_maes)), 'rmse': float(np.mean(cv_rmses)), 'mape': float(np.mean(cv_mapes))}
        },
        'feature_importances': feat_imp_df,
        'split_info': {
            'train_min': str(train_df['datetime'].min()),
            'train_max': str(train_df['datetime'].max()),
            'test_min': str(test_df['datetime'].min()),
            'test_max': str(test_df['datetime'].max())
        }
    }
    
    model_path = os.path.join(model_dir, "xgb_model.pkl")
    joblib.dump(model_payload, model_path)
    print(f"\nTrained model and metrics payload saved to: {model_path}")

if __name__ == "__main__":
    main()
