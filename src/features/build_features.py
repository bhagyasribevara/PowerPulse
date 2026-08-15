import os
import sys
import numpy as np
import pandas as pd
import holidays

def compute_engineered_features(df):
    """
    Engineers time-series features with strict anti-leakage guarantees.
    Modular design allows reuse both for offline batch features and online live inference.
    """
    df = df.copy()
    
    # Ensure datetime index/column
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
    
    # Extract time components
    if 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        start_year = df['datetime'].dt.year.min()
        end_year = df['datetime'].dt.year.max()
    else:
        start_year, end_year = 2000, 2024

    # Weekend flag
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    # India holidays
    india_holidays = holidays.India(years=range(int(start_year), int(end_year) + 2))
    if 'datetime' in df.columns:
        df['is_holiday'] = df['datetime'].dt.date.apply(lambda d: 1 if d in india_holidays else 0)
    else:
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = 0

    # Cyclical Encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # Weather proxies (explicitly documented as proxy features)
    temp_col = 'temperature_2m' if 'temperature_2m' in df.columns else 'temperature'
    rh_col = 'relative_humidity_2m' if 'relative_humidity_2m' in df.columns else ('humidity' if 'humidity' in df.columns else None)
    
    temp = df[temp_col] if temp_col in df.columns else 25.0
    rh = df[rh_col] if (rh_col and rh_col in df.columns) else 50.0

    df['heat_index'] = temp + 0.5 * rh
    df['ac_load_proxy'] = np.where(temp > 30, np.exp((temp - 30.0) / 5.0), 0.0)

    # Anti-leakage Lag & Rolling Features (only computed if load column is present and dataset is time-series)
    if 'load' in df.columns and len(df) > 168:
        # Shift load by 1 before any lag/rolling calculation to prevent current/future leakage
        df['lag_1h'] = df['load'].shift(1)
        df['lag_24h'] = df['load'].shift(24)
        df['lag_168h'] = df['load'].shift(168) # Prior week same hour
        df['rolling_mean_24h'] = df['load'].shift(1).rolling(24).mean()

    return df

def main():
    processed_dir = os.path.join("data", "processed")
    merged_path = os.path.join(processed_dir, "merged.csv")
    
    if not os.path.exists(merged_path):
        print(f"[ERROR] Merged dataset missing at: {merged_path}")
        sys.exit(1)
        
    print(f"Loading merged dataset from: {merged_path}")
    df_merged = pd.read_csv(merged_path)
    
    print("Building engineered features...")
    df_features = compute_engineered_features(df_merged)
    
    # Drop NaNs resulting from lag/rolling shifts
    initial_count = len(df_features)
    df_features.dropna(subset=['lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h'], inplace=True)
    final_count = len(df_features)
    
    print(f"\nDropped {initial_count - final_count} initial rows with unaligned lags.")
    print(f"Final Feature Matrix Row Count: {final_count}")
    
    feature_cols = [c for c in df_features.columns if c not in ['datetime', 'timestamp']]
    print(f"Engineered Feature Columns ({len(feature_cols)}):")
    print(feature_cols)
    
    output_path = os.path.join(processed_dir, "features.csv")
    df_features.to_csv(output_path, index=False)
    print(f"Saved feature dataset to: {output_path}")

if __name__ == "__main__":
    main()
