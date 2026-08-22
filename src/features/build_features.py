import os
import sys
import numpy as np
import pandas as pd
import holidays

def compute_engineered_features(df: pd.DataFrame, genesis_time=None) -> pd.DataFrame:
    """
    Engineers time-series features with strict anti-leakage guarantees.
    Supports both offline batch feature generation and live single-row inference.
    """
    df = df.copy()
    
    # Standardize timestamp column name
    if 'timestamp' in df.columns:
        ts_col = 'timestamp'
    elif 'datetime' in df.columns:
        ts_col = 'datetime'
    else:
        ts_col = None

    if ts_col is not None:
        df['datetime'] = pd.to_datetime(df[ts_col])
        df['timestamp'] = df['datetime']
        df = df.sort_values('timestamp').reset_index(drop=True)

    # 1. Temporal & Secular Trend Indices
    if 'timestamp' in df.columns and len(df) > 0:
        if genesis_time is not None:
            ref_time = pd.to_datetime(genesis_time)
        else:
            ref_time = pd.to_datetime(df['timestamp'].min())
        df['time_step'] = (df['timestamp'] - ref_time).dt.total_seconds() / 3600.0
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        start_year = df['timestamp'].dt.year.min()
        end_year = df['timestamp'].dt.year.max()
    else:
        if 'time_step' not in df.columns:
            df['time_step'] = 0.0
        start_year, end_year = 2000, 2024

    # 2. Socio-Temporal Flags
    if 'dayofweek' in df.columns:
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    else:
        df['is_weekend'] = 0

    india_holidays = holidays.India(years=range(int(start_year), int(end_year) + 2))
    if 'timestamp' in df.columns:
        df['is_holiday'] = df['timestamp'].dt.date.apply(lambda d: 1 if d in india_holidays else 0)
    else:
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = 0

    # 3. Cyclical Transforms
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # 4. Thermal & Cooling Demand Proxies
    temp_col = 'temperature_2m' if 'temperature_2m' in df.columns else ('temperature' if 'temperature' in df.columns else None)
    rh_col = 'relative_humidity_2m' if 'relative_humidity_2m' in df.columns else ('humidity' if 'humidity' in df.columns else None)

    temp = df[temp_col] if (temp_col and temp_col in df.columns) else 25.0
    rh = df[rh_col] if (rh_col and rh_col in df.columns) else 50.0

    df['heat_index'] = temp + 0.5 * rh
    df['ac_load_proxy'] = np.where(temp > 30, np.exp((temp - 30.0) / 5.0), 0.0)

    # 5. Precipitation Dynamics
    if 'precipitation' not in df.columns:
        if 'rain' in df.columns:
            df['precipitation'] = df['rain']
        else:
            df['precipitation'] = 0.0
    df['precipitation'] = df['precipitation'].fillna(0.0)
    df['is_raining'] = (df['precipitation'] > 0.1).astype(int)
    df['rain_rolling_3h'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
    df['rain_rolling_6h'] = df['precipitation'].rolling(window=6, min_periods=1).sum()

    # 6. Anti-Leakage Autoregressive Lags & Rolling Statistics
    if 'load' in df.columns and len(df) > 168:
        df['lag_1h'] = df['load'].shift(1)
        df['lag_24h'] = df['load'].shift(24)
        df['lag_168h'] = df['load'].shift(168)
        df['rolling_mean_24h'] = df['load'].shift(1).rolling(window=24, min_periods=24).mean()

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_res = compute_engineered_features(df)
    if 'load' in df_res.columns:
        df_res = df_res.dropna(subset=['lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h']).reset_index(drop=True)
    return df_res

def main():
    processed_dir = os.path.join("data", "processed")
    merged_path = os.path.join(processed_dir, "merged.csv")
    output_path = os.path.join(processed_dir, "features.csv")
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(merged_path):
        print(f"[ERROR] Merged dataset missing at: {merged_path}")
        sys.exit(1)
        
    print(f"Loading merged dataset from: {merged_path}")
    merged_df = pd.read_csv(merged_path)
    features_df = engineer_features(merged_df)
    features_df.to_csv(output_path, index=False)
    print(f"[✓] Feature engineering complete. Matrix shape: {features_df.shape}")

if __name__ == "__main__":
    main()

