import os
import sys
import pandas as pd

def main():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    power_path = os.path.join(raw_dir, "hourly data(2000-2023).csv")
    weather_path = os.path.join(raw_dir, "delhi_weather.csv")
    
    if not os.path.exists(power_path):
        print(f"[ERROR] Power consumption CSV missing at: {power_path}")
        sys.exit(1)
        
    if not os.path.exists(weather_path):
        print(f"[ERROR] Weather CSV missing at: {weather_path}")
        sys.exit(1)
        
    print("Loading power consumption data...")
    df_power = pd.read_csv(power_path)
    df_power['datetime'] = pd.to_datetime(df_power['timestamp']).dt.floor('h')
    
    # Rename electricity_demand column to load if necessary
    if 'electricity_demand' in df_power.columns and 'load' not in df_power.columns:
        df_power.rename(columns={'electricity_demand': 'load'}, inplace=True)
        
    print("Loading weather data...")
    df_weather = pd.read_csv(weather_path)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime']).dt.floor('h')
    
    print("Merging on datetime timestamp...")
    df_merged = pd.merge(df_power, df_weather, on='datetime', how='left')
    
    # Check for missing values
    missing_before = df_merged.isnull().sum()
    print("\n--- Missing Values Before Imputation ---")
    print(missing_before[missing_before > 0] if missing_before.sum() > 0 else "None")
    
    # Causal Forward Fill: Ensure zero future-data leakage
    df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
    df_merged.ffill(inplace=True)
    
    missing_after = df_merged.isnull().sum()
    print("\n--- Missing Values After Imputation ---")
    print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "0 missing values remaining.")
    
    row_count = len(df_merged)
    print(f"\nFinal Merged Row Count: {row_count}")
    
    if row_count < 1000:
        print("[ERROR] Merged dataset row count is unusually small. Halting.")
        sys.exit(1)
        
    output_path = os.path.join(processed_dir, "merged.csv")
    df_merged.to_csv(output_path, index=False)
    print(f"Clean merged data successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
