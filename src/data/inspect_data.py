import os
import pandas as pd

def main():
    raw_dir = os.path.join("data", "raw")
    csv_path = os.path.join(raw_dir, "hourly data(2000-2023).csv")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Main dataset not found at {csv_path}")
        return

    print(f"Loading primary power consumption dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("\n--- DATASET SUMMARY ---")
    print(f"Shape: {df.shape}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\n--- COLUMNS & TYPES ---")
    print(df.dtypes)
    print("\n--- FIRST 10 ROWS ---")
    print(df.head(10))
    print("\n--- NULL COUNTS ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
