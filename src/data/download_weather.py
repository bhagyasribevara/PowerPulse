import os
import sys
import requests
import pandas as pd

LATITUDE = 28.61
LONGITUDE = 77.20
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

def get_power_data_date_range(raw_dir):
    primary_file = os.path.join(raw_dir, "hourly data(2000-2023).csv")
    if not os.path.exists(primary_file):
        print(f"[ERROR] Primary CSV file not found: {primary_file}")
        sys.exit(1)
        
    print(f"Reading date range from primary file: {primary_file}")
    df = pd.read_csv(primary_file, usecols=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    min_date = df['timestamp'].min().strftime("%Y-%m-%d")
    max_date = df['timestamp'].max().strftime("%Y-%m-%d")
    
    print(f"Dataset date range: {min_date} to {max_date}")
    return min_date, max_date

def fetch_weather_chunk(start_date, end_date):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
        "timezone": "UTC"
    }
    print(f"Fetching Open-Meteo weather data: {start_date} -> {end_date}...")
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    hourly = data.get("hourly", {})
    df_chunk = pd.DataFrame({
        "datetime": pd.to_datetime(hourly.get("time", [])),
        "temperature_2m": hourly.get("temperature_2m", []),
        "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
        "windspeed_10m": hourly.get("windspeed_10m", [])
    })
    return df_chunk

def main():
    raw_dir = os.path.join("data", "raw")
    start_date_str, end_date_str = get_power_data_date_range(raw_dir)
    
    start_dt = pd.to_datetime(start_date_str)
    end_dt = pd.to_datetime(end_date_str)
    
    chunks = []
    curr_start = start_dt
    while curr_start <= end_dt:
        curr_end = min(curr_start + pd.DateOffset(years=2), end_dt)
        s_str = curr_start.strftime("%Y-%m-%d")
        e_str = curr_end.strftime("%Y-%m-%d")
        
        df_chunk = fetch_weather_chunk(s_str, e_str)
        chunks.append(df_chunk)
        
        curr_start = curr_end + pd.Timedelta(days=1)
        
    df_weather = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    
    output_path = os.path.join(raw_dir, "delhi_weather.csv")
    df_weather.to_csv(output_path, index=False)
    print(f"Successfully saved full weather dataset ({len(df_weather)} rows) to: {output_path}")

if __name__ == "__main__":
    main()
