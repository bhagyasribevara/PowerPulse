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
        return "2000-01-01", "2023-12-31"
        
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
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "windspeed_10m",
            "precipitation",
            "rain"
        ],
        "timezone": "Asia/Kolkata"
    }
    print(f"Fetching Open-Meteo weather data: {start_date} -> {end_date}...")
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    hourly = data.get("hourly", {})
    df_chunk = pd.DataFrame({
        "datetime": pd.to_datetime(hourly.get("time", [])),
        "timestamp": pd.to_datetime(hourly.get("time", [])),
        "temperature_2m": hourly.get("temperature_2m", []),
        "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
        "windspeed_10m": hourly.get("windspeed_10m", []),
        "precipitation": hourly.get("precipitation", []),
        "rain": hourly.get("rain", [])
    })
    return df_chunk

def fetch_delhi_weather(start_date="2000-01-01", end_date="2023-12-31", output_path="data/raw/delhi_weather.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_dir = os.path.dirname(output_path)
    
    if os.path.exists(os.path.join(raw_dir, "hourly data(2000-2023).csv")):
        s_str, e_str = get_power_data_date_range(raw_dir)
    else:
        s_str, e_str = start_date, end_date
        
    start_dt = pd.to_datetime(s_str)
    end_dt = pd.to_datetime(e_str)
    
    chunks = []
    curr_start = start_dt
    while curr_start <= end_dt:
        curr_end = min(curr_start + pd.DateOffset(years=2), end_dt)
        chunk_s = curr_start.strftime("%Y-%m-%d")
        chunk_e = curr_end.strftime("%Y-%m-%d")
        
        df_chunk = fetch_weather_chunk(chunk_s, chunk_e)
        chunks.append(df_chunk)
        
        curr_start = curr_end + pd.Timedelta(days=1)
        
    df_weather = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df_weather.to_csv(output_path, index=False)
    print(f"[✓] Weather dataset saved to {output_path} ({len(df_weather)} rows)")

if __name__ == "__main__":
    fetch_delhi_weather()

