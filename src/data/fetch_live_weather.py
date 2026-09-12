import os
import requests
import pandas as pd
import numpy as np

DEFAULT_WEATHER_API_KEY = "62e9b799365941cfa6534158262008"

def fetch_5day_hourly_weather(api_key: str = None) -> pd.DataFrame:
    """
    Fetches hourly weather forecast for Delhi (Lat 28.61, Lon 77.20) for 7 full days (168 hours).
    Prioritizes comprehensive multi-day data from Open-Meteo (7-day hourly API) and WeatherAPI.
    Ensures timestamps are localized and aligned to Asia/Kolkata (IST).
    """
    lat, lon = 28.61, 77.20
    active_key = api_key or os.environ.get("WEATHER_API_KEY", DEFAULT_WEATHER_API_KEY)
    
    # 1. Fetch full 7-day hourly forecast from Open-Meteo (high reliability, 168 hours)
    df_om = None
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "windspeed_10m", "precipitation"],
            "forecast_days": 7,
            "timezone": "Asia/Kolkata"
        }
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if "hourly" in data and "time" in data["hourly"]:
                df_om = pd.DataFrame(data["hourly"])
                df_om["timestamp"] = pd.to_datetime(df_om["time"])
                df_om.drop(columns=["time"], inplace=True, errors="ignore")
                df_om = df_om.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"[Weather Fetch Warning] Open-Meteo query failed: {e}")

    # 2. Attempt WeatherAPI.com (if key provided, up to 3 days available on free tier)
    df_wapi = None
    if active_key:
        try:
            url = f"https://api.weatherapi.com/v1/forecast.json?key={active_key}&q={lat},{lon}&days=7&aqi=no&alerts=no"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if "forecast" in data and "forecastday" in data["forecast"]:
                    rows = []
                    for day in data["forecast"]["forecastday"]:
                        for hr in day.get("hour", []):
                            rows.append({
                                "timestamp": pd.to_datetime(hr["time"]),
                                "temperature_2m": float(hr.get("temp_c", 25.0)),
                                "relative_humidity_2m": float(hr.get("humidity", 50.0)),
                                "windspeed_10m": float(hr.get("wind_kph", 10.0)),
                                "precipitation": float(hr.get("precip_mm", 0.0))
                            })
                    if len(rows) > 0:
                        df_wapi = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            print(f"[Weather Fetch Warning] WeatherAPI query failed: {e}")

    # Merge or fallback
    if df_om is not None and not df_om.empty:
        if df_wapi is not None and not df_wapi.empty:
            # Overwrite overlapping timestamps with WeatherAPI station data, keep full 7 days from Open-Meteo
            wapi_ts = set(df_wapi["timestamp"])
            df_remaining = df_om[~df_om["timestamp"].isin(wapi_ts)]
            df_combined = pd.concat([df_wapi, df_remaining], ignore_index=True)
            return df_combined.sort_values("timestamp").reset_index(drop=True)
        return df_om

    if df_wapi is not None and not df_wapi.empty:
        return df_wapi

    # 3. Synthetic fallback if both APIs are offline
    print("[Weather Fetch Fallback] Using synthetic 7-day meteorological forecast.")
    now = pd.Timestamp.now().floor("h")
    times = [now + pd.Timedelta(hours=i) for i in range(168)]
    return pd.DataFrame({
        "timestamp": times,
        "temperature_2m": [30.0 + 6.0 * np.sin((i - 6) * np.pi / 12.0) for i in range(168)],
        "relative_humidity_2m": [60.0 + 15.0 * np.cos((i - 6) * np.pi / 12.0) for i in range(168)],
        "windspeed_10m": [10.0 + 3.0 * np.sin(i * np.pi / 24.0) for i in range(168)],
        "precipitation": [0.0] * 168
    })
