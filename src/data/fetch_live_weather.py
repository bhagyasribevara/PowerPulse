import os
import requests
import pandas as pd
import numpy as np

DEFAULT_WEATHER_API_KEY = "62e9b799365941cfa6534158262008"

def fetch_5day_hourly_weather(api_key: str = None) -> pd.DataFrame:
    """
    Fetches hourly weather forecast for Delhi (Lat 28.61, Lon 77.20) for up to 6 days.
    Prioritizes WeatherAPI.com (if key provided or default), falls back to Open-Meteo.
    Ensures timestamps are localized and aligned to Asia/Kolkata (IST).
    """
    lat, lon = 28.61, 77.20
    active_key = api_key or os.environ.get("WEATHER_API_KEY", DEFAULT_WEATHER_API_KEY)
    
    # 1. Attempt WeatherAPI.com
    if active_key:
        try:
            url = f"https://api.weatherapi.com/v1/forecast.json?key={active_key}&q={lat},{lon}&days=6&aqi=no&alerts=no"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if "forecast" in data and "forecastday" in data["forecast"]:
                    rows = []
                    for day in data["forecast"]["forecastday"]:
                        for hr in day.get("hour", []):
                            # hr["time"] is in local Delhi time e.g. "2026-08-20 00:00"
                            rows.append({
                                "timestamp": pd.to_datetime(hr["time"]),
                                "temperature_2m": float(hr.get("temp_c", 25.0)),
                                "relative_humidity_2m": float(hr.get("humidity", 50.0)),
                                "windspeed_10m": float(hr.get("wind_kph", 10.0)),
                                "precipitation": float(hr.get("precip_mm", 0.0))
                            })
                    if len(rows) > 0:
                        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
                        return df
        except Exception as e:
            print(f"[Weather Fetch Warning] WeatherAPI query failed: {e}. Falling back to Open-Meteo.")

    # 2. Fallback to Open-Meteo Keyless API
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "windspeed_10m", "precipitation"],
            "forecast_days": 6,
            "timezone": "Asia/Kolkata"
        }
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        df = pd.DataFrame(data["hourly"])
        df["timestamp"] = pd.to_datetime(df["time"])
        df.drop(columns=["time"], inplace=True, errors="ignore")
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"[Weather Fetch Error] Open-Meteo fallback also failed: {e}")
        # Synthetic fallback if offline or network error
        now = pd.Timestamp.now().floor("h")
        times = [now + pd.Timedelta(hours=i) for i in range(144)]
        return pd.DataFrame({
            "timestamp": times,
            "temperature_2m": [30.0 + 5.0 * np.sin(i * np.pi / 12.0) for i in range(144)],
            "relative_humidity_2m": [60.0 + 15.0 * np.cos(i * np.pi / 12.0) for i in range(144)],
            "windspeed_10m": [10.0] * 144,
            "precipitation": [0.0] * 144
        })
