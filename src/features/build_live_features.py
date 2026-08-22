import numpy as np
import pandas as pd
import holidays

def create_future_feature_matrix(
    weather_df: pd.DataFrame,
    genesis_time: pd.Timestamp,
    historical_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs future feature matrix for live operational forecasting.
    Includes secular trend step, calendar flags, cyclical features,
    thermal/rain indices, and historical seasonal median autoregressive anchors.
    """
    df = weather_df.sort_values("timestamp").reset_index(drop=True).copy()
    
    # 1. Macro Trend Step (Hours from genesis)
    ref_time = pd.to_datetime(genesis_time)
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["datetime"]
    df["time_step"] = (df["timestamp"] - ref_time).dt.total_seconds() / 3600.0
    
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    # 2. Socio-temporal Flags
    start_yr = int(df["year"].min())
    end_yr = int(df["year"].max())
    india_holidays = holidays.India(years=range(start_yr, end_yr + 2))
    
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["timestamp"].dt.date.apply(lambda d: 1 if d in india_holidays else 0)

    # 3. Cyclical Transforms
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 4. Thermal & Rain Indicators
    df["heat_index"] = df["temperature_2m"] + 0.5 * df["relative_humidity_2m"]
    df["ac_load_proxy"] = np.where(
        df["temperature_2m"] > 30.0,
        np.exp((df["temperature_2m"] - 30.0) / 5.0),
        0.0
    )
    df["precipitation"] = df["precipitation"].fillna(0.0)
    df["is_raining"] = (df["precipitation"] > 0.1).astype(int)
    df["rain_rolling_3h"] = df["precipitation"].rolling(window=3, min_periods=1).sum()
    df["rain_rolling_6h"] = df["precipitation"].rolling(window=6, min_periods=1).sum()

    # 5. Historical Autoregressive Anchors
    # Group historical data by (month, dayofweek, hour) to derive median seasonal demand profile
    if "load" in historical_features_df.columns:
        anchors = historical_features_df.groupby(["month", "dayofweek", "hour"])["load"].median().reset_index()
        anchors.rename(columns={"load": "lag_anchor"}, inplace=True)
        df = df.merge(anchors, on=["month", "dayofweek", "hour"], how="left")
        
        # Fallback if any unmapped combinations
        global_median = float(historical_features_df["load"].median())
        df["lag_anchor"] = df["lag_anchor"].fillna(global_median)
    else:
        df["lag_anchor"] = 3500.0

    df["day_of_week"] = df["dayofweek"]
    df["hour_of_day"] = df["hour"]
    df["temperature"] = df["temperature_2m"]
    df["humidity"] = df["relative_humidity_2m"]
    df["solar_generation"] = 0.0

    df["lag_1h"] = df["lag_anchor"]
    df["lag_24h"] = df["lag_anchor"]
    df["lag_168h"] = df["lag_anchor"]
    df["rolling_mean_24h"] = df["lag_anchor"]
    
    return df

def execute_recursive_forecast(
    feature_matrix: pd.DataFrame,
    trend_model,
    xgb_model,
    trend_features: list,
    xgb_features: list
) -> pd.DataFrame:
    """
    Executes chronological recursive multi-step forecasting across the 5-day horizon.
    Updates lag_1h, lag_24h, and rolling_mean_24h dynamically from preceding step forecasts.
    """
    df = feature_matrix.copy()
    n_rows = len(df)
    forecasts = np.zeros(n_rows)

    # Make sure all features exist
    for f in list(set(trend_features + xgb_features)):
        if f not in df.columns:
            df[f] = 0.0

    recent_predictions = []

    for i in range(n_rows):
        # Update autoregressive lags from earlier predictions if available
        if i > 0:
            df.loc[i, "lag_1h"] = forecasts[i - 1]
        if i >= 24:
            df.loc[i, "lag_24h"] = forecasts[i - 24]
        if i >= 168:
            df.loc[i, "lag_168h"] = forecasts[i - 168]

        if len(recent_predictions) > 0:
            window = recent_predictions[-24:]
            df.loc[i, "rolling_mean_24h"] = float(np.mean(window))

        # Predict Stage 1 Trend
        row_trend_input = df.loc[[i], trend_features]
        if trend_model is not None:
            try:
                base_trend = float(trend_model.predict(row_trend_input)[0])
            except Exception:
                base_trend = 0.0
        else:
            base_trend = 0.0

        # Predict Stage 2 Residual
        row_xgb_input = df.loc[[i], xgb_features]
        residual = float(xgb_model.predict(row_xgb_input)[0])

        step_pred = base_trend + residual
        forecasts[i] = step_pred
        recent_predictions.append(step_pred)

    df["forecast_mw"] = forecasts
    return df
