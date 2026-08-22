import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add root directory to sys.path to import src.features.build_features
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features.build_features import compute_engineered_features
from src.data.fetch_live_weather import fetch_5day_hourly_weather
from src.features.build_live_features import create_future_feature_matrix, execute_recursive_forecast
from src.data.delhi_geo_nodes import DELHI_ZONAL_HUBS

st.set_page_config(
    page_title="GridSathi — Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Delhi Grid Dark visual theme
st.markdown("""
<style>
    .stApp {
        background-color: #0a0e1a;
        color: #f3f4f6;
    }
    .main {
        background-color: #0a0e1a;
    }
    .metric-card {
        background-color: #111827;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #1f2937;
        text-align: center;
    }
    .metric-value-amber {
        font-size: 30px;
        font-weight: bold;
        color: #fbbf24;
    }
    .metric-value-blue {
        font-size: 30px;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #9ca3af;
        margin-top: 4px;
    }
    .risk-box-high {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #ef4444;
        padding: 14px;
        border-radius: 10px;
        font-weight: 600;
        text-align: left;
    }
    .risk-box-warning {
        background-color: rgba(234, 179, 8, 0.15);
        border: 1px solid #eab308;
        color: #eab308;
        padding: 14px;
        border-radius: 10px;
        font-weight: 600;
        text-align: left;
    }
    .risk-box-normal {
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #22c55e;
        padding: 14px;
        border-radius: 10px;
        font-weight: 600;
        text-align: left;
    }
    .header-title {
        font-size: 34px;
        font-weight: 800;
        color: #fbbf24;
        margin-bottom: 0px;
    }
    .header-subtitle {
        font-size: 16px;
        font-weight: 400;
        color: #9ca3af;
        margin-bottom: 20px;
    }
    .footer-text {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def mean_absolute_percentage_error_custom(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

@st.cache_resource
def load_pipeline():
    model_path = os.path.join(ROOT_DIR, "model", "xgb_model.pkl")
    features_path = os.path.join(ROOT_DIR, "data", "processed", "features.csv")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train model first.")
        st.stop()
        
    if not os.path.exists(features_path):
        st.error(f"Features file not found at {features_path}. Please build features first.")
        st.stop()
        
    artifacts = joblib.load(model_path)
    df_features = pd.read_csv(features_path)
    if 'timestamp' in df_features.columns:
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        df_features['datetime'] = df_features['timestamp']
    elif 'datetime' in df_features.columns:
        df_features['datetime'] = pd.to_datetime(df_features['datetime'])
        df_features['timestamp'] = df_features['datetime']
    df_features = df_features.sort_values('timestamp').reset_index(drop=True)
    
    return artifacts, df_features

@st.cache_resource
def get_shap_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)

def predict_demand(input_dict: dict, forecast_timestamp: pd.Timestamp, artifacts: dict) -> float:
    genesis_time_val = artifacts.get("genesis_time", "2000-01-01")
    genesis_time = pd.to_datetime(genesis_time_val)
    
    forecast_ts = pd.to_datetime(forecast_timestamp)
    time_step = (forecast_ts - genesis_time).total_seconds() / 3600.0
    input_dict["time_step"] = time_step
    input_dict["timestamp"] = forecast_ts
    input_dict["datetime"] = forecast_ts
    
    df_in = compute_engineered_features(pd.DataFrame([input_dict]), genesis_time=genesis_time)
    
    trend_model = artifacts.get("trend_model")
    xgb_model = artifacts.get("xgb_model", artifacts.get("model"))
    trend_features = artifacts.get("trend_features", ["time_step"])
    xgb_features = artifacts.get("xgb_features", artifacts.get("feature_cols", []))

    # Fill any missing feature columns with 0.0
    for f in list(set(trend_features + xgb_features)):
        if f not in df_in.columns or pd.isna(df_in[f].values[0]):
            df_in[f] = 0.0

    if trend_model is not None:
        try:
            base_trend = float(trend_model.predict(df_in[trend_features])[0])
        except Exception:
            base_trend = 0.0
    else:
        base_trend = 0.0

    residual = float(xgb_model.predict(df_in[xgb_features])[0])
    return base_trend + residual

def predict_demand_batch(df_subset: pd.DataFrame, artifacts: dict) -> np.ndarray:
    trend_model = artifacts.get("trend_model")
    xgb_model = artifacts.get("xgb_model", artifacts.get("model"))
    trend_features = artifacts.get("trend_features", ["time_step"])
    xgb_features = artifacts.get("xgb_features", artifacts.get("feature_cols", []))
    
    df_copy = df_subset.copy()
    for f in list(set(trend_features + xgb_features)):
        if f not in df_copy.columns:
            df_copy[f] = 0.0
            
    if trend_model is not None:
        try:
            base_trend = trend_model.predict(df_copy[trend_features])
        except Exception:
            base_trend = np.zeros(len(df_copy))
    else:
        base_trend = np.zeros(len(df_copy))
        
    residual = xgb_model.predict(df_copy[xgb_features])
    return base_trend + residual

FEATURE_META = {
    "temperature_2m": "Temperature (°C)",
    "temperature": "Temperature (°C)",
    "relative_humidity_2m": "Relative Humidity (%)",
    "humidity": "Relative Humidity (%)",
    "windspeed_10m": "Wind Speed (km/h)",
    "heat_index": "Heat Index (°C)",
    "ac_load_proxy": "AC Load Proxy Index",
    "hour": "Hour of Day",
    "hour_of_day": "Hour of Day",
    "hour_sin": "Diurnal Sine Wave",
    "hour_cos": "Diurnal Cosine Wave",
    "month": "Month of Year",
    "month_sin": "Seasonal Sine Wave",
    "month_cos": "Seasonal Cosine Wave",
    "dayofweek": "Day of Week",
    "day_of_week": "Day of Week",
    "is_weekend": "Weekend Flag",
    "is_holiday": "Public Holiday Flag",
    "solar_generation": "Solar Generation Offset",
    "precipitation": "Precipitation (mm)",
    "is_raining": "Rain Indicator",
    "rain_rolling_3h": "Rain 3h Rolling",
    "rain_rolling_6h": "Rain 6h Rolling",
    "lag_1h": "Lag 1-Hour Prior (MW)",
    "lag_24h": "Lag 24-Hours Prior (MW)",
    "lag_168h": "Lag 7-Days Prior (MW)",
    "rolling_mean_24h": "24h Rolling Mean (MW)",
    "time_step": "Long-Term Secular Trend"
}

@st.cache_data
def compute_global_shap_data(_xgb_model, _df_features, xgb_features: tuple):
    step = max(1, len(_df_features) // 300)
    sample_df = _df_features.iloc[::step][list(xgb_features)].copy().fillna(0.0)
    explainer = shap.TreeExplainer(_xgb_model)
    shap_vals = explainer.shap_values(sample_df)
    mean_abs_impact = np.mean(np.abs(shap_vals), axis=0)
    return sample_df, shap_vals, mean_abs_impact

def render_dynamic_shap_tab(artifacts: dict, input_dict: dict, forecast_ts: pd.Timestamp, df_features: pd.DataFrame):
    st.subheader("🔍 Explainable AI Analytics Hub (TreeSHAP)")
    st.caption("Game-theoretic Shapley attributions decomposing both local individual prediction instances and global feature interaction mechanics.")

    xgb_model = artifacts.get("xgb_model", artifacts.get("model"))
    xgb_features = artifacts.get("xgb_features", artifacts.get("feature_cols", []))
    trend_model = artifacts.get("trend_model")
    trend_features = artifacts.get("trend_features", ["time_step"])
    genesis_time = pd.to_datetime(artifacts.get("genesis_time", artifacts.get("split_info", {}).get("train_min", "2000-01-01")))

    explainer = get_shap_explainer(xgb_model)
    raw_base_val = explainer.expected_value
    base_val = float(np.array(raw_base_val).flatten()[0])

    view_mode = st.radio(
        "Select XAI Analysis View",
        ["📍 Local Prediction Attribution (Waterfall & Narrative)", "🌐 Global Feature Importance & Non-Linear Dependence"],
        horizontal=True
    )

    if "Local" in view_mode:
        mode = st.radio(
            "Instance to Explain",
            ["🕹️ Active Sidebar Scenario", "📅 Historical Record Picker"],
            horizontal=True
        )

        if mode == "🕹️ Active Sidebar Scenario":
            inst_dict = input_dict.copy()
            inst_ts = forecast_ts
            df_in = compute_engineered_features(pd.DataFrame([inst_dict]), genesis_time=genesis_time)
            instance_label = f"Sidebar Scenario ({forecast_ts.strftime('%Y-%m-%d %H:00')}, Temp: {inst_dict['temperature_2m']}°C, RH: {inst_dict['relative_humidity_2m']}%)"
        else:
            min_date = df_features["timestamp"].dt.date.min()
            max_date = df_features["timestamp"].dt.date.max()
            col_hdate, col_hhour = st.columns([2, 1])
            with col_hdate:
                hist_date = st.date_input("Pick Historical Date", value=max_date, min_value=min_date, max_value=max_date)
            with col_hhour:
                hist_hour = st.slider("Hour of Day", min_value=0, max_value=23, value=19, step=1, key="hist_hour_slider")

            target_hist_ts = pd.Timestamp(f"{hist_date} {hist_hour:02d}:00:00")
            hist_rows = df_features[df_features["timestamp"] == target_hist_ts]
            if len(hist_rows) == 0:
                closest_idx = (df_features["timestamp"] - target_hist_ts).abs().idxmin()
                hist_rows = df_features.iloc[[closest_idx]]
                target_hist_ts = hist_rows["timestamp"].values[0]
                st.info(f"Exact timestamp not available. Showing closest record: `{pd.to_datetime(target_hist_ts)}`.")

            df_in = hist_rows.copy()
            instance_label = f"Historical Record ({pd.to_datetime(target_hist_ts).strftime('%Y-%m-%d %H:00')}, Actual: {df_in['load'].values[0]:.1f} MW)"

        # Ensure all features exist
        for f in list(set(trend_features + xgb_features)):
            if f not in df_in.columns or pd.isna(df_in[f].values[0]):
                df_in[f] = 0.0

        # Trend baseline
        if trend_model is not None:
            try:
                base_trend = float(trend_model.predict(df_in[trend_features])[0])
            except Exception:
                base_trend = 0.0
        else:
            base_trend = 0.0

        # Compute SHAP values
        shap_vals = explainer.shap_values(df_in[xgb_features])[0]
        total_shap = float(np.sum(shap_vals))
        final_pred = base_trend + base_val + total_shap

        feature_rows = []
        for f, sv in zip(xgb_features, shap_vals):
            val = df_in[f].values[0] if f in df_in.columns else 0.0
            fname = FEATURE_META.get(f, f)
            feature_rows.append({
                "Feature_Code": f,
                "Feature": fname,
                "Raw_Value": val,
                "SHAP Contribution (MW)": float(sv),
                "Abs_Impact": abs(float(sv)),
                "Direction": "🔺 Pushing Demand UP" if sv >= 0 else "🔻 Pushing Demand DOWN"
            })

        contrib_df = pd.DataFrame(feature_rows).sort_values(by="Abs_Impact", ascending=False).reset_index(drop=True)

        # Attribution Breakdown Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TreeSHAP Base Value", f"{base_val:.1f} MW")
        if trend_model is not None:
            m2.metric("Base Secular Trend", f"{base_trend:.1f} MW")
        else:
            m2.metric("Baseline Offset", f"{base_val:.1f} MW")
        m3.metric("Net SHAP Feature Impact", f"{total_shap:+.1f} MW", delta=f"{total_shap:+.1f} MW", delta_color="inverse")
        m4.metric("Instance Demand Forecast", f"{final_pred:.1f} MW")

        # Natural Language AI Executive Narrative
        pos_drivers = contrib_df[contrib_df["SHAP Contribution (MW)"] > 0].head(2)
        neg_drivers = contrib_df[contrib_df["SHAP Contribution (MW)"] < 0].head(1)
        
        narrative_parts = [f"🤖 **Executive XAI Briefing:** For `{instance_label}`, the model predicts a demand of **{final_pred:.1f} MW** (baseline: **{base_val+base_trend:.1f} MW**)."]
        if not pos_drivers.empty:
            pos_str = " and ".join([f"**{r['Feature']}** (<span style='color: #ef4444;'>+{r['SHAP Contribution (MW)']:.1f} MW</span>)" for _, r in pos_drivers.iterrows()])
            narrative_parts.append(f"Primary upward load pressure is driven by {pos_str}.")
        if not neg_drivers.empty:
            neg_str = f"**{neg_drivers.iloc[0]['Feature']}** (<span style='color: #38bdf8;'>{neg_drivers.iloc[0]['SHAP Contribution (MW)']:.1f} MW</span>)"
            narrative_parts.append(f"Downside relief is contributed by {neg_str}.")
        
        st.markdown(f"""
        <div style="background-color: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; padding: 12px 16px; border-radius: 6px; margin: 12px 0 18px 0; font-size: 14px; line-height: 1.5;">
            {' '.join(narrative_parts)}
        </div>
        """, unsafe_allow_html=True)

        # Plotly Waterfall Plot
        top_n = min(10, len(contrib_df))
        top_contrib = contrib_df.head(top_n).copy()
        other_impact = total_shap - top_contrib["SHAP Contribution (MW)"].sum()

        wf_x = ["Base Value"]
        wf_y = [base_val + base_trend]
        wf_text = [f"{base_val + base_trend:.1f} MW"]
        wf_measures = ["absolute"]

        for _, row in top_contrib.iterrows():
            val_str = f"{row['Raw_Value']:.1f}" if isinstance(row['Raw_Value'], (float, int, np.floating)) else str(row['Raw_Value'])
            wf_x.append(f"{row['Feature']}<br>({val_str})")
            wf_y.append(row["SHAP Contribution (MW)"])
            wf_text.append(f"{row['SHAP Contribution (MW)']:+.1f} MW")
            wf_measures.append("relative")

        if abs(other_impact) > 0.01:
            wf_x.append("Other Features")
            wf_y.append(other_impact)
            wf_text.append(f"{other_impact:+.1f} MW")
            wf_measures.append("relative")

        wf_x.append("Final Forecast")
        wf_y.append(final_pred)
        wf_text.append(f"{final_pred:.1f} MW")
        wf_measures.append("total")

        col_wf, col_bar = st.columns([1.15, 0.85])
        with col_wf:
            fig_wf = go.Figure(go.Waterfall(
                name="SHAP Waterfall",
                orientation="v",
                measure=wf_measures,
                x=wf_x,
                y=wf_y,
                text=wf_text,
                textposition="outside",
                connector={"line": {"color": "#4b5563"}},
                decreasing={"marker": {"color": "#38bdf8"}},
                increasing={"marker": {"color": "#ef4444"}},
                totals={"marker": {"color": "#fbbf24"}}
            ))
            fig_wf.update_layout(
                title="Local Prediction Waterfall: Feature Push from Baseline",
                yaxis_title="MW",
                paper_bgcolor='#111827',
                plot_bgcolor='#0a0e1a',
                font=dict(color='#f3f4f6'),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        with col_bar:
            fig_bar = px.bar(
                top_contrib.sort_values(by="SHAP Contribution (MW)", ascending=True),
                x="SHAP Contribution (MW)",
                y="Feature",
                orientation="h",
                title="Top Factors Shifting Demand",
                color="SHAP Contribution (MW)",
                color_continuous_scale="RdBu_r"
            )
            fig_bar.update_layout(
                paper_bgcolor='#111827',
                plot_bgcolor='#0a0e1a',
                font=dict(color='#f3f4f6'),
                coloraxis_colorbar=dict(title="Δ MW"),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed Table
        st.markdown("#### 📋 Detailed Feature Attribution Decomposition")
        table_display = contrib_df[["Feature", "Raw_Value", "SHAP Contribution (MW)", "Direction"]].copy()
        table_display["Raw_Value"] = table_display["Raw_Value"].apply(lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
        table_display["SHAP Contribution (MW)"] = table_display["SHAP Contribution (MW)"].apply(lambda v: f"{v:+.2f} MW")
        st.dataframe(table_display, use_container_width=True, hide_index=True)

    else:
        # Global View
        st.markdown("#### 🌐 Global Model Attribution & Non-Linear Dependence Analysis")
        st.caption("Quantifies overall feature impact magnitude across historical observations and visualizes non-linear behavioral curves.")
        
        sample_df, shap_matrix, mean_abs = compute_global_shap_data(xgb_model, df_features, tuple(xgb_features))
        
        glob_df = pd.DataFrame({
            "Feature_Code": xgb_features,
            "Feature": [FEATURE_META.get(f, f) for f in xgb_features],
            "Mean_Abs_SHAP": mean_abs
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        c_g1, c_g2 = st.columns([1, 1])
        with c_g1:
            fig_g_bar = px.bar(
                glob_df.head(10).sort_values(by="Mean_Abs_SHAP", ascending=True),
                x="Mean_Abs_SHAP",
                y="Feature",
                orientation="h",
                title="Global Mean Absolute SHAP Impact (MW)",
                color="Mean_Abs_SHAP",
                color_continuous_scale=["#38bdf8", "#f43f5e"]
            )
            fig_g_bar.update_layout(
                paper_bgcolor='#111827',
                plot_bgcolor='#0a0e1a',
                font=dict(color='#f3f4f6'),
                xaxis_title="Average |SHAP Value| (MW)",
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_g_bar, use_container_width=True)

        with c_g2:
            st.markdown("**🔍 Interactive Non-Linear Feature Dependence**")
            dep_options = [f for f in ["temperature_2m", "ac_load_proxy", "heat_index", "hour", "lag_24h", "relative_humidity_2m"] if f in xgb_features]
            dep_feat = st.selectbox(
                "Primary Feature (X-Axis)",
                dep_options,
                index=0
            )
            color_options = [f for f in ["relative_humidity_2m", "temperature_2m", "hour", "heat_index"] if f in xgb_features]
            color_feat = st.selectbox(
                "Secondary Interaction Color (Z-Axis)",
                color_options,
                index=0
            )

            feat_idx = xgb_features.index(dep_feat) if dep_feat in xgb_features else 0
            dep_x = sample_df[dep_feat].values if dep_feat in sample_df.columns else np.zeros(len(sample_df))
            dep_y = shap_matrix[:, feat_idx]
            dep_color = sample_df[color_feat].values if color_feat in sample_df.columns else np.zeros(len(sample_df))

            fig_dep = px.scatter(
                x=dep_x,
                y=dep_y,
                color=dep_color,
                labels={"x": FEATURE_META.get(dep_feat, dep_feat), "y": f"SHAP Impact for {FEATURE_META.get(dep_feat, dep_feat)} (MW)", "color": FEATURE_META.get(color_feat, color_feat)},
                title=f"Dependence: {FEATURE_META.get(dep_feat, dep_feat)} vs SHAP (MW)",
                color_continuous_scale="Plasma"
            )
            fig_dep.update_layout(
                paper_bgcolor='#111827',
                plot_bgcolor='#0a0e1a',
                font=dict(color='#f3f4f6'),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_dep, use_container_width=True)

def render_dynamic_actual_vs_pred_tab(df_features: pd.DataFrame, artifacts: dict, scenario_info: dict):
    st.subheader("📉 Dynamic Historical Evaluation (Actual vs Predicted)")
    st.caption("Interactively explore model fidelity across historical seasons, heatwaves, and holidays with live metric recalculation and sidebar scenario overlays.")

    min_date = df_features["timestamp"].dt.date.min()
    max_date = df_features["timestamp"].dt.date.max()

    col_preset, col_dates = st.columns([1, 2])
    with col_preset:
        preset_choice = st.selectbox(
            "⚡ Quick Horizon Preset",
            [
                "Recent 14 Days (Default Holdout)",
                "Recent 30 Days",
                "Peak Summer Heatwave (June 2023)",
                "Monsoon Humidity Surge (August 2023)",
                "Winter Cold Wave Peak (Dec 2023)",
                "Full Historical Dataset (Sampled)",
                "Custom Date Range"
            ]
        )

    if preset_choice == "Recent 14 Days (Default Holdout)":
        default_start = max_date - pd.Timedelta(days=14)
        default_end = max_date
    elif preset_choice == "Recent 30 Days":
        default_start = max_date - pd.Timedelta(days=30)
        default_end = max_date
    elif preset_choice == "Peak Summer Heatwave (June 2023)":
        default_start = pd.to_datetime("2023-06-01").date()
        default_end = pd.to_datetime("2023-06-30").date()
    elif preset_choice == "Monsoon Humidity Surge (August 2023)":
        default_start = pd.to_datetime("2023-08-01").date()
        default_end = pd.to_datetime("2023-08-31").date()
    elif preset_choice == "Winter Cold Wave Peak (Dec 2023)":
        default_start = pd.to_datetime("2023-12-01").date()
        default_end = pd.to_datetime("2023-12-31").date()
    elif preset_choice == "Full Historical Dataset (Sampled)":
        default_start = min_date
        default_end = max_date
    else:
        default_start = max_date - pd.Timedelta(days=14)
        default_end = max_date

    with col_dates:
        date_selection = st.date_input(
            "Select Evaluation Range",
            value=[default_start, default_end],
            min_value=min_date,
            max_value=max_date,
            key=f"date_picker_{preset_choice}"
        )

    if isinstance(date_selection, (list, tuple)) and len(date_selection) == 2:
        start_d, end_d = date_selection
    elif isinstance(date_selection, (list, tuple)) and len(date_selection) == 1:
        start_d = end_d = date_selection[0]
    else:
        start_d = default_start
        end_d = default_end

    mask = (df_features["timestamp"].dt.date >= start_d) & (df_features["timestamp"].dt.date <= end_d)
    subset = df_features.loc[mask].copy()

    if len(subset) == 0:
        st.warning("No historical observations found in the selected date range.")
        return

    # Compute batch predictions
    subset["pred_load"] = predict_demand_batch(subset, artifacts)

    actual = subset["load"]
    pred = subset["pred_load"]

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error_custom(actual, pred)

    has_naive = "lag_24h" in subset.columns and not subset["lag_24h"].isna().all()
    if has_naive:
        naive_mae = mean_absolute_error(actual, subset["lag_24h"])
        naive_rmse = np.sqrt(mean_squared_error(actual, subset["lag_24h"]))
        naive_mape = mean_absolute_percentage_error_custom(actual, subset["lag_24h"])
        mae_delta = f"{mae - naive_mae:+.2f} MW vs Naive"
        rmse_delta = f"{rmse - naive_rmse:+.2f} MW vs Naive"
        mape_delta = f"{mape - naive_mape:+.2f}% vs Naive"
    else:
        mae_delta = None
        rmse_delta = None
        mape_delta = None

    peak_actual = actual.max()
    peak_pred = pred.max()
    peak_err = peak_pred - peak_actual

    # Live Metrics Strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Window MAE", f"{mae:.2f} MW", delta=mae_delta, delta_color="inverse")
    m2.metric("Selected Window RMSE", f"{rmse:.2f} MW", delta=rmse_delta, delta_color="inverse")
    m3.metric("Selected Window MAPE", f"{mape:.2f}%", delta=mape_delta, delta_color="inverse")
    m4.metric("Window Peak Load", f"{peak_actual:.1f} MW", delta=f"{peak_err:+.1f} MW peak error", delta_color="off")

    # Options Row
    c_ctrl1, c_ctrl2 = st.columns(2)
    with c_ctrl1:
        overlay_scenario = st.checkbox("📌 Overlay Active Sidebar Scenario Forecast onto Historical Curve", value=True)
    with c_ctrl2:
        show_residuals = st.checkbox("📊 Show Residual Error Deviation Subplot", value=False)

    # Plotly Time-Series Line Chart
    hist_x = [ts.strftime("%Y-%m-%d %H:%M") for ts in subset['timestamp']]
    hist_actual = [float(v) for v in subset['load']]
    hist_pred = [float(v) for v in subset['pred_load']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=hist_actual,
        mode='lines',
        name='Actual Load (MW)',
        line=dict(color='#38bdf8', width=2),
        hovertemplate='<b>Actual:</b> %{y:.1f} MW<br><b>Time:</b> %{x}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=hist_pred,
        mode='lines',
        name='Predicted Load (MW)',
        line=dict(color='#fbbf24', width=1.8, dash='dash'),
        hovertemplate='<b>Predicted:</b> %{y:.1f} MW<br><b>Time:</b> %{x}<extra></extra>'
    ))

    if overlay_scenario and scenario_info is not None:
        scen_ts = scenario_info.get("timestamp")
        scen_pred = scenario_info.get("pred_demand")
        fig.add_trace(go.Scatter(
            x=[scen_ts],
            y=[scen_pred],
            mode='markers+text',
            name='Sidebar Scenario Simulation',
            marker=dict(size=14, color='#ef4444', symbol='diamond', line=dict(width=2, color='#ffffff')),
            text=[f"Scenario: {scen_pred:.1f} MW"],
            textposition='top center',
            hovertemplate=f"<b>Simulated Scenario:</b> {scen_pred:.1f} MW<br><b>Timestamp:</b> {scen_ts}<extra></extra>"
        ))
        fig.add_hline(
            y=scen_pred,
            line_dash="dot",
            line_color="rgba(239, 68, 68, 0.6)",
            annotation_text=f"Scenario Level ({scen_pred:.1f} MW)",
            annotation_position="bottom right",
            annotation_font_color="#ef4444"
        )

    fig.update_layout(
        title=f"Delhi Electricity Load: Actual vs Predicted ({start_d} to {end_d}) [N={len(subset)} hours]",
        xaxis_title="Timestamp",
        yaxis_title="Demand (MW)",
        paper_bgcolor='#111827',
        plot_bgcolor='#0a0e1a',
        font=dict(color='#f3f4f6'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True, bgcolor="#1f2937"))
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_residuals:
        subset["residual_mw"] = subset["load"] - subset["pred_load"]
        fig_res = px.area(
            subset,
            x="timestamp",
            y="residual_mw",
            title="Model Residuals (Actual Load - Predicted Load) [MW]",
            labels={"residual_mw": "Residual Error (MW)", "timestamp": "Timestamp"},
            color_discrete_sequence=["#a855f7"]
        )
        fig_res.add_hline(y=0, line_color="#9ca3af", line_dash="dash")
        fig_res.update_layout(
            paper_bgcolor='#111827',
            plot_bgcolor='#0a0e1a',
            font=dict(color='#f3f4f6')
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with st.expander("📋 Detailed Window Extremes & Worst-Error Analysis", expanded=False):
        subset["abs_error_mw"] = (subset["load"] - subset["pred_load"]).abs()
        subset["pct_error"] = (subset["abs_error_mw"] / subset["load"].replace(0, np.nan)) * 100
        worst_5 = subset.sort_values(by="abs_error_mw", ascending=False).head(5)

        display_worst = worst_5[["timestamp", "load", "pred_load", "abs_error_mw", "pct_error", "temperature_2m", "relative_humidity_2m"]].copy()
        display_worst.columns = ["Timestamp", "Actual (MW)", "Predicted (MW)", "Absolute Error (MW)", "Error (%)", "Temp (°C)", "Humidity (%)"]
        display_worst["Actual (MW)"] = display_worst["Actual (MW)"].map("{:.1f}".format)
        display_worst["Predicted (MW)"] = display_worst["Predicted (MW)"].map("{:.1f}".format)
        display_worst["Absolute Error (MW)"] = display_worst["Absolute Error (MW)"].map("{:.1f}".format)
        display_worst["Error (%)"] = display_worst["Error (%)"].map("{:.2f}%".format)

        st.markdown("**Top 5 Peak Discrepancy Moments in Selected Window:**")
        st.dataframe(display_worst, use_container_width=True, hide_index=True)

        if len(worst_5) > 0:
            st.markdown("---")
            st.markdown("**🔍 Root-Cause Post-Mortem Diagnostics (TreeSHAP on Worst Errors):**")
            st.caption("Inspect the exact feature pushes at the selected discrepancy timestamp to diagnose weather shocks vs autoregressive lag carryover.")
            
            xgb_features = artifacts.get("xgb_features", artifacts.get("feature_cols", []))
            selected_err_idx = st.selectbox(
                "Select a discrepancy moment to diagnose root cause:",
                range(len(worst_5)),
                format_func=lambda i: f"Discrepancy #{i+1}: {worst_5.iloc[i]['timestamp']} (Error: {worst_5.iloc[i]['abs_error_mw']:.1f} MW)",
                key="err_diag_select"
            )
            err_row = worst_5.iloc[selected_err_idx]
            
            # Prepare row for SHAP
            err_input = err_row[xgb_features].to_frame().T.fillna(0.0)
            explainer = get_shap_explainer(artifacts.get("xgb_model", artifacts.get("model")))
            err_shap = explainer.shap_values(err_input)[0]
            
            err_contrib = pd.DataFrame({
                "Feature": [FEATURE_META.get(f, f) for f in xgb_features],
                "Raw_Value": err_input.iloc[0].values,
                "SHAP_MW": err_shap,
                "Abs_Impact": np.abs(err_shap)
            }).sort_values(by="Abs_Impact", ascending=False)

            c_err1, c_err2 = st.columns([1.1, 0.9])
            with c_err1:
                fig_err_bar = px.bar(
                    err_contrib.head(6).sort_values(by="SHAP_MW", ascending=True),
                    x="SHAP_MW",
                    y="Feature",
                    orientation="h",
                    title=f"Feature Push Breakdown at {pd.to_datetime(err_row['timestamp']).strftime('%Y-%m-%d %H:00')}",
                    color="SHAP_MW",
                    color_continuous_scale="RdBu_r"
                )
                fig_err_bar.update_layout(
                    paper_bgcolor='#111827',
                    plot_bgcolor='#0a0e1a',
                    font=dict(color='#f3f4f6'),
                    margin=dict(t=40, b=10, l=10, r=10),
                    xaxis_title="SHAP Impact (MW)"
                )
                st.plotly_chart(fig_err_bar, use_container_width=True)

            with c_err2:
                top_err_f = err_contrib.iloc[0]
                is_weather = any(w in top_err_f['Feature'] for w in ['Temp', 'Heat', 'Humidity', 'Rain', 'AC'])
                root_cause_text = "Severe Weather Anomaly (Sudden Meteorological Shock)" if is_weather else "Autoregressive Lag Inertia (Demand Shift vs Previous Day Lags)"
                
                st.markdown(f"""
                <div style="background-color: #111827; padding: 16px; border-radius: 10px; border: 1px solid #1f2937; margin-top: 10px;">
                    <div style="color: #fbbf24; font-weight: 700; font-size: 15px; margin-bottom: 6px;">🩺 Post-Mortem Diagnosis</div>
                    <div style="font-size: 13px; color: #d1d5db; line-height: 1.6;">
                        • <b>Timestamp:</b> <code>{err_row['timestamp']}</code><br>
                        • <b>Actual Load:</b> <span style="color: #38bdf8;">{err_row['load']:.1f} MW</span><br>
                        • <b>Model Prediction:</b> <span style="color: #fbbf24;">{err_row['pred_load']:.1f} MW</span><br>
                        • <b>Error Residual:</b> <span style="color: #ef4444;">{err_row['pred_load'] - err_row['load']:+.1f} MW</span><br>
                        • <b>Primary Driver:</b> <b>{top_err_f['Feature']}</b> ({top_err_f['SHAP_MW']:+.1f} MW)<br>
                        • <b>Root-Cause Class:</b> <span style="color: #38bdf8; font-weight: 600;">{root_cause_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_spatial_disaggregation_tab(forecast_matrix_df, target_year: int = 2026, sidebar_pred_mw: float = None):
    st.subheader("🗺️ Delhi Zonal Disaggregation & Substation Stress Map")
    st.caption("Live spatial mapping of aggregate grid demand across 20 primary power distribution nodes and transformer clusters.")

    # Determine whether we have the full forecast matrix or just the sidebar value
    has_forecast = forecast_matrix_df is not None and not forecast_matrix_df.empty

    if has_forecast:
        live_max = float(forecast_matrix_df["forecast_mw"].max())
        live_avg = float(forecast_matrix_df["forecast_mw"].mean())
        
        src_choice = st.radio(
            "⚡ Select Demand Input Source for Spatial Zonal Disaggregation:",
            [
                "📅 5-Day Forecast Timestamp Window (Date & Hour Picker)",
                f"🕹️ Active Sidebar Scenario ({sidebar_pred_mw:.1f} MW)" if sidebar_pred_mw else "🕹️ Active Sidebar Scenario",
                f"🔮 5-Day Forecast Peak ({live_max:.1f} MW)",
                f"🔮 5-Day Forecast Average ({live_avg:.1f} MW)"
            ],
            horizontal=True,
            key="spatial_src_radio"
        )

        if "Timestamp" in src_choice:
            col_date, col_hour, col_kpi = st.columns([1.2, 1.2, 1.6])

            unique_dates = sorted(forecast_matrix_df["timestamp"].dt.date.unique())
            with col_date:
                selected_date = st.selectbox("📅 Select Horizon Date", unique_dates, index=0, key="spatial_date_pick")

            with col_hour:
                selected_hour = st.slider("⏰ Select Hour (IST)", 0, 23, 19, format="%02d:00", key="spatial_hour_pick")

            target_row = forecast_matrix_df[
                (forecast_matrix_df["timestamp"].dt.date == selected_date) &
                (forecast_matrix_df["hour"] == selected_hour)
            ]

            if target_row.empty:
                st.warning("Selected date/hour is not present in the forecast results.")
                return

            aggregate_mw = float(target_row["forecast_mw"].values[0])
            ambient_temp = float(target_row["temperature_2m"].values[0])
            is_rain = float(target_row["precipitation"].values[0]) > 0.1

            with col_kpi:
                st.metric(
                    label=f"Delhi Aggregate Demand @ {selected_hour:02d}:00 IST",
                    value=f"{aggregate_mw:.2f} MW",
                    delta=f"Temp: {ambient_temp:.1f}°C {'🌧️' if is_rain else '☀️'}"
                )
            context_label = f"{selected_date.strftime('%b %d, %Y')} @ {selected_hour:02d}:00 IST"

        elif "Sidebar" in src_choice:
            aggregate_mw = sidebar_pred_mw if sidebar_pred_mw else 0.0
            context_label = f"Sidebar Scenario ({aggregate_mw:.1f} MW)"
        elif "Peak" in src_choice:
            aggregate_mw = live_max
            context_label = f"5-Day Forecast Peak ({live_max:.1f} MW)"
        else:
            aggregate_mw = live_avg
            context_label = f"5-Day Forecast Average ({live_avg:.1f} MW)"
    else:
        st.info("💡 Run the 5-day operational forecast pipeline in Tab 4 to enable timestamp window disaggregation. Currently displaying active sidebar scenario.")
        aggregate_mw = sidebar_pred_mw if sidebar_pred_mw else 0.0
        context_label = f"Sidebar Scenario ({aggregate_mw:.1f} MW)"

    if aggregate_mw <= 0:
        st.warning("No demand value available for spatial disaggregation.")
        return

    # 3. Dynamic Compound Load Allocation across 20 Zonal Hubs
    years_elapsed = max(0, target_year - 2020)
    df_hubs = pd.DataFrame(DELHI_ZONAL_HUBS)

    df_hubs["growth_multiplier"] = (1 + df_hubs["growth_rate"]) ** years_elapsed
    df_hubs["dynamic_weight"] = df_hubs["base_weight"] * df_hubs["growth_multiplier"]
    df_hubs["normalized_share"] = df_hubs["dynamic_weight"] / df_hubs["dynamic_weight"].sum()

    df_hubs["allocated_mw"] = aggregate_mw * df_hubs["normalized_share"]
    df_hubs["stress_pct"] = (df_hubs["allocated_mw"] / df_hubs["rated_capacity_mw"]) * 100.0

    def get_zone_status(p):
        if p >= 85.0:
            return "🔴 Critical Congestion"
        if p >= 70.0:
            return "🟡 Elevated Stress"
        return "🟢 Normal Operating"

    df_hubs["status"] = df_hubs["stress_pct"].apply(get_zone_status)

    # Summary Metrics Row
    c1, c2, c3 = st.columns(3)
    c1.metric("Aggregate Forecast Demand", f"{aggregate_mw:.2f} MW")
    c2.metric("Target Horizon Year", f"{target_year} (+{years_elapsed}y growth)")
    worst_hub = df_hubs.loc[df_hubs["stress_pct"].idxmax()]
    c3.metric("Highest Stress Substation Node", f"{worst_hub['name']} ({worst_hub['stress_pct']:.1f}%)")

    # 4. Mapbox Dark-Matter Geospatial Scatter Map
    st.markdown("---")

    customdata = [
        [
            row["name"],
            row["discom"],
            row["type"],
            float(row["allocated_mw"]),
            float(row["rated_capacity_mw"]),
            float(row["stress_pct"]),
            row["status"]
        ]
        for _, row in df_hubs.iterrows()
    ]

    map_title = f"<b>Delhi Power Distribution Grid Stress Map</b> · {context_label}"

    fig_map = go.Figure(go.Scattermapbox(
        lat=[float(x) for x in df_hubs["lat"]],
        lon=[float(x) for x in df_hubs["lon"]],
        mode="markers+text",
        text=[str(x) for x in df_hubs["name"]],
        textposition="bottom right",
        textfont=dict(size=10, color="#cbd5e1"),
        customdata=customdata,
        marker=dict(
            size=[max(14, min(34, float(mw) * 0.45 + 8)) for mw in df_hubs["allocated_mw"]],
            color=[float(s) for s in df_hubs["stress_pct"]],
            colorscale=[
                [0.0, "#22c55e"],   # Green (<70%)
                [0.5, "#eab308"],   # Amber (75%)
                [0.75, "#f97316"],  # Orange (85%)
                [1.0, "#ef4444"]    # Red (90%+)
            ],
            cmin=40.0,
            cmax=95.0,
            showscale=True,
            colorbar=dict(
                title=dict(text="Transformer<br>Stress (%)", font=dict(color="#f8fafc", size=12)),
                ticksuffix="%",
                len=0.8,
                tickfont=dict(color="#f8fafc")
            ),
            opacity=0.92,
            allowoverlap=True
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Utility / DISCOM: <b>%{customdata[1]}</b><br>"
            "Zone Type: %{customdata[2]}<br>"
            "Allocated Demand: <b>%{customdata[3]:.2f} MW</b><br>"
            "Rated Substation Limit: <b>%{customdata[4]:.0f} MW</b><br>"
            "Transformer Loading: <b>%{customdata[5]:.1f}%</b><br>"
            "Status: <b>%{customdata[6]}</b>"
            "<extra></extra>"
        )
    ))

    fig_map.update_layout(
        title=dict(
            text=map_title,
            font=dict(size=15, color="#f8fafc")
        ),
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=28.635, lon=77.160),
            zoom=10.2
        ),
        paper_bgcolor="#181b20",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        margin=dict(t=40, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # 5. Diagnostic Substation Table
    st.markdown("#### 📋 20-Node Granular Substation Diagnostics")
    table_df = df_hubs[["name", "discom", "type", "allocated_mw", "rated_capacity_mw", "stress_pct", "status"]].copy()
    table_df.sort_values(by="stress_pct", ascending=False, inplace=True)

    table_df.rename(columns={
        "name": "Substation / Power Circle",
        "discom": "Utility",
        "type": "Demographic Type",
        "allocated_mw": "Allocated Demand (MW)",
        "rated_capacity_mw": "Rated Limit (MW)",
        "stress_pct": "Grid Loading (%)",
        "status": "Operational State"
    }, inplace=True)

    st.dataframe(
        table_df.style.format({
            "Allocated Demand (MW)": "{:.2f}",
            "Rated Limit (MW)": "{:.0f}",
            "Grid Loading (%)": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


def main():
    artifacts, df_features = load_pipeline()
    metrics = artifacts.get('metrics', {})
    xgb_features = artifacts.get("xgb_features", artifacts.get("feature_cols", []))
    trend_features = artifacts.get("trend_features", ["time_step"])
    genesis = artifacts.get("genesis_time", artifacts.get("split_info", {}).get("train_min", "2000-01-01"))
    trend_model = artifacts.get("trend_model")
    xgb_model = artifacts.get("xgb_model", artifacts.get("model"))

    st.markdown('<div class="header-title">⚡ GridSathi</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Delhi Electricity Demand Forecasting & Grid Risk Monitor</div>', unsafe_allow_html=True)

    st.sidebar.header("🕹️ Scenario Inputs")
    latest_row = df_features.iloc[-1]

    temp_input = st.sidebar.slider("Temperature (°C)", min_value=5.0, max_value=48.0, value=float(latest_row.get('temperature_2m', 30.0)), step=0.5)
    rh_input = st.sidebar.slider("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=float(latest_row.get('relative_humidity_2m', 60.0)), step=1.0)
    raw_precip = latest_row.get('precipitation', 0.0)
    precip_default = 0.0 if (pd.isna(raw_precip) or raw_precip is None) else float(raw_precip)
    precip_input = st.sidebar.slider("Precipitation (mm)", min_value=0.0, max_value=50.0, value=precip_default, step=0.5)
    
    hour_input = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=19, step=1)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_input = st.sidebar.selectbox("Day of Week", days, index=2)
    dayofweek_val = days.index(day_input)

    year_input = st.sidebar.number_input("Forecast Year", min_value=2000, max_value=2035, value=2026, step=1)
    month_input = st.sidebar.slider("Month", min_value=1, max_value=12, value=6, step=1)
    is_holiday_input = st.sidebar.checkbox("Is Public Holiday?", value=False)

    forecast_ts = pd.Timestamp(f"{year_input}-{month_input:02d}-15 {hour_input:02d}:00:00")

    input_dict = {
        'timestamp': forecast_ts,
        'datetime': forecast_ts,
        'temperature_2m': temp_input,
        'temperature': temp_input,
        'relative_humidity_2m': rh_input,
        'humidity': rh_input,
        'windspeed_10m': 10.0,
        'precipitation': precip_input,
        'hour': hour_input,
        'hour_of_day': hour_input,
        'dayofweek': dayofweek_val,
        'day_of_week': dayofweek_val,
        'month': month_input,
        'year': year_input,
        'is_holiday': 1 if is_holiday_input else 0,
        'is_weekend': 1 if dayofweek_val in [5, 6] else 0,
        'load': latest_row['load']
    }

    for col in ['lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h']:
        if col in latest_row:
            input_dict[col] = float(latest_row[col])

    pred_demand = predict_demand(input_dict, forecast_ts, artifacts)
    
    processed_single = compute_engineered_features(pd.DataFrame([input_dict]))
    ac_proxy = float(processed_single['ac_load_proxy'].values[0]) if 'ac_load_proxy' in processed_single.columns else 1.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value-amber">{pred_demand:.1f} MW</div>
            <div class="metric-label">PREDICTED ELECTRICITY DEMAND</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value-blue">{ac_proxy:.2f}</div>
            <div class="metric-label">AC LOAD PROXY INDEX</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        is_high_risk = (18 <= hour_input <= 21) and (month_input in [5, 6, 7, 8]) and (ac_proxy > 1.5)
        is_warning_risk = not is_high_risk and ((ac_proxy > 1.0) or (temp_input > 35.0) or ((18 <= hour_input <= 21) and month_input in [5, 6, 7, 8]))

        if is_high_risk:
            st.markdown("""
            <div class="risk-box-high">
                🔴 HIGH RISK — Rule-Based Risk Indicator<br>
                <span style="font-size: 13px; font-weight: normal; color: #f3f4f6;">High evening demand/ramp conditions detected.</span>
            </div>
            """, unsafe_allow_html=True)
        elif is_warning_risk:
            st.markdown("""
            <div class="risk-box-warning">
                🟡 WARNING — Rule-Based Risk Indicator<br>
                <span style="font-size: 13px; font-weight: normal; color: #f3f4f6;">Elevated demand/ramp conditions detected. Increased monitoring recommended.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-box-normal">
                🟢 NORMAL — Rule-Based Risk Indicator<br>
                <span style="font-size: 13px; font-weight: normal; color: #f3f4f6;">Demand conditions are within the expected operating range.</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1f2937;'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Model Evaluation & Baseline", 
        "🔍 Local Feature Attribution (SHAP)", 
        "📉 Dynamic Historical Evaluation", 
        "🔮 5-Day Operational Forecast & Grid Risk",
        "🗺️ Area-Wise Disaggregation"
    ])

    with tab1:
        st.subheader("Model Performance Benchmark (Holdout Test Set)")
        st.caption("Evaluating Two-Stage Hybrid Ridge + XGBoost against a 24-Hour Naive Baseline on the chronological holdout set.")
        
        xgb_m = metrics.get('xgb', {'mae': 0, 'rmse': 0, 'mape': 0})
        naive_m = metrics.get('naive', {'mae': 0, 'rmse': 0, 'mape': 0})
        cv_m = metrics.get('cv_xgb', {})
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Hybrid Pipeline MAE", f"{xgb_m['mae']:.2f} MW", delta=f"{xgb_m['mae'] - naive_m['mae']:.2f} vs Naive", delta_color="inverse")
        with m_col2:
            st.metric("Hybrid Pipeline RMSE", f"{xgb_m['rmse']:.2f} MW", delta=f"{xgb_m['rmse'] - naive_m['rmse']:.2f} vs Naive", delta_color="inverse")
        with m_col3:
            st.metric("Hybrid Pipeline MAPE", f"{xgb_m['mape']:.2f}%", delta=f"{xgb_m['mape'] - naive_m['mape']:.2f}% vs Naive", delta_color="inverse")
            
        benchmark_data = {
            "Model": ["24h Naive Baseline", "Hybrid Ridge + XGBoost", "XGBoost 5-Fold CV (Residuals)"],
            "MAE (MW)": [f"{naive_m['mae']:.2f}", f"{xgb_m['mae']:.2f}", f"{cv_m.get('mae', 0):.2f}"],
            "RMSE (MW)": [f"{naive_m['rmse']:.2f}", f"{xgb_m['rmse']:.2f}", f"{cv_m.get('rmse', 0):.2f}"],
            "MAPE (%)": [f"{naive_m['mape']:.2f}%", f"{xgb_m['mape']:.2f}%", f"{cv_m.get('mape', 0):.2f}%"]
        }
        st.table(pd.DataFrame(benchmark_data))

    with tab2:
        render_dynamic_shap_tab(artifacts, input_dict, forecast_ts, df_features)

    with tab3:
        render_dynamic_actual_vs_pred_tab(
            df_features=df_features,
            artifacts=artifacts,
            scenario_info={"timestamp": forecast_ts, "pred_demand": pred_demand}
        )

    with tab4:
        st.subheader("🔮 Multi-Day Real-Time Demand Forecasting Engine")
        st.caption("Fetches live meteorological feeds for Delhi (IST) and executes recursive two-stage hybrid inference across 5-day horizon.")

        # Selection Controls
        c1, c2, c3 = st.columns([1.5, 2, 1.2])
        
        now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
        today_date = now_ist.date()
        max_date = today_date + pd.Timedelta(days=5)

        with c1:
            selected_date = st.date_input(
                "Select Target Horizon Date",
                value=today_date,
                min_value=today_date,
                max_value=max_date
            )

        with c2:
            selected_hours = st.slider(
                "Select Hourly Time Window (IST)",
                min_value=0,
                max_value=23,
                value=(0, 23),
                step=1
            )

        with c3:
            st.write("")
            st.write("")
            trigger_analysis = st.button("⚡ Run Forecasting Pipeline", type="primary", use_container_width=True)

        # Forecast Pipeline Trigger & Execution
        if trigger_analysis or "live_forecast_results" not in st.session_state:
            with st.spinner("Connecting to Weather API & executing recursive hybrid model inference..."):
                weather_raw = fetch_5day_hourly_weather()
                genesis = artifacts.get("genesis_time", artifacts.get("split_info", {}).get("train_min", "2000-01-01"))
                trend_model = artifacts.get("trend_model")
                xgb_model = artifacts.get("xgb_model", artifacts.get("model"))
                trend_features = artifacts.get("trend_features", ["time_step"])
                xgb_features = artifacts.get("xgb_features", artifacts.get("feature_cols", []))

                matrix = create_future_feature_matrix(weather_raw, genesis, df_features)
                df_forecast_res = execute_recursive_forecast(
                    matrix, trend_model, xgb_model, trend_features, xgb_features
                )
                st.session_state["live_forecast_results"] = df_forecast_res

        df_full_forecast = st.session_state["live_forecast_results"]
        
        # Apply Horizon Filters
        mask = (df_full_forecast["timestamp"].dt.date == selected_date) & \
               (df_full_forecast["hour"] >= selected_hours[0]) & \
               (df_full_forecast["hour"] <= selected_hours[1])
        df_target_window = df_full_forecast.loc[mask].copy()
        if df_target_window.empty:
            df_target_window = df_full_forecast.iloc[:24].copy()

        # Display Operational Summary Metrics
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        avg_load = float(df_target_window["forecast_mw"].mean())
        max_idx = df_target_window["forecast_mw"].idxmax()
        max_row = df_target_window.loc[max_idx]
        avg_temp = float(df_target_window["temperature_2m"].mean())
        rain_detected = float(df_target_window["precipitation"].sum()) > 0.5

        m1.metric("Window Average Demand", f"{avg_load:.2f} MW")
        m2.metric("Peak Window Demand", f"{max_row['forecast_mw']:.2f} MW", delta=f"Hour {int(max_row['hour']):02d}:00")
        m3.metric("Window Avg Temp", f"{avg_temp:.1f} °C")
        m4.metric("Precipitation Alert", "Rain / Showers 🌧️" if rain_detected else "Clear Conditions ☀️")

        # 1. Cast time series data to clean native Python lists to prevent Plotly 6 bdata binary serialization issues
        x_timestamps = [ts.strftime("%Y-%m-%d %H:%M") for ts in df_target_window["timestamp"]]
        temp_vals = [float(t) for t in df_target_window["temperature_2m"]]
        load_vals = [float(l) for l in df_target_window["forecast_mw"]]

        min_load = float(min(load_vals))
        max_load = float(max(load_vals))
        load_padding = max(15.0, (max_load - min_load) * 0.25)

        min_temp = float(min(temp_vals))
        max_temp = float(max(temp_vals))

        # 2. Initialize Dual-Axis Canvas
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 3. Add Vertical Temperature Bars (Secondary Y-Axis / Background Layer)
        fig.add_trace(
            go.Bar(
                x=x_timestamps,
                y=temp_vals,
                name="Temperature (°C)",
                marker_color="rgba(56, 189, 248, 0.35)",  # Translucent Sky Blue
                marker_line=dict(width=1.5, color="rgba(56, 189, 248, 0.85)"),
                hovertemplate="<b>%{x}</b><br>Temperature: <b>%{y:.1f} °C</b><extra></extra>",
            ),
            secondary_y=True,
        )

        # 4. Add Forecast Total Load Line (Primary Y-Axis / Foreground Layer) - Vibrant Red Line
        fig.add_trace(
            go.Scatter(
                x=x_timestamps,
                y=load_vals,
                name="Forecast Total Load (MW)",
                mode="lines+markers",
                line=dict(color="#ef4444", width=3.5, shape="spline"),  # Bright Red Line
                marker=dict(size=7, color="#ef4444", line=dict(width=1.5, color="#ffffff")),
                hovertemplate="<b>%{x}</b><br>Forecast Demand: <b>%{y:.2f} MW</b><extra></extra>",
            ),
            secondary_y=False,
        )

        # 5. Apply High-End Dark Card Layout
        formatted_date = selected_date.strftime("%B %d, %Y")
        fig.update_layout(
            title=dict(
                text=f"<b>Delhi Grid Projected Hourly Load</b> · Forecast Horizon · {formatted_date}",
                font=dict(size=15, color="#f8fafc"),
                x=0.01,
                y=0.96,
            ),
            paper_bgcolor="#181b20",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="sans-serif"),
            margin=dict(t=60, b=50, l=40, r=40),
            bargap=0.35,  # Separates vertical bars cleanly
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(color="#cbd5e1"),
            ),
            # Top-right indicator badge
            annotations=[
                dict(
                    text="TIME SERIES WITH DUAL Y-AXIS",
                    xref="paper",
                    yref="paper",
                    x=1.0,
                    y=1.12,
                    showarrow=False,
                    font=dict(size=10, color="#64748b"),
                )
            ],
        )

        # Primary Y-Axis (Left: MW Demand - Red Theme)
        fig.update_yaxes(
            title_text="Demand (MW)",
            title_font=dict(color="#ef4444", size=13),
            tickfont=dict(color="#ef4444"),
            range=[max(0.0, min_load - load_padding), max_load + load_padding],
            gridcolor="#262c36",
            zeroline=False,
            secondary_y=False,
        )

        # Secondary Y-Axis (Right: Temperature °C - Sky Blue Theme)
        fig.update_yaxes(
            title_text="Temperature (°C)",
            title_font=dict(color="#38bdf8", size=13),
            tickfont=dict(color="#38bdf8"),
            range=[0.0, max(50.0, max_temp * 1.35)],  # Keeps vertical bars anchored cleanly at bottom
            showgrid=False,
            zeroline=False,
            secondary_y=True,
        )

        # X-Axis Styling
        fig.update_xaxes(
            title_text="Timestamp (IST)",
            gridcolor="#262c36",
            zeroline=False,
            showline=True,
            linecolor="#334155",
            tickfont=dict(color="#cbd5e1"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tabular View
        with st.expander("📋 Detailed Hourly Meteorological & Load Data", expanded=False):
            disp_df = df_target_window[["timestamp", "temperature_2m", "relative_humidity_2m", "precipitation", "forecast_mw"]].copy()
            disp_df.columns = ["Timestamp (IST)", "Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "Forecast Load (MW)"]
            disp_df["Temperature (°C)"] = disp_df["Temperature (°C)"].map("{:.1f}".format)
            disp_df["Humidity (%)"] = disp_df["Humidity (%)"].map("{:.0f}%".format)
            disp_df["Rainfall (mm)"] = disp_df["Rainfall (mm)"].map("{:.2f}".format)
            disp_df["Forecast Load (MW)"] = disp_df["Forecast Load (MW)"].map("{:.2f}".format)
            st.dataframe(disp_df, use_container_width=True, hide_index=True)

        # ---------------- Operational Peak Demand SHAP Decomposition ----------------
        st.markdown("---")
        st.markdown("#### 🔍 Operational Peak Demand XAI Decomposition")
        st.caption(f"Deconstructs the exact meteorological and cyclical feature attributions driving the window peak of **{max_row['forecast_mw']:.2f} MW** at **Hour {int(max_row['hour']):02d}:00 IST**.")

        explainer = get_shap_explainer(artifacts.get("xgb_model", artifacts.get("model")))
        peak_input = max_row[xgb_features].to_frame().T.fillna(0.0)
        peak_shap = explainer.shap_values(peak_input)[0]

        peak_contrib = pd.DataFrame({
            "Feature": [FEATURE_META.get(f, f) for f in xgb_features],
            "Raw_Value": peak_input.iloc[0].values,
            "SHAP_MW": peak_shap,
            "Abs_Impact": np.abs(peak_shap)
        }).sort_values(by="Abs_Impact", ascending=False)

        top_pos = peak_contrib[peak_contrib["SHAP_MW"] > 0].head(3)
        c_pk1, c_pk2 = st.columns([1.2, 0.8])
        with c_pk1:
            fig_pk_bar = px.bar(
                peak_contrib.head(7).sort_values(by="SHAP_MW", ascending=True),
                x="SHAP_MW",
                y="Feature",
                orientation="h",
                title=f"Peak Hour Load Drivers — {selected_date.strftime('%b %d')} @ {int(max_row['hour']):02d}:00 IST",
                color="SHAP_MW",
                color_continuous_scale="Reds"
            )
            fig_pk_bar.update_layout(
                paper_bgcolor='#111827',
                plot_bgcolor='#0a0e1a',
                font=dict(color='#f3f4f6'),
                margin=dict(t=40, b=10, l=10, r=10),
                xaxis_title="Push on Peak Demand (MW)"
            )
            st.plotly_chart(fig_pk_bar, use_container_width=True)

        with c_pk2:
            st.markdown(f"""
            <div style="background-color: #111827; padding: 18px; border-radius: 10px; border: 1px solid #1f2937; margin-top: 10px;">
                <div style="color: #fbbf24; font-weight: 700; font-size: 15px; margin-bottom: 8px;">⚡ Peak Ramp Operational Intelligence</div>
                <div style="font-size: 13px; color: #d1d5db; line-height: 1.6;">
                    The operational peak of <b>{max_row['forecast_mw']:.2f} MW</b> is driven primarily by:
                    <ul style="margin-top: 6px; margin-bottom: 10px; padding-left: 18px;">
                        {"".join([f"<li><b>{r['Feature']}</b>: <span style='color: #ef4444; font-weight: 600;'>+{r['SHAP_MW']:.1f} MW</span></li>" for _, r in top_pos.iterrows()])}
                    </ul>
                    <b>Grid Recommendation:</b> Pre-schedule spinning reserves and initiate automated demand response 45 minutes prior to Hour {int(max_row['hour']):02d}:00 IST.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab5:
        live_forecast_df = st.session_state.get("live_forecast_results", None)
        render_spatial_disaggregation_tab(
            forecast_matrix_df=live_forecast_df,
            target_year=year_input,
            sidebar_pred_mw=pred_demand
        )

    st.markdown("<hr style='border-color: #1f2937;'>", unsafe_allow_html=True)
    st.markdown('<div class="footer-text">GridSathi | Delhi Electricity Demand Forecasting | Hybrid Two-Stage Pipeline</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
