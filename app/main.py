import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add root directory to sys.path to import src.features.build_features
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features.build_features import compute_engineered_features

st.set_page_config(
    page_title="GridSathi — Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00d2ff;
    }
    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
    }
    .risk-box-alert {
        background-color: rgba(255, 75, 75, 0.2);
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-box-normal {
        background-color: rgba(9, 171, 59, 0.2);
        border: 1px solid #09ab3b;
        color: #09ab3b;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    model_path = os.path.join(ROOT_DIR, "model", "xgb_model.pkl")
    features_path = os.path.join(ROOT_DIR, "data", "processed", "features.csv")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train model first.")
        st.stop()
        
    if not os.path.exists(features_path):
        st.error(f"Features file not found at {features_path}. Please build features first.")
        st.stop()
        
    payload = joblib.load(model_path)
    df_features = pd.read_csv(features_path)
    df_features['datetime'] = pd.to_datetime(df_features['datetime'])
    df_features = df_features.sort_values('datetime').reset_index(drop=True)
    
    return payload, df_features

def main():
    payload, df_features = load_model_and_data()
    model = payload['model']
    feature_cols = payload['feature_cols']
    metrics = payload['metrics']
    feat_imp = payload['feature_importances']
    
    st.title("⚡ GridSathi — Delhi Electricity Demand Forecaster")
    st.markdown("Real-data demand forecasting for Delhi power grid powered by XGBoost machine learning.")
    
    st.sidebar.header("🕹️ Scenario Inputs")
    
    # Recent reference values for baseline defaults
    latest_row = df_features.iloc[-1]
    
    temp_input = st.sidebar.slider("Temperature (°C)", min_value=5.0, max_value=48.0, value=float(latest_row.get('temperature_2m', 30.0)), step=0.5)
    rh_input = st.sidebar.slider("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=float(latest_row.get('relative_humidity_2m', 60.0)), step=1.0)
    hour_input = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=19, step=1)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_input = st.sidebar.selectbox("Day of Week", days, index=2)
    dayofweek_val = days.index(day_input)
    
    month_input = st.sidebar.slider("Month", min_value=1, max_value=12, value=6, step=1)
    is_holiday_input = st.sidebar.checkbox("Is Public Holiday?", value=False)
    
    # Construct input row using recent historical lag averages to ensure complete feature matrix
    input_dict = {
        'datetime': pd.Timestamp(f"2023-{month_input:02d}-15 {hour_input:02d}:00:00"),
        'temperature_2m': temp_input,
        'temperature': temp_input,
        'relative_humidity_2m': rh_input,
        'humidity': rh_input,
        'windspeed_10m': 10.0,
        'hour': hour_input,
        'hour_of_day': hour_input,
        'dayofweek': dayofweek_val,
        'day_of_week': dayofweek_val,
        'month': month_input,
        'is_holiday': 1 if is_holiday_input else 0,
        'is_weekend': 1 if dayofweek_val in [5, 6] else 0,
        'solar_generation': 0.0 if (hour_input < 6 or hour_input > 19) else 150.0,
        'load': latest_row['load']
    }
    
    # Use latest historical lag/rolling values as reference context
    for col in ['lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h']:
        input_dict[col] = float(latest_row[col])
        
    input_df = pd.DataFrame([input_dict])
    processed_input_df = compute_engineered_features(input_df)
    
    # Align exact feature matrix columns
    X_single = processed_input_df[feature_cols]
    pred_demand = float(model.predict(X_single)[0])
    
    # Main Dashboard Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pred_demand:.1f} MW</div>
            <div class="metric-label">Predicted Electricity Demand</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        ac_proxy = float(processed_input_df['ac_load_proxy'].values[0])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{ac_proxy:.2f}</div>
            <div class="metric-label">AC Load Proxy Index</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Duck Curve Risk Indicator (rule-based analytical indicator)
        is_duck_curve_risk = (18 <= hour_input <= 21) and (month_input in [5, 6, 7, 8]) and (ac_proxy > 1.5)
        if is_duck_curve_risk:
            st.markdown("""
            <div class="risk-box-alert">
                ⚠️ Duck Curve Risk Indicator: HIGH RISK<br>
                <span style="font-size: 12px; font-weight: normal;">Evening peak demand ramp (18:00-21:00) during summer AC proxy surge.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-box-normal">
                ✅ Duck Curve Risk Indicator: NORMAL<br>
                <span style="font-size: 12px; font-weight: normal;">Grid operating within standard ramping limits.</span>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Tabs for detailed views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Evaluation & Baseline", 
        "🔍 Feature Importances", 
        "📈 Historical Actual vs Predicted", 
        "🎛️ What-If Temperature Sensitivity"
    ])
    
    with tab1:
        st.subheader("Model Performance Benchmark (Holdout Test Set)")
        st.caption("Evaluating XGBoost against a 24-Hour Naive Baseline on the chronological 3-month holdout set.")
        
        xgb_m = metrics['xgb']
        naive_m = metrics['naive']
        cv_m = metrics.get('cv_xgb', {})
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("XGBoost MAE", f"{xgb_m['mae']:.2f} MW", delta=f"{xgb_m['mae'] - naive_m['mae']:.2f} vs Naive", delta_color="inverse")
        with m_col2:
            st.metric("XGBoost RMSE", f"{xgb_m['rmse']:.2f} MW", delta=f"{xgb_m['rmse'] - naive_m['rmse']:.2f} vs Naive", delta_color="inverse")
        with m_col3:
            st.metric("XGBoost MAPE", f"{xgb_m['mape']:.2f}%", delta=f"{xgb_m['mape'] - naive_m['mape']:.2f}% vs Naive", delta_color="inverse")
            
        benchmark_data = {
            "Model": ["24h Naive Baseline", "XGBoost Regressor", "XGBoost 5-Fold CV (Train)"],
            "MAE (MW)": [f"{naive_m['mae']:.2f}", f"{xgb_m['mae']:.2f}", f"{cv_m.get('mae', 0):.2f}"],
            "RMSE (MW)": [f"{naive_m['rmse']:.2f}", f"{xgb_m['rmse']:.2f}", f"{cv_m.get('rmse', 0):.2f}"],
            "MAPE (%)": [f"{naive_m['mape']:.2f}%", f"{xgb_m['mape']:.2f}%", f"{cv_m.get('mape', 0):.2f}%"]
        }
        st.table(pd.DataFrame(benchmark_data))
        
    with tab2:
        st.subheader("XGBoost Top 10 Feature Importances")
        fig_imp = px.bar(
            feat_imp.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance Scores",
            color="importance",
            color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with tab3:
        st.subheader("Historical Actual vs Predicted Load Curve")
        sample_df = df_features.tail(500).copy()
        X_sample = sample_df[feature_cols]
        sample_df['pred_load'] = model.predict(X_sample)
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=sample_df['datetime'], y=sample_df['load'], mode='lines', name='Actual Load (MW)', line=dict(color='#00d2ff', width=2)))
        fig_curve.add_trace(go.Scatter(x=sample_df['datetime'], y=sample_df['pred_load'], mode='lines', name='Predicted Load (MW)', line=dict(color='#ff9900', width=1.5, dash='dash')))
        fig_curve.update_layout(title="Delhi Power Consumption (Recent Holdout Window)", xaxis_title="Timestamp", yaxis_title="Load (MW)", template="plotly_dark")
        st.plotly_chart(fig_curve, use_container_width=True)
        
    with tab4:
        st.subheader("What-If Temperature Sensitivity Analysis")
        st.caption("Simulate demand delta by perturbing temperature around current scenario.")
        
        temp_delta = st.slider("Shift Temperature Delta (°C)", min_value=-5.0, max_value=5.0, value=0.0, step=0.5)
        
        deltas = np.linspace(-5.0, 5.0, 11)
        sim_results = []
        
        for d in deltas:
            sim_dict = input_dict.copy()
            sim_dict['temperature_2m'] = temp_input + d
            sim_dict['temperature'] = temp_input + d
            sim_df = compute_engineered_features(pd.DataFrame([sim_dict]))
            X_sim = sim_df[feature_cols]
            sim_pred = float(model.predict(X_sim)[0])
            sim_results.append({
                'Temp Delta (°C)': d,
                'Simulated Temp (°C)': temp_input + d,
                'Predicted Demand (MW)': sim_pred,
                'Delta (MW)': sim_pred - pred_demand
            })
            
        sim_res_df = pd.DataFrame(sim_results)
        
        fig_sim = px.line(
            sim_res_df,
            x="Simulated Temp (°C)",
            y="Predicted Demand (MW)",
            markers=True,
            title="Temperature vs Demand Response Curve",
            template="plotly_dark"
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        st.dataframe(sim_res_df[['Temp Delta (°C)', 'Simulated Temp (°C)', 'Predicted Demand (MW)', 'Delta (MW)']])

if __name__ == "__main__":
    main()
