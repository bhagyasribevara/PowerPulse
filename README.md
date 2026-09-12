# ⚡ PowerPulse — Intelligent Delhi Electricity Demand Forecasting & Grid Risk Management System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble%20ML-EB6E08?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/XAI-TreeSHAP-00ADD8)](https://shap.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly%20%26%20Mapbox-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PowerPulse** is an advanced, production-grade machine learning decision-support platform designed to forecast electricity demand, monitor duck curve ramping risks, decompose peak load drivers with Game-Theoretic Explainable AI (TreeSHAP), and spatially disaggregate transformer loading across **20 primary power distribution substations** in the National Capital Territory of Delhi.

---

## 📌 Problem Statement

Delhi's electricity grid is among the most demanding and dynamic urban power systems in the world, serving over 30 million residents across five distinct distribution utilities (**BRPL, TPDDL, BYPL, NDMC, and MES**). Grid operators face severe operational and economic challenges:

1. **Extreme Weather Shocks & Heatwaves**: Summer temperatures in Delhi frequently exceed **45°C**, triggering non-linear surges in domestic and commercial air conditioning demand. Cooling accounts for over **50% of peak summer electricity consumption**.
2. **Aggressive Evening Ramp ("Duck Curve" Stress)**: In late afternoon and early evening (18:00–22:00 IST), solar irradiance collapses while domestic cooling and lighting loads surge simultaneously, causing steep ramps exceeding **50–100 MW/minute**.
3. **Substation Congestion & Localized Blackouts**: Uneven real-estate growth across Delhi creates severe spatial bottlenecks where localized transformer clusters exceed **85–90% rated capacity**, risking equipment failure and rolling blackouts.
4. **Economic Penalties Under Deviation Settlement Mechanism (DSM)**: Over-forecasting leads to costly idle spinning reserve commitments, while under-forecasting incurs heavy grid deviation penalties under the **Indian Electricity Grid Code (IEGC)**.

---

## 💡 The PowerPulse Solution

**PowerPulse** addresses these challenges through a unified empirical architecture combining econometric trend modeling, gradient-boosted residual learning, real-time live meteorological telemetry, and geospatial GIS intelligence:

```
                                  ┌──────────────────────────────────────────────┐
                                  │           Open-Meteo & WeatherAPI            │
                                  │   (Live 7-Day / 168-Hour Hourly Feeds)       │
                                  └──────────────────────┬───────────────────────┘
                                                         │
┌───────────────────────────────┐                        ▼
│ 23 Years Hourly Data (Delhi)  │         ┌──────────────────────────────┐
│  (210,000+ Observations)      ├────────►│ Anti-Leakage Feature Engine  │
└───────────────────────────────┘         │ (Cyclical, Thermal, Lags)    │
                                          └──────────────┬───────────────┘
                                                         │
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │ Two-Stage Hybrid Pipeline    │
                                          │ • Stage 1: Ridge Trend       │
                                          │ • Stage 2: XGBoost Residuals │
                                          └──────────────┬───────────────┘
                                                         │
             ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
             ▼                                           ▼                                           ▼
┌─────────────────────────┐                 ┌─────────────────────────┐                 ┌─────────────────────────┐
│   Game-Theoretic XAI    │                 │   Operational 7-Day     │                 │   20-Node Geospatial    │
│ (TreeSHAP Attributions) │                 │ Forecast & Ramp Monitor │                 │ Substation Stress Map   │
└─────────────────────────┘                 └─────────────────────────┘                 └─────────────────────────┘
```

---

## 🌟 Key Modules & Features

### 1. 🕹️ Interactive Scenario Simulator (Sidebar)
- Live sliders for **Temperature (°C)**, **Relative Humidity (%)**, **Precipitation (mm)**, **Hour of Day (0–23)**, **Day of Week**, **Month (1–12)**, **Target Year (2000–2035)**, and **Public Holiday Flags**.
- **Real-Time AC Load Proxy Index**: Evaluates exponential thermal stress activation ($e^{\frac{T - 30}{5}}$ for $T > 30^\circ\text{C}$).
- **Rule-Based Duck Curve Risk Indicator**: Real-time traffic-light alert (🟢 Normal, 🟡 Warning, 🔴 High Risk) flagging hazardous evening ramp windows during heatwave months.

### 2. 📈 Model Evaluation & Empirical Baseline (Tab 1)
- Evaluates the Two-Stage Hybrid model against a **24-Hour Naive Baseline** on a chronological holdout test set.
- Reports **MAE (MW)**, **RMSE (MW)**, and **MAPE (%)** across holdout test data and 5-Fold `TimeSeriesSplit` cross-validation.
- Demonstrates a **~38.8% relative error reduction** over naive persistence methods.

### 3. 🔍 Explainable AI Analytics Hub — TreeSHAP (Tab 2)
- **Local Prediction Attribution**: Deconstructs any active simulation or historical timestamp into an additive Waterfall chart showing exact individual feature pushes (in $\pm\text{MW}$) from the base expected value.
- **Natural Language AI Executive Briefing**: Automatically synthesized operational narrative identifying primary upward demand drivers and downside relief factors.
- **Global Feature Importance & Non-Linear Dependence**: Visualizes global mean absolute SHAP values and interactive 3D dependence curves (e.g., Temperature vs. SHAP MW colored by Relative Humidity).

### 4. 📉 Dynamic Historical Evaluation & Root-Cause Post-Mortem (Tab 3)
- Interactive time-window selector with presets (*Peak Summer Heatwave 2023, Monsoon Surge, Winter Cold Wave, Full Dataset*).
- Live recalculation of MAE, RMSE, and MAPE metrics over arbitrary historical date ranges.
- **Root-Cause Post-Mortem Diagnostics**: Automated TreeSHAP inspection on the top 5 peak error moments to classify discrepancy origins into *Severe Meteorological Shocks* vs. *Autoregressive Lag Inertia*.

### 5. 🔮 Live 7-Day Operational Forecast & Grid Ramp Monitor (Tab 4)
- **Multi-Day Horizon Engine**: Queries live meteorological feeds for Delhi (IST) across **7 full days (168 hours)** from Open-Meteo and WeatherAPI.
- **Dual-Axis High-End Visualizer**:
  - **Vertical Sky-Blue Temperature Bars** on Secondary Y-Axis (Right: $0\text{–}50^\circ\text{C}$).
  - **Vibrant Red Spline Forecast Demand Curve** with white-halo data markers on Primary Y-Axis (Left: $\text{MW}$).
- **Operational Peak Ramp Decomposition**: Automatically identifies the diurnal peak hour and isolates the top contributing meteorological and cyclical drivers with spinning reserve dispatch recommendations.

### 6. 🗺️ 20-Node Geospatial Zonal Disaggregation & Substation Stress Map (Tab 5)
- **Mapbox Carto Dark-Matter Spatial Map**: Renders **20 primary power distribution nodes and transformer clusters** across Delhi:
  - **BRPL (South & West Delhi)**: Dwarka Sub-City, Saket & Hauz Khas, Janakpuri, Vasant Kunj, Nehru Place, Najafgarh.
  - **TPDDL (North & North-West Delhi)**: Rohini Sector Complex, Narela Industrial, Pitampura, Badli & Bawana, Model Town.
  - **BYPL (Central & East Delhi)**: Laxmi Nagar, Mayur Vihar, Chandni Chowk, Shahdara, Patparganj.
  - **NDMC & MES (Institutional Core)**: Connaught Place, Chanakyapuri Diplomatic Enclave, Delhi Cantonment.
- **Dynamic Compound Growth Allocation**: Adjusts nodal demand weights according to demographic real-estate growth rates:
  $$\text{Allocated Load}_i(t) = \text{Demand}_{\text{agg}}(t) \times \frac{W_{i,0} \cdot (1 + g_i)^{\Delta t}}{\sum_j W_{j,0} \cdot (1 + g_j)^{\Delta t}}$$
- **Transformer Stress Color Scale**: Green ($<70\%$), Amber ($75\%$), Orange ($85\%$), Red ($90\%+$ Critical Congestion).
- **Substation Diagnostics Table**: 20-row sortable operational state matrix with formatted MW demand, rated limits, and loading percentages.

---

## 📊 Model Performance Benchmark

| Model Architecture | MAE (MW) | RMSE (MW) | MAPE (%) | Relative Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **24-Hour Naive Baseline** ($Load_t = Load_{t-24h}$) | 64.97 MW | 81.55 MW | 13.17% | Baseline (0.0%) |
| **Two-Stage Hybrid (Ridge + XGBoost)** | **40.03 MW** | **50.46 MW** | **8.06%** | **+38.8% Improvement** |
| **XGBoost (5-Fold TimeSeriesSplit CV)** | **40.02 MW** | **50.15 MW** | **7.96%** | **+39.5% Improvement** |

---

## 🧮 Mathematical Formulation

### 1. Two-Stage Hybrid Decomposition
$$\hat{Y}(t) = \hat{Y}_{\text{trend}}(t) + \hat{R}(X(t))$$
- **Stage 1 (Secular Macro Trend)**: Fitted using Linear Ridge Regression on long-term time steps from genesis ($t_0$):
  $$\hat{Y}_{\text{trend}}(t) = \beta_0 + \beta_1 \cdot \text{time\_step}(t)$$
- **Stage 2 (Non-Linear Residual Dynamics)**: Fitted using Extreme Gradient Boosting (XGBoost) on weather, calendar, and autoregressive lag residuals:
  $$\hat{R}(X(t)) = \sum_{k=1}^K f_k(X(t)), \quad f_k \in \mathcal{F}$$

### 2. Cooling Proxy & Thermal Activation
$$\text{Heat Index} = T + 0.5 \cdot RH$$
$$\text{AC Load Proxy} = \begin{cases} \exp\left(\frac{T - 30.0}{5.0}\right) & \text{if } T > 30.0^\circ\text{C} \\ 0.0 & \text{otherwise} \end{cases}$$

### 3. Diurnal & Seasonal Cyclical Transforms
$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

---

## 📂 Project Directory Structure

```text
PowerPulse/
├── app/
│   └── main.py                     # Streamlit Decision-Support Web Application
├── data/
│   ├── raw/                        # Raw Historical CSVs (Kaggle & Open-Meteo)
│   └── processed/
│       └── features.csv            # Engineered Anti-Leakage Feature Matrix (Tracked)
├── docs/
│   └── demo_script.md              # Demonstration script for evaluations & presentations
├── model/
│   └── xgb_model.pkl               # Serialized Two-Stage Pipeline Artifacts & Metrics
├── src/
│   ├── data/
│   │   ├── delhi_geo_nodes.py      # 20-Node Delhi Substation Metadata & GIS Coordinates
│   │   ├── download_weather.py     # Open-Meteo Archive API Historical Weather Downloader
│   │   ├── fetch_live_weather.py   # Live 7-Day (168h) Weather Ingestion (Open-Meteo + WeatherAPI)
│   │   ├── inspect_data.py         # Raw schema and missing-value diagnostics
│   │   └── merge_data.py           # Causal merge & forward-fill data alignment
│   ├── features/
│   │   ├── build_features.py       # Anti-leakage offline time-series feature pipeline
│   │   └── build_live_features.py  # Live operational feature builder & recursive multi-step forecaster
│   └── models/
│       └── train_model.py          # Baseline benchmark, 5-fold CV, Ridge + XGBoost training
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive Documentation
```

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/bhagyasribevara/PowerPulse.git
cd PowerPulse
```

### 2. Create and Activate Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app/main.py
```
*The interactive platform will open automatically at `http://localhost:8501`.*

---

## 🔄 Full Reproduction Pipeline (From Scratch)

To re-download historical meteorological archives, rebuild the 210,000-row feature matrix, and retrain the two-stage hybrid ML model from scratch:

```bash
# Step 1: Download 23 years of hourly Delhi weather data (2000–2023)
python src/data/download_weather.py

# Step 2: Causally merge power load and meteorological observations
python src/data/merge_data.py

# Step 3: Engineer anti-leakage time-series features and thermal proxies
python src/features/build_features.py

# Step 4: Evaluate naive baseline, run 5-fold CV, and train Hybrid Ridge + XGBoost
python src/models/train_model.py
```

---

## 🛠️ Technology Stack

- **Machine Learning & Econometrics**: XGBoost, Scikit-Learn (Ridge, TimeSeriesSplit), TreeSHAP
- **Web Application & UI**: Streamlit, Custom Dark Visual Design (CSS)
- **Data Visualization & GIS**: Plotly Express, Plotly Graph Objects, Mapbox Dark-Matter, Subplots
- **Live Meteorological APIs**: Open-Meteo 7-Day Hourly API, WeatherAPI.com
- **Data Engineering**: Pandas, NumPy, Holidays (India)
- **Serialization**: Joblib

---

## 👥 Contributors & Maintainers

- **Bhagya Sri Bevara** — *Lead Developer & Data Science Architect* ([@bhagyasribevara](https://github.com/bhagyasribevara))

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
