# GridSathi — Delhi Electricity Demand Forecasting App ⚡

GridSathi is an electricity demand forecasting and duck curve risk monitoring application for Delhi. Built using real-world hourly power consumption data (2000–2023) and Open-Meteo historical weather data, GridSathi leverages XGBoost machine learning to predict demand and evaluate grid operational risk.

---

## 🌟 Key Features

1. **Real Data Ingestion & Alignment**: Integrates 210,000+ hourly observations from the Kaggle Delhi Power Consumption dataset (2000–2023) aligned with historical weather data from the Open-Meteo Archive API.
2. **Anti-Leakage Feature Engineering**: Employs strict past-only time series features including lag terms (`lag_1h`, `lag_24h`, `lag_168h`), shifted 24-hour rolling averages, and cyclical encodings (`hour_sin/cos`, `month_sin/cos`).
3. **Engineered Weather Proxies**: Incorporates `heat_index` and `ac_load_proxy` (exponential heat activation term above 30°C) as proxy variables for cooling-driven power demand.
4. **Empirical Baseline Benchmark**: Evaluates XGBoost against a **24-Hour Naive Baseline** ($Load_t = Load_{t-24h}$) on a chronological 3-month holdout set and 5-fold `TimeSeriesSplit` cross-validation.
5. **Duck Curve Risk Indicator**: Rule-based alert mechanism notifying grid operators of potential evening ramp surges (18:00–21:00) driven by high AC proxy load during summer months.
6. **Interactive Streamlit Dashboard**: Provides scenario simulation, XGBoost feature importance visualization, actual-vs-predicted load curves, and live "What-If" temperature sensitivity analysis.

---

## 📊 Model Performance Summary (Holdout Test Set: Oct–Dec 2023)

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **24-Hour Naive Baseline** | 64.97 MW | 81.55 MW | 13.17% |
| **XGBoost Regressor** | **40.03 MW** | **50.46 MW** | **8.06%** |
| **XGBoost (5-Fold TimeSeriesSplit CV)** | **40.02 MW** | **50.15 MW** | **7.96%** |

*XGBoost achieves a ~38.8% relative error reduction over the 24-hour naive baseline.*

---

## 🛠️ Project Structure

```text
Gridsathi/
├── app/
│   └── main.py              # Streamlit Web Application
├── data/
│   ├── raw/                 # Raw Kaggle & Open-Meteo CSVs (Excluded from Git)
│   └── processed/
│       └── features.csv     # Pre-built feature matrix (Tracked in Git)
├── docs/
│   └── demo_script.md       # Empirical demo script for presentation
├── model/
│   └── xgb_model.pkl        # Serialized model & benchmark payload (Tracked in Git)
├── src/
│   ├── data/
│   │   ├── inspect_data.py   # Raw dataset schema inspection
│   │   ├── download_weather.py # Open-Meteo weather API downloader
│   │   └── merge_data.py    # Causal missing-value merge script
│   ├── features/
│   │   └── build_features.py# Anti-leakage time-series feature builder
│   └── models/
│       └── train_model.py   # Baseline, CV, XGBoost training & evaluation
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Local Launch

> **Included Runtime Artifacts**: To allow instant out-of-the-box execution upon cloning, this repository tracks the pre-built feature matrix (`data/processed/features.csv`) and trained model weights (`model/xgb_model.pkl`).
>
> **Excluded Raw Data**: The `data/raw/` directory contains raw source datasets (~32 MB) and is intentionally excluded from Git.

1. **Activate Virtual Environment**:
   ```bash
   .\.venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Streamlit Dashboard Directly**:
   ```bash
   streamlit run app/main.py
   ```
   *Starts the interactive web application immediately on `http://localhost:8501`.*

---

## 🔄 Full Reproduction Pipeline (Optional)

For users who wish to regenerate the processed data and retrain the XGBoost model from scratch using raw data sources:

1. **Download Historical Weather Data**:
   ```bash
   python src/data/download_weather.py
   ```
   *Fetches hourly historical meteorological data (`delhi_weather.csv`) from Open-Meteo Archive API covering 2000–2023.*

2. **Causal Data Cleaning & Merge**:
   ```bash
   python src/data/merge_data.py
   ```
   *Merges power demand data with weather data, applying causal forward fill (`ffill()`) to produce `data/processed/merged.csv`.*

3. **Anti-Leakage Feature Engineering**:
   ```bash
   python src/features/build_features.py
   ```
   *Engineers time-series features (lags, rolling means, cyclical encodings, heat index, AC load proxy) with past-only indexing to output `data/processed/features.csv`.*

4. **Model Training & Evaluation**:
   ```bash
   python src/models/train_model.py
   ```
   *Evaluates the 24-hour naive baseline, performs 5-fold TimeSeriesSplit cross-validation, trains XGBoost, and serializes `model/xgb_model.pkl`.*
