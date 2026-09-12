# PowerPulse — Presentation Demo Script for Judges

## 1. The Hook (Real Problem Context)
"Delhi's power grid faces immense seasonal demand swings, with peak electricity load surpassing 8,000 MW during peak summer months. Rapid evening load ramping driven by domestic air conditioning poses major operational challenges for grid stability, forcing reliance on expensive peak power purchases if load is under-forecasted."

## 2. What We Built
"PowerPulse is an empirical machine-learning demand forecasting system and Duck Curve Risk Indicator for Delhi's electricity grid, trained on over two decades of real hourly consumption and meteorological data with a Two-Stage Hybrid Ridge + XGBoost architecture, TreeSHAP explainability, live 7-day weather forecasting, and 20-node geospatial substation disaggregation."

## 3. Live Demo Walkthrough (Step-by-Step Clicks)
1. **Initial Dashboard Load**:
   - Point to the top metric cards showing the live predicted MW load and the calculated **AC Load Proxy Index**.
   - Note the **Duck Curve Risk Indicator** widget displaying the real-time operational state.
2. **Interactive Scenario Simulation (Sidebar)**:
   - In the sidebar, set **Hour of Day** to `19:00` (7 PM) and **Month** to `6` (June).
   - Increase **Temperature** to `42.0 °C`.
   - Observe the **Duck Curve Risk Indicator** instantly flag **HIGH RISK**, highlighting the steep evening ramping hazard.
3. **Model Benchmark & Rigor (Tab 1)**:
   - Switch to the **Model Evaluation & Baseline** tab.
   - Show judges the empirical comparison table comparing the Hybrid Model against the 24-Hour Naive Baseline on the chronological holdout set.
4. **Explainable AI Analytics Hub (Tab 2 - TreeSHAP)**:
   - Switch to **Local Feature Attribution (SHAP)**.
   - Show the interactive Waterfall chart and the automated Natural Language AI Executive Narrative explaining exactly which weather and cyclical factors pushed demand up or down.
5. **Dynamic Historical Evaluation & Root-Cause Post-Mortem (Tab 3)**:
   - Switch to **Dynamic Historical Evaluation**.
   - Filter historical seasons, view residual error curves, and inspect the automated Root-Cause Post-Mortem on worst discrepancy moments.
6. **Live Multi-Day Operational Forecaster (Tab 4)**:
   - Switch to **5-Day Operational Forecast & Grid Risk**.
   - Show the dual-axis chart with vertical meteorological temperature bars and the prominent red projected demand curve across the full multi-day horizon.
7. **Geospatial Zonal Disaggregation & Substation Stress Map (Tab 5)**:
   - Switch to **Area-Wise Disaggregation**.
   - Display the high-density Mapbox dark-matter spatial stress map across 20 primary power distribution nodes and the granular substation diagnostics table.

## 4. Real Model Performance (Empirical Holdout Metrics)
*Evaluated on chronological holdout set:*
- **Hybrid Pipeline MAPE**: `8.06%` (Mean Absolute Percentage Error)
- **Hybrid Pipeline MAE**: `40.03 MW` (Mean Absolute Error)
- **Hybrid Pipeline RMSE**: `50.46 MW` (Root Mean Squared Error)
- **Baseline Comparison**: Outperforms the 24-Hour Naive Baseline (`13.17%` MAPE, `64.97 MW` MAE) by **38.8%** relative error reduction.
- **5-Fold TimeSeriesSplit CV**: `7.96%` average out-of-fold MAPE.

## 5. System Highlights
- **Two-Stage Hybrid ML**: Decoupled long-term secular growth trend (Stage 1 Ridge) from non-linear weather-driven residual volatility (Stage 2 XGBoost).
- **Game-Theoretic TreeSHAP**: Provides exact additive feature attributions for regulatory transparency.
- **Geospatial Substation Granularity**: 20-node disaggregation with compound demographic growth rates across BRPL, TPDDL, BYPL, NDMC, and MES.
- **Live 7-Day API Integration**: Automated weather ingestion from Open-Meteo and WeatherAPI.
