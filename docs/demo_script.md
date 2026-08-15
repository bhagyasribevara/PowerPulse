# GridSathi — Presentation Demo Script for Judges

## 1. The Hook (Real Problem Context)
"Delhi's power grid faces immense seasonal demand swings, with peak electricity load surpassing 8,000 MW during peak summer months. Rapid evening load ramping driven by domestic air conditioning poses major operational challenges for grid stability, forcing reliance on expensive peak power purchases if load is under-forecasted."

## 2. What We Built
"GridSathi is an empirical machine-learning demand forecasting system and Duck Curve Risk Indicator for Delhi's electricity grid, trained on over two decades of real hourly consumption and meteorological data."

## 3. Live Demo Walkthrough (Step-by-Step Clicks)
1. **Initial Dashboard Load**:
   - Point to the top metric cards showing the live predicted MW load and the calculated **AC Load Proxy Index**.
   - Note the **Duck Curve Risk Indicator** widget displaying the status.
2. **Scenario Simulation**:
   - In the sidebar, set **Hour of Day** to `19:00` (7 PM) and **Month** to `6` (June).
   - Increase **Temperature** to `42.0 °C`.
   - Observe the **Duck Curve Risk Indicator** instantly flag **HIGH RISK**, highlighting the steep evening ramping hazard.
3. **Model Benchmark & Rigor**:
   - Switch to the **Model Evaluation & Baseline** tab.
   - Show judges the empirical comparison table comparing XGBoost against the 24-Hour Naive Baseline on the chronological 3-month holdout set.
4. **Feature Importance Insights**:
   - Switch to the **Feature Importances** tab.
   - Point out top drivers: `hour_sin` (diurnal cycle), `solar_generation`, and `hour_of_day`.
5. **What-If Sensitivity Analysis**:
   - Switch to the **What-If Temperature Sensitivity** tab.
   - Move the temperature delta slider to `+3.0 °C` and show judges the live computed demand response curve.

## 4. Real Model Performance (Empirical Holdout Metrics)
*Evaluated on chronological 3-month holdout set (Oct 1, 2023 – Dec 31, 2023):*
- **XGBoost MAPE**: `8.06%` (Mean Absolute Percentage Error)
- **XGBoost MAE**: `40.03 MW` (Mean Absolute Error)
- **XGBoost RMSE**: `50.46 MW` (Root Mean Squared Error)
- **Baseline Comparison**: Outperforms the 24-Hour Naive Baseline (`13.17%` MAPE, `64.97 MW` MAE) by **38.8%** relative error reduction.
- **5-Fold TimeSeriesSplit CV**: `7.96%` average out-of-fold MAPE.

## 5. Known Limitations
- **Data Recency**: Dataset spans 2000–2023; real-time operational deployment requires streaming feeds.
- **Weather Uncertainty**: Model accuracy depends on weather forecast precision; weather errors compound into demand uncertainty.
- **Proxy Features**: `heat_index` and `ac_load_proxy` are engineered mathematical proxies, not direct physical SCADA telemetry from individual appliances.
- **SLDC Integration**: Currently a standalone decision-support dashboard, not directly wired into State Load Despatch Centre (SLDC) automatic generation control (AGC).

## 6. Future Scope
- Direct API integration with SLDC telemetry and live SCADA feeds.
- Integration of spatial sub-station level granularity across Delhi discoms (BRPL, BYPL, TPDDL).
- Probabilistic forecasting intervals (quantile regression) to quantify peak load tail risk.

*(Financial cost-savings metric placeholder: [NEEDS VERIFIED TARIFF SOURCE])*
