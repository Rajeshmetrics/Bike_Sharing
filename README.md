# SIGMOID | Bike-Sharing Demand and Fleet Decision Case Study

## Deliverables
- `analysis.py`: reproducible quality checks, feature engineering, hypothesis test, time-ordered model comparison, model serialization, figures, and fleet scenarios.
- `app.py`: natural-language, rule-based Streamlit operations prototype (no API key required).
- `CASE_STUDY_REPORT.pdf`: analysis, results, caveats, and business recommendations.
- `outputs/results.json`, `outputs/test_predictions.csv`, `outputs/fleet_scenarios.csv`, `outputs/demand_model.joblib`.
- `figures/`: generated charts.
- Original `hour.csv`, `day.csv`, and `Readme.txt`.

## Setup
From this folder:

```bash
python -m pip install -r requirements.txt
python analysis.py
streamlit run app.py
```

Open the local URL printed by Streamlit (typically `http://localhost:8501`).

## Design choices
The target is hourly **completed rentals** (`cnt`). `casual` and `registered` are excluded as predictors because their sum is the target. `instant` is a row index, not a predictive signal. Data are split chronologically: training through September 2012, validation October 2012, and test November–December 2012. Model selection is based only on validation MAE, and the chosen model is refit on training + validation before its final test score. A dummy mean predictor provides a sanity-check baseline. Weather variables in the holdout are *observed historical values*; deployment needs weather forecasts and monitoring for forecast error.

The hypothesis compares working-day commute hours (07–09 and 17–19) against other working-day hours. Welch's t-test is descriptive; a day-block bootstrap gives a more appropriate confidence interval with within-day correlation. Observational association does not prove a causal impact.

The fleet scenario uses predicted 95th-percentile hourly trip starts multiplied by assumed average trip duration, divided by assumed fleet utilization. This is a citywide concurrency proxy, not a count of bikes that should be purchased. No station inventory, rejected rentals, trip durations, bike repair status, or rebalancing logs were provided.

## Suggested interview demo
1. Open the PDF report and explain the CEO's question.
2. Show data quality and leakage prevention.
3. Show hypothesis test and chart of demand by hour.
4. Explain chronological split and compare baseline, random forest, gradient boosting.
5. Run `streamlit run app.py`; ask “Should we buy more bikes?” and “How many bikes do we need for a 30 minute trip?”
6. Explain why station-level data are needed before any capital expenditure decision.

## Dataset attribution
Fanaee-T, H. and Gama, J. (2013), “Event labeling combining ensemble detectors and background knowledge,” *Progress in Artificial Intelligence*, DOI: 10.1007/s13748-013-0040-3.
