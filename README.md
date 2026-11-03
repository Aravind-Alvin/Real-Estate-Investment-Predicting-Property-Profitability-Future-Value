# Real Estate Investment Advisor

Predicts whether a property is a **Good Investment** and estimates its **price after 5 years**, with a Streamlit dashboard.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
Everything the app needs (models, cleaned data) is already included. To rebuild from scratch:
```bash
python src/01_preprocessing.py   # cleaning, features, labels  -> data/processed/
python src/02_eda.py             # 20 EDA questions + extras   -> reports/figures, reports/eda_findings.md
python src/03_modeling.py        # train/evaluate/save models  -> models/, reports/model_metrics.json
python tests/smoke_test_app.py   # runs every app page against a stubbed Streamlit
```

## Layout
| Path | Contents |
|---|---|
| `data/raw/` | original CSV |
| `data/processed/india_housing_cleaned.csv` | cleaned + engineered dataset (deliverable) |
| `data/processed/app_data.csv.gz` | slim copy used by the app |
| `src/` | `config.py`, `features.py` (features + growth model), `01_preprocessing.py`, `02_eda.py`, `03_modeling.py`, `advisor.py` |
| `app/streamlit_app.py` | the dashboard |
| `models/` | classifier, regressor, feature importances, city benchmarks |
| `reports/` | EDA figures, `eda_findings.md`, `model_metrics.json`, `data_quality_report.json` |
| `docs/PROJECT_DOCUMENTATION.md` | methodology, findings, use cases, limitations |

## Read this first
The raw data has no price history and its columns are statistically independent of price (it appears synthetic).
Appreciation and "Good Investment" are therefore defined by a **documented rule-based growth model**, and the ML models learn to
reproduce it. Their very high scores measure rule recovery, not real-market forecasting skill. Details in `docs/PROJECT_DOCUMENTATION.md`.
