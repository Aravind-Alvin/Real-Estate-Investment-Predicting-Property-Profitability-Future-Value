# Real Estate Investment Advisor

Predicts whether an Indian property listing is a **Good Investment**, projects its future value, and lets users
screen 250,000 listings in a Streamlit dashboard.

## Quick start
```bash
pip install -r requirements.txt
# put india_housing_prices.csv in data/raw/   (only needed to rebuild; processed data + models are included)
streamlit run app.py
# full rebuild:  ./run_pipeline.sh
```

## Layout
| File | Purpose |
|---|---|
| `config.py` | Every business assumption (growth weights, label thresholds) in one place |
| `features.py` | Feature engineering, appreciation rule, label - shared by pipeline and app |
| `01_preprocess.py` | Cleaning, validation, feature engineering, label -> `data/processed/housing_clean.csv` |
| `02_eda.py` | All 20 EDA questions + extras -> `eda_outputs/figures/`, `eda_findings.json` |
| `03_train_models.py` | Classifier, appreciation regressor, price diagnostic -> `models/` |
| `advisor.py` | Recommendation logic used by the app (no UI code) |
| `app.py` | Streamlit app: Advisor form, Property Finder, Market Insights, Methodology |
| `DOCUMENTATION.md` | Methodology, findings, limitations, use cases |

**Read the Limitations section of DOCUMENTATION.md before using the outputs for decisions.**
