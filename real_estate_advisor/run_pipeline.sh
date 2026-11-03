#!/usr/bin/env bash
# Rebuild everything from the raw CSV (place it at data/raw/india_housing_prices.csv first)
set -e
python 01_preprocess.py      # -> data/processed/housing_clean.csv, models/reference_stats.json
python 02_eda.py             # -> eda_outputs/figures/*.png, eda_outputs/eda_findings.json
python 03_train_models.py    # -> models/*.joblib, models/metrics.json
streamlit run app.py
