# Real Estate Investment Advisor — Project Documentation

## 1. Objective
Help investors, buyers and listing platforms (a) flag whether a property is a *Good Investment* and (b) estimate its value after 5 years,
delivered through an interactive Streamlit application.

## 2. Data
`india_housing_prices.csv`: 250,000 listings, 23 columns, 20 states, 42 cities, 500 locality labels, 3 property types.

### 2.1 Data-quality findings (`reports/data_quality_report.json`)
| Check | Result | Action |
|---|---|---|
| Missing values | 0 | generic median/mode fallback kept in the script |
| Duplicates (full row, and excluding ID) | 0 | de-duplication step retained |
| `Price_per_SqFt` | in **lakhs** per sq ft, rounded to 2 dp | recomputed in ₹/sq ft (`Price_per_SqFt_INR`) |
| `Age_of_Property` vs `Year_Built` | consistent (reference year 2025) | none |
| `Floor_No` > `Total_Floors` | 116,304 rows (46.5%) | flagged; `Floor_No` capped at `Total_Floors` |
| Status `Under_Construction` but age > 0 | all 125,035 such rows | flagged only (`Status_Age_Conflict`) |
| Implausible sq ft per BHK (<250 or >2,500) | 44,710 rows | flagged only |
| Price/sqft outliers (IQR) | 19,723 rows (7.9%) | flagged, kept |
| Size / price outliers | 0 | – |
| `Sold` availability status (in data dictionary) | absent from file | noted |
| Locality | the same 500 labels recur in all 42 cities | locality treated as a label, not a geography |

### 2.2 Critical observation
Price (10–500 lakhs) and size (500–5,000 sq ft) are uniformly distributed, and **no feature is related to price** (|r| < 0.01 for
everything except mechanically derived columns; city, state, property type, furnishing, facing, amenities, parking and transport differ by ≤ 3%).
A gradient-boosting model predicting price from listing attributes scores **R² = −0.0002** (MAE 122 L, identical to guessing the mean).
The data behaves like randomly generated records. This drives the design below.

## 3. Methodology
### 3.1 Feature engineering (`src/features.py`)
`Price_per_SqFt_INR`, `City_Median_PPSF`, `Price_vs_City_Median`, `School_Density_Score`, `Healthcare_Score`, `Transport_Score`,
`Infrastructure_Score` (0.35 schools + 0.35 hospitals + 0.30 transport, 0–10), `Amenity_Count`, `Amenity_Score`, `Parking_Flag`, `Security_Flag`,
`Is_Metro`, `Floor_Ratio`, `SqFt_per_BHK`. Categorical variables are one-hot encoded (property type, furnishing, facing, owner, availability);
location enters through city-level benchmarks (price/sqft median, metro flag) rather than 42 sparse dummies. Scaling: `StandardScaler` inside the
linear-model pipelines (fit on training data only); tree models are scale-invariant.

### 3.2 Targets — why a rule-based growth model
There is no time dimension (one snapshot, no sale dates), so appreciation cannot be learned. We therefore define it transparently:

`annual growth = 5.0% base + metro (+1.0pp) + infrastructure (±1.5pp) + amenities/security (0–1.25pp) + age (−0.75…+0.5pp) + under-construction (+0.5pp) + house/villa (+0.25pp) + valuation vs city median (±1.5pp)`, clipped to 2–12%.

* **Estimated price after 5 years** = price × (1 + growth)⁵.
* **Good Investment = 1** if growth ≥ **7%** *and* price/sqft ≤ city median. Resulting class balance: 30.1% positive.

All coefficients are assumptions an analyst should replace with calibrated values from real transaction histories.

### 3.3 Models (80/20 stratified split, seed 42)
| Task | Candidates | Selected |
|---|---|---|
| Classification | Logistic Regression, Random Forest, Gradient Boosting | Gradient Boosting |
| Regression (annual growth rate) | Ridge, Random Forest, Gradient Boosting | Gradient Boosting |

## 4. Results
### 4.1 Classifier (test set, 50,000 rows)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 3-fold CV F1 (40k) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.951 | 0.918 | 0.919 | 0.918 | 0.990 | 0.918 |
| Random Forest | 0.971 | 0.947 | 0.955 | 0.951 | 0.997 | 0.917 |
| **Gradient Boosting** | **0.993** | **0.987** | **0.990** | **0.988** | **0.9998** | 0.976 |

Majority-class baseline accuracy: 69.9%. Confusion matrix [[TN 34,742, FP 199], [FN 150, TP 14,909]].
**Confidence:** 92% of predictions have ≥ 90% confidence (mean 97.4%). Calibration is good at the extremes but the model is *under*-confident in the
0.5–0.7 band (predicted ≈ 0.61, observed ≈ 0.76).

### 4.2 Regressor
| Model | R² (rate) | MAE (pp/yr) | MAE of 5-yr price (lakhs) |
|---|---|---|---|
| Ridge | 0.935 | 0.267 | 3.74 |
| Random Forest | 0.989 | 0.111 | 1.83 |
| **Gradient Boosting** | **0.998** | **0.048** | **0.78** |
| Constant-rate baseline | 0 | – | 17.56 |

### 4.3 How to read these scores
Both targets are deterministic functions of the input columns, so near-perfect scores show the models recover the documented rules — they are **not**
evidence of real-world predictive accuracy. They are useful because they serve the rules to new listings instantly, with a probability and
feature attributions. Permutation importance (top features, classifier): `Price_vs_City_Median`, `Infrastructure_Score`, `Age_of_Property`,
`Is_Metro`, `Amenity_Score`, `Availability_Status` — i.e. exactly the inputs of the growth model.

## 5. EDA findings (full numbers in `reports/eda_findings.md`, charts in `reports/figures/`)
All 20 questions are answered; the headline is consistent:
* Prices and sizes are uniform; size–price correlation −0.003 (Q1–Q4). Price/sqft outliers: 7.9% of rows (Q5).
* State and city averages differ by only 2.5% (price/sqft) and 3.1% (price) (Q6–Q7); BHK mix is ~20% per category in every city (Q9).
* Schools, hospitals, furnishing, facing, parking, amenities and transport show no meaningful relationship with price or price/sqft (Q12–Q15, Q18–Q20).
  A few p-values fall below 0.05 (e.g. property type p = 0.019, Gym p = 0.02) but the differences are ≈1% — negligible, and unconvincing after multiple testing.
* Owner types and availability are split ~⅓ each / ~50-50 (Q16–Q17).
* Q10 (price trends of top-5 localities): no dates exist; year-built cohorts were used as a proxy and show no consistent trend.
* Brief items: **area vs return** rises with size only because larger homes at random prices look "undervalued" per sq ft (label artefact);
  **crime rate** is not in the data (Security flag used as a weak proxy; 32.2% vs 28.0% good-rate, driven by the growth model);
  **infrastructure vs resale value**: correlation with current price −0.001; the rise in projected value is the model's assumption.

## 6. Streamlit application (`app/streamlit_app.py`)
1. **Investment Advisor** – form for all property attributes → "Good Investment?" with probability and confidence, 5-year price (base / conservative / optimistic), growth-driver chart, valuation versus city median.
2. **Explore Listings** – filter by state, city, type, BHK, area, price, availability, good-investment only; KPIs, sortable table, CSV download.
3. **Market Insights** – state × type heat-map, city comparison, build-year trend, size-vs-price scatter coloured by growth.
4. **Model Insights** – metrics, confusion matrix, calibration, permutation feature importance, plus the "fair-price" reality check.
5. **About & Caveats**.
Verification: `tests/smoke_test_app.py` executes every page against a stubbed Streamlit API (no errors). It was **not** run inside a live Streamlit server in the build environment, so do a quick visual pass after `streamlit run`.

## 7. Use cases
* **Investors** – screen listings by projected growth and valuation vs city median; compare conservative/base/optimistic outcomes.
* **Buyers** – filter by budget, BHK and area and shortlist properties flagged as good investments.
* **Platforms** – batch-score listings with `src/advisor.py`; show probability and drivers next to each listing to build trust.
* **Analysts** – replace the assumption-based growth model with one calibrated on real resale histories; the pipeline needs no other changes.

## 8. Limitations & next steps
1. Obtain real transaction history (sale dates/prices) to learn appreciation; then retrain and report honest out-of-time metrics.
2. Join external data: crime statistics, metro/road projects, locality-level geography (locality names are not unique to a city here).
3. Replace scenario bands (±2pp) with empirical prediction intervals (quantile regression) once real targets exist.
4. Fix source-data issues: floor numbers, under-construction ages, property types with 30 floors.
5. Not financial advice.
