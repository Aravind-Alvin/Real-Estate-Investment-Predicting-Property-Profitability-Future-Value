# Real Estate Investment Advisor - Project Documentation

## 1. Summary
A complete analytics pipeline and Streamlit application built on 250,000 Indian property listings (20 states, 42 cities,
500 locality labels). It cleans and engineers the data, answers the 20 EDA questions, labels each property as a
"Good Investment" using explicit domain rules, trains models, and serves an interactive advisor.

**The most important finding is about the data itself:** apart from mechanical relationships, the raw attributes are
statistically independent of price (details in section 4). That shapes what the models can and cannot claim (section 6).

## 2. Methodology

### 2.1 Preprocessing (`01_preprocess.py`)
| Check | Result | Action |
|---|---|---|
| Missing values | 0 | Generic median/mode imputation coded for reuse |
| Duplicates (ignoring ID) | 0 | Drop logic in place |
| Age vs Year_Built | Consistent (reference year 2025) | None |
| `Floor_No` > `Total_Floors` | **116,304 rows (46.5%)** | `Total_Floors` raised to `Floor_No`; row flagged `Floor_Data_Repaired` |
| Raw `Price_per_SqFt` | In lakhs, rounded to 2 dp; 2,534 rows became 0.00 | Recomputed in INR = Price x 100,000 / Size |
| Outliers | 19,723 price/sq ft (IQR), 0 size | Flagged (`Outlier_PPSF`), not deleted - values are plausible |
| `Locality_N` | Same label appears in all 42 cities | Used `City | Locality` as the key |

Scaling/encoding: min-max scaled copies of key numerics are stored in the CSV; models apply `StandardScaler` + one-hot
(linear models) or ordinal encoding (gradient boosting) **inside sklearn pipelines**, so the app transforms inputs exactly as in training.

### 2.2 Engineered features (`features.py`)
Price/sq ft (INR), price vs city median, School Density Score, Healthcare Score, Transport Score (Low/Med/High → 0/5/10),
**Infrastructure Score** (30% schools, 30% hospitals, 40% transport; 0-10), amenity flags and count, parking/security flags,
floor ratio, size per BHK, age bucket, build period.

### 2.3 Good-Investment label and forecast (all weights in `config.py`)
Estimated annual appreciation = 5.0% base + up to 2.5 pp infrastructure + up to 1.0 pp amenities + 0.5 pp security
+ 0.5 pp parking + 1.0 pp under-construction premium − 0.04 pp per year of age ± value-for-money adjustment
(cheaper than the city median per sq ft adds, up to ±1.5 pp), bounded to 2-15%.

**Good_Investment = 1** when estimated appreciation ≥ **8%** *and* price/sq ft ≤ **1.25x** the city median. This labels 24.9% of properties.
Future value = price x (1 + rate)^years, with conservative / optimistic scenarios at ∓2 pp.

### 2.4 Models (`03_train_models.py`, 80/20 split, seed 42)
| Task | Model | Hold-out result |
|---|---|---|
| Good_Investment classifier | HistGradientBoosting (vs Logistic Regression) | Accuracy 99.1%, precision 98.4%, recall 98.1%, F1 0.983, ROC-AUC 0.9997 (LogReg: F1 0.948) |
| Annual appreciation regressor | HistGradientBoosting (vs Ridge) | R² 0.9985, MAE 0.038 pp (Ridge R² 0.967) |
| Diagnostic: predict current price | HistGradientBoosting | **R² = -0.0003**, MAE 122.3 L - same as predicting the mean (122.3 L) |

Top classifier drivers (permutation importance): price vs city median, infrastructure score, ready/under-construction,
property age, amenity count, security, parking.

## 3. Streamlit app (`app.py`)
* **Investment Advisor** - form (state→city, type, BHK, size, price, year built, status, floors, schools/hospitals, transport, parking, security, amenities, facing, owner). Returns a verdict, probability, estimated appreciation, projected value for 1-10 years with scenario band, a driver breakdown, a comparable-listing benchmark (percentile of asking price and price/sq ft), and a negotiation target price at which the rule would pass.
* **Property Finder** - filter by state, city, locality text, type, BHK, price, area, availability, furnishing, minimum infrastructure score and "Good Investment only"; sort, view, chart and download results (CSV).
* **Market Insights** - all EDA figures with plain-language findings.
* **Methodology & Model** - assumptions, metrics and caveats inside the app.

## 4. EDA findings (numbers from `eda_outputs/eda_findings.json`)

**Price & size**
1. *Price distribution*: uniform, 10-500 lakhs, mean 254.6 L, skew ≈ 0.
2. *Size distribution*: uniform, 500-5,000 sq ft, mean ≈ 2,750.
3. *Price/sq ft by type*: Apartment ₹13,047, Independent House ₹13,102, Villa ₹13,026 - a 0.6% spread.
4. *Size vs price*: uncorrelated (ρ = -0.003). Size vs price/sq ft ρ = -0.60 - a mechanical effect of price being independent of size.
5. *Outliers*: 7.9% of listings are price/sq ft outliers (up to ₹99,182; 99th percentile ₹65,823), almost all small units (median 799 sq ft vs 2,747). No size outliers.

**Location**
6. *Price/sq ft by state*: ₹12,932 (Punjab) to ₹13,252 (Karnataka) - 2.5% spread.
7. *Average price by city*: 250.8 L (Cuttack) to 258.5 L (Bangalore) - 3.1% spread.
8. *Median age by locality*: overall 18 years; locality medians 3-34.5 but with only ~12 listings each they are noisy.
9. *BHK by city*: ≈ 20% for each BHK in every city (max deviation ≈ 2 pp).
10. *Top-5 expensive localities*: Chennai Locality_227 (₹33.9k/sq ft), Mangalore Locality_459, Mysore Locality_381, Jaipur Locality_213, Amritsar Locality_436 - each based on only 8-13 listings, so this ranking is largely sampling noise. The data has **no sale dates**, so "trend" is shown by year built; city-wide it is flat (ρ = -0.001).

**Relationships**
11. *Correlations*: raw listing attributes are mutually uncorrelated (max |r| = 0.004). Visible correlations are mechanical or engineered.
12-13. *Schools / hospitals vs price/sq ft*: ρ = -0.002 each - no relationship.
14. *Furnishing*: 254.3-255.0 L (0.25% spread, ANOVA p = 0.61).
15. *Facing*: ≤ 0.9% difference, not significant (p = 0.35).

**Ownership / amenities**
16. *Owner type*: Broker 83,479; Owner 83,268; Builder 83,253.
17. *Availability*: Under_Construction 125,035; Ready_to_Move 124,965 (no "Sold" status exists).
18. *Parking*: no effect (254.8 L vs 254.4 L, p = 0.58).
19. *Amenities*: each amenity moves price/sq ft < 1% (gym +0.75% largest).
20. *Transport*: price/sq ft differs < 0.5% by level. Good-Investment rate is 14% (Low), 24% (Medium), 36% (High) - **by construction of the rule**, not an observed market effect.

**Project-specific questions**
* *Crime rate vs Good Investment*: the dataset has **no crime field**. Using the Security flag as the nearest proxy, 30.3% of secured properties are labelled Good vs 19.5% unsecured (rule-driven).
* *Infrastructure vs resale value*: no resale data exists. Against price, infrastructure shows no relationship (ρ = -0.003).

## 5. Business use cases
* **Investors** - screen listings by projected return and see the assumptions behind each estimate.
* **Buyers** - filter by budget, size, BHK and location; find properties priced below their city benchmark.
* **Real-estate platforms** - automate a consistent investment score for each listing; the rule weights are configurable.
* **Trust & transparency** - every verdict shows its drivers and a comparable-listing benchmark.

## 6. Limitations (please read)
1. **No price history.** Appreciation rates are assumptions in `config.py`, not learned from data. Forecasts are scenarios.
2. **Circular classifier.** The label is a deterministic function of the inputs; the ~99% accuracy shows the model reproduces the rule, not that it predicts real returns.
3. **Label effects in EDA are built in.** Higher Good-Investment rates for better infrastructure, security, amenities or transport follow from the rule's weights.
4. **Data looks synthetic.** Uniform price and size, no attribute-price correlations, equal category splits, floors above building height. Real-world conclusions (e.g. "metros are pricier") cannot be drawn from it.
5. **Current price is unpredictable** from the listing attributes (R² ≈ 0), so the app benchmarks asking price against comparables rather than predicting a fair price.
6. **"Value" rewards large units.** Because price is independent of size, large units look cheap per sq ft and score as good value.
7. Missing from the brief's wording: crime rate, an "infrastructure score" and appreciation are not in the data; the latter two are engineered, crime is unavailable.
8. Not financial advice.

## 7. Suggested next steps
Add dated transactions or rental yields to learn real appreciation; add crime/connectivity/geo data; recalibrate `config.GROWTH` against
observed price indices; add time-based validation.
