"""
Central configuration for the Real Estate Investment Advisor project.

Every business assumption used to build the "Good Investment" label and the
price forecast lives here so it can be reviewed and tuned in one place.
NOTE: the dataset has no historical prices, so the appreciation model below is a
transparent, rule-based *scenario* model (domain assumptions), not a rate learned
from observed price history.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "india_housing_prices.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "housing_clean.csv.gz"
MODEL_DIR = ROOT / "models"
EDA_DIR = ROOT / "eda_outputs"
FIG_DIR = EDA_DIR / "figures"

RANDOM_STATE = 42
LAKH = 100_000  # 1 lakh = 100,000 INR

# ---- Feature-engineering scales -------------------------------------------------
SCHOOLS_MAX = 10          # Nearby_Schools observed range 1-10
HOSPITALS_MAX = 10        # Nearby_Hospitals observed range 1-10
TRANSPORT_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
INFRA_WEIGHTS = {"schools": 0.30, "hospitals": 0.30, "transport": 0.40}  # sum = 1
AMENITY_LIST = ["Gym", "Pool", "Clubhouse", "Playground", "Garden"]

# ---- Appreciation (growth) assumptions, percentage points / year -----------------
GROWTH = {
    "base": 5.0,                 # baseline annual appreciation (%)
    "infra_max": 2.5,            # +0 .. +2.5 pp scaled by Infrastructure_Score / 10
    "amenity_max": 1.0,          # +0 .. +1.0 pp scaled by Amenity_Count / 5
    "security": 0.5,             # gated / secured community
    "parking": 0.5,              # parking available
    "under_construction": 1.0,   # pre-completion appreciation premium
    "age_penalty_per_year": 0.04,  # older stock appreciates slower (age capped at 35)
    "value_slope": 0.9,          # pp per unit of -ln(price_per_sqft / city median)
    "value_clip": (-1.5, 1.5),   # bounds of the value adjustment (pp)
    "floor": 2.0,                # min annual appreciation (%)
    "cap": 15.0,                 # max annual appreciation (%)
}

# ---- "Good Investment" label rule ----------------------------------------------
GOOD_APPRECIATION_THRESHOLD = 8.0   # est. annual appreciation (%) must be >= this
GOOD_MAX_PPSF_VS_CITY = 1.25        # and not priced > 125% of the city median INR/sqft
FORECAST_HORIZON_YEARS = 5
SCENARIO_SPREAD_PP = 2.0            # conservative / optimistic = base -/+ 2 pp
