"""Shared configuration: paths, constants and domain assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "india_housing_prices.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "india_housing_cleaned.csv"
APP_DATA = ROOT / "data" / "processed" / "app_data.csv.gz"
CITY_STATS_JSON = ROOT / "models" / "city_stats.json"
DQ_REPORT_JSON = ROOT / "reports" / "data_quality_report.json"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# ---- dataset conventions -------------------------------------------------
REFERENCE_YEAR = 2025          # Age_of_Property == 2025 - Year_Built in the raw file
LAKH = 100_000                 # 1 lakh = 100,000 INR
FORECAST_YEARS = 5

# ---- categorical levels (fixed so the app and training agree) ------------
PROPERTY_TYPES = ["Apartment", "Independent House", "Villa"]
FURNISHED = ["Unfurnished", "Semi-furnished", "Furnished"]
FACING = ["East", "North", "South", "West"]
OWNER_TYPES = ["Builder", "Broker", "Owner"]
AVAILABILITY = ["Ready_to_Move", "Under_Construction"]
TRANSPORT = ["Low", "Medium", "High"]
AMENITY_LIST = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]

# ---- domain assumptions for the growth model (see docs) -------------------
METRO_CITIES = {
    "Mumbai", "New Delhi", "Dwarka", "Gurgaon", "Noida", "Bangalore",
    "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad",
}
GOOD_CAGR_THRESHOLD = 0.07     # projected annual appreciation must be >= 7 %
GOOD_MAX_PRICE_RATIO = 1.00    # and price/sqft must not exceed the city median
RANDOM_STATE = 42
