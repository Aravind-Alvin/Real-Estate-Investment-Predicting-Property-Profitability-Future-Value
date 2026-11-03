"""Feature engineering + the transparent growth (appreciation) model.

IMPORTANT: the raw dataset is a single snapshot - it has no price history.
Appreciation therefore cannot be *learned* from it. We encode a documented,
rule-based appreciation model (domain assumptions) and use it to (a) build the
"Good Investment" label and (b) project 5-year prices. ML models are then
trained as surrogates of these rules so they can be served to new listings.
"""
import numpy as np
import pandas as pd
from . import config as C


# --------------------------------------------------------------------- basics
def count_amenities(s: pd.Series) -> pd.Series:
    return s.fillna("").apply(lambda x: len([a for a in str(x).split(",") if a.strip()]))


def compute_city_stats(df: pd.DataFrame) -> dict:
    """City-level benchmarks (median price/sqft in INR) used as valuation anchors."""
    ppsf = df["Price_in_Lakhs"] * C.LAKH / df["Size_in_SqFt"]
    stats = ppsf.groupby(df["City"]).median().round(2).to_dict()
    stats["__GLOBAL__"] = float(round(ppsf.median(), 2))
    return stats


# ---------------------------------------------------------------- engineering
def add_engineered_features(df: pd.DataFrame, city_stats: dict) -> pd.DataFrame:
    d = df.copy()
    d["Price_per_SqFt_INR"] = d["Price_in_Lakhs"] * C.LAKH / d["Size_in_SqFt"]
    d["City_Median_PPSF"] = d["City"].map(city_stats).fillna(city_stats["__GLOBAL__"])
    d["Price_vs_City_Median"] = d["Price_per_SqFt_INR"] / d["City_Median_PPSF"]

    d["School_Density_Score"] = (d["Nearby_Schools"] - 1) / 9 * 10          # 0-10
    d["Healthcare_Score"] = (d["Nearby_Hospitals"] - 1) / 9 * 10            # 0-10
    d["Transport_Score"] = d["Public_Transport_Accessibility"].map(
        {"Low": 2.0, "Medium": 6.0, "High": 10.0})
    d["Infrastructure_Score"] = (0.35 * d["School_Density_Score"]
                                 + 0.35 * d["Healthcare_Score"]
                                 + 0.30 * d["Transport_Score"])             # 0-10

    d["Amenity_Count"] = count_amenities(d["Amenities"])
    d["Parking_Flag"] = (d["Parking_Space"] == "Yes").astype(int)
    d["Security_Flag"] = (d["Security"] == "Yes").astype(int)
    d["Amenity_Score"] = (d["Amenity_Count"] / 5 * 6 + d["Security_Flag"] * 2
                          + d["Parking_Flag"] * 2)                          # 0-10

    d["Is_Metro"] = d["City"].isin(C.METRO_CITIES).astype(int)
    d["Floor_Ratio"] = d["Floor_No"] / d["Total_Floors"].clip(lower=1)
    d["SqFt_per_BHK"] = d["Size_in_SqFt"] / d["BHK"].clip(lower=1)
    return d


# --------------------------------------------------- transparent growth model
def growth_components(d: pd.DataFrame) -> pd.DataFrame:
    """Annual appreciation = base + additive adjustments (decimal, 0.01 = 1pp)."""
    comp = pd.DataFrame(index=d.index)
    comp["Base market growth"] = 0.050
    comp["Metro city"] = 0.010 * d["Is_Metro"]
    comp["Infrastructure"] = ((d["Infrastructure_Score"] - 5) / 5 * 0.015).clip(-0.015, 0.015)
    comp["Amenities & security"] = d["Amenity_Score"] / 10 * 0.0125
    age = d["Age_of_Property"]
    comp["Property age"] = np.select([age <= 5, age <= 15, age <= 25], [0.005, 0.0025, 0.0], -0.0075)
    comp["Under construction"] = np.where(d["Availability_Status"] == "Under_Construction", 0.005, 0.0)
    comp["Land-content (house/villa)"] = np.where(d["Property_Type"] == "Apartment", 0.0, 0.0025)
    ratio = d["Price_vs_City_Median"].clip(lower=0.25, upper=4)
    comp["Valuation vs city median"] = (-0.010 * np.log2(ratio)).clip(-0.015, 0.015)
    return comp


def annual_growth(d: pd.DataFrame) -> pd.Series:
    return growth_components(d).sum(axis=1).clip(0.02, 0.12)


def add_targets(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Annual_Appreciation_Rate"] = annual_growth(d)
    d["Future_Price_5Y_Lakhs"] = d["Price_in_Lakhs"] * (1 + d["Annual_Appreciation_Rate"]) ** C.FORECAST_YEARS
    d["Good_Investment"] = ((d["Annual_Appreciation_Rate"] >= C.GOOD_CAGR_THRESHOLD)
                            & (d["Price_vs_City_Median"] <= C.GOOD_MAX_PRICE_RATIO)).astype(int)
    return d


# ------------------------------------------------------------- model matrix
NUMERIC = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property",
           "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Total_Floors", "Floor_Ratio",
           "SqFt_per_BHK", "School_Density_Score", "Healthcare_Score", "Transport_Score",
           "Infrastructure_Score", "Amenity_Count", "Amenity_Score", "Parking_Flag",
           "Security_Flag", "Is_Metro", "City_Median_PPSF", "Price_vs_City_Median"]
ONEHOT = {"Property_Type": C.PROPERTY_TYPES, "Furnished_Status": C.FURNISHED,
          "Facing": C.FACING, "Owner_Type": C.OWNER_TYPES,
          "Availability_Status": C.AVAILABILITY}


def model_matrix(d: pd.DataFrame) -> pd.DataFrame:
    """Numeric design matrix with fixed one-hot columns (identical in training & app)."""
    X = d[NUMERIC].astype(float).copy()
    for col, levels in ONEHOT.items():
        for lv in levels:
            X[f"{col}={lv}"] = (d[col] == lv).astype(float)
    return X
