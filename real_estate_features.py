"""Shared feature-engineering logic (used by the notebook AND the Streamlit app)."""
import numpy as np
import pandas as pd

TRANSPORT_MAP = {"Low": 0.33, "Medium": 0.66, "High": 1.0}
AMENITY_LIST = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]
HORIZON_YEARS = 5


def engineer_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Add engineered columns. `ref` holds reference stats (city medians etc.)."""
    d = df.copy()
    d["Price_per_SqFt_INR"] = d["Price_in_Lakhs"] * 1e5 / d["Size_in_SqFt"]
    d["Infrastructure_Score"] = 100 * (
        0.40 * d["Nearby_Schools"] / 10
        + 0.30 * d["Nearby_Hospitals"] / 10
        + 0.30 * d["Public_Transport_Accessibility"].map(TRANSPORT_MAP)
    )
    city_avg_schools = d["City"].map(ref["city_avg_schools"]).fillna(ref["global_avg_schools"])
    d["School_Density_Score"] = (d["Nearby_Schools"] / city_avg_schools).round(3)
    d["Amenity_Count"] = d["Amenities"].fillna("").apply(lambda s: sum(a in s for a in AMENITY_LIST))
    d["Floor_Ratio"] = (d["Floor_No"] / d["Total_Floors"].clip(lower=1)).clip(upper=1)
    d["Sqft_per_BHK"] = d["Size_in_SqFt"] / d["BHK"]
    d["City_Median_PPSF"] = d["City"].map(ref["city_median_ppsf"]).fillna(ref["global_median_ppsf"])
    d["Price_vs_City_Median"] = d["Price_per_SqFt_INR"] / d["City_Median_PPSF"]
    d["Log_PPSF"] = np.log1p(d["Price_per_SqFt_INR"])
    d["Parking_Space"] = d["Parking_Space"].astype(str)
    d["Security"] = d["Security"].astype(str)
    return d


def add_targets(d: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Create the two target variables from transparent, documented domain rules."""
    d = d.copy()
    tier = d["City"].map(ref["city_tier"]).fillna(0)
    type_adj = d["Property_Type"].map({"Villa": 0.005, "Independent House": 0.0025}).fillna(0.0)
    infra_adj = (d["Infrastructure_Score"] - 50) / 50 * 0.01
    build_adj = np.where(d["Availability_Status"] == "Under_Construction", 0.01, 0.0)
    age_adj = -(d["Age_of_Property"].clip(upper=35) / 35) * 0.01
    d["Annual_Growth_Rate"] = (0.06 + tier + type_adj + infra_adj + build_adj + age_adj).clip(0.04, 0.12)
    d["Future_Price_5Y"] = d["Price_in_Lakhs"] * (1 + d["Annual_Growth_Rate"]) ** HORIZON_YEARS
    score = (
        (d["Price_per_SqFt_INR"] <= d["City_Median_PPSF"]).astype(int)
        + (d["Annual_Growth_Rate"] >= ref["median_growth"]).astype(int)
        + (d["BHK"] >= 3).astype(int)
        + (d["Availability_Status"] == "Ready_to_Move").astype(int)
        + (d["Infrastructure_Score"] >= ref["median_infra"]).astype(int)
    )
    d["Investment_Score"] = score
    d["Good_Investment"] = (score >= 3).astype(int)
    return d
