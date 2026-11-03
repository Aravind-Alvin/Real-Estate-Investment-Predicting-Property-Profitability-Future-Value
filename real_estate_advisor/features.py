"""
Shared feature engineering. Used by BOTH the offline pipeline and the Streamlit app
so a property typed into the form is transformed exactly like the training data.
"""
import numpy as np
import pandas as pd
import config as C


def amenity_flags(series: pd.Series) -> pd.DataFrame:
    """Turn the free-text 'Gym, Pool, ...' column into 0/1 flags."""
    out = {}
    s = series.fillna("").astype(str)
    for a in C.AMENITY_LIST:
        out[f"Has_{a}"] = s.str.contains(rf"\b{a}\b", regex=True).astype(int)
    return pd.DataFrame(out, index=series.index)


def city_ppsf_table(df: pd.DataFrame) -> dict:
    """Median INR price/sqft per city (reference for value-for-money)."""
    return (df["Price_in_Lakhs"] * C.LAKH / df["Size_in_SqFt"]).groupby(df["City"]).median().to_dict()


def engineer(df: pd.DataFrame, city_ppsf: dict) -> pd.DataFrame:
    """Add all engineered columns to a frame holding the raw property columns."""
    d = df.copy()
    d["Price_per_SqFt_INR"] = d["Price_in_Lakhs"] * C.LAKH / d["Size_in_SqFt"]
    d["City_Median_PPSF_INR"] = d["City"].map(city_ppsf)
    d["PPSF_vs_City"] = d["Price_per_SqFt_INR"] / d["City_Median_PPSF_INR"]

    d["School_Density_Score"] = (d["Nearby_Schools"] / C.SCHOOLS_MAX * 10).clip(0, 10)
    d["Healthcare_Score"] = (d["Nearby_Hospitals"] / C.HOSPITALS_MAX * 10).clip(0, 10)
    d["Transport_Score"] = d["Public_Transport_Accessibility"].map(C.TRANSPORT_MAP) * 10
    w = C.INFRA_WEIGHTS
    d["Infrastructure_Score"] = (w["schools"] * d["School_Density_Score"]
                                 + w["hospitals"] * d["Healthcare_Score"]
                                 + w["transport"] * d["Transport_Score"])

    d = pd.concat([d, amenity_flags(d["Amenities"])], axis=1)
    d["Amenity_Count"] = d[[f"Has_{a}" for a in C.AMENITY_LIST]].sum(axis=1)
    d["Parking_Flag"] = (d["Parking_Space"] == "Yes").astype(int)
    d["Security_Flag"] = (d["Security"] == "Yes").astype(int)
    d["Is_Ready"] = (d["Availability_Status"] == "Ready_to_Move").astype(int)
    d["Floor_Ratio"] = d["Floor_No"] / d["Total_Floors"].clip(lower=1)
    d["Sqft_per_BHK"] = d["Size_in_SqFt"] / d["BHK"].clip(lower=1)
    return d


def appreciation_components(d: pd.DataFrame) -> pd.DataFrame:
    """Percentage-point contribution of each driver to the estimated annual appreciation."""
    g = C.GROWTH
    comp = pd.DataFrame(index=d.index)
    comp["Base market growth"] = g["base"]
    comp["Infrastructure (schools, hospitals, transport)"] = g["infra_max"] * d["Infrastructure_Score"] / 10
    comp["Amenities"] = g["amenity_max"] * d["Amenity_Count"] / len(C.AMENITY_LIST)
    comp["Security"] = g["security"] * d["Security_Flag"]
    comp["Parking"] = g["parking"] * d["Parking_Flag"]
    comp["Under-construction premium"] = g["under_construction"] * (1 - d["Is_Ready"])
    comp["Property age"] = -g["age_penalty_per_year"] * d["Age_of_Property"].clip(0, 35)
    comp["Value vs city median price/sq ft"] = (-g["value_slope"] * np.log(d["PPSF_vs_City"].clip(lower=0.05))
                                                 ).clip(*g["value_clip"])
    return comp


def appreciation_rate(d: pd.DataFrame) -> pd.Series:
    """Rule-based estimated annual appreciation (%) - see config.GROWTH."""
    return appreciation_components(d).sum(axis=1).clip(C.GROWTH["floor"], C.GROWTH["cap"])


def label_good_investment(d: pd.DataFrame, rate: pd.Series) -> pd.Series:
    return ((rate >= C.GOOD_APPRECIATION_THRESHOLD)
            & (d["PPSF_vs_City"] <= C.GOOD_MAX_PPSF_VS_CITY)).astype(int)


def future_value(price_lakhs, rate_pct, years):
    return price_lakhs * (1 + rate_pct / 100.0) ** years


# Columns fed to the ML models (identical in training and in the app)
NUM_FEATURES = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Age_of_Property", "Floor_No", "Total_Floors",
                "Nearby_Schools", "Nearby_Hospitals", "Price_per_SqFt_INR", "PPSF_vs_City",
                "School_Density_Score", "Healthcare_Score", "Transport_Score", "Infrastructure_Score",
                "Amenity_Count", "Parking_Flag", "Security_Flag", "Is_Ready", "Floor_Ratio", "Sqft_per_BHK"]
CAT_FEATURES = ["City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
                "Facing", "Owner_Type"]
MODEL_FEATURES = NUM_FEATURES + CAT_FEATURES
