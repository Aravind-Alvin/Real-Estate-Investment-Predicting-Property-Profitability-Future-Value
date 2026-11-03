"""Step 1 - Data preprocessing.

Run from project root:  python -m src.01_preprocessing   (or: python src/01_preprocessing.py)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src import config as C
from src import features as F


def iqr_flag(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    return (s < q1 - k * (q3 - q1)) | (s > q3 + k * (q3 - q1))


def main():
    df = pd.read_csv(C.RAW_CSV)
    dq = {"raw_rows": int(len(df)), "raw_columns": int(df.shape[1])}

    # ---- 1. text hygiene, missing values, duplicates ---------------------
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
    dq["missing_values_total"] = int(df.isna().sum().sum())
    dq["duplicate_rows_full"] = int(df.duplicated().sum())
    dq["duplicate_rows_excluding_ID"] = int(df.drop(columns="ID").duplicated().sum())
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"]).reset_index(drop=True)
    # (generic safety net - fill any NaN if a future file has them)
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in df.select_dtypes(exclude="number").columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    # ---- 2. consistency checks / fixes ----------------------------------
    pps_inr = df["Price_in_Lakhs"] * C.LAKH / df["Size_in_SqFt"]
    dq["price_per_sqft_unit_note"] = ("Raw Price_per_SqFt is in LAKHS per sq ft, rounded to 2 dp "
                                      "(e.g. 0.10). Recomputed in INR/sq ft as Price_per_SqFt_INR.")
    dq["raw_pps_max_abs_error_lakhs"] = float((df["Price_in_Lakhs"] / df["Size_in_SqFt"]
                                               - df["Price_per_SqFt"]).abs().max())
    dq["age_mismatch_rows"] = int((df["Age_of_Property"] != C.REFERENCE_YEAR - df["Year_Built"]).sum())

    df["Floor_Data_Inconsistent"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)
    dq["floor_no_gt_total_floors_rows"] = int(df["Floor_Data_Inconsistent"].sum())
    # Fix: a flat cannot sit above the top of its building -> cap Floor_No at Total_Floors.
    df["Floor_No"] = np.minimum(df["Floor_No"], df["Total_Floors"])

    df["Status_Age_Conflict"] = ((df["Availability_Status"] == "Under_Construction")
                                 & (df["Age_of_Property"] > 0)).astype(int)
    dq["under_construction_but_aged_rows"] = int(df["Status_Age_Conflict"].sum())

    df["BHK_Size_Implausible"] = ((df["Size_in_SqFt"] / df["BHK"] < 250)
                                  | (df["Size_in_SqFt"] / df["BHK"] > 2500)).astype(int)
    dq["implausible_sqft_per_bhk_rows"] = int(df["BHK_Size_Implausible"].sum())

    # ---- 3. outliers: flagged, NOT dropped (they are in-range for this data) ----
    df["Outlier_PPSF"] = iqr_flag(pps_inr).astype(int)
    df["Outlier_Size"] = iqr_flag(df["Size_in_SqFt"]).astype(int)
    df["Outlier_Price"] = iqr_flag(df["Price_in_Lakhs"]).astype(int)
    dq["outliers_ppsf_iqr"] = int(df["Outlier_PPSF"].sum())
    dq["outliers_size_iqr"] = int(df["Outlier_Size"].sum())
    dq["outliers_price_iqr"] = int(df["Outlier_Price"].sum())

    # ---- 4. feature engineering + labels --------------------------------
    city_stats = F.compute_city_stats(df)
    df = F.add_engineered_features(df, city_stats)
    df = F.add_targets(df)

    dq["good_investment_rate"] = float(df["Good_Investment"].mean())
    dq["annual_appreciation_summary"] = df["Annual_Appreciation_Rate"].describe().round(4).to_dict()
    dq["final_rows"] = int(len(df))

    # ---- 5. save --------------------------------------------------------
    C.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(C.CLEAN_CSV, index=False, compression="gzip")   # gzip keeps it GitHub-friendly
    keep = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt_INR", "Year_Built", "Age_of_Property", "Furnished_Status", "Nearby_Schools",
            "Nearby_Hospitals", "Public_Transport_Accessibility", "Parking_Space", "Security",
            "Facing", "Owner_Type", "Availability_Status", "Infrastructure_Score",
            "Annual_Appreciation_Rate", "Future_Price_5Y_Lakhs", "Good_Investment"]
    df[keep].round(4).to_csv(C.APP_DATA, index=False, compression="gzip")
    C.CITY_STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.CITY_STATS_JSON.write_text(json.dumps(city_stats, indent=2))
    C.DQ_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.DQ_REPORT_JSON.write_text(json.dumps(dq, indent=2))
    print(json.dumps(dq, indent=2))


if __name__ == "__main__":
    main()
