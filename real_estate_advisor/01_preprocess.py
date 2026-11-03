"""
STEP 1 - Data preprocessing
  * missing values, duplicates, type/consistency checks
  * recompute Price_per_SqFt in INR (the raw column is in lakhs, rounded to 2 dp)
  * repair impossible floor data, flag outliers (IQR) instead of deleting them
  * feature engineering + rule-based appreciation + 'Good_Investment' label
Outputs: data/processed/housing_clean.csv, models/reference_stats.json, eda_outputs/preprocessing_report.json
"""
import json
import numpy as np
import pandas as pd
import config as C
import features as F


def iqr_flag(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((s < q1 - k * iqr) | (s > q3 + k * iqr)).astype(int)


def main():
    report = {}
    df = pd.read_csv(C.RAW_CSV)
    report["rows_raw"] = len(df)

    # ---- text hygiene -----------------------------------------------------------
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # ---- missing values (none expected; handled generically so the script is reusable)
    report["missing_before"] = int(df.isna().sum().sum())
    num_cols = df.select_dtypes(include="number").columns.drop("ID")
    cat_cols = df.select_dtypes(exclude="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # ---- duplicates: exact duplicates ignoring the ID column --------------------------
    before = len(df)
    df = df.drop_duplicates(subset=df.columns.drop("ID")).drop_duplicates(subset="ID")
    report["duplicates_removed"] = before - len(df)

    # ---- sanity filters -----------------------------------------------------------------
    invalid = (df["Size_in_SqFt"] <= 0) | (df["Price_in_Lakhs"] <= 0) | (df["BHK"] <= 0)
    report["invalid_rows_removed"] = int(invalid.sum())
    df = df[~invalid].copy()

    # Age_of_Property is consistent with Year_Built in the raw file (reference year 2025)
    ref_year = int((df["Year_Built"] + df["Age_of_Property"]).mode().iloc[0])
    report["age_reference_year"] = ref_year
    report["age_inconsistent_rows"] = int(((ref_year - df["Year_Built"]) != df["Age_of_Property"]).sum())

    # ---- floor logic: Floor_No cannot exceed Total_Floors ------------------------------------
    bad_floor = df["Floor_No"] > df["Total_Floors"]
    report["floor_gt_total_rows"] = int(bad_floor.sum())
    report["floor_gt_total_pct"] = round(float(bad_floor.mean()) * 100, 2)
    df["Floor_Data_Repaired"] = bad_floor.astype(int)
    df["Total_Floors"] = np.maximum(df["Total_Floors"], df["Floor_No"])  # building is at least as tall as the floor

    # ---- feature engineering -------------------------------------------------------------------
    city_ppsf = F.city_ppsf_table(df)
    raw_pps_zero = int((df["Price_per_SqFt"] == 0).sum())
    report["raw_price_per_sqft_rounded_to_zero"] = raw_pps_zero
    df = df.rename(columns={"Price_per_SqFt": "Price_per_SqFt_Lakh_raw"})
    df = F.engineer(df, city_ppsf)
    df["Locality_ID"] = df["City"] + " | " + df["Locality"]       # 'Locality_N' repeats across cities
    df["Age_Bucket"] = pd.cut(df["Age_of_Property"], [-1, 5, 10, 20, 35],
                              labels=["0-5y", "6-10y", "11-20y", "21-35y"]).astype(str)
    df["Build_Period"] = (df["Year_Built"] // 5 * 5).astype(int)

    # ---- outlier flags (kept, not deleted: all values are plausible & bounded) ---------------------------
    df["Outlier_PPSF"] = iqr_flag(df["Price_per_SqFt_INR"])
    df["Outlier_Size"] = iqr_flag(df["Size_in_SqFt"])
    report["outliers_ppsf"] = int(df["Outlier_PPSF"].sum())
    report["outliers_size"] = int(df["Outlier_Size"].sum())

    # ---- appreciation, forecast and label -------------------------------------------------------------------
    df["Est_Annual_Appreciation_Pct"] = F.appreciation_rate(df).round(3)
    df["Future_Price_5Y_Lakhs"] = F.future_value(df["Price_in_Lakhs"], df["Est_Annual_Appreciation_Pct"],
                                                 C.FORECAST_HORIZON_YEARS).round(2)
    df["Good_Investment"] = F.label_good_investment(df, df["Est_Annual_Appreciation_Pct"])

    # ---- min-max scaled copies of key numerics (model pipelines also scale internally) ----------------------
    for c in ["Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property", "Infrastructure_Score"]:
        lo, hi = df[c].min(), df[c].max()
        df[f"{c}_scaled"] = ((df[c] - lo) / (hi - lo)).round(5)

    df = df.sort_values("ID").reset_index(drop=True)
    fcols = df.select_dtypes(include="float").columns
    df[fcols] = df[fcols].round(3)           # keeps the CSV compact; labels were computed before rounding
    df.to_csv(C.CLEAN_CSV, index=False)

    report["rows_clean"] = len(df)
    report["columns_clean"] = df.shape[1]
    report["good_investment_rate_pct"] = round(float(df["Good_Investment"].mean()) * 100, 2)
    C.EDA_DIR.mkdir(exist_ok=True)
    (C.EDA_DIR / "preprocessing_report.json").write_text(json.dumps(report, indent=2))

    # reference stats consumed by the app / models
    ref = {"city_ppsf_inr": city_ppsf,
           "age_reference_year": ref_year,
           "state_city": df.groupby("State")["City"].unique().apply(sorted).to_dict()}
    C.MODEL_DIR.mkdir(exist_ok=True)
    (C.MODEL_DIR / "reference_stats.json").write_text(json.dumps(ref, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
