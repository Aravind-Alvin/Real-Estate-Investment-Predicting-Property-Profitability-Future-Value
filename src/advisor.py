"""Inference helpers shared by the Streamlit app and tests."""
import json
import joblib
import pandas as pd
from . import config as C
from . import features as F


def train_serving_models(sample: int = 120_000):
    """Fallback: retrain the two serving models from the cleaned data (~1 min).
    Used when the pickled models cannot be loaded (e.g. scikit-learn version mismatch)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    d = pd.read_csv(C.CLEAN_CSV)
    d = d.sample(min(sample, len(d)), random_state=C.RANDOM_STATE)
    X = F.model_matrix(d)
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=C.RANDOM_STATE).fit(X, d["Good_Investment"])
    reg = HistGradientBoostingRegressor(max_iter=250, random_state=C.RANDOM_STATE).fit(X, d["Annual_Appreciation_Rate"])
    return clf, reg, list(X.columns)


def load_artifacts():
    city_stats = json.loads(C.CITY_STATS_JSON.read_text())
    try:
        clf = joblib.load(C.MODEL_DIR / "classifier.joblib")
        reg = joblib.load(C.MODEL_DIR / "regressor.joblib")
        cols = joblib.load(C.MODEL_DIR / "feature_columns.joblib")
    except Exception as exc:          # version mismatch, corrupt pickle, ...
        print(f"Could not load saved models ({type(exc).__name__}: {exc}); retraining.")
        clf, reg, cols = train_serving_models()
    return {"clf": clf, "reg": reg, "cols": cols, "city_stats": city_stats}


def build_row(i: dict, city_stats: dict) -> pd.DataFrame:
    """i: dict of user inputs -> engineered one-row DataFrame."""
    raw = pd.DataFrame([{
        "City": i["City"], "Property_Type": i["Property_Type"], "BHK": i["BHK"],
        "Size_in_SqFt": i["Size_in_SqFt"], "Price_in_Lakhs": i["Price_in_Lakhs"],
        "Age_of_Property": max(C.REFERENCE_YEAR - i["Year_Built"], 0),
        "Furnished_Status": i["Furnished_Status"],
        "Floor_No": min(i["Floor_No"], i["Total_Floors"]), "Total_Floors": i["Total_Floors"],
        "Nearby_Schools": i["Nearby_Schools"], "Nearby_Hospitals": i["Nearby_Hospitals"],
        "Public_Transport_Accessibility": i["Public_Transport_Accessibility"],
        "Parking_Space": "Yes" if i["Parking"] else "No", "Security": "Yes" if i["Security"] else "No",
        "Amenities": ", ".join(i["Amenities"]), "Facing": i["Facing"], "Owner_Type": i["Owner_Type"],
        "Availability_Status": i["Availability_Status"]}])
    return F.add_engineered_features(raw, city_stats)


def assess(i: dict, art: dict) -> dict:
    d = build_row(i, art["city_stats"])
    X = F.model_matrix(d)[art["cols"]]
    p_good = float(art["clf"].predict_proba(X)[0, 1])
    rate = float(art["reg"].predict(X)[0])
    rate_rule = float(F.annual_growth(d).iloc[0])
    price = float(d["Price_in_Lakhs"].iloc[0])
    n = C.FORECAST_YEARS
    scen = {name: price * (1 + max(rate + delta, 0)) ** n
            for name, delta in [("Conservative (-2pp/yr)", -0.02), ("Base case", 0.0), ("Optimistic (+2pp/yr)", 0.02)]}
    comps = F.growth_components(d).iloc[0]
    return {"p_good": p_good, "is_good": p_good >= 0.5, "confidence": max(p_good, 1 - p_good),
            "rate": rate, "rate_rule": rate_rule, "price": price, "future": scen,
            "gain": scen["Base case"] - price, "components": comps,
            "ppsf": float(d["Price_per_SqFt_INR"].iloc[0]), "city_median_ppsf": float(d["City_Median_PPSF"].iloc[0]),
            "ratio": float(d["Price_vs_City_Median"].iloc[0]), "infra": float(d["Infrastructure_Score"].iloc[0])}
