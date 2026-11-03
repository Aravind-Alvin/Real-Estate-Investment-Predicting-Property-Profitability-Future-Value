"""
Decision logic behind the Streamlit app (no Streamlit imports -> unit-testable).
"""
import json
import joblib
import numpy as np
import pandas as pd
import config as C
import features as F

RAW_COLS = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built",
            "Furnished_Status", "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
            "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities", "Facing", "Owner_Type",
            "Availability_Status"]


def load_artifacts():
    ref = json.loads((C.MODEL_DIR / "reference_stats.json").read_text())
    clf = joblib.load(C.MODEL_DIR / "good_investment_classifier.joblib")
    reg = joblib.load(C.MODEL_DIR / "appreciation_regressor.joblib")
    metrics = json.loads((C.MODEL_DIR / "metrics.json").read_text())
    return ref, clf, reg, metrics


def to_frame(form: dict, ref: dict) -> pd.DataFrame:
    """Single-row raw frame from the form values."""
    f = dict(form)
    f["Age_of_Property"] = max(0, ref["age_reference_year"] - int(f["Year_Built"]))
    f["Total_Floors"] = max(int(f["Total_Floors"]), int(f["Floor_No"]))
    f["Amenities"] = ", ".join(f.get("Amenities_list", [])) or "None"
    f.setdefault("Locality", "")
    return pd.DataFrame([{c: f[c] for c in RAW_COLS}])


def _predict(raw: pd.DataFrame, ref, clf, reg):
    d = F.engineer(raw, ref["city_ppsf_inr"])
    X = d[F.MODEL_FEATURES]
    prob = clf.predict_proba(X)[:, 1]
    rate_ml = reg.predict(X)
    rate_rule = F.appreciation_rate(d).values
    return d, prob, rate_ml, rate_rule


def max_price_for_good(raw_row: pd.DataFrame, ref: dict, steps: int = 80):
    """Highest asking price (lakhs) at which the rule-based 'Good Investment' test passes, or None."""
    base = float(raw_row["Price_in_Lakhs"].iloc[0])
    grid = np.linspace(base * 0.30, base * 1.0, steps)
    rows = pd.concat([raw_row] * steps, ignore_index=True)
    rows["Price_in_Lakhs"] = grid
    d = F.engineer(rows, ref["city_ppsf_inr"])
    ok = F.label_good_investment(d, F.appreciation_rate(d)).values.astype(bool)
    return float(grid[ok].max()) if ok.any() else None


def assess(form: dict, ref, clf, reg, market: pd.DataFrame, years_max: int = 10) -> dict:
    raw = to_frame(form, ref)
    d, prob, rate_ml, rate_rule = _predict(raw, ref, clf, reg)
    price = float(raw["Price_in_Lakhs"].iloc[0])
    rate = float(np.clip(rate_ml[0], C.GROWTH["floor"], C.GROWTH["cap"]))
    p = float(prob[0])

    horizon = np.arange(0, years_max + 1)
    sp = C.SCENARIO_SPREAD_PP
    paths = {"Conservative": F.future_value(price, max(rate - sp, 0), horizon),
             "Expected": F.future_value(price, rate, horizon),
             "Optimistic": F.future_value(price, rate + sp, horizon)}

    comp = F.appreciation_components(d).iloc[0]
    # comparables: same city, type and BHK
    comp_mask = (market["City"] == form["City"]) & (market["Property_Type"] == form["Property_Type"]) & (market["BHK"] == form["BHK"])
    comps = market[comp_mask]
    if len(comps) < 30:
        comps = market[market["City"] == form["City"]]
    ppsf = float(d["Price_per_SqFt_INR"].iloc[0])
    bench = {"n_comparables": int(len(comps)),
             "median_price_lakhs": float(comps["Price_in_Lakhs"].median()),
             "median_ppsf_inr": float(comps["Price_per_SqFt_INR"].median()),
             "price_percentile": float((comps["Price_in_Lakhs"] < price).mean() * 100),
             "ppsf_percentile": float((comps["Price_per_SqFt_INR"] < ppsf).mean() * 100),
             "city_median_ppsf_inr": float(ref["city_ppsf_inr"][form["City"]]),
             "ppsf_vs_city": float(d["PPSF_vs_City"].iloc[0])}

    if p >= 0.75 and rate >= C.GOOD_APPRECIATION_THRESHOLD:
        verdict, tone = "Strong investment candidate", "success"
    elif p >= 0.5:
        verdict, tone = "Good investment (moderate confidence)", "success"
    elif p >= 0.25:
        verdict, tone = "Borderline - negotiate or compare alternatives", "warning"
    else:
        verdict, tone = "Not recommended at this price", "error"

    return {"raw": raw, "engineered": d, "prob_good": p, "rate": rate, "rate_rule": float(rate_rule[0]),
            "future_5y": float(F.future_value(price, rate, C.FORECAST_HORIZON_YEARS)),
            "paths": pd.DataFrame(paths, index=horizon), "components": comp, "benchmark": bench,
            "verdict": verdict, "tone": tone, "price": price, "ppsf": ppsf,
            "infra": float(d["Infrastructure_Score"].iloc[0]),
            "max_price_for_good": max_price_for_good(raw, ref)}


def explain(res: dict) -> list:
    """Plain-English reasons (strongest drivers first)."""
    c = res["components"].drop("Base market growth")
    out = []
    for name, v in c.reindex(c.abs().sort_values(ascending=False).index).head(4).items():
        out.append(f"{'+' if v >= 0 else '-'}{abs(v):.2f} pp/year from {name.lower()}")
    b = res["benchmark"]
    if b["ppsf_vs_city"] > C.GOOD_MAX_PPSF_VS_CITY:
        out.append(f"Priced at {b['ppsf_vs_city']:.1f}x the city median per sq ft "
                   f"(rule requires <= {C.GOOD_MAX_PPSF_VS_CITY:.2f}x)")
    return out
