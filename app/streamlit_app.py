"""Real Estate Investment Advisor - Streamlit app.   Run:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src import advisor as A
from src import config as C

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_resource
def get_artifacts():
    return A.load_artifacts()


@st.cache_data
def get_data():
    return pd.read_csv(C.APP_DATA)


@st.cache_data
def get_metrics():
    return json.loads((C.REPORT_DIR / "model_metrics.json").read_text())


art, data, metrics = get_artifacts(), get_data(), get_metrics()
state_city = data.groupby("State")["City"].unique().apply(sorted).to_dict()

page = st.sidebar.radio("Navigate", ["🔮 Investment Advisor", "🔎 Explore Listings",
                                     "📊 Market Insights", "🧠 Model Insights", "ℹ️ About & Caveats"])
st.sidebar.info("Forecasts come from a documented growth model, not observed price history "
                "(the dataset has none). See **About & Caveats**.")


def fmt_lakh(x):
    return f"₹{x:,.1f} L" if x < 100 else f"₹{x / 100:,.2f} Cr"


# =============================================================== ADVISOR
if page.startswith("🔮"):
    st.title("🏠 Real Estate Investment Advisor")
    st.caption("Enter a property's details to see whether it is a good investment and its estimated value in 5 years.")
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            state = st.selectbox("State", sorted(state_city), index=sorted(state_city).index("Maharashtra"))
            city = st.selectbox("City", state_city[state])
            ptype = st.selectbox("Property type", C.PROPERTY_TYPES)
            bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
            size = st.number_input("Size (sq ft)", 500, 5000, 1500, step=50)
            price = st.number_input("Asking price (₹ lakhs)", 10.0, 500.0, 150.0, step=5.0)
        with c2:
            year = st.slider("Year built", 1990, 2025, 2015)
            furn = st.selectbox("Furnishing", C.FURNISHED, index=1)
            total_fl = st.slider("Total floors in building", 1, 30, 12)
            floor = st.slider("Floor number", 0, 30, 4)
            facing = st.selectbox("Facing", C.FACING)
            owner = st.selectbox("Seller type", C.OWNER_TYPES)
        with c3:
            schools = st.slider("Nearby schools", 1, 10, 6)
            hosp = st.slider("Nearby hospitals", 1, 10, 6)
            transport = st.select_slider("Public-transport accessibility", C.TRANSPORT, value="Medium")
            avail = st.selectbox("Availability", C.AVAILABILITY)
            parking = st.checkbox("Parking space", True)
            security = st.checkbox("Security (gated / CCTV / guard)", True)
            amen = st.multiselect("Amenities", C.AMENITY_LIST, default=["Gym", "Garden"])
        submitted = st.form_submit_button("Analyse investment", type="primary")

    if submitted:
        res = A.assess(dict(City=city, Property_Type=ptype, BHK=bhk, Size_in_SqFt=size, Price_in_Lakhs=price,
                            Year_Built=year, Furnished_Status=furn, Floor_No=floor, Total_Floors=total_fl,
                            Nearby_Schools=schools, Nearby_Hospitals=hosp, Public_Transport_Accessibility=transport,
                            Parking=parking, Security=security, Amenities=amen, Facing=facing, Owner_Type=owner,
                            Availability_Status=avail), art)
        st.divider()
        a, b, c, d_ = st.columns(4)
        a.metric("Is this a Good Investment?", "✅ Yes" if res["is_good"] else "❌ No")
        b.metric("Model confidence", f"{res['confidence']:.0%}", help="Confidence = probability of the predicted class.")
        c.metric("Estimated price after 5 years", fmt_lakh(res["future"]["Base case"]),
                 delta=f"{res['gain']:+,.1f} L vs today")
        d_.metric("Projected annual appreciation", f"{res['rate']:.2%}")

        if res["is_good"]:
            st.success(f"Probability of 'Good Investment': {res['p_good']:.1%}. The projected growth clears the "
                       f"{C.GOOD_CAGR_THRESHOLD:.0%} threshold and the price/sqft is at or below the {city} median.")
        else:
            reasons = []
            if res["rate"] < C.GOOD_CAGR_THRESHOLD:
                reasons.append(f"projected growth {res['rate']:.2%} is below the {C.GOOD_CAGR_THRESHOLD:.0%} threshold")
            if res["ratio"] > C.GOOD_MAX_PRICE_RATIO:
                reasons.append(f"price/sqft is {res['ratio']:.0%} of the {city} median")
            st.warning(f"Probability of 'Good Investment': {res['p_good']:.1%}. Main reasons: " + "; ".join(reasons) + ".")

        left, right = st.columns(2)
        with left:
            st.subheader("5-year value scenarios")
            sc = pd.Series(res["future"]); sc["Today"] = res["price"]
            order = ["Today", "Conservative (-2pp/yr)", "Base case", "Optimistic (+2pp/yr)"]
            fig, ax = plt.subplots(figsize=(6, 3.6))
            bars = ax.bar(range(4), sc[order].values, color=["#7f7f7f", "#ff9896", "#1f77b4", "#98df8a"])
            ax.set_xticks(range(4)); ax.set_xticklabels(["Today", "Conservative", "Base", "Optimistic"])
            ax.set_ylabel("₹ lakhs")
            for r_, v in zip(bars, sc[order].values):
                ax.annotate(f"{v:,.0f}", (r_.get_x() + r_.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
            st.pyplot(fig); plt.close(fig)
            st.caption("Scenarios shift the annual growth rate by ±2 percentage points. They are assumptions, not statistical intervals.")
        with right:
            st.subheader("What drives the growth estimate (pp / year)")
            comp = (res["components"] * 100).round(2)
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh(comp.index[::-1], comp.values[::-1], color=["#2ca02c" if v >= 0 else "#d62728" for v in comp.values[::-1]])
            ax.set_xlabel("percentage points"); st.pyplot(fig); plt.close(fig)
            st.caption(f"Rule-model total {res['rate_rule']:.2%} (clipped to 2–12%). ML model estimate: {res['rate']:.2%}.")

        st.subheader("Valuation check")
        v1, v2, v3 = st.columns(3)
        v1.metric("Your price per sq ft", f"₹{res['ppsf']:,.0f}")
        v2.metric(f"{city} median price per sq ft", f"₹{res['city_median_ppsf']:,.0f}")
        v3.metric("Infrastructure score (0–10)", f"{res['infra']:.1f}")

# =============================================================== EXPLORER
elif page.startswith("🔎"):
    st.title("🔎 Explore & filter listings")
    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("State", sorted(state_city))
    cities_pool = sorted({c for s in (states or state_city) for c in state_city[s]})
    cities = f2.multiselect("City", cities_pool)
    ptypes = f3.multiselect("Property type", C.PROPERTY_TYPES)
    g1, g2, g3 = st.columns(3)
    bhks = g1.multiselect("BHK", [1, 2, 3, 4, 5])
    size_rng = g2.slider("Area (sq ft)", 500, 5000, (500, 5000), step=50)
    price_rng = g3.slider("Price (₹ lakhs)", 10, 500, (10, 500), step=5)
    h1, h2, h3 = st.columns(3)
    avail = h1.multiselect("Availability", C.AVAILABILITY)
    only_good = h2.checkbox("Only 'Good Investment' listings")
    sort_by = h3.selectbox("Sort by", ["Annual_Appreciation_Rate", "Future_Price_5Y_Lakhs", "Price_in_Lakhs", "Size_in_SqFt"])

    m = data[data["Size_in_SqFt"].between(*size_rng) & data["Price_in_Lakhs"].between(*price_rng)]
    for col, vals in [("State", states), ("City", cities), ("Property_Type", ptypes), ("BHK", bhks), ("Availability_Status", avail)]:
        if vals:
            m = m[m[col].isin(vals)]
    if only_good:
        m = m[m["Good_Investment"] == 1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching listings", f"{len(m):,}")
    k2.metric("Avg price", fmt_lakh(m["Price_in_Lakhs"].mean()) if len(m) else "–")
    k3.metric("Avg price/sqft", f"₹{m['Price_per_SqFt_INR'].mean():,.0f}" if len(m) else "–")
    k4.metric("% Good Investment", f"{m['Good_Investment'].mean():.1%}" if len(m) else "–")
    if len(m):
        show = m.sort_values(sort_by, ascending=False).head(500)
        st.dataframe(show[["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                           "Price_per_SqFt_INR", "Availability_Status", "Annual_Appreciation_Rate",
                           "Future_Price_5Y_Lakhs", "Good_Investment"]].round(3), use_container_width=True, hide_index=True)
        st.caption("Showing top 500 rows of the filtered result.")
        st.download_button("Download filtered results (CSV)", m.to_csv(index=False).encode(), "filtered_listings.csv", "text/csv")
    else:
        st.info("No listings match these filters.")

# =============================================================== MARKET
elif page.startswith("📊"):
    st.title("📊 Market insights")
    st.caption("Note: in this dataset prices are almost uniformly distributed and barely differ by location or feature — see About & Caveats.")
    t1, t2, t3, t4 = st.tabs(["Location heatmap", "City comparison", "Trend by build year", "Size vs price"])
    with t1:
        hm = data.pivot_table(index="State", columns="Property_Type", values="Price_per_SqFt_INR", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7, 8)); sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
        ax.set_title("Avg price per sq ft (₹): state × property type"); st.pyplot(fig); plt.close(fig)
    with t2:
        metric = st.selectbox("Metric", ["Price_in_Lakhs", "Price_per_SqFt_INR", "Annual_Appreciation_Rate", "Good_Investment"])
        cs = data.groupby("City")[metric].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 9)); cs.plot.barh(ax=ax, color="#1f77b4"); ax.set_xlabel(metric)
        lo, hi = cs.min(), cs.max(); ax.set_xlim(lo - (hi - lo) * 0.2, hi + (hi - lo) * 0.05)
        st.pyplot(fig); plt.close(fig); st.caption("Axis is truncated to make small differences visible.")
    with t3:
        sel = st.multiselect("Cities", sorted(data["City"].unique()), default=["Mumbai", "Bangalore", "Chennai"])
        if sel:
            sub = data[data["City"].isin(sel)].copy()
            sub["Build period"] = pd.cut(sub["Year_Built"], [1989, 1995, 2000, 2005, 2010, 2015, 2020, 2023],
                                         labels=["1990-95", "1996-00", "2001-05", "2006-10", "2011-15", "2016-20", "2021-23"])
            tr = sub.groupby(["Build period", "City"], observed=True)["Price_per_SqFt_INR"].median().unstack()
            fig, ax = plt.subplots(figsize=(9, 4.5)); tr.plot(marker="o", ax=ax); ax.set_ylabel("Median ₹/sqft")
            st.pyplot(fig); plt.close(fig)
            st.caption("Year-built cohorts are only a proxy for time — the data has no transaction dates.")
    with t4:
        smp = data.sample(5000, random_state=0)
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(smp["Size_in_SqFt"], smp["Price_in_Lakhs"], c=smp["Annual_Appreciation_Rate"] * 100, s=6, cmap="viridis", alpha=.6)
        plt.colorbar(sc, label="Projected growth %/yr"); ax.set(xlabel="Size (sq ft)", ylabel="Price (₹ lakhs)")
        st.pyplot(fig); plt.close(fig)

# =============================================================== MODEL
elif page.startswith("🧠"):
    st.title("🧠 Model performance, confidence & feature importance")
    cm_ = metrics["classification"]; rm = metrics["regression"]
    st.subheader(f"Classifier — {cm_['best_model']}")
    tbl = pd.DataFrame(cm_["candidates"]).T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1_mean_40k"]].round(4)
    st.dataframe(tbl, use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Predictions with ≥90% confidence", f"{cm_['share_predictions_with_confidence_ge_0.9']:.1%}")
    b.metric("Mean confidence", f"{cm_['mean_confidence']:.1%}")
    c.metric("Majority-class baseline accuracy", f"{cm_['majority_class_baseline_accuracy']:.1%}")
    cm = np.array(cm_["confusion_matrix_[[TN,FP],[FN,TP]]"])
    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"], ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)"); st.pyplot(fig); plt.close(fig)
    with r:
        cal = pd.DataFrame(cm_["calibration"])
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.plot(cal["mean_pred"], cal["actual_rate"], marker="o")
        ax.set(xlabel="Predicted probability", ylabel="Observed share good", title="Calibration"); st.pyplot(fig); plt.close(fig)
    st.subheader(f"Regressor (annual appreciation rate) — {rm['best_model']}")
    rt = pd.DataFrame(rm["candidates"]).T.round(4)
    rt.loc["Constant-rate baseline"] = [0.0, np.nan, rm["baseline_constant_rate"]["mae_future_price_lakhs"], np.nan]
    st.dataframe(rt, use_container_width=True)
    st.subheader("Feature importance (permutation)")
    fc = pd.read_csv(C.MODEL_DIR / "feature_importance_classifier.csv").head(12)
    fr = pd.read_csv(C.MODEL_DIR / "feature_importance_regressor.csv").head(12)
    l, r = st.columns(2)
    for col, f, t in [(l, fc, "Classifier (drop in F1)"), (r, fr, "Regressor (drop in R²)")]:
        with col:
            fig, ax = plt.subplots(figsize=(5, 4.2)); ax.barh(f["feature"][::-1], f["importance"][::-1], color="#ff7f0e")
            ax.set_title(t); st.pyplot(fig); plt.close(fig)
    fp = metrics["fair_price_check"]
    st.error(f"**Reality check:** a model asked to predict today's price from the listing attributes scores R² = {fp['r2_test']:.3f} "
             f"(MAE {fp['mae_lakhs']:.0f} L vs {fp['mae_if_predicting_mean_lakhs']:.0f} L for guessing the mean). "
             "Prices in this dataset are unrelated to the other columns, so the high scores above reflect how well the models "
             "reproduce the documented rule-based labels — not an ability to forecast the real market.")

# =============================================================== ABOUT
else:
    st.title("ℹ️ About & caveats")
    st.markdown(f"""
**What the numbers mean**
- **Good Investment** = projected annual appreciation ≥ **{C.GOOD_CAGR_THRESHOLD:.0%}** *and* price/sqft ≤ **{C.GOOD_MAX_PRICE_RATIO:.0%}** of the city median.
- **Price after 5 years** = price × (1 + rate)^5, where *rate* is estimated by a gradient-boosting model trained to reproduce a transparent growth model
  (base 5% + metro, infrastructure, amenities, age, construction status, house/villa and valuation adjustments; clipped to 2–12%).

**Important limitations**
1. The dataset is a single snapshot with **no price history**, so appreciation cannot be learned from data; the growth model encodes domain assumptions.
2. Listing attributes show **no statistical relationship with price** in this data (it looks synthetic), so no model can estimate a "fair price".
3. There is **no crime-rate column**; the Security flag is used only as a rough proxy in the EDA.
4. About 46% of rows had Floor_No > Total_Floors (fixed by capping); all 'Under_Construction' rows have a non-zero age.
5. Not financial advice. Validate any decision with local market data and a qualified advisor.
""")
