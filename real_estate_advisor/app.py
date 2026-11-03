"""
Real Estate Investment Advisor - Streamlit app
Run:  streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import config as C
import advisor as A

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

FINDER_COLS = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
               "Price_per_SqFt_INR", "Year_Built", "Age_of_Property", "Furnished_Status", "Availability_Status",
               "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenity_Count", "Infrastructure_Score",
               "Est_Annual_Appreciation_Pct", "Future_Price_5Y_Lakhs", "Good_Investment", "Owner_Type", "Facing"]


@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    return A.load_artifacts()


@st.cache_data(show_spinner="Loading property data...")
def get_market() -> pd.DataFrame:
    return pd.read_csv(C.CLEAN_CSV, usecols=FINDER_COLS)


@st.cache_data
def get_findings() -> dict:
    p = C.EDA_DIR / "eda_findings.json"
    return json.loads(p.read_text()) if p.exists() else {}


if not (C.CLEAN_CSV.exists() and (C.MODEL_DIR / "good_investment_classifier.joblib").exists()):
    st.error("Processed data / models not found. Run `python 01_preprocess.py` then `python 03_train_models.py` first.")
    st.stop()

ref, clf, reg, metrics = get_artifacts()
market = get_market()

st.title("🏠 Real Estate Investment Advisor")
st.caption("Predict whether a property is a good investment, project its future value, and screen listings across India. "
           "Educational analytics on a synthetic-looking dataset - not financial advice.")

tab_adv, tab_find, tab_ins, tab_meth = st.tabs(["Investment Advisor", "Property Finder", "Market Insights", "Methodology & Model"])

# ====================================================================== TAB 1
with tab_adv:
    st.subheader("Evaluate a property")
    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", sorted(ref["state_city"]), key="a_state")
        city = st.selectbox("City", ref["state_city"][state], key="a_city")
        ptype = st.selectbox("Property type", ["Apartment", "Independent House", "Villa"], key="a_type")
        bhk = st.slider("BHK", 1, 5, 3, key="a_bhk")
        size = st.number_input("Size (sq ft)", 300, 6000, 1500, step=50, key="a_size")
        price = st.number_input("Asking price (Lakhs)", 5.0, 1000.0, 150.0, step=5.0, key="a_price")
    with c2:
        year = st.number_input("Year built / expected completion", 1980, 2030, 2018, key="a_year")
        status = st.radio("Availability", ["Ready_to_Move", "Under_Construction"], horizontal=True, key="a_status")
        furn = st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"], key="a_furn")
        floor = st.number_input("Floor number", 0, 40, 3, key="a_floor")
        tfloors = st.number_input("Total floors in building", 1, 40, 10, key="a_tfloors")
        facing = st.selectbox("Facing", ["East", "North", "South", "West"], key="a_facing")
    with c3:
        schools = st.slider("Nearby schools (1-10)", 1, 10, 6, key="a_sch")
        hosp = st.slider("Nearby hospitals (1-10)", 1, 10, 6, key="a_hos")
        transport = st.select_slider("Public transport accessibility", ["Low", "Medium", "High"], value="Medium", key="a_tr")
        parking = st.radio("Parking space", ["Yes", "No"], horizontal=True, key="a_park")
        security = st.radio("Security (gated / CCTV / guard)", ["Yes", "No"], horizontal=True, key="a_sec")
        owner = st.selectbox("Listed by", ["Builder", "Owner", "Broker"], key="a_owner")
    amen = st.multiselect("Amenities", C.AMENITY_LIST, default=["Gym", "Garden"], key="a_amen")

    if st.button("Analyze investment", type="primary", key="a_go"):
        form = dict(State=state, City=city, Locality="", Property_Type=ptype, BHK=bhk, Size_in_SqFt=size,
                    Price_in_Lakhs=price, Year_Built=year, Furnished_Status=furn, Floor_No=floor, Total_Floors=tfloors,
                    Nearby_Schools=schools, Nearby_Hospitals=hosp, Public_Transport_Accessibility=transport,
                    Parking_Space=parking, Security=security, Amenities_list=amen, Facing=facing, Owner_Type=owner,
                    Availability_Status=status)
        st.session_state["result"] = A.assess(form, ref, clf, reg, market)

    res = st.session_state.get("result")
    if res:
        st.divider()
        getattr(st, res["tone"])(f"**{res['verdict']}**  -  probability of 'Good Investment': {res['prob_good']:.0%}")
        years = st.slider("Forecast horizon (years)", 1, 10, C.FORECAST_HORIZON_YEARS, key="a_years")
        exp_val = float(res["paths"]["Expected"].loc[years])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. annual appreciation", f"{res['rate']:.2f}%")
        m2.metric(f"Projected value in {years}y", f"{exp_val:,.0f} L", f"{(exp_val / res['price'] - 1) * 100:+.0f}%")
        m3.metric("Price per sq ft", f"₹{res['ppsf']:,.0f}", f"{(res['benchmark']['ppsf_vs_city'] - 1) * 100:+.0f}% vs city median")
        m4.metric("Infrastructure score", f"{res['infra']:.1f} / 10")

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            p = res["paths"]
            ax.fill_between(p.index, p["Conservative"], p["Optimistic"], alpha=.2, color="#2a9d8f")
            ax.plot(p.index, p["Expected"], color="#264653", lw=2.5, label="Expected")
            ax.plot(p.index, p["Conservative"], "--", color="#e76f51", label="Conservative (-2pp)")
            ax.plot(p.index, p["Optimistic"], "--", color="#2a9d8f", label="Optimistic (+2pp)")
            ax.axvline(years, color="grey", ls=":")
            ax.set_xlabel("Years from now"); ax.set_ylabel("Value (Lakhs)"); ax.set_title("Projected property value"); ax.legend(fontsize=8)
            st.pyplot(fig); plt.close(fig)
        with g2:
            comp = res["components"]
            fig, ax = plt.subplots(figsize=(6, 3.8))
            colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in comp.values]
            ax.barh([n[:38] for n in comp.index][::-1], comp.values[::-1], color=colors[::-1])
            ax.set_xlabel("Percentage points of annual appreciation"); ax.set_title("What drives the estimate")
            st.pyplot(fig); plt.close(fig)

        b = res["benchmark"]
        st.markdown("#### Market benchmark & reasoning")
        b1, b2 = st.columns(2)
        with b1:
            st.write(f"Compared with **{b['n_comparables']:,}** similar listings (same city/type/BHK): median asking price "
                     f"**{b['median_price_lakhs']:,.0f} L**, median **₹{b['median_ppsf_inr']:,.0f}/sq ft**. "
                     f"Your price is higher than **{b['price_percentile']:.0f}%** of them "
                     f"(per sq ft: **{b['ppsf_percentile']:.0f}%**).")
            mp = res["max_price_for_good"]
            if mp is None:
                st.warning("No price level makes this property pass the 'Good Investment' rule - the fundamentals "
                           "(location score, age, amenities) are the limiting factor.")
            elif mp >= res["price"] - 0.01:
                st.info("The asking price already satisfies the Good Investment rule.")
            else:
                st.info(f"Negotiation target: the rule is met at about **{mp:,.0f} L or below** "
                        f"({(1 - mp / res['price']) * 100:.0f}% under the ask).")
        with b2:
            for line in A.explain(res):
                st.write("• " + line)
        st.caption(f"Model-estimated rate {res['rate']:.2f}% vs rule-based {res['rate_rule']:.2f}%. Forecasts are scenario "
                   f"projections built on stated assumptions (see Methodology), not guarantees.")

# ====================================================================== TAB 2
with tab_find:
    st.subheader("Filter and rank properties")
    f1, f2, f3 = st.columns(3)
    with f1:
        sel_states = st.multiselect("State", sorted(market["State"].unique()), key="f_state")
        city_opts = sorted(market.loc[market["State"].isin(sel_states), "City"].unique()) if sel_states else sorted(market["City"].unique())
        sel_cities = st.multiselect("City", city_opts, key="f_city")
        loc_txt = st.text_input("Locality contains (e.g. 'Locality_12')", key="f_loc")
        sel_types = st.multiselect("Property type", sorted(market["Property_Type"].unique()), key="f_type")
    with f2:
        bhk_r = st.slider("BHK", 1, 5, (1, 5), key="f_bhk")
        price_r = st.slider("Price (Lakhs)", 10, 500, (10, 500), key="f_price")
        size_r = st.slider("Area (sq ft)", 500, 5000, (500, 5000), step=50, key="f_size")
    with f3:
        sel_av = st.multiselect("Availability", sorted(market["Availability_Status"].unique()), key="f_av")
        sel_fu = st.multiselect("Furnishing", sorted(market["Furnished_Status"].unique()), key="f_fu")
        min_infra = st.slider("Min infrastructure score", 0.0, 10.0, 0.0, 0.5, key="f_infra")
        only_good = st.checkbox("Only 'Good Investment' properties", value=True, key="f_good")
    s1, s2 = st.columns(2)
    sort_by = s1.selectbox("Sort by", ["Est_Annual_Appreciation_Pct", "Price_in_Lakhs", "Price_per_SqFt_INR",
                                       "Size_in_SqFt", "Infrastructure_Score"], key="f_sort")
    ascending = s2.radio("Order", ["Descending", "Ascending"], horizontal=True, key="f_order") == "Ascending"

    m = market["BHK"].between(*bhk_r) & market["Price_in_Lakhs"].between(*price_r) & market["Size_in_SqFt"].between(*size_r)
    m &= market["Infrastructure_Score"] >= min_infra
    if sel_states: m &= market["State"].isin(sel_states)
    if sel_cities: m &= market["City"].isin(sel_cities)
    if sel_types: m &= market["Property_Type"].isin(sel_types)
    if sel_av: m &= market["Availability_Status"].isin(sel_av)
    if sel_fu: m &= market["Furnished_Status"].isin(sel_fu)
    if loc_txt.strip(): m &= market["Locality"].str.contains(loc_txt.strip(), case=False, regex=False)
    if only_good: m &= market["Good_Investment"] == 1
    out = market[m].sort_values(sort_by, ascending=ascending)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching properties", f"{len(out):,}")
    if len(out):
        k2.metric("Avg price", f"{out['Price_in_Lakhs'].mean():,.0f} L")
        k3.metric("Avg est. appreciation", f"{out['Est_Annual_Appreciation_Pct'].mean():.2f}%")
        k4.metric("Avg price/sq ft", f"₹{out['Price_per_SqFt_INR'].mean():,.0f}")
        top_n = st.number_input("Rows to display", 10, 1000, 100, step=10, key="f_rows")
        st.dataframe(out.head(int(top_n)).drop(columns=["Good_Investment"]), hide_index=True)
        st.download_button("Download filtered results (CSV, up to 50,000 rows)", out.head(50000).to_csv(index=False).encode(),
                           "filtered_properties.csv", "text/csv", key="f_dl")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.5)); ax.hist(out["Price_in_Lakhs"], bins=40, color="#2a6f97")
            ax.set_title("Price distribution of matches (Lakhs)"); st.pyplot(fig); plt.close(fig)
        with c2:
            cs = out.groupby("City").size().sort_values(ascending=False).head(10)[::-1]
            fig, ax = plt.subplots(figsize=(6, 3.5)); ax.barh(cs.index, cs.values, color="#2a9d8f")
            ax.set_title("Top cities among matches"); st.pyplot(fig); plt.close(fig)
    else:
        st.warning("No properties match these filters - try widening them.")

# ====================================================================== TAB 3
CAPTIONS = {
    "q01_price_distribution": ("Q1  Price distribution", "Prices are almost perfectly uniform between 10 and 500 lakhs (mean 254.6 L, skew ≈ 0) - unlike real markets there is no long right tail."),
    "q02_size_distribution": ("Q2  Size distribution", "Sizes are uniform between 500 and 5,000 sq ft (mean ≈ 2,750)."),
    "q03_ppsf_by_property_type": ("Q3  Price/sq ft by property type", "Mean ₹/sq ft is within 0.6% across Apartment, House and Villa - property type does not move price per sq ft here."),
    "q04_size_vs_price": ("Q4  Size vs price", "Size and total price are uncorrelated (ρ = -0.003). Price per sq ft falls with size (ρ = -0.60) purely because price is independent of size."),
    "q05_outliers": ("Q5  Outliers", "7.9% of listings (19,723) are IQR outliers on price/sq ft - almost all are small units (median 800 sq ft vs 2,747 overall). No size outliers exist."),
    "q06_ppsf_by_state": ("Q6  Price/sq ft by state", "State averages span only ₹12,932 (Punjab) to ₹13,252 (Karnataka) - a 2.5% spread."),
    "q07_price_by_city": ("Q7  Price by city", "City average prices range 250.8 L (Cuttack) to 258.5 L (Bangalore) - a 3% spread. Note the truncated axis."),
    "q08_median_age_by_locality": ("Q8  Median age by locality", "Overall median age is 18 years. Locality medians range 3-34.5 yrs, but each locality has only ~12 listings, so these are noisy."),
    "q09_bhk_by_city": ("Q9  BHK mix by city", "Every BHK type is ≈ 20% in every city (max deviation ≈ 2 pp)."),
    "q10_top5_localities": ("Q10  Top-5 localities & trend", "Top localities (8-13 listings each) average ₹30-34k/sq ft. The data has no sale dates, so 'trend' is by year built - and it is flat (ρ = -0.001)."),
    "q11_correlation_matrix": ("Q11  Correlation matrix", "Raw listing attributes are mutually uncorrelated (max |r| = 0.004). The visible correlations are mechanical (price vs price/sq ft) or come from engineered features."),
    "q12_schools_vs_ppsf": ("Q12  Schools vs price/sq ft", "No relationship (ρ = -0.002); group means differ ≤ 3% with no pattern."),
    "q13_hospitals_vs_ppsf": ("Q13  Hospitals vs price/sq ft", "No relationship (ρ = -0.002)."),
    "q14_price_by_furnished": ("Q14  Furnishing", "Average price differs by only 0.25% across furnishing levels (ANOVA p = 0.61)."),
    "q15_ppsf_by_facing": ("Q15  Facing direction", "Differences ≤ 0.9% and not significant (p = 0.35)."),
    "q16_owner_type": ("Q16  Owner type", "Listings are split evenly: Broker 33.4%, Owner 33.3%, Builder 33.3%."),
    "q17_availability": ("Q17  Availability", "50/50 Ready_to_Move vs Under_Construction; the dataset has no 'Sold' status."),
    "q18_parking": ("Q18  Parking", "Parking has no price effect (254.8 L vs 254.4 L, p = 0.58)."),
    "q19_amenities": ("Q19  Amenities", "Each amenity shifts ₹/sq ft by < 1% (gym +0.75% the largest). More amenities raise the Good-Investment rate (17% → 34%), but that is by design of the rule."),
    "q20_transport": ("Q20  Public transport", "Price/sq ft differs < 0.5% by transport level. Good-Investment rate rises 14% → 24% → 36% (Low → High), again by design of the growth rule."),
    "x1_security_vs_good_investment": ("Crime proxy  Security", "The data has no crime-rate field. Secured communities are 'Good Investments' 30% of the time vs 20% without security (rule-driven)."),
    "x2_infrastructure_vs_value": ("Infrastructure vs value", "Infrastructure score has no relationship with price or price/sq ft (ρ ≈ -0.003) - there is no resale-value signal in the data."),
    "x3_label_overview": ("Label balance", "About 25% of properties are labelled Good Investment (est. appreciation ≥ 8% and ≤ 1.25x city median ₹/sq ft)."),
}

with tab_ins:
    st.subheader("Exploratory analysis")
    st.info("Headline finding: apart from mechanical relationships, the raw attributes in this dataset are statistically "
            "independent of price. Charts below show exactly what the data supports.")
    keys = [k for k in CAPTIONS if (C.FIG_DIR / f"{k}.png").exists()]
    if not keys:
        st.warning("Run `python 02_eda.py` to generate the figures.")
    else:
        pick = st.selectbox("Choose an analysis", keys, format_func=lambda k: CAPTIONS[k][0], key="i_pick")
        st.image(str(C.FIG_DIR / f"{pick}.png"))
        st.write(CAPTIONS[pick][1])

# ====================================================================== TAB 4
with tab_meth:
    st.subheader("How the numbers are produced")
    st.markdown(f"""
**1. Data preparation** - 250,000 listings; no missing values or duplicates. Price/sq ft recomputed in INR (the raw column was rounded to 2 decimals of a lakh, zeroing 2,534 rows). Floor numbers above building height (46.5% of rows) were repaired and flagged.

**2. Engineered features** - Infrastructure score (30% schools, 30% hospitals, 40% transport), amenity count, price vs city median, floor ratio, size per BHK.

**3. 'Good Investment' label (domain rule)** - estimated annual appreciation ≥ **{C.GOOD_APPRECIATION_THRESHOLD}%** *and* price/sq ft ≤ **{C.GOOD_MAX_PPSF_VS_CITY}x** the city median.

**4. Appreciation estimate** - base {C.GROWTH['base']}% + infrastructure (up to {C.GROWTH['infra_max']} pp) + amenities, security, parking, under-construction premium, minus an age penalty, ± a value-for-money adjustment. All weights are in `config.py`.

**5. Forecast** - value × (1 + rate)^years, with conservative/optimistic scenarios at ±{C.SCENARIO_SPREAD_PP} pp.
""")
    st.warning("**Read this before relying on the output.** The dataset contains no historical prices, so appreciation rates are "
               "*assumptions*, not learned from market history. The classifier reproduces the rule that defines the label, so its "
               "~99% accuracy shows rule fidelity, not predictive power on real outcomes. Current price could not be predicted from "
               "listing attributes (R² ≈ 0), so the app benchmarks asking prices against comparable listings instead.")
    cm = metrics["classifier"]["models"]
    st.markdown("**Classifier (hold-out 20%)**")
    st.dataframe(pd.DataFrame(cm).T[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4))
    st.markdown("**Appreciation regressor**")
    st.dataframe(pd.DataFrame(metrics["appreciation_regressor"]["models"]).T.round(4))
    d = metrics["current_price_diagnostic"]
    st.markdown(f"**Current-price diagnostic:** gradient-boosting R² = {d['gbm_r2']:.4f}, MAE {d['gbm_mae_lakhs']:.1f} L vs "
                f"{d['predict_mean_mae_lakhs']:.1f} L for always predicting the mean.")
