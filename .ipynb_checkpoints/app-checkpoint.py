"""Real Estate Investment Advisor — Streamlit app.
Run:  streamlit run app.py     (needs ./models and ./data produced by the notebook)"""
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from real_estate_features import engineer_features, add_targets, AMENITY_LIST, HORIZON_YEARS

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
REF_YEAR = 2025


@st.cache_resource
def load_assets():
    clf = joblib.load("models/clf_pipeline.joblib")
    reg = joblib.load("models/reg_pipeline.joblib")
    ref = json.load(open("models/reference.json"))
    meta = json.load(open("models/model_meta.json"))
    return clf, reg, ref, meta


@st.cache_data
def load_data():
    return pd.read_csv("data/app_sample.csv")


clf, reg, REF, META = load_assets()
data = load_data()
FEATURES = META["features"]
city_state = data.drop_duplicates("City").set_index("City")["State"].to_dict()
localities = sorted(data["Locality"].unique(), key=lambda s: int(s.split("_")[1]))

st.title("🏠 Real Estate Investment Advisor")
st.caption("Predicts whether a property is a **Good Investment** and its **estimated price after 5 years** · "
           f"models: {META['best_classifier']} (classification) · {META['best_regressor']} (regression)")
tab_adv, tab_mkt, tab_mod = st.tabs(["🔮 Investment Advisor", "📊 Market Explorer", "🧠 Model Insights"])

# ------------------------------------------------------------------------------------------------ Advisor
with tab_adv:
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("City", sorted(city_state), index=0)
            ptype = st.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
            locality = st.selectbox("Locality", localities, index=0)
            bhk = st.slider("BHK", 1, 5, 3)
            size = st.number_input("Size (sq ft)", 300, 6000, 1800, step=50)
            price = st.number_input("Asking price (₹ Lakhs)", 5.0, 1000.0, 150.0, step=5.0)
        with c2:
            year_built = st.slider("Year built", 1990, REF_YEAR, 2012)
            floor_no = st.number_input("Floor number", 0, 40, 5)
            total_floors = st.number_input("Total floors in building", 1, 40, 12)
            furnished = st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
            availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
            facing = st.selectbox("Facing", ["North", "East", "South", "West"])
        with c3:
            schools = st.slider("Schools nearby", 0, 10, 6)
            hospitals = st.slider("Hospitals nearby", 0, 10, 5)
            transport = st.select_slider("Public-transport access", ["Low", "Medium", "High"], value="Medium")
            parking = st.radio("Parking", ["Yes", "No"], horizontal=True)
            security = st.radio("Security", ["Yes", "No"], horizontal=True)
            owner = st.selectbox("Owner type", ["Owner", "Builder", "Broker"])
            amenities = st.multiselect("Amenities", AMENITY_LIST, default=["Gym", "Garden"])
        submitted = st.form_submit_button("Analyse this property", type="primary", width="stretch")

    if submitted:
        row = pd.DataFrame([dict(
            State=city_state[city], City=city, Locality=locality, Property_Type=ptype, BHK=bhk, Size_in_SqFt=size,
            Price_in_Lakhs=price, Age_of_Property=REF_YEAR - year_built, Furnished_Status=furnished,
            Floor_No=min(floor_no, max(total_floors, floor_no)), Total_Floors=max(total_floors, floor_no),
            Nearby_Schools=schools, Nearby_Hospitals=hospitals, Public_Transport_Accessibility=transport,
            Parking_Space=parking, Security=security, Amenities=", ".join(amenities), Facing=facing,
            Owner_Type=owner, Availability_Status=availability)])
        eng = add_targets(engineer_features(row, REF), REF)
        X = eng[FEATURES]
        p_good = float(clf.predict_proba(X)[0, 1])
        future = float(reg.predict(X)[0])
        gain = future / price - 1
        cagr = (future / price) ** (1 / HORIZON_YEARS) - 1

        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Verdict", "✅ Good Investment" if p_good >= 0.5 else "⚠️ Not a Good Investment")
        k2.metric("Model confidence", f"{min(max(p_good, 1 - p_good), 0.999):.1%}")
        k3.metric(f"Estimated price in {HORIZON_YEARS} yrs", f"₹{future:,.1f} L", f"{gain:+.1%} total")
        k4.metric("Implied annual growth", f"{cagr:.2%}")

        g1, g2 = st.columns([1, 1.4])
        with g1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=p_good * 100, number={"suffix": "%"},
                                         title={"text": "Probability of being a Good Investment"},
                                         gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563EB"},
                                                "steps": [{"range": [0, 50], "color": "#FEE2E2"}, {"range": [50, 100], "color": "#DCFCE7"}],
                                                "threshold": {"line": {"color": "#111827", "width": 3}, "value": 50}}))
            fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
            st.plotly_chart(fig, width="stretch")
        with g2:
            years = np.arange(0, HORIZON_YEARS + 1)
            path = price * (future / price) ** (years / HORIZON_YEARS)
            band = META["rmse"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=path + band * years / HORIZON_YEARS, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=years, y=path - band * years / HORIZON_YEARS, fill="tonexty", fillcolor="rgba(37,99,235,.15)",
                                     line=dict(width=0), name=f"± model RMSE (₹{band:,.1f} L)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=years, y=path, mode="lines+markers", line=dict(color="#2563EB", width=3), name="Projected value"))
            fig.update_layout(title="Projected value path", xaxis_title="Years from today", yaxis_title="₹ Lakhs", height=300, margin=dict(t=50, b=30))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Why? — the five investment criteria")
        city_med = REF["city_median_ppsf"].get(city, REF["global_median_ppsf"])
        crit = [("Price/sq ft at or below the city median", eng["Price_per_SqFt_INR"].iloc[0] <= city_med,
                 f"₹{eng['Price_per_SqFt_INR'].iloc[0]:,.0f} vs city median ₹{city_med:,.0f}"),
                ("Growth outlook above median", eng["Annual_Growth_Rate"].iloc[0] >= REF["median_growth"],
                 f"{eng['Annual_Growth_Rate'].iloc[0]:.2%} vs median {REF['median_growth']:.2%} per year"),
                ("3+ BHK", bhk >= 3, f"{bhk} BHK"),
                ("Ready to move", availability == "Ready_to_Move", availability.replace("_", " ")),
                ("Infrastructure score above median", eng["Infrastructure_Score"].iloc[0] >= REF["median_infra"],
                 f"{eng['Infrastructure_Score'].iloc[0]:.0f} vs median {REF['median_infra']:.0f}")]
        for name, ok, detail in crit:
            st.write(f"{'✅' if ok else '❌'} **{name}** — {detail}")
        st.caption(f"Rule score: {int(eng['Investment_Score'].iloc[0])}/5 (≥ 3 = Good). Price range is indicative: ±1 RMSE on held-out data.")
    else:
        st.info("Fill in the property details and press **Analyse this property**.")

# ------------------------------------------------------------------------------------------------ Explorer
with tab_mkt:
    f1, f2, f3, f4 = st.columns(4)
    states = f1.multiselect("State", sorted(data["State"].unique()))
    ptypes = f2.multiselect("Property type", sorted(data["Property_Type"].unique()))
    bhks = f3.multiselect("BHK", sorted(data["BHK"].unique()))
    pr = f4.slider("Price (₹ L)", float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max()),
                   (float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max())))
    sr = st.slider("Size (sq ft)", int(data["Size_in_SqFt"].min()), int(data["Size_in_SqFt"].max()),
                   (int(data["Size_in_SqFt"].min()), int(data["Size_in_SqFt"].max())))
    d = data[data["Price_in_Lakhs"].between(*pr) & data["Size_in_SqFt"].between(*sr)]
    if states: d = d[d["State"].isin(states)]
    if ptypes: d = d[d["Property_Type"].isin(ptypes)]
    if bhks: d = d[d["BHK"].isin(bhks)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching listings", f"{len(d):,}")
    m2.metric("Avg price", f"₹{d['Price_in_Lakhs'].mean():,.0f} L" if len(d) else "–")
    m3.metric("Avg ₹/sq ft", f"₹{d['Price_per_SqFt_INR'].mean():,.0f}" if len(d) else "–")
    m4.metric("% Good Investment", f"{d['Good_Investment'].mean():.1%}" if len(d) else "–")
    if len(d):
        a, b = st.columns(2)
        hm = d.pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt_INR", aggfunc="mean")
        a.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="Blues", title="Avg ₹/sq ft — city × property type",
                                 labels=dict(color="₹/sq ft")).update_layout(height=520), width="stretch")
        top = d.groupby("City")["Good_Investment"].mean().nlargest(12).sort_values() * 100
        b.plotly_chart(px.bar(top, orientation="h", title="Top cities by % Good Investment", labels=dict(value="% Good", City="")
                              ).update_layout(height=520, showlegend=False), width="stretch")
        d = d.assign(Build_Period=pd.cut(d["Year_Built"], [1989, 1994, 1999, 2004, 2009, 2014, 2019, 2023],
                                         labels=["1990-94", "1995-99", "2000-04", "2005-09", "2010-14", "2015-19", "2020-23"]))
        tr = d.groupby("Build_Period", observed=True)["Price_per_SqFt_INR"].median().reset_index()
        st.plotly_chart(px.line(tr, x="Build_Period", y="Price_per_SqFt_INR", markers=True,
                                title="Median ₹/sq ft by construction period").update_layout(height=360), width="stretch")
        st.dataframe(d.sort_values("Future_Price_5Y", ascending=False).head(200)[
            ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Future_Price_5Y",
             "Infrastructure_Score", "Good_Investment"]], width="stretch", hide_index=True)

# ------------------------------------------------------------------------------------------------ Model insights
with tab_mod:
    c1, c2 = st.columns(2)
    c1.subheader("Classification — Good Investment")
    c1.dataframe(pd.DataFrame(META["classification"]).T[["accuracy", "precision", "recall", "f1", "roc_auc"]].style.format("{:.4f}"))
    c2.subheader("Regression — 5-year price")
    c2.dataframe(pd.DataFrame(META["regression"]).T[["rmse", "mae", "r2", "mape"]].style.format("{:.3f}"))
    a, b = st.columns(2)
    ci = pd.Series(META["clf_importance"]).sort_values()
    a.plotly_chart(px.bar(ci, orientation="h", title="Feature importance — classifier").update_layout(showlegend=False, height=480), width="stretch")
    ri = pd.Series(META["reg_importance"]).sort_values()
    b.plotly_chart(px.bar(ri, orientation="h", title="Feature importance — regressor").update_layout(showlegend=False, height=480), width="stretch")
    ab = META["ablation_no_price"]
    st.info(f"**Ablation:** without any price features the classifier still reaches {ab['accuracy']:.1%} accuracy (AUC {ab['roc_auc']:.3f}). "
            "Targets are rule-based (see notebook Step 1.4): the 5-year price assumes growth rates driven by city tier, property type, infrastructure, "
            "construction status and age — it is a scenario model, not a market forecast.")
