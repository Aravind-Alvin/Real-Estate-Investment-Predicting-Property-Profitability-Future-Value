"""Real Estate Investment Advisor — Streamlit app.

Run locally:   streamlit run app.py
Needs:         ./models/*.joblib, ./models/*.json, ./data/app_sample.csv  (all written by the notebook)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Insights import render_lab
from real_estate_features import AMENITY_LIST, HORIZON_YEARS, add_targets, engineer_features

# ──────────────────────────────────────────────────────────────────────────── config
AUTHOR = "P Suman Sangeet"
REF_YEAR = 2025
ROOT = Path(__file__).parent
BLUE, GREEN, RED, AMBER = "#2563EB", "#10B981", "#EF4444", "#F59E0B"

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [BLUE, AMBER, GREEN, RED, "#8B5CF6", "#06B6D4", "#EC4899", "#64748B"]

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background: linear-gradient(135deg,#4F46E5 0%,#7C3AED 55%,#EC4899 100%); color:#fff;
       padding: 26px 30px; border-radius: 16px; margin-bottom: 14px;}
.hero h1 {color:#fff; margin:0 0 4px 0; font-size: 2.0rem;}
.hero p  {color:#E0E7FF; margin:0 0 12px 0; font-size: 1.02rem;}
.chip {display:inline-block; background:rgba(255,255,255,.2); padding:3px 11px; border-radius:20px;
       margin:0 6px 4px 0; font-size:.82rem;}
.verdict {border-radius:14px; padding:18px 22px; margin-bottom:10px; border-left:8px solid;}
.verdict.good {background:#ECFDF5; border-color:#10B981; color:#064E3B;}
.verdict.bad  {background:#FEF2F2; border-color:#EF4444; color:#7F1D1D;}
.verdict .t {font-size:1.55rem; font-weight:700; line-height:1.2;}
.verdict .s {font-size:.95rem; opacity:.85; margin-top:4px;}
.bigprice {border-radius:14px; padding:18px 22px; margin-bottom:10px; background:#EFF6FF;
           border-left:8px solid #2563EB; color:#1E3A8A;}
.bigprice .t {font-size:1.9rem; font-weight:700; line-height:1.2;}
.bigprice .s {font-size:.95rem; opacity:.85; margin-top:4px;}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────── loading
@st.cache_resource(show_spinner="Loading trained models…")
def load_assets():
    clf = joblib.load(ROOT / "models" / "clf_pipeline.joblib")
    reg = joblib.load(ROOT / "models" / "reg_pipeline.joblib")
    ref = json.loads((ROOT / "models" / "reference.json").read_text())
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    return clf, reg, ref, meta


@st.cache_data(show_spinner="Loading market data…")
def load_data() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "app_sample.csv")
    d["Gain_5Y_pct"] = (d["Future_Price_5Y"] / d["Price_in_Lakhs"] - 1) * 100
    d["Good_Investment_pct"] = d["Good_Investment"] * 100
    d["Build_Start"] = (d["Year_Built"] // 5 * 5).astype(int)
    return d


try:
    clf, reg, REF, META = load_assets()
    data = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: `{e.filename}`. Run the notebook first so it writes `models/` and `data/`, "
             "then restart the app (see DEPLOYMENT.md).")
    st.stop()

FEATURES = META["features"]
CITY_STATE = data.drop_duplicates("City").set_index("City")["State"].to_dict()
ALL_CITIES = sorted(CITY_STATE)


def _loc_key(s: str):
    tail = s.rsplit("_", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, s)


ALL_LOCALITIES = sorted(data["Locality"].unique(), key=_loc_key)
LOC_BY_CITY = data.groupby("City")["Locality"].unique().to_dict()
P_MIN, P_MAX = float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max())
S_MIN, S_MAX = int(data["Size_in_SqFt"].min()), int(data["Size_in_SqFt"].max())


def fmt_price(lakhs: float) -> str:
    """₹ Lakhs → friendly string (₹ Cr above 100 L)."""
    return f"₹{lakhs / 100:,.2f} Cr" if lakhs >= 100 else f"₹{lakhs:,.1f} L"


# ──────────────────────────────────────────────────────────────────────────── inference
def predict(rows: pd.DataFrame):
    """Exactly the notebook's inference path: features → targets-helpers → pipelines."""
    eng = add_targets(engineer_features(rows, REF), REF)
    X = eng[FEATURES]
    return eng, clf.predict_proba(X)[:, 1], reg.predict(X)


INPUT_KEYS = ["in_city", "in_ptype", "in_locality", "in_bhk", "in_size", "in_price", "in_year", "in_floor",
              "in_total", "in_furn", "in_avail", "in_facing", "in_schools", "in_hosp", "in_transport",
              "in_parking", "in_security", "in_owner", "in_amen"]


def build_row(s: dict, price: float | None = None) -> pd.DataFrame:
    total = max(int(s["in_total"]), int(s["in_floor"]))
    return pd.DataFrame([dict(
        State=CITY_STATE[s["in_city"]], City=s["in_city"], Locality=s["in_locality"],
        Property_Type=s["in_ptype"], BHK=int(s["in_bhk"]), Size_in_SqFt=float(s["in_size"]),
        Price_in_Lakhs=float(price if price is not None else s["in_price"]),
        Age_of_Property=REF_YEAR - int(s["in_year"]), Furnished_Status=s["in_furn"],
        Floor_No=int(s["in_floor"]), Total_Floors=total, Nearby_Schools=int(s["in_schools"]),
        Nearby_Hospitals=int(s["in_hosp"]), Public_Transport_Accessibility=s["in_transport"],
        Parking_Space=s["in_parking"], Security=s["in_security"], Amenities=", ".join(s["in_amen"]),
        Facing=s["in_facing"], Owner_Type=s["in_owner"], Availability_Status=s["in_avail"])])


def analyse(s: dict) -> dict:
    eng, p, f = predict(build_row(s))
    prices = float(s["in_price"]) * np.linspace(0.5, 1.5, 31)          # "negotiation curve"
    _, pc, fc = predict(pd.concat([build_row(s, x) for x in prices], ignore_index=True))
    return dict(inputs=s, eng=eng.iloc[0], p_good=float(p[0]), future=float(f[0]),
                curve=pd.DataFrame(dict(price=prices, p_good=pc, future=fc)))


def break_even_message(curve: pd.DataFrame, price: float, good: bool) -> str:
    i0 = int((curve["price"] - price).abs().idxmin())
    ok = (curve["p_good"] >= 0.5).to_numpy()
    if good:
        j = i0
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        if j == len(ok) - 1:
            return "✅ The verdict **holds across the whole −50 % … +50 % price range** — it is driven by fundamentals, not by the asking price."
        p = curve["price"].iloc[j]
        return f"⚠️ Stays a *Good Investment* up to about **{fmt_price(p)}** ({p / price - 1:+.0%}); above that the verdict flips."
    j = i0
    while j - 1 >= 0 and not ok[j - 1]:
        j -= 1
    if j == 0:
        return "❌ **No price down to −50 % flips the verdict** — the weak spots are fundamentals (BHK, readiness, infrastructure), not the price."
    p = curve["price"].iloc[j - 1]
    return f"💡 Becomes a *Good Investment* at roughly **{fmt_price(p)}** ({p / price - 1:+.0%}) — a data-backed negotiation target."


# ──────────────────────────────────────────────────────────────────────────── form state & presets
def pick_city(*prefs: str) -> str:
    return next((c for c in prefs if c in CITY_STATE), ALL_CITIES[0])


PRESETS = {
    "🏢 Starter apartment": dict(in_city=pick_city("Pune", "Bangalore"), in_ptype="Apartment", in_bhk=2, in_size=950,
                                 in_price=60.0, in_year=2018, in_floor=4, in_total=10, in_avail="Ready_to_Move",
                                 in_schools=6, in_hosp=5, in_transport="High", in_amen=["Gym"]),
    "🏡 Family home": dict(in_city=pick_city("Hyderabad", "Chennai"), in_ptype="Independent House", in_bhk=3,
                           in_size=1800, in_price=150.0, in_year=2012, in_floor=0, in_total=2,
                           in_avail="Ready_to_Move", in_schools=7, in_hosp=5, in_transport="Medium",
                           in_amen=["Garden", "Playground"]),
    "🏰 Premium villa": dict(in_city=pick_city("Mumbai", "Delhi"), in_ptype="Villa", in_bhk=4, in_size=3500,
                             in_price=450.0, in_year=2024, in_floor=0, in_total=2, in_avail="Under_Construction",
                             in_schools=8, in_hosp=7, in_transport="Medium", in_amen=["Pool", "Clubhouse", "Garden"]),
}
DEFAULTS = dict(in_city=pick_city("Pune"), in_ptype="Apartment", in_bhk=3, in_size=1800, in_price=150.0, in_year=2012,
                in_floor=5, in_total=12, in_furn="Semi-furnished", in_avail="Ready_to_Move", in_facing="East",
                in_schools=6, in_hosp=5, in_transport="Medium", in_parking="Yes", in_security="Yes",
                in_owner="Owner", in_amen=["Gym", "Garden"])
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[k] = min(max(v, P_MIN), P_MAX) if k == "in_price" else v
    st.session_state["in_size"] = int(min(max(st.session_state["in_size"], S_MIN), S_MAX))
    st.session_state.pop("in_locality", None)


# ──────────────────────────────────────────────────────────────────────────── sidebar filters
FILTER_DEFAULTS = ["f_states", "f_cities", "f_types", "f_bhk", "f_avail", "f_price", "f_size"]
for k in FILTER_DEFAULTS[:5]:
    st.session_state.setdefault(k, [])


def reset_filters():
    for k in FILTER_DEFAULTS:          # deleting the keys lets every widget fall back to its own default
        st.session_state.pop(k, None)


with st.sidebar:
    st.header("🔎 Market filters")
    st.caption("These filters drive the **Market Explorer** tab.")
    st.multiselect("State", sorted(data["State"].unique()), key="f_states")
    city_opts = sorted(data.loc[data["State"].isin(st.session_state["f_states"]), "City"].unique()
                       if st.session_state["f_states"] else data["City"].unique())
    st.session_state["f_cities"] = [c for c in st.session_state["f_cities"] if c in city_opts]
    st.multiselect("City", city_opts, key="f_cities")
    st.multiselect("Property type", sorted(data["Property_Type"].unique()), key="f_types")
    st.multiselect("BHK", sorted(data["BHK"].unique()), key="f_bhk")
    st.multiselect("Availability", sorted(data["Availability_Status"].unique()), key="f_avail")
    st.slider("Asking price (₹ Lakhs)", P_MIN, P_MAX, value=(P_MIN, P_MAX), step=5.0, key="f_price")
    st.slider("Built-up area (sq ft)", S_MIN, S_MAX, value=(S_MIN, S_MAX), step=50, key="f_size")
    st.button("↺ Reset filters", on_click=reset_filters, width="stretch")
    st.divider()
    st.caption(f"Built by **{AUTHOR}** · Python · scikit-learn · XGBoost · MLflow · Streamlit")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    s = st.session_state
    m = df["Price_in_Lakhs"].between(*s["f_price"]) & df["Size_in_SqFt"].between(*s["f_size"])
    for col, key in [("State", "f_states"), ("City", "f_cities"), ("Property_Type", "f_types"),
                     ("BHK", "f_bhk"), ("Availability_Status", "f_avail")]:
        if s[key]:
            m &= df[col].isin(s[key])
    return df[m]


# ──────────────────────────────────────────────────────────────────────────── header
clf_name, reg_name = META["best_classifier"], META["best_regressor"]
st.markdown(
    f"""<div class="hero"><h1>🏠 Real Estate Investment Advisor</h1>
<p>Is this property a good investment — and what will it be worth in {HORIZON_YEARS} years? Predict, compare and explore the Indian housing market.</p>
<span class="chip">Classification · {clf_name}</span><span class="chip">Regression · {reg_name}</span>
<span class="chip">{len(data):,} listings explored</span><span class="chip">MLflow-tracked</span>
<span class="chip">+ KMeans · IsolationForest · KNN · Monte Carlo · Counterfactuals</span></div>""",
    unsafe_allow_html=True)
tab_adv, tab_lab, tab_mkt, tab_mod = st.tabs(["🔮 Investment Advisor", "🧪 Insight Lab", "📊 Market Explorer", "🧠 Model & Methodology"])

# ════════════════════════════════════════════════════════════════════════════ TAB 1 · ADVISOR
with tab_adv:
    st.markdown("##### Describe the property  ·  or start from an example")
    pc = st.columns(len(PRESETS) + 3)
    for col, name in zip(pc, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**📍 Location & type**")
        st.selectbox("City", ALL_CITIES, key="in_city")
        st.selectbox("Property type", ["Apartment", "Independent House", "Villa"], key="in_ptype")
        locs = sorted(LOC_BY_CITY.get(st.session_state["in_city"], ALL_LOCALITIES), key=_loc_key)
        locs = locs if len(locs) >= 3 else ALL_LOCALITIES
        if st.session_state.get("in_locality") not in locs:
            st.session_state["in_locality"] = locs[0]
        st.selectbox("Locality", locs, key="in_locality")
        st.slider("BHK", int(data["BHK"].min()), int(data["BHK"].max()), key="in_bhk")
    with c2.container(border=True):
        st.markdown("**🏗️ Size, price & age**")
        st.number_input("Built-up area (sq ft)", 100, S_MAX * 2, step=50, key="in_size")
        st.number_input("Asking price (₹ Lakhs)", 1.0, P_MAX * 3, step=5.0, key="in_price")
        st.slider("Year built", int(data["Year_Built"].min()), REF_YEAR, key="in_year")
        st.number_input("Floor number", 0, 60, key="in_floor")
        st.number_input("Total floors in building", 1, 60, key="in_total")
        st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"], key="in_avail",
                     format_func=lambda x: x.replace("_", " "))
    with c3.container(border=True):
        st.markdown("**🌆 Surroundings & extras**")
        st.slider("Schools nearby", 0, 10, key="in_schools")
        st.slider("Hospitals nearby", 0, 10, key="in_hosp")
        st.select_slider("Public-transport access", ["Low", "Medium", "High"], key="in_transport")
        r1, r2 = st.columns(2)
        r1.radio("Parking", ["Yes", "No"], horizontal=True, key="in_parking")
        r2.radio("Security", ["Yes", "No"], horizontal=True, key="in_security")
        st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"], key="in_furn")
        f1, f2 = st.columns(2)
        f1.selectbox("Facing", ["North", "East", "South", "West"], key="in_facing")
        f2.selectbox("Owner type", ["Owner", "Builder", "Broker"], key="in_owner")
        st.multiselect("Amenities", AMENITY_LIST, key="in_amen")

    if st.session_state["in_price"] > P_MAX or st.session_state["in_price"] < P_MIN or \
            not S_MIN <= st.session_state["in_size"] <= S_MAX:
        st.warning(f"Price or size is outside the range the models were trained on "
                   f"({fmt_price(P_MIN)} – {fmt_price(P_MAX)}, {S_MIN:,} – {S_MAX:,} sq ft). Treat the result with caution.")

    if st.button("🔍 Analyse this property", type="primary", width="stretch"):
        with st.spinner("Scoring the property…"):
            st.session_state["result"] = analyse({k: st.session_state[k] for k in INPUT_KEYS})

    res = st.session_state.get("result")
    if res is None:
        st.info("Fill in the details (or pick an example) and press **Analyse this property**.")
    else:
        s, eng, p_good, future = res["inputs"], res["eng"], res["p_good"], res["future"]
        price, good = float(s["in_price"]), res["p_good"] >= 0.5
        gain, cagr = future / price - 1, (future / price) ** (1 / HORIZON_YEARS) - 1
        city_med = REF["city_median_ppsf"].get(s["in_city"], REF["global_median_ppsf"])
        ppsf = float(eng["Price_per_SqFt_INR"])
        rmse = META["rmse"]

        st.divider()
        st.subheader(f"Result — {s['in_bhk']} BHK {s['in_ptype']} in {s['in_city']} · {fmt_price(price)}")
        v1, v2 = st.columns(2)
        v1.markdown(
            f"""<div class="verdict {'good' if good else 'bad'}"><div class="t">{'✅ Good Investment' if good else '⚠️ Not a Good Investment'}</div>
<div class="s">Classifier confidence {min(max(p_good, 1 - p_good), 0.999):.1%} · rule score {int(eng['Investment_Score'])}/5 (≥ 3 = good)</div></div>""",
            unsafe_allow_html=True)
        v2.markdown(
            f"""<div class="bigprice"><div class="t">{fmt_price(future)} <span style="font-size:1rem">in {REF_YEAR + HORIZON_YEARS}</span></div>
<div class="s">{gain:+.1%} total · {cagr:.2%} per year · typical model error ± {fmt_price(rmse)}</div></div>""",
            unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Price per sq ft", f"₹{ppsf:,.0f}", f"{ppsf / city_med - 1:+.0%} vs {s['in_city']} median", delta_color="inverse")
        k2.metric("Infrastructure score", f"{eng['Infrastructure_Score']:.0f} / 100",
                  f"{eng['Infrastructure_Score'] - REF['median_infra']:+.0f} vs median")
        k3.metric("Assumed growth outlook", f"{eng['Annual_Growth_Rate']:.2%} / yr",
                  f"{(eng['Annual_Growth_Rate'] - REF['median_growth']) * 100:+.2f} pp vs median")
        k4.metric("Space per bedroom", f"{eng['Sqft_per_BHK']:,.0f} sq ft")

        g1, g2 = st.columns([1, 1.5])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p_good * 100, number={"suffix": "%"},
            title={"text": "Probability of being a Good Investment"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": GREEN if good else RED},
                   "steps": [{"range": [0, 50], "color": "#FEE2E2"}, {"range": [50, 100], "color": "#DCFCE7"}],
                   "threshold": {"line": {"color": "#111827", "width": 3}, "value": 50}}))
        gauge.update_layout(height=320, margin=dict(t=70, b=10, l=30, r=30))
        g1.plotly_chart(gauge, width="stretch", key="gauge")

        yrs = np.arange(HORIZON_YEARS + 1)
        path = price * (future / price) ** (yrs / HORIZON_YEARS)
        band = rmse * yrs / HORIZON_YEARS
        proj = go.Figure()
        proj.add_trace(go.Scatter(x=REF_YEAR + yrs, y=path + band, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        proj.add_trace(go.Scatter(x=REF_YEAR + yrs, y=path - band, fill="tonexty", fillcolor="rgba(37,99,235,.15)",
                                  line=dict(width=0), name=f"± model RMSE ({fmt_price(rmse)})", hoverinfo="skip"))
        proj.add_trace(go.Scatter(x=REF_YEAR + yrs, y=path, mode="lines+markers", line=dict(color=BLUE, width=3),
                                  name="Projected value", hovertemplate="%{x}: ₹%{y:,.1f} L<extra></extra>"))
        proj.update_layout(title="Projected value path", xaxis=dict(dtick=1, title="Year"), yaxis_title="₹ Lakhs",
                           height=320, margin=dict(t=60, b=30), legend=dict(orientation="h", y=-0.25))
        g2.plotly_chart(proj, width="stretch", key="proj")

        w1, w2 = st.columns(2)
        with w1:
            curve = res["curve"]
            wf = go.Figure()
            wf.add_hrect(y0=50, y1=100, fillcolor="#DCFCE7", opacity=.5, line_width=0)
            wf.add_hrect(y0=0, y1=50, fillcolor="#FEE2E2", opacity=.5, line_width=0)
            wf.add_trace(go.Scatter(x=curve["price"], y=curve["p_good"] * 100, mode="lines", line=dict(color=BLUE, width=3),
                                    customdata=curve["future"], name="P(Good)",
                                    hovertemplate="Ask ₹%{x:,.1f} L<br>P(Good) %{y:.0f}%<br>5-yr value ₹%{customdata:,.1f} L<extra></extra>"))
            wf.add_trace(go.Scatter(x=[price], y=[p_good * 100], mode="markers", marker=dict(size=14, color="#111827"),
                                    name="Your asking price"))
            wf.update_layout(title="What-if: asking price vs. investment verdict", xaxis_title="Asking price (₹ Lakhs)",
                             yaxis=dict(title="P(Good Investment) %", range=[0, 100]), height=340,
                             margin=dict(t=60, b=30), legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(wf, width="stretch", key="whatif")
            st.markdown(break_even_message(curve, price, good))
        with w2:
            city_df = data.loc[data["City"] == s["in_city"], "Price_per_SqFt_INR"]
            hist = px.histogram(city_df[city_df <= city_df.quantile(.99)], nbins=40, title=f"How {s['in_city']} prices per sq ft compare",
                                labels=dict(value="₹ per sq ft"))
            hist.add_vline(x=ppsf, line_color=RED, line_width=3, annotation_text="This property")
            hist.add_vline(x=city_med, line_color="#111827", line_dash="dash", annotation_text="City median", annotation_position="bottom right")
            hist.update_layout(height=340, showlegend=False, yaxis_title="Listings", margin=dict(t=60, b=30))
            st.plotly_chart(hist, width="stretch", key="pphist")
            pct = (city_df <= ppsf).mean()
            st.markdown(f"This property is cheaper per sq ft than **{1 - pct:.0%}** of listings in {s['in_city']}.")

        st.markdown("##### Why? The five investment criteria")
        crit = [("Price/sq ft ≤ city median", ppsf <= city_med, f"₹{ppsf:,.0f} vs ₹{city_med:,.0f}"),
                ("Growth outlook ≥ median", eng["Annual_Growth_Rate"] >= REF["median_growth"],
                 f"{eng['Annual_Growth_Rate']:.2%} vs {REF['median_growth']:.2%}"),
                ("3+ BHK", s["in_bhk"] >= 3, f"{s['in_bhk']} BHK"),
                ("Ready to move", s["in_avail"] == "Ready_to_Move", s["in_avail"].replace("_", " ")),
                ("Infrastructure ≥ median", eng["Infrastructure_Score"] >= REF["median_infra"],
                 f"{eng['Infrastructure_Score']:.0f} vs {REF['median_infra']:.0f}")]
        for col, (name, ok, detail) in zip(st.columns(5), crit):
            with col.container(border=True):
                st.markdown(f"{'✅' if ok else '❌'} **{name}**")
                st.caption(detail)

        sim = data[(data["City"] == s["in_city"]) & (data["BHK"] == s["in_bhk"])
                   & data["Price_in_Lakhs"].between(price * .75, price * 1.25)]
        with st.expander(f"🔁 Comparable listings — {len(sim):,} in {s['in_city']}, {s['in_bhk']} BHK, within ±25 % of your price"):
            if sim.empty:
                st.write("No comparable listings in the sample.")
            else:
                st.dataframe(sim.sort_values("Gain_5Y_pct", ascending=False).head(15).assign(
                    Good_Investment=lambda x: x["Good_Investment"].astype(bool))[
                    ["Locality", "Property_Type", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR",
                     "Future_Price_5Y", "Gain_5Y_pct", "Good_Investment"]],
                    width="stretch", hide_index=True,
                    column_config={"Price_in_Lakhs": st.column_config.NumberColumn("Price (₹ L)", format="%.1f"),
                                   "Price_per_SqFt_INR": st.column_config.NumberColumn("₹/sq ft", format="localized"),
                                   "Future_Price_5Y": st.column_config.NumberColumn("5-yr value (₹ L)", format="%.1f"),
                                   "Gain_5Y_pct": st.column_config.NumberColumn("5-yr gain (%)", format="%.0f"),
                                   "Good_Investment": st.column_config.CheckboxColumn("Good?")})
        st.caption("Estimates come from a scenario model trained on rule-based targets (see *Model & Methodology*). "
                   "Not financial advice.")

# ════════════════════════════════════════════════════════════════════════════ TAB 1b · INSIGHT LAB
with tab_lab:
    render_lab(st.session_state.get("result"), data, dict(
        predict=predict, build_row=build_row, fmt=fmt_price, ref_year=REF_YEAR, horizon=HORIZON_YEARS,
        amenities=AMENITY_LIST, rmse=META["rmse"]))

# ════════════════════════════════════════════════════════════════════════════ TAB 2 · EXPLORER
with tab_mkt:
    d = apply_filters(data)
    st.caption(f"Showing **{len(d):,}** of {len(data):,} sampled listings — adjust the filters in the sidebar.")
    if d.empty:
        st.warning("No listings match these filters. Widen the price/area range or clear some selections.")
    else:
        m = st.columns(5)
        m[0].metric("Matching listings", f"{len(d):,}")
        m[1].metric("Avg asking price", fmt_price(d["Price_in_Lakhs"].mean()))
        m[2].metric("Median ₹ / sq ft", f"₹{d['Price_per_SqFt_INR'].median():,.0f}")
        m[3].metric("Avg projected 5-yr gain", f"{d['Gain_5Y_pct'].mean():.0f}%")
        m[4].metric("% Good Investment", f"{d['Good_Investment'].mean():.1%}")

        # ---- heatmap (interactive) + treemap
        METRICS = {"Avg price per sq ft (₹)": ("Price_per_SqFt_INR", ",.0f"), "Avg asking price (₹ L)": ("Price_in_Lakhs", ",.0f"),
                   "% Good Investment": ("Good_Investment_pct", ".0f"), "Avg projected 5-yr gain (%)": ("Gain_5Y_pct", ".0f"),
                   "Avg infrastructure score": ("Infrastructure_Score", ".0f")}
        h1, h2 = st.columns([1.35, 1])
        with h1:
            o1, o2, o3 = st.columns(3)
            metric = o1.selectbox("Heatmap metric", list(METRICS), key="hm_metric")
            rows_by = o2.selectbox("Rows", ["City", "State"], key="hm_rows")
            cols_by = o3.selectbox("Columns", ["BHK", "Property_Type", "Availability_Status", "Public_Transport_Accessibility"], key="hm_cols")
            col, fmt = METRICS[metric]
            pv = d.pivot_table(index=rows_by, columns=cols_by, values=col, aggfunc="mean")
            if cols_by == "Public_Transport_Accessibility":
                pv = pv.reindex(columns=[c for c in ["Low", "Medium", "High"] if c in pv.columns])
            pv.columns = [f"{c} BHK" if cols_by == "BHK" else str(c).replace("_", " ") for c in pv.columns]
            hm = px.imshow(pv, text_auto=fmt, aspect="auto", color_continuous_scale="Viridis" if col == "Good_Investment_pct" else "Blues",
                           title=f"{metric} — {rows_by.lower()} × {cols_by.replace('_', ' ').lower()}", labels=dict(color=""))
            hm.update_layout(height=520, margin=dict(t=60, b=20))
            st.plotly_chart(hm, width="stretch", key="heatmap")
        with h2:
            tm = d.groupby(["State", "City"]).agg(Listings=("ID", "size"), Median_PPSF=("Price_per_SqFt_INR", "median")).reset_index()
            fig = px.treemap(tm, path=[px.Constant("India"), "State", "City"], values="Listings", color="Median_PPSF",
                             color_continuous_scale="Blues", title="Where the listings are (size) and how pricey (colour)",
                             labels=dict(Median_PPSF="Median ₹/sq ft"))
            fig.update_layout(height=572, margin=dict(t=60, b=10, l=10, r=10))
            st.plotly_chart(fig, width="stretch", key="treemap")

        # ---- trend charts
        top_cities = d["City"].value_counts().head(6).index
        t1, t2 = st.columns(2)
        tr = d[d["City"].isin(top_cities)].groupby(["Build_Start", "City"])["Price_per_SqFt_INR"].median().reset_index()
        tr["Period"] = tr["Build_Start"].astype(str) + "–" + (tr["Build_Start"] + 4).astype(str).str[-2:]
        f1 = px.line(tr.sort_values("Build_Start"), x="Period", y="Price_per_SqFt_INR", color="City", markers=True,
                     title="Median ₹/sq ft by construction period (top cities)", labels=dict(Price_per_SqFt_INR="₹ per sq ft"))
        f1.update_layout(height=400, margin=dict(t=60, b=30), legend=dict(orientation="h", y=-0.25))
        t1.plotly_chart(f1, width="stretch", key="trend_vintage")

        proj_rows = []
        sub = d[d["City"].isin(top_cities)]
        for y in range(HORIZON_YEARS + 1):
            val = (sub["Price_in_Lakhs"] * (1 + sub["Annual_Growth_Rate"]) ** y).groupby(sub["City"]).mean()
            proj_rows += [dict(Year=REF_YEAR + y, City=c, Avg_Price=v) for c, v in val.items()]
        f2 = px.line(pd.DataFrame(proj_rows), x="Year", y="Avg_Price", color="City", markers=True,
                     title=f"Projected average price, {REF_YEAR}–{REF_YEAR + HORIZON_YEARS} (top cities)", labels=dict(Avg_Price="₹ Lakhs"))
        f2.update_layout(height=400, margin=dict(t=60, b=30), xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.25))
        t2.plotly_chart(f2, width="stretch", key="trend_proj")

        # ---- opportunity quadrant + infra
        q1, q2 = st.columns(2)
        cq = d.groupby("City").agg(Listings=("ID", "size"), Median_PPSF=("Price_per_SqFt_INR", "median"),
                                   Pct_Good=("Good_Investment_pct", "mean"), Avg_Gain=("Gain_5Y_pct", "mean")).reset_index()
        fq = px.scatter(cq, x="Median_PPSF", y="Pct_Good", size="Listings", color="Avg_Gain", text="City",
                        color_continuous_scale="Viridis", title="Opportunity map: price level vs. share of good investments",
                        labels=dict(Median_PPSF="Median ₹ per sq ft", Pct_Good="% Good Investment", Avg_Gain="Avg 5-yr gain %"))
        fq.add_vline(x=cq["Median_PPSF"].median(), line_dash="dot", line_color="#94A3B8")
        fq.add_hline(y=cq["Pct_Good"].median(), line_dash="dot", line_color="#94A3B8")
        fq.update_traces(textposition="top center")
        fq.update_layout(height=430, margin=dict(t=60, b=30))
        q1.plotly_chart(fq, width="stretch", key="quadrant")
        q1.caption("Top-left = cheaper than peers *and* a higher share of good investments.")

        d2 = d.assign(Infra_Band=pd.cut(d["Infrastructure_Score"], [0, 40, 55, 70, 85, 100], include_lowest=True,
                                        labels=["0–40", "40–55", "55–70", "70–85", "85–100"]))
        ib = d2.groupby("Infra_Band", observed=True).agg(Gain=("Gain_5Y_pct", "mean"), Good=("Good_Investment_pct", "mean")).reset_index()
        fi = go.Figure()
        fi.add_bar(x=ib["Infra_Band"].astype(str), y=ib["Gain"], name="Avg 5-yr gain %", marker_color=BLUE)
        fi.add_trace(go.Scatter(x=ib["Infra_Band"].astype(str), y=ib["Good"], name="% Good Investment", yaxis="y2",
                                mode="lines+markers", line=dict(color=AMBER, width=3)))
        fi.update_layout(title="Infrastructure score vs. appreciation & hit-rate", height=430, margin=dict(t=60, b=30),
                         xaxis_title="Infrastructure score band", yaxis=dict(title="Avg 5-yr gain %"),
                         yaxis2=dict(title="% Good", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=-0.25))
        q2.plotly_chart(fi, width="stretch", key="infra")

        # ---- leaderboard + listings
        st.markdown("##### 🏆 Locality leaderboard")
        lb = d.groupby("Locality").agg(Listings=("ID", "size"), Median_PPSF=("Price_per_SqFt_INR", "median"),
                                       Pct_Good=("Good_Investment_pct", "mean"), Avg_Gain=("Gain_5Y_pct", "mean"),
                                       Infra=("Infrastructure_Score", "mean")).reset_index()
        lb = lb[lb["Listings"] >= min(10, max(lb["Listings"].max(), 1))].sort_values(["Pct_Good", "Avg_Gain"], ascending=False).head(15)
        st.dataframe(lb, width="stretch", hide_index=True, column_config={
            "Median_PPSF": st.column_config.NumberColumn("Median ₹/sq ft", format="localized"),
            "Pct_Good": st.column_config.ProgressColumn("% Good", min_value=0, max_value=100, format="%.0f"),
            "Avg_Gain": st.column_config.NumberColumn("Avg 5-yr gain %", format="%.1f"),
            "Infra": st.column_config.ProgressColumn("Infrastructure", min_value=0, max_value=100, format="%.0f")})

        st.markdown("##### 📋 Listings")
        sort_by = st.radio("Sort by", ["Highest projected gain", "Lowest ₹/sq ft", "Lowest price", "Best infrastructure"],
                           horizontal=True, key="sort_by")
        col, asc = {"Highest projected gain": ("Gain_5Y_pct", False), "Lowest ₹/sq ft": ("Price_per_SqFt_INR", True),
                    "Lowest price": ("Price_in_Lakhs", True), "Best infrastructure": ("Infrastructure_Score", False)}[sort_by]
        show_cols = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR",
                     "Future_Price_5Y", "Gain_5Y_pct", "Infrastructure_Score", "Good_Investment"]
        out = d.sort_values(col, ascending=asc).assign(Good_Investment=lambda x: x["Good_Investment"].astype(bool))[show_cols]
        st.dataframe(out.head(200), width="stretch", hide_index=True, column_config={
            "Price_in_Lakhs": st.column_config.NumberColumn("Price (₹ L)", format="%.1f"),
            "Price_per_SqFt_INR": st.column_config.NumberColumn("₹/sq ft", format="localized"),
            "Future_Price_5Y": st.column_config.NumberColumn("5-yr value (₹ L)", format="%.1f"),
            "Gain_5Y_pct": st.column_config.NumberColumn("5-yr gain (%)", format="%.0f"),
            "Infrastructure_Score": st.column_config.ProgressColumn("Infra", min_value=0, max_value=100, format="%.0f"),
            "Good_Investment": st.column_config.CheckboxColumn("Good?")})
        st.download_button("⬇️ Download filtered listings (CSV)", out.head(5000).to_csv(index=False).encode(),
                           "filtered_listings.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════ TAB 3 · MODEL
with tab_mod:
    bc, br = META["classification"][clf_name], META["regression"][reg_name]
    st.markdown("##### The two models behind the verdict")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown(f"**Classification — *Is this a Good Investment?*** · {clf_name}")
        x = st.columns(3)
        x[0].metric("Accuracy", f"{bc['accuracy']:.2%}"); x[1].metric("F1", f"{bc['f1']:.3f}"); x[2].metric("ROC-AUC", f"{bc['roc_auc']:.3f}")
    with b.container(border=True):
        st.markdown(f"**Regression — *Estimated price after {HORIZON_YEARS} years*** · {reg_name}")
        x = st.columns(3)
        x[0].metric("RMSE", f"₹{br['rmse']:.2f} L"); x[1].metric("MAE", f"₹{br['mae']:.2f} L"); x[2].metric("R²", f"{br['r2']:.4f}")

    st.markdown("##### Model comparison (held-out test set)")
    a, b = st.columns(2)
    a.dataframe(pd.DataFrame(META["classification"]).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
                .style.format("{:.4f}").highlight_max(color="#BBF7D0"), width="stretch")
    b.dataframe(pd.DataFrame(META["regression"]).T[["rmse", "mae", "r2", "mape"]]
                .style.format("{:.3f}").highlight_min(subset=["rmse", "mae", "mape"], color="#BBF7D0")
                .highlight_max(subset=["r2"], color="#BBF7D0"), width="stretch")

    a, b = st.columns(2)
    for col, key, title, color in [(a, "clf_importance", "What drives *Good Investment*", BLUE), (b, "reg_importance", f"What drives the {HORIZON_YEARS}-year price", GREEN)]:
        imp = pd.Series(META[key]).sort_values()
        fig = px.bar(imp, orientation="h", title=title.replace("*", ""), labels=dict(value="Relative importance (%)", index=""))
        fig.update_traces(marker_color=color)
        fig.update_layout(showlegend=False, height=450, margin=dict(t=60))
        col.plotly_chart(fig, width="stretch", key=key)

    ab = META.get("ablation_no_price")
    if ab:
        st.info(f"**Sanity check (ablation):** with every price feature removed, the classifier still reaches "
                f"**{ab['accuracy']:.1%} accuracy** (ROC-AUC {ab['roc_auc']:.3f}) — it relies on property attributes, not just a price comparison.")

    st.markdown("##### How the app works")
    steps = ["**1 · Form**\n\nYou describe the property.", "**2 · Feature engineering**\n\nSame `real_estate_features.py` as in training — no train/serve skew.",
             "**3 · Classifier**\n\nProbability of *Good Investment*.", "**4 · Regressor**\n\nEstimated price after 5 years (± RMSE).",
             "**5 · Insights**\n\nWhat-if curve, city comparison, comparable listings."]
    for col, txt in zip(st.columns(5), steps):
        col.container(border=True).markdown(txt)

    with st.expander("⚠️ Read this before trusting the numbers (limitations)", expanded=True):
        st.markdown(
            "- **Targets are rule-based scenarios.** The dataset has no price history or realised returns, so *Good Investment* "
            "(5-point rule score ≥ 3) and *5-year price* (growth driven by city tier, property type, infrastructure, "
            "construction status and age) are engineered. High scores show the model **learned the rules**, not that it forecasts the market.\n"
            "- **The data looks synthetic** (e.g. price is uncorrelated with size), so market charts show patterns in this dataset only.\n"
            "- **Crime rate / RERA are not in the data**; `Security` and an `Infrastructure_Score` are used as proxies.\n"
            "- The notebook's robustness check adds hidden market noise to the growth rate; accuracy then settles lower — "
            "that is the more realistic expectation for real-world use.\n- **Not financial advice.**")

st.caption(f"© {REF_YEAR} · {AUTHOR} · Real Estate Investment Advisor")