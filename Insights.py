"""Insight Lab — extra live models layered on top of the advisor's classifier + regressor.

Models added (all trained on app_sample.csv at start-up, cached):
  1. KMeans + PCA          → "Property DNA": market segment + 2-D map of where the property sits
  2. Isolation Forest      → how typical / unusual the listing is
  3. KNN (non-price feats) → "Twin finder": fair-price band from structurally similar listings
  4. Monte Carlo (GBM)     → live risk simulator calibrated on the regressor's RMSE
  5. Counterfactual engine → re-scores the property under what-if levers (uses the real clf/reg)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BLUE, GREEN, RED, AMBER = "#2563EB", "#10B981", "#EF4444", "#F59E0B"
NON_PRICE = [1, 2, 4, 5, 6]          # columns of _feat() used by the twin finder (no price → no circularity)


def _feat(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "lp": np.log1p(df["Price_in_Lakhs"]), "ls": np.log1p(df["Size_in_SqFt"]), "bhk": df["BHK"],
        "lppsf": np.log1p(df["Price_per_SqFt_INR"]), "infra": df["Infrastructure_Score"],
        "growth": df["Annual_Growth_Rate"], "age": df["Age"]}).astype(float)


def _name_segments(prof: pd.DataFrame) -> dict:
    names, free = {}, list(prof.index)
    for label, col, asc in [("💎 Premium core", "Price_per_SqFt_INR", False), ("🚀 High-growth frontier", "Annual_Growth_Rate", False),
                            ("🪙 Value entry", "Price_per_SqFt_INR", True), ("🏛️ Mature resale", "Age", False)]:
        if free:
            c = prof.loc[free, col].sort_values(ascending=asc).index[0]
            names[c] = label
            free.remove(c)
    for c in free:
        names[c] = f"⚖️ Balanced mainstream {c}"
    return names


@st.cache_resource(show_spinner="Training segmentation, anomaly & similarity models…")
def fit_models(_data: pd.DataFrame, ref_year: int, k: int = 5) -> dict:
    d = _data.assign(Age=ref_year - _data["Year_Built"]).reset_index(drop=True)
    X = _feat(d)
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    km = KMeans(k, n_init=10, random_state=42).fit(Z)
    pca = PCA(2, random_state=42).fit(Z)
    iso = IsolationForest(n_estimators=200, random_state=42).fit(Z)
    d["Seg"] = km.labels_
    names = _name_segments(d.groupby("Seg")[["Price_per_SqFt_INR", "Annual_Growth_Rate", "Age"]].mean())
    d["Segment"] = d["Seg"].map(names)
    xy = pca.transform(Z)
    d["PC1"], d["PC2"] = xy[:, 0], xy[:, 1]
    d["iso"] = iso.score_samples(Z)
    nn = NearestNeighbors(n_neighbors=min(30, len(d))).fit(Z[:, NON_PRICE])
    return dict(d=d, sc=sc, km=km, pca=pca, iso=iso, nn=nn, names=names, Z=Z)


# ───────────────────────────────────────────────────────────── counterfactual engine
def _levers(s: dict, amenities: list) -> dict:
    L = {f"Negotiate −{p}% off price": (lambda c, p=p: {"in_price": c["in_price"] * (1 - p / 100)}) for p in (5, 10, 20)}
    if s["in_avail"] != "Ready_to_Move":
        L["Pick a ready-to-move unit"] = lambda c: {"in_avail": "Ready_to_Move"}
    if s["in_transport"] != "High":
        L["Choose high public-transport access"] = lambda c: {"in_transport": "High"}
    if s["in_bhk"] < 3:
        L["Step up to 3 BHK"] = lambda c: {"in_bhk": 3}
    if s["in_schools"] < 10:
        L["+2 schools nearby"] = lambda c: {"in_schools": min(10, c["in_schools"] + 2)}
    if s["in_hosp"] < 10:
        L["+2 hospitals nearby"] = lambda c: {"in_hosp": min(10, c["in_hosp"] + 2)}
    if s["in_year"] < 2022:
        L["Newer build (2022)"] = lambda c: {"in_year": max(c["in_year"], 2022)}
    for a in amenities:
        if a not in s["in_amen"]:
            L[f"Add amenity: {a}"] = lambda c, a=a: {"in_amen": list(c["in_amen"]) + [a]}
    return L


def _score(variants: list, ctx: dict):
    rows = pd.concat([ctx["build_row"](v) for v in variants], ignore_index=True)
    _, p, f = ctx["predict"](rows)
    return p, f


@st.cache_data(show_spinner="Re-scoring what-if scenarios…")
def counterfactuals(inputs_json: str, _ctx: dict):
    s = json.loads(inputs_json)
    L = _levers(s, _ctx["amenities"])
    (p0,), (f0,) = _score([s], _ctx)
    p, f = _score([{**s, **fn(s)} for fn in L.values()], _ctx)
    df = pd.DataFrame(dict(Lever=list(L), dP=(p - p0) * 100, dFuture=f - f0, P_new=p * 100))
    df["Flips verdict"] = (df["P_new"] >= 50) != (p0 >= 0.5)
    # greedy "best path": keep applying the lever that lifts P(Good) the most
    cur, pool, path = dict(s), dict(L), [("Current", p0 * 100)]
    for _ in range(4):
        if not pool:
            break
        cands = {k: {**cur, **fn(cur)} for k, fn in pool.items()}
        pp, _ = _score(list(cands.values()), _ctx)
        i = int(np.argmax(pp))
        if pp[i] * 100 <= path[-1][1] + 0.05:
            break
        k = list(cands)[i]
        cur = cands[k]
        path.append((k, pp[i] * 100))
        pool.pop(k)
        if k.startswith("Negotiate"):
            pool = {a: b for a, b in pool.items() if not a.startswith("Negotiate")}
    return df.sort_values("dP", ascending=False), path


# ───────────────────────────────────────────────────────────── Monte Carlo
def monte_carlo(price, future, rmse, horizon, years, vol_mult, shift_pp, n=4000):
    mu = np.log(future / price) / horizon + shift_pp / 100
    sigma = max(rmse / future, 0.03) / np.sqrt(horizon) * vol_mult
    z = np.random.default_rng(7).standard_normal((n, years))
    paths = price * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) + sigma * z, axis=1))
    return np.hstack([np.full((n, 1), price), paths])


# ───────────────────────────────────────────────────────────── render
def render_brand_header():
    """Render a compact, text-first brand mark for the Property Insight Lab."""
    st.markdown(
        """
        <style>
        .insight-brand {display:flex; align-items:center; gap:14px; padding:16px 18px;
          margin:2px 0 20px; border:1px solid rgba(100,116,139,.28); border-radius:16px;
          background:linear-gradient(115deg, rgba(37,99,235,.10), rgba(16,185,129,.06));}
        .insight-brand-mark {width:54px; height:54px; flex:0 0 54px; display:flex;
          align-items:center; justify-content:center; border-radius:14px; background:#0F172A;
          color:#F8FAFC; font-size:25px; font-weight:800; letter-spacing:-2px;
          box-shadow:0 5px 16px rgba(15,23,42,.18);}
        .insight-brand-name {color:var(--text-color); font-size:19px; font-weight:800;
          letter-spacing:.08em; line-height:1.2;}
        .insight-brand-sub {color:var(--text-color); opacity:.72; font-size:12px;
          margin-top:5px; letter-spacing:.04em;}
        .insight-brand-tag {margin-left:auto; padding:6px 10px; border-radius:999px;
          border:1px solid rgba(37,99,235,.35); color:#2563EB; font-size:11px;
          font-weight:700; letter-spacing:.06em; white-space:nowrap;}
        @media(max-width:560px) {.insight-brand-tag {display:none;} .insight-brand-name {font-size:15px;}}
        </style>
        <div class="insight-brand">
          <div class="insight-brand-mark" aria-label="Property Insight Lab logo">P<span style="color:#34D399">I</span></div>
          <div><div class="insight-brand-name">PROPERTY INSIGHT LAB</div>
            <div class="insight-brand-sub">REAL ESTATE ANALYTICS · INVESTMENT INTELLIGENCE</div></div>
          <div class="insight-brand-tag">DATA-DRIVEN DECISIONS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lab(res, data: pd.DataFrame, ctx: dict):
    render_brand_header()
    st.markdown("## Investment Insights & Interactive Analytics")
    st.caption("Explore market positioning, comparable properties, simulated outcomes and model-scored what-if scenarios.")
    if res is None:
        st.info("🧪 Run **Analyse this property** in the *Investment Advisor* tab first — the Lab then builds "
                "segment, risk and what-if models around that exact property.")
        return
    s, eng, future = res["inputs"], res["eng"], res["future"]
    price, ref_year, horizon, fmt = float(s["in_price"]), ctx["ref_year"], ctx["horizon"], ctx["fmt"]
    M = fit_models(data, ref_year)
    D = M["d"]

    u = pd.DataFrame([dict(Price_in_Lakhs=price, Size_in_SqFt=float(s["in_size"]), BHK=int(s["in_bhk"]),
                           Price_per_SqFt_INR=float(eng["Price_per_SqFt_INR"]), Infrastructure_Score=float(eng["Infrastructure_Score"]),
                           Annual_Growth_Rate=float(eng["Annual_Growth_Rate"]), Age=ref_year - int(s["in_year"]))])
    Zu = M["sc"].transform(_feat(u))
    seg = M["names"][int(M["km"].predict(Zu)[0])]
    px_, py_ = M["pca"].transform(Zu)[0]
    typical = float((D["iso"] < M["iso"].score_samples(Zu)[0]).mean())
    ppsf = float(u["Price_per_SqFt_INR"][0])

    st.markdown("### 🧬 1 · Property DNA — segmentation & anomaly models")
    st.caption("KMeans clusters the whole market into segments; PCA flattens it into a map; Isolation Forest measures how unusual the listing is.")
    a, b = st.columns([1.6, 1])
    with a:
        samp = D.sample(min(2500, len(D)), random_state=1)
        fm = px.scatter(samp, x="PC1", y="PC2", color="Segment", opacity=.45, title="Market map — every dot is a listing",
                        hover_data=["City", "Locality", "Price_in_Lakhs"])
        fm.add_trace(go.Scatter(x=[px_], y=[py_], mode="markers+text", text=["YOU"], textposition="top center",
                                marker=dict(symbol="star", size=24, color="#111827", line=dict(color="white", width=2)), name="This property"))
        fm.update_layout(height=430, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.15, title=""))
        st.plotly_chart(fm, width="stretch", key="lab_map")
    with b:
        pct = lambda col, v: float((D[col] <= v).mean())
        dims = {"Affordability": 1 - pct("Price_per_SqFt_INR", ppsf), "Infrastructure": pct("Infrastructure_Score", u["Infrastructure_Score"][0]),
                "Growth outlook": pct("Annual_Growth_Rate", u["Annual_Growth_Rate"][0]), "Space": pct("Size_in_SqFt", u["Size_in_SqFt"][0]),
                "Newness": 1 - pct("Age", u["Age"][0]), "Bedrooms": pct("BHK", u["BHK"][0])}
        fr = go.Figure(go.Scatterpolar(r=[v * 100 for v in dims.values()] + [list(dims.values())[0] * 100], theta=list(dims) + [list(dims)[0]],
                                       fill="toself", line=dict(color=BLUE), name="Percentile vs market"))
        fr.update_layout(title="DNA radar (percentile vs market)", polar=dict(radialaxis=dict(range=[0, 100])), height=300,
                         margin=dict(t=60, b=10), showlegend=False)
        st.plotly_chart(fr, width="stretch", key="lab_radar")
        sg = D[D["Segment"] == seg]
        st.markdown(f"**Segment:** {seg}")
        c1, c2 = st.columns(2)
        c1.metric("Segment % good", f"{sg['Good_Investment_pct'].mean():.0f}%", f"{sg['Good_Investment_pct'].mean() - D['Good_Investment_pct'].mean():+.0f} pp vs market")
        c2.metric("Typicality", f"{typical:.0%}", "more typical than this share of listings", delta_color="off")

    st.markdown("### 👯 2 · Twin finder — is the asking price fair? (KNN)")
    st.caption("Finds listings that match on size, BHK, infrastructure, growth outlook and age — **ignoring price** — then compares ₹/sq ft.")
    dist, idx = M["nn"].kneighbors(Zu[:, NON_PRICE])
    tw = D.iloc[idx[0]]
    lo, med, hi = tw["Price_per_SqFt_INR"].quantile([.25, .5, .75])
    verdict = "🟢 below" if ppsf < lo else ("🔴 above" if ppsf > hi else "🟡 within")
    t1, t2 = st.columns([1, 1.3])
    with t1:
        fh = px.histogram(tw, x="Price_per_SqFt_INR", nbins=14, title=f"₹/sq ft of {len(tw)} structural twins", labels=dict(Price_per_SqFt_INR="₹ per sq ft"))
        fh.add_vline(x=ppsf, line_color=RED, line_width=3, annotation_text="This property")
        fh.add_vrect(x0=lo, x1=hi, fillcolor=GREEN, opacity=.12, line_width=0)
        fh.update_layout(height=330, margin=dict(t=50, b=20), yaxis_title="Twins", showlegend=False)
        st.plotly_chart(fh, width="stretch", key="lab_twins")
        st.markdown(f"Asking **₹{ppsf:,.0f}/sq ft** is {verdict} the twins' fair band (₹{lo:,.0f}–{hi:,.0f}, median ₹{med:,.0f}) → "
                    f"implied fair price ≈ **{fmt(med * float(s['in_size']) / 1e5)}**.")
    with t2:
        st.dataframe(tw.head(8)[["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Gain_5Y_pct", "Good_Investment"]]
                     .assign(Good_Investment=lambda x: x["Good_Investment"].astype(bool)), hide_index=True, width="stretch",
                     column_config={"Gain_5Y_pct": st.column_config.NumberColumn("5-yr gain %", format="%.0f"),
                                    "Price_per_SqFt_INR": st.column_config.NumberColumn("₹/sq ft", format="localized"),
                                    "Good_Investment": st.column_config.CheckboxColumn("Good?")})

    st.markdown("### 🎲 3 · Risk simulator — Monte Carlo (live)")
    st.caption(f"4,000 simulated price paths. Drift = the regressor's {horizon}-year forecast; volatility is calibrated on the model's RMSE. Drag the sliders — it recomputes instantly.")
    k1, k2, k3, k4 = st.columns(4)
    yrs = k1.slider("Holding period (years)", 1, 10, horizon, key="lab_years")
    vm = k2.slider("Market volatility ×", 0.5, 3.0, 1.0, 0.25, key="lab_vol")
    sh = k3.slider("Growth shock (pp / yr)", -5.0, 5.0, 0.0, 0.5, key="lab_shift")
    fd = k4.slider("Benchmark FD rate (%)", 3.0, 10.0, 7.0, 0.5, key="lab_fd")
    P = monte_carlo(price, future, ctx["rmse"], horizon, yrs, vm, sh)
    final, x = P[:, -1], np.arange(yrs + 1) + ref_year
    q = np.percentile(P, [5, 25, 50, 75, 95], axis=0)
    bench = price * (1 + fd / 100) ** np.arange(yrs + 1)
    m = st.columns(4)
    m[0].metric("Median value", fmt(float(np.median(final))), f"{np.median(final) / price - 1:+.0%}")
    m[1].metric("Chance of loss", f"{(final < price).mean():.1%}")
    m[2].metric(f"Beats {fd:.1f}% FD", f"{(final > bench[-1]).mean():.1%}")
    m[3].metric("Worst 5% case", fmt(float(q[0, -1])), f"{q[0, -1] / price - 1:+.0%}", delta_color="off")
    r1, r2 = st.columns([1.5, 1])
    ff = go.Figure()
    ff.add_trace(go.Scatter(x=x, y=q[4], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    ff.add_trace(go.Scatter(x=x, y=q[0], fill="tonexty", fillcolor="rgba(37,99,235,.12)", line=dict(width=0), name="5–95 %"))
    ff.add_trace(go.Scatter(x=x, y=q[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    ff.add_trace(go.Scatter(x=x, y=q[1], fill="tonexty", fillcolor="rgba(37,99,235,.28)", line=dict(width=0), name="25–75 %"))
    ff.add_trace(go.Scatter(x=x, y=q[2], line=dict(color=BLUE, width=3), name="Median"))
    ff.add_trace(go.Scatter(x=x, y=bench, line=dict(color=AMBER, dash="dash"), name=f"FD @ {fd:.1f}%"))
    ff.update_layout(title="Fan chart of simulated value", yaxis_title="₹ Lakhs", xaxis=dict(dtick=1), height=360,
                     margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.2))
    r1.plotly_chart(ff, width="stretch", key="lab_fan")
    fo = px.histogram(x=final, nbins=45, title=f"Distribution of value in {ref_year + yrs}", labels=dict(x="₹ Lakhs"))
    fo.add_vline(x=price, line_color=RED, annotation_text="Buy price")
    fo.add_vline(x=float(np.median(final)), line_dash="dash", annotation_text="Median", annotation_position="top right")
    fo.update_layout(height=360, margin=dict(t=50, b=20), showlegend=False, yaxis_title="Paths")
    r2.plotly_chart(fo, width="stretch", key="lab_hist")

    st.markdown("### 🧭 4 · What would change the verdict? — counterfactual engine")
    st.caption("Every lever below is re-scored through the **real classifier and regressor** — a model-agnostic sensitivity analysis.")
    df, path = counterfactuals(json.dumps(s, sort_keys=True, default=str), ctx)
    c1, c2 = st.columns([1.3, 1])
    fb = px.bar(df.sort_values("dP"), x="dP", y="Lever", orientation="h", color="Flips verdict",
                color_discrete_map={True: GREEN, False: "#94A3B8"}, title="Δ P(Good Investment), percentage points",
                labels=dict(dP="Δ probability (pp)", Lever=""), hover_data={"dFuture": ":.1f", "P_new": ":.0f"})
    fb.update_layout(height=max(340, 26 * len(df) + 90), margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.12))
    c1.plotly_chart(fb, width="stretch", key="lab_cf")
    with c2:
        fw = go.Figure(go.Waterfall(x=[k for k, _ in path], y=[path[0][1]] + [path[i][1] - path[i - 1][1] for i in range(1, len(path))],
                                    measure=["absolute"] + ["relative"] * (len(path) - 1), connector=dict(line=dict(color="#CBD5E1")),
                                    increasing=dict(marker=dict(color=GREEN)), totals=dict(marker=dict(color=BLUE))))
        fw.add_hline(y=50, line_dash="dot", line_color="#111827")
        fw.update_layout(title="Greedy best path to a better verdict", yaxis=dict(title="P(Good) %", range=[0, 100]), height=340, margin=dict(t=50, b=20))
        st.plotly_chart(fw, width="stretch", key="lab_wf")
        st.markdown(f"Best combination lifts confidence from **{path[0][1]:.0f}%** to **{path[-1][1]:.0f}%**." if len(path) > 1
                    else "No single lever improves this property's probability further.")
    st.caption("Scenario analysis on a rule-derived dataset — illustrative, not financial advice.")