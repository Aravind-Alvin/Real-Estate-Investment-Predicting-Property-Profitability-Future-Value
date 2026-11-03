"""Step 2 - Exploratory Data Analysis (20 questions + 4 project-brief analyses).

Outputs: reports/figures/*.png and reports/eda_findings.md (numbers computed here).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from src import config as C

sns.set_theme(style="whitegrid", context="notebook")
C.FIG_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(C.CLEAN_CSV)
PPSF = "Price_per_SqFt_INR"
out = []          # markdown lines


def save(name):
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"{name}.png", dpi=110)
    plt.close()


def note(q, text):
    out.append(f"**{q}** {text}\n")


def kw_p(col, val):
    groups = [g[val].values for _, g in df.groupby(col)]
    return stats.kruskal(*groups).pvalue


def sig(p):
    return "no statistically significant difference" if p > 0.05 else "statistically significant difference"


# ===================================================== PRICE & SIZE ANALYSIS
out.append("## Price & size analysis\n")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Price_in_Lakhs"], bins=50, ax=ax[0], color="#1f77b4")
ax[0].set(title="Q1  Distribution of price (INR lakhs)", xlabel="Price (lakhs)")
sns.histplot(df["Size_in_SqFt"], bins=50, ax=ax[1], color="#2ca02c")
ax[1].set(title="Q2  Distribution of size (sq ft)", xlabel="Size (sq ft)")
save("q01_q02_price_size_distribution")
p, s = df["Price_in_Lakhs"], df["Size_in_SqFt"]
note("Q1 Price distribution:", f"Price ranges {p.min():.0f}-{p.max():.0f} lakhs, mean {p.mean():.1f}, median {p.median():.1f}, "
     f"skew {p.skew():.2f}. The histogram is essentially flat (uniform), unlike real markets which are right-skewed.")
note("Q2 Size distribution:", f"Size ranges {s.min()}-{s.max()} sq ft, mean {s.mean():.0f}, median {s.median():.0f}, skew {s.skew():.2f}; also near-uniform.")

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=df, x="Property_Type", y=PPSF, ax=ax, showfliers=False)
ax.set(title="Q3  Price per sq ft (INR) by property type", ylabel="INR / sq ft")
save("q03_ppsf_by_property_type")
g = df.groupby("Property_Type")[PPSF].agg(["mean", "median"]).round(0)
note("Q3 Price/sqft by property type:", "median INR/sqft - " + ", ".join(f"{k}: {v:,.0f}" for k, v in g["median"].items())
     + f". Kruskal-Wallis p={kw_p('Property_Type', PPSF):.3f}, but the gap between types is only {(g['median'].max()/g['median'].min()-1):.1%} - "
     "statistically detectable with 250k rows yet practically negligible.")

smp = df.sample(6000, random_state=1)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(smp["Size_in_SqFt"], smp["Price_in_Lakhs"], s=5, alpha=.35)
ax.set(title="Q4  Size vs price (6k-row sample)", xlabel="Size (sq ft)", ylabel="Price (lakhs)")
save("q04_size_vs_price")
r_p = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]); r_s = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"], method="spearman")
note("Q4 Size vs price:", f"Pearson r={r_p:.3f}, Spearman rho={r_s:.3f} - effectively zero. In a real market larger homes cost more; here size does not explain price.")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, c, t in zip(ax, [PPSF, "Size_in_SqFt", "Price_in_Lakhs"], ["Price/sqft (INR)", "Size (sqft)", "Price (lakhs)"]):
    sns.boxplot(y=df[c], ax=a, color="#9ecae1"); a.set_title(f"Q5  {t}")
save("q05_outliers")
note("Q5 Outliers:", f"IQR rule flags {int(df.Outlier_PPSF.sum()):,} rows ({df.Outlier_PPSF.mean():.1%}) as price/sqft outliers "
     f"(max {df[PPSF].max():,.0f} INR/sqft vs median {df[PPSF].median():,.0f}); 0 outliers for size and price (both are bounded/uniform). "
     "PPSF outliers arise from small flats with high prices - they are flagged, not removed.")

# ============================================================ LOCATION
out.append("## Location-based analysis\n")
st = df.groupby("State")[PPSF].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 6)); st.plot.barh(ax=ax, color="#ff7f0e")
ax.set(title="Q6  Average price per sq ft by state (INR)", xlabel="INR / sq ft")
save("q06_ppsf_by_state")
note("Q6 Avg price/sqft by state:", f"highest {st.index[-1]} ({st.iloc[-1]:,.0f}), lowest {st.index[0]} ({st.iloc[0]:,.0f}); the spread between states is only "
     f"{(st.iloc[-1]/st.iloc[0]-1):.1%}. Kruskal p={kw_p('State', PPSF):.3f} -> {sig(kw_p('State', PPSF))}.")

ct = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 9)); ct.plot.barh(ax=ax, color="#1f77b4")
ax.set(title="Q7  Average property price by city (lakhs)", xlabel="Lakhs"); ax.set_xlim(240, ct.max() + 2)
save("q07_price_by_city")
note("Q7 Avg price by city:", f"range {ct.iloc[0]:.1f}-{ct.iloc[-1]:.1f} lakhs ({ct.index[0]} lowest, {ct.index[-1]} highest) - a spread of only "
     f"{(ct.iloc[-1]/ct.iloc[0]-1):.1%} across 42 cities. Note the x-axis is truncated at 240 to make differences visible.")

la = df.groupby("Locality")["Age_of_Property"].median().sort_values()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(la, bins=15, ax=ax[0]); ax[0].set(title="Q8  Distribution of median age across 500 localities", xlabel="Median age (yrs)")
pd.concat([la.head(8), la.tail(8)]).plot.barh(ax=ax[1], color="#9467bd"); ax[1].set_title("Youngest & oldest 8 localities")
save("q08_median_age_by_locality")
note("Q8 Median age by locality:", f"median age per locality ranges {la.min():.0f}-{la.max():.0f} yrs (overall median {df.Age_of_Property.median():.0f}); "
     "differences are small and consistent with sampling noise.")

bh = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(7, 11)); sns.heatmap(bh, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={"label": "% of city listings"})
ax.set_title("Q9  BHK mix by city (% of listings)")
save("q09_bhk_by_city")
note("Q9 BHK across cities:", f"every city has a near-uniform mix - each BHK category is {bh.min().min():.0f}-{bh.max().max():.0f}% of a city's listings (ideal uniform = 20%).")

top_loc = df.groupby("Locality")[PPSF].mean().nlargest(5).index.tolist()
sub = df[df["Locality"].isin(top_loc)].copy()
sub["Build_Period"] = pd.cut(sub["Year_Built"], [1989, 1995, 2000, 2005, 2010, 2015, 2020, 2023],
                             labels=["1990-95", "1996-00", "2001-05", "2006-10", "2011-15", "2016-20", "2021-23"])
tr = sub.groupby(["Build_Period", "Locality"], observed=True)[PPSF].median().unstack()
fig, ax = plt.subplots(figsize=(9, 4.8)); tr.plot(marker="o", ax=ax)
ax.set(title="Q10  Price/sqft by build period - top-5 priciest localities", ylabel="Median INR / sqft", xlabel="Year-built cohort (proxy for time)")
save("q10_top5_locality_trend")
note("Q10 Price trends, top-5 localities:", f"top 5 by average price/sqft: {', '.join(top_loc)}. The dataset has NO transaction dates, so true price trends cannot be computed; "
     "the chart uses year-built cohorts as a proxy and shows no consistent upward trend (lines cross and are noisy).")

# ============================================ FEATURE RELATIONSHIPS
out.append("## Feature relationships & correlation\n")
nc = ["Size_in_SqFt", "Price_in_Lakhs", PPSF, "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
      "Floor_No", "Total_Floors", "Parking_Flag", "Security_Flag", "Amenity_Count", "Infrastructure_Score"]
cm = df[nc].corr()
fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Q11  Correlation matrix (numeric features)")
save("q11_correlation_matrix")
pairs = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack().abs().sort_values(ascending=False)
note("Q11 Correlations:", "strongest pairs: " + "; ".join(f"{a}~{b} ({cm.loc[a, b]:+.2f})" for (a, b) in pairs.index[:4])
     + ". Apart from mechanical relationships (price/sqft is derived from price and size; infrastructure score from schools/hospitals; total vs floor number), "
     "all other pairs have |r| < 0.01.")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, c, t in zip(ax, ["Nearby_Schools", "Nearby_Hospitals"], ["Q12  Schools", "Q13  Hospitals"]):
    gm = df.groupby(c)[PPSF].mean(); a.plot(gm.index, gm.values, marker="o"); a.set(title=f"{t} vs avg price/sqft", xlabel=c, ylabel="INR / sqft")
save("q12_q13_schools_hospitals_vs_ppsf")
rs = df["Nearby_Schools"].corr(df[PPSF], method="spearman"); rh = df["Nearby_Hospitals"].corr(df[PPSF], method="spearman")
note("Q12 Schools vs price/sqft:", f"Spearman rho={rs:+.3f} (p={stats.spearmanr(df.Nearby_Schools, df[PPSF])[1]:.2f}) - no relationship.")
note("Q13 Hospitals vs price/sqft:", f"Spearman rho={rh:+.3f} (p={stats.spearmanr(df.Nearby_Hospitals, df[PPSF])[1]:.2f}) - no relationship.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax[0], showfliers=False); ax[0].set_title("Q14  Price by furnished status")
sns.boxplot(data=df, x="Facing", y=PPSF, ax=ax[1], showfliers=False); ax[1].set_title("Q15  Price/sqft by facing")
save("q14_q15_furnishing_facing")
f1 = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().round(1); f2 = df.groupby("Facing")[PPSF].mean().round(0)
note("Q14 Price by furnished status:", "mean lakhs - " + ", ".join(f"{k}: {v}" for k, v in f1.items()) + f". Kruskal p={kw_p('Furnished_Status', 'Price_in_Lakhs'):.3f} -> {sig(kw_p('Furnished_Status', 'Price_in_Lakhs'))}.")
note("Q15 Price/sqft by facing:", "mean INR/sqft - " + ", ".join(f"{k}: {v:,.0f}" for k, v in f2.items()) + f". Kruskal p={kw_p('Facing', PPSF):.3f} -> {sig(kw_p('Facing', PPSF))}.")

# ============================================= INVESTMENT / OWNERSHIP
out.append("## Investment, amenities & ownership\n")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
oc = df["Owner_Type"].value_counts(); ac = df["Availability_Status"].value_counts()
oc.plot.bar(ax=ax[0], color="#17becf", rot=0); ax[0].set_title("Q16  Properties by owner type")
ac.plot.bar(ax=ax[1], color="#bcbd22", rot=0); ax[1].set_title("Q17  Properties by availability status")
for a in ax:
    for i, v in enumerate(a.patches): a.annotate(f"{int(v.get_height()):,}", (v.get_x() + v.get_width() / 2, v.get_height()), ha="center", va="bottom", fontsize=8)
save("q16_q17_owner_availability")
note("Q16 Owner type:", ", ".join(f"{k}: {v:,}" for k, v in oc.items()) + " (near-equal thirds).")
note("Q17 Availability:", ", ".join(f"{k}: {v:,}" for k, v in ac.items()) + ". The 'Sold' status listed in the data dictionary does not occur in the file.")

fig, ax = plt.subplots(figsize=(6, 4.5)); sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=ax, errorbar="ci")
ax.set(title="Q18  Avg price by parking availability", ylabel="Lakhs"); ax.set_ylim(240, 270)
save("q18_parking_vs_price")
pg = df.groupby("Parking_Space")["Price_in_Lakhs"].mean()
note("Q18 Parking vs price:", f"with parking {pg['Yes']:.1f} vs without {pg['No']:.1f} lakhs (diff {pg['Yes'] - pg['No']:+.2f}); "
     f"Mann-Whitney p={stats.mannwhitneyu(df[df.Parking_Space == 'Yes'].Price_in_Lakhs, df[df.Parking_Space == 'No'].Price_in_Lakhs).pvalue:.3f}.")

am = df["Amenities"].str.split(", ").explode().str.strip()
rows = []
for a_ in C.AMENITY_LIST:
    has = df["Amenities"].str.contains(a_)
    rows.append((a_, df.loc[has, PPSF].mean(), df.loc[~has, PPSF].mean(), stats.mannwhitneyu(df.loc[has, PPSF], df.loc[~has, PPSF]).pvalue))
am_df = pd.DataFrame(rows, columns=["Amenity", "With", "Without", "p"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
am_df.set_index("Amenity")[["With", "Without"]].plot.bar(ax=ax[0], rot=0); ax[0].set(title="Q19  Avg price/sqft with vs without amenity", ylabel="INR / sqft")
sns.barplot(data=df, x="Amenity_Count", y=PPSF, ax=ax[1]); ax[1].set_title("Price/sqft by number of amenities")
save("q19_amenities_vs_ppsf")
note("Q19 Amenities vs price/sqft:", "; ".join(f"{r.Amenity}: {r.With:,.0f} vs {r.Without:,.0f} (p={r.p:.2f})" for r in am_df.itertuples()) + ". Differences are under 1% (the Gym p-value is borderline and, with 5 tests, not convincing) - no amenity carries a meaningful premium.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(data=df, x="Public_Transport_Accessibility", y=PPSF, order=C.TRANSPORT, ax=ax[0]); ax[0].set_title("Q20  Price/sqft by transport access")
gi = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(C.TRANSPORT) * 100
gi.plot.bar(ax=ax[1], rot=0, color="#2ca02c"); ax[1].set(title="% rated Good Investment by transport access", ylabel="%")
save("q20_transport_vs_ppsf_investment")
tp = df.groupby("Public_Transport_Accessibility")[PPSF].mean()
note("Q20 Transport access:", f"avg INR/sqft - Low {tp['Low']:,.0f}, Medium {tp['Medium']:,.0f}, High {tp['High']:,.0f} (Kruskal p={kw_p('Public_Transport_Accessibility', PPSF):.2f}). "
     "Good-Investment share (model label, which rewards infrastructure by construction): " + ", ".join(f"{k} {v:.1f}%" for k, v in gi.items()) + ".")

# ===================================== PROJECT-BRIEF EXTRA ANALYSES
out.append("## Additional analyses requested in the project brief\n")
fig, ax = plt.subplots(figsize=(11, 8)); cp = df.groupby("City")[PPSF].mean().sort_values()
# heatmap: state x property type
hm = df.pivot_table(index="State", columns="Property_Type", values=PPSF, aggfunc="mean")
plt.close(); fig, ax = plt.subplots(figsize=(7, 8)); sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
ax.set_title("E0  Location heatmap: avg INR/sqft, state x property type")
save("e0_state_property_heatmap")

df["Size_Band"] = pd.qcut(df["Size_in_SqFt"], 5, labels=["XS", "S", "M", "L", "XL"])
sb = df.groupby("Size_Band", observed=True).agg(cagr=("Annual_Appreciation_Rate", "mean"), good=("Good_Investment", "mean"))
fig, ax = plt.subplots(figsize=(7, 4)); (sb["good"] * 100).plot.bar(ax=ax, rot=0, color="#d62728")
ax.set(title="E1  Area (size band) vs % Good Investment", ylabel="%")
save("e1_area_vs_investment")
note("E1 Area vs investment return:", f"corr(size, projected CAGR)={df.Size_in_SqFt.corr(df.Annual_Appreciation_Rate):+.3f}; share of Good Investments by size quintile: "
     + ", ".join(f"{k} {v * 100:.1f}%" for k, v in sb["good"].items()) + ". Caution: price is independent of size in this data, so a bigger home at the same price has a lower price/sqft "
     "and looks 'undervalued' versus the city median. The growth model rewards that, so this rising pattern is a label-construction effect, not market evidence.")

out.append("**E2 Crime rate vs investment:** the dataset has **no crime-rate column**, so this cannot be measured directly. "
           "The closest available safety proxy is the `Security` flag; " + (lambda g: f"Good-Investment rate with security {g['Yes']:.1%} vs without {g['No']:.1%}")(df.groupby('Security')['Good_Investment'].mean()) +
           ". A real crime-rate feed (e.g. NCRB district data) should be joined before drawing conclusions.\n")
fig, ax = plt.subplots(figsize=(6, 4)); (df.groupby("Security")["Good_Investment"].mean() * 100).plot.bar(ax=ax, rot=0, color="#8c564b")
ax.set(title="E2  Security (crime-rate proxy) vs % Good Investment", ylabel="%")
save("e2_security_vs_investment")

df["Infra_Band"] = pd.cut(df["Infrastructure_Score"], [0, 3, 5, 7, 10], include_lowest=True, labels=["0-3", "3-5", "5-7", "7-10"])
ib = df.groupby("Infra_Band", observed=True).agg(ppsf=(PPSF, "mean"), price=("Price_in_Lakhs", "mean"), fut=("Future_Price_5Y_Lakhs", "mean"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ib["price"].plot.bar(ax=ax[0], rot=0); ax[0].set(title="E3  Avg current price by infrastructure score", ylabel="Lakhs", ylim=(240, 270))
ib["fut"].plot.bar(ax=ax[1], rot=0, color="#2ca02c"); ax[1].set(title="Projected 5-yr value by infrastructure score", ylabel="Lakhs", ylim=(300, 400))
save("e3_infra_vs_value")
note("E3 Infrastructure vs resale value:", f"corr(infrastructure score, current price)={df.Infrastructure_Score.corr(df.Price_in_Lakhs):+.3f} (none). "
     "Projected 5-yr value does rise with infrastructure, but that is the growth model's assumption (+/-1.5pp), not an empirical finding.")

(C.REPORT_DIR / "eda_findings.md").write_text("# EDA findings (auto-generated by src/02_eda.py)\n\n" + "\n".join(out))
print(f"Saved {len(list(C.FIG_DIR.glob('*.png')))} figures and eda_findings.md")
