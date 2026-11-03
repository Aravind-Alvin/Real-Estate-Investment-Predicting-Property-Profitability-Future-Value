"""
STEP 2 - Exploratory Data Analysis (Q1-Q20 + investment-specific checks)
Saves one figure per question in eda_outputs/figures and every number quoted in the
documentation to eda_outputs/eda_findings.json.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import config as C

sns.set_theme(style="whitegrid", context="notebook")
C.FIG_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(C.CLEAN_CSV)
F = {}          # findings


def save(name):
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"{name}.png", dpi=110)
    plt.close()


def r2(x):
    return round(float(x), 3)


def spearman(a, b):
    rho, p = stats.spearmanr(a, b)
    return {"spearman_rho": r2(rho), "p_value": float(f"{p:.3g}")}


def group_gap(col, target="Price_per_SqFt_INR"):
    g = df.groupby(col)[target].mean()
    return {"means": g.round(2).to_dict(), "max_vs_min_pct": r2((g.max() / g.min() - 1) * 100)}


# =============================== PRICE & SIZE ==================================
# Q1 price distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Price_in_Lakhs"], bins=50, ax=ax[0], color="#2a6f97"); ax[0].set_title("Q1 Price distribution (Lakhs)")
sns.boxplot(x=df["Price_in_Lakhs"], ax=ax[1], color="#89c2d9"); ax[1].set_title("Price boxplot")
save("q01_price_distribution")
F["q1_price"] = {**df["Price_in_Lakhs"].describe().round(2).to_dict(), "skew": r2(df["Price_in_Lakhs"].skew())}

# Q2 size distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Size_in_SqFt"], bins=50, ax=ax[0], color="#468faf"); ax[0].set_title("Q2 Size distribution (sq ft)")
sns.boxplot(x=df["Size_in_SqFt"], ax=ax[1], color="#a9d6e5"); ax[1].set_title("Size boxplot")
save("q02_size_distribution")
F["q2_size"] = {**df["Size_in_SqFt"].describe().round(2).to_dict(), "skew": r2(df["Size_in_SqFt"].skew())}

# Q3 price/sqft by property type
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt_INR", showfliers=False, ax=ax, palette="Blues")
ax.set_title("Q3 Price per sq ft by property type (INR, outliers hidden)")
save("q03_ppsf_by_property_type")
F["q3_ppsf_by_type"] = {"mean": group_gap("Property_Type"),
                        "median": df.groupby("Property_Type")["Price_per_SqFt_INR"].median().round(1).to_dict(),
                        "kruskal_p": float(f"{stats.kruskal(*[g['Price_per_SqFt_INR'].values for _, g in df.groupby('Property_Type')])[1]:.3g}")}

# Q4 size vs price
samp = df.sample(8000, random_state=C.RANDOM_STATE)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(samp["Size_in_SqFt"], samp["Price_in_Lakhs"], s=4, alpha=.35, color="#1d3557")
ax[0].set_xlabel("Size (sq ft)"); ax[0].set_ylabel("Price (Lakhs)"); ax[0].set_title("Q4 Size vs price (8k sample)")
ax[1].scatter(samp["Size_in_SqFt"], samp["Price_per_SqFt_INR"], s=4, alpha=.35, color="#e76f51")
ax[1].set_xlabel("Size (sq ft)"); ax[1].set_ylabel("Price per sq ft (INR)"); ax[1].set_title("Size vs price per sq ft")
save("q04_size_vs_price")
F["q4_size_vs_price"] = {"price": spearman(df["Size_in_SqFt"], df["Price_in_Lakhs"]),
                         "ppsf": spearman(df["Size_in_SqFt"], df["Price_per_SqFt_INR"]),
                         "pearson_price": r2(df[["Size_in_SqFt", "Price_in_Lakhs"]].corr().iloc[0, 1])}

# Q5 outliers
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x=df["Price_per_SqFt_INR"], ax=ax[0], color="#f4a261"); ax[0].set_title("Q5 Price per sq ft (INR) - IQR outliers")
sns.boxplot(x=df["Size_in_SqFt"], ax=ax[1], color="#2a9d8f"); ax[1].set_title("Size (sq ft)")
save("q05_outliers")
F["q5_outliers"] = {"ppsf_outliers": int(df["Outlier_PPSF"].sum()),
                    "ppsf_outliers_pct": r2(df["Outlier_PPSF"].mean() * 100),
                    "size_outliers": int(df["Outlier_Size"].sum()),
                    "ppsf_p99": r2(df["Price_per_SqFt_INR"].quantile(.99)),
                    "ppsf_max": r2(df["Price_per_SqFt_INR"].max()),
                    "outlier_median_size": float(df.loc[df.Outlier_PPSF == 1, "Size_in_SqFt"].median()),
                    "overall_median_size": float(df["Size_in_SqFt"].median())}

# =============================== LOCATION ======================================
# Q6 avg ppsf by state
s6 = df.groupby("State")["Price_per_SqFt_INR"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 6)); s6.plot.barh(ax=ax, color="#2a6f97")
ax.set_title("Q6 Average price per sq ft by state (INR)"); ax.set_xlabel("INR / sq ft")
save("q06_ppsf_by_state")
F["q6_ppsf_state"] = {"top3": s6.tail(3).round(1)[::-1].to_dict(), "bottom3": s6.head(3).round(1).to_dict(),
                      "spread_pct": r2((s6.max() / s6.min() - 1) * 100)}

# Q7 avg price by city
s7 = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 9)); s7.plot.barh(ax=ax, color="#468faf")
ax.set_xlim(s7.min() * 0.97, s7.max() * 1.01)
ax.set_title("Q7 Average property price by city (Lakhs; axis truncated)"); ax.set_xlabel("Lakhs")
save("q07_price_by_city")
F["q7_price_city"] = {"top3": s7.tail(3).round(2)[::-1].to_dict(), "bottom3": s7.head(3).round(2).to_dict(),
                      "spread_pct": r2((s7.max() / s7.min() - 1) * 100), "overall_mean": r2(df["Price_in_Lakhs"].mean())}

# Q8 median age by locality
s8 = df.groupby("Locality_ID")["Age_of_Property"].agg(["median", "size"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(s8["median"], bins=25, ax=ax[0], color="#6a994e"); ax[0].set_title("Q8 Median property age across 21,000 localities")
top = s8[s8["size"] >= 8].sort_values("median")
pd.concat([top.head(8), top.tail(8)])["median"].plot.barh(ax=ax[1], color="#bc6c25")
ax[1].set_title("Youngest / oldest localities (>=8 listings)")
save("q08_median_age_by_locality")
F["q8_age_locality"] = {"overall_median_age": float(df["Age_of_Property"].median()),
                        "locality_median_range": [float(s8["median"].min()), float(s8["median"].max())],
                        "n_localities": int(len(s8)), "median_listings_per_locality": float(s8["size"].median()),
                        "youngest": top.head(3)["median"].to_dict(), "oldest": top.tail(3)["median"].to_dict()}

# Q9 BHK across cities
ct = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(7, 11)); sns.heatmap(ct, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={"label": "% of city listings"})
ax.set_title("Q9 BHK mix by city (%)")
save("q09_bhk_by_city")
F["q9_bhk_city"] = {"overall_pct": (df["BHK"].value_counts(normalize=True).sort_index() * 100).round(1).to_dict(),
                    "max_deviation_from_20pct": r2((ct - 20).abs().max().max())}

# Q10 top-5 most expensive localities & trend by build period
loc = df.groupby("Locality_ID").agg(avg_ppsf=("Price_per_SqFt_INR", "mean"), n=("ID", "size")).query("n >= 8")
top5 = loc.sort_values("avg_ppsf", ascending=False).head(5)
t5 = df[df["Locality_ID"].isin(top5.index)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
top5["avg_ppsf"].sort_values().plot.barh(ax=ax[0], color="#c1121f"); ax[0].set_title("Q10 Top 5 localities by avg price/sq ft (INR)")
trend = t5.groupby(["Build_Period", "Locality_ID"])["Price_per_SqFt_INR"].mean().unstack()
trend.plot(ax=ax[1], marker="o"); ax[1].set_title("Price/sq ft by year-built period (no sale dates exist)")
ax[1].set_xlabel("Year built (5-yr bucket)"); ax[1].legend(fontsize=7)
save("q10_top5_localities")
all_trend = df.groupby("Build_Period")["Price_per_SqFt_INR"].mean()
F["q10_top5_localities"] = {"top5_avg_ppsf": top5["avg_ppsf"].round(1).to_dict(), "listings_each": top5["n"].to_dict(),
                            "city_wide_trend_by_build_period": all_trend.round(0).to_dict(),
                            "trend_slope_per_year_citywide": r2(np.polyfit(df["Year_Built"], df["Price_per_SqFt_INR"], 1)[0]),
                            "citywide_spearman_year_vs_ppsf": spearman(df["Year_Built"], df["Price_per_SqFt_INR"])}

# =============================== CORRELATIONS ==================================
# Q11 correlation matrix
numc = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property", "Floor_No", "Total_Floors",
        "Nearby_Schools", "Nearby_Hospitals", "Infrastructure_Score", "Amenity_Count", "Parking_Flag", "Security_Flag",
        "Est_Annual_Appreciation_Pct"]
corr = df[numc].corr()
fig, ax = plt.subplots(figsize=(11, 8.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Q11 Correlation matrix (numeric features)")
save("q11_correlation_matrix")
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
strong = pairs[pairs.abs() > 0.3].round(3)
F["q11_correlations"] = {"pairs_abs_gt_0.3": {f"{a} ~ {b}": v for (a, b), v in strong.items()},
                         "max_abs_among_raw_listing_attrs": r2(df[["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Age_of_Property", "Floor_No",
                                                                  "Nearby_Schools", "Nearby_Hospitals"]].corr().where(
                             ~np.eye(7, dtype=bool)).abs().max().max())}

# Q12 / Q13 schools & hospitals vs ppsf
for q, col, nm in [(12, "Nearby_Schools", "schools"), (13, "Nearby_Hospitals", "hospitals")]:
    g = df.groupby(col)["Price_per_SqFt_INR"].mean()
    fig, ax = plt.subplots(figsize=(7, 4)); g.plot(marker="o", ax=ax, color="#264653")
    ax.set_title(f"Q{q} Avg price/sq ft by number of nearby {nm}"); ax.set_ylabel("INR / sq ft")
    save(f"q{q}_{nm}_vs_ppsf")
    F[f"q{q}_{nm}_vs_ppsf"] = {**spearman(df[col], df["Price_per_SqFt_INR"]), "group_means": g.round(0).to_dict(),
                               "max_vs_min_pct": r2((g.max() / g.min() - 1) * 100)}

# Q14 furnished status
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax[0], palette="Blues", errorbar="ci"); ax[0].set_title("Q14 Avg price (Lakhs) by furnishing")
sns.barplot(data=df, x="Furnished_Status", y="Price_per_SqFt_INR", ax=ax[1], palette="Oranges", errorbar="ci"); ax[1].set_title("Avg price/sq ft (INR)")
save("q14_price_by_furnished")
F["q14_furnished"] = {"price": group_gap("Furnished_Status", "Price_in_Lakhs"), "ppsf": group_gap("Furnished_Status"),
                      "anova_price_p": float(f"{stats.f_oneway(*[g['Price_in_Lakhs'].values for _, g in df.groupby('Furnished_Status')])[1]:.3g}")}

# Q15 facing
fig, ax = plt.subplots(figsize=(7, 4.5)); sns.barplot(data=df, x="Facing", y="Price_per_SqFt_INR", ax=ax, palette="viridis", errorbar="ci")
ax.set_title("Q15 Avg price/sq ft by facing (INR)")
save("q15_ppsf_by_facing")
F["q15_facing"] = {"ppsf": group_gap("Facing"),
                   "kruskal_p": float(f"{stats.kruskal(*[g['Price_per_SqFt_INR'].values for _, g in df.groupby('Facing')])[1]:.3g}")}

# =============================== INVESTMENT / OWNERSHIP =========================
# Q16 owner type
oc = df["Owner_Type"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4)); sns.barplot(x=oc.index, y=oc.values, ax=ax, palette="Set2"); ax.set_title("Q16 Listings by owner type")
for i, v in enumerate(oc.values): ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
save("q16_owner_type")
F["q16_owner_type"] = {"counts": oc.to_dict(), "pct": (oc / oc.sum() * 100).round(1).to_dict(),
                       "good_investment_rate_pct": (df.groupby("Owner_Type")["Good_Investment"].mean() * 100).round(1).to_dict()}

# Q17 availability
ac = df["Availability_Status"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4)); sns.barplot(x=ac.index, y=ac.values, ax=ax, palette="Set1"); ax.set_title("Q17 Listings by availability status")
for i, v in enumerate(ac.values): ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
save("q17_availability")
F["q17_availability"] = {"counts": ac.to_dict(), "pct": (ac / ac.sum() * 100).round(1).to_dict(),
                         "good_investment_rate_pct": (df.groupby("Availability_Status")["Good_Investment"].mean() * 100).round(1).to_dict(),
                         "note": "dataset has only Ready_to_Move and Under_Construction (no 'Sold')"}

# Q18 parking
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=ax[0], palette="Blues", errorbar="ci"); ax[0].set_title("Q18 Avg price (Lakhs) by parking")
sns.barplot(data=df, x="Parking_Space", y="Price_per_SqFt_INR", ax=ax[1], palette="Oranges", errorbar="ci"); ax[1].set_title("Avg price/sq ft (INR)")
save("q18_parking")
F["q18_parking"] = {"price": group_gap("Parking_Space", "Price_in_Lakhs"), "ppsf": group_gap("Parking_Space"),
                    "ttest_price_p": float(f"{stats.ttest_ind(df.loc[df.Parking_Space=='Yes','Price_in_Lakhs'], df.loc[df.Parking_Space=='No','Price_in_Lakhs'])[1]:.3g}")}

# Q19 amenities
rows = []
for a in C.AMENITY_LIST:
    m = df[f"Has_{a}"] == 1
    rows.append({"amenity": a, "with": df.loc[m, "Price_per_SqFt_INR"].mean(), "without": df.loc[~m, "Price_per_SqFt_INR"].mean()})
am = pd.DataFrame(rows).set_index("amenity")
am["diff_pct"] = (am["with"] / am["without"] - 1) * 100
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
am[["with", "without"]].plot.bar(ax=ax[0], color=["#2a9d8f", "#e9c46a"]); ax[0].set_title("Q19 Avg price/sq ft with vs without amenity"); ax[0].set_ylabel("INR/sq ft")
df.groupby("Amenity_Count")["Price_per_SqFt_INR"].mean().plot(marker="o", ax=ax[1], color="#264653"); ax[1].set_title("By number of amenities")
save("q19_amenities")
F["q19_amenities"] = {"per_amenity_diff_pct": am["diff_pct"].round(2).to_dict(),
                      "by_count": df.groupby("Amenity_Count")["Price_per_SqFt_INR"].mean().round(0).to_dict(),
                      "spearman_count_vs_ppsf": spearman(df["Amenity_Count"], df["Price_per_SqFt_INR"]),
                      "good_investment_rate_by_count_pct": (df.groupby("Amenity_Count")["Good_Investment"].mean() * 100).round(1).to_dict()}

# Q20 transport
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
order = ["Low", "Medium", "High"]
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt_INR", order=order, ax=ax[0], palette="Blues", errorbar="ci"); ax[0].set_title("Q20 Avg price/sq ft")
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Est_Annual_Appreciation_Pct", order=order, ax=ax[1], palette="Greens", errorbar="ci"); ax[1].set_title("Est. annual appreciation (%)")
(df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(order) * 100).plot.bar(ax=ax[2], color="#e76f51"); ax[2].set_title("% Good Investments")
save("q20_transport")
F["q20_transport"] = {"ppsf": group_gap("Public_Transport_Accessibility"),
                      "est_appreciation_pct": df.groupby("Public_Transport_Accessibility")["Est_Annual_Appreciation_Pct"].mean().round(2).to_dict(),
                      "good_investment_rate_pct": (df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean() * 100).round(1).to_dict()}

# =============================== PROJECT-SPECIFIC CHECKS ========================
# Crime rate is NOT in the dataset -> closest available safety proxy is the Security flag
sec = df.groupby("Security")["Good_Investment"].mean() * 100
fig, ax = plt.subplots(figsize=(5.5, 4)); sec.plot.bar(ax=ax, color=["#adb5bd", "#2a9d8f"]); ax.set_ylabel("% Good Investments")
ax.set_title("Safety proxy: secured community vs Good Investment")
save("x1_security_vs_good_investment")
F["x1_security_proxy"] = {"good_investment_rate_pct": sec.round(1).to_dict(),
                          "ppsf": group_gap("Security"), "note": "No crime-rate column exists in the dataset."}

# Infrastructure score vs price (resale value proxy)
df["Infra_Band"] = pd.cut(df["Infrastructure_Score"], [0, 3, 5, 7, 10.01], labels=["0-3", "3-5", "5-7", "7-10"], include_lowest=True)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
df.groupby("Infra_Band", observed=True)["Price_per_SqFt_INR"].mean().plot.bar(ax=ax[0], color="#264653"); ax[0].set_title("Avg price/sq ft by infrastructure band")
df.groupby("Infra_Band", observed=True)["Price_in_Lakhs"].mean().plot.bar(ax=ax[1], color="#2a9d8f"); ax[1].set_title("Avg price (Lakhs)")
(df.groupby("Infra_Band", observed=True)["Good_Investment"].mean() * 100).plot.bar(ax=ax[2], color="#e76f51"); ax[2].set_title("% Good Investments")
save("x2_infrastructure_vs_value")
F["x2_infrastructure"] = {"spearman_vs_ppsf": spearman(df["Infrastructure_Score"], df["Price_per_SqFt_INR"]),
                          "spearman_vs_price": spearman(df["Infrastructure_Score"], df["Price_in_Lakhs"]),
                          "price_by_band": df.groupby("Infra_Band", observed=True)["Price_in_Lakhs"].mean().round(1).to_dict(),
                          "good_rate_by_band_pct": (df.groupby("Infra_Band", observed=True)["Good_Investment"].mean() * 100).round(1).to_dict()}

# Label overview
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df["Good_Investment"].map({0: "Not good", 1: "Good"}).value_counts().plot.pie(ax=ax[0], autopct="%.1f%%", colors=["#adb5bd", "#2a9d8f"]); ax[0].set_ylabel("")
ax[0].set_title("Good Investment label balance")
sns.histplot(df["Est_Annual_Appreciation_Pct"], bins=50, ax=ax[1], color="#2a6f97"); ax[1].axvline(C.GOOD_APPRECIATION_THRESHOLD, color="red", ls="--")
ax[1].set_title("Estimated annual appreciation (red = threshold)")
save("x3_label_overview")
F["x3_label"] = {"good_pct": r2(df["Good_Investment"].mean() * 100), "appreciation_mean": r2(df["Est_Annual_Appreciation_Pct"].mean()),
                 "appreciation_range": [r2(df["Est_Annual_Appreciation_Pct"].min()), r2(df["Est_Annual_Appreciation_Pct"].max())],
                 "top_cities_good_rate_pct": (df.groupby("City")["Good_Investment"].mean() * 100).sort_values(ascending=False).head(5).round(1).to_dict(),
                 "bottom_cities_good_rate_pct": (df.groupby("City")["Good_Investment"].mean() * 100).sort_values().head(5).round(1).to_dict(),
                 "good_rate_by_property_type_pct": (df.groupby("Property_Type")["Good_Investment"].mean() * 100).round(1).to_dict()}

(C.EDA_DIR / "eda_findings.json").write_text(json.dumps(F, indent=2, default=str))
print("EDA complete:", len(list(C.FIG_DIR.glob('*.png'))), "figures")
