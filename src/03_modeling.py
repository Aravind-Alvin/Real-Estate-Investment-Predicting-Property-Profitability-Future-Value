"""Step 3 - Modeling: Good-Investment classifier + appreciation-rate regressor.

The regression target is the projected annual appreciation rate; the 5-year price is
Price * (1 + rate) ** 5. See docs for why (no price history in the raw data).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src import config as C
from src import features as F

RS = C.RANDOM_STATE
df = pd.read_csv(C.CLEAN_CSV)
X = F.model_matrix(df)
y_c = df["Good_Investment"]
y_r = df["Annual_Appreciation_Rate"]
idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=RS, stratify=y_c)
Xtr, Xte = X.loc[idx_tr], X.loc[idx_te]
metrics = {"n_train": int(len(idx_tr)), "n_test": int(len(idx_te)), "n_features": int(X.shape[1])}
sub = np.random.RandomState(RS).choice(idx_tr, 40_000, replace=False)       # for CV only


def timed(label, fn):
    t = time.time(); r = fn(); print(f"  {label}: {time.time() - t:.0f}s"); return r


# ------------------------------------------------------------ classification
print("Classification")
clfs = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0)),
    "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20,
                                            max_samples=0.5, n_jobs=-1, random_state=RS),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=RS),
}
cres = {}
for name, m in clfs.items():
    timed(name, lambda: m.fit(Xtr, y_c.loc[idx_tr]))
    pr = m.predict_proba(Xte)[:, 1]; pd_ = (pr >= 0.5).astype(int)
    cv = cross_val_score(m, X.loc[sub], y_c.loc[sub], cv=3, scoring="f1")
    cres[name] = {"accuracy": accuracy_score(y_c.loc[idx_te], pd_), "precision": precision_score(y_c.loc[idx_te], pd_),
                  "recall": recall_score(y_c.loc[idx_te], pd_), "f1": f1_score(y_c.loc[idx_te], pd_),
                  "roc_auc": roc_auc_score(y_c.loc[idx_te], pr), "cv_f1_mean_40k": cv.mean(), "cv_f1_std_40k": cv.std()}
    print("   ", {k: round(v, 4) for k, v in cres[name].items()})
best_c = max(cres, key=lambda k: cres[k]["f1"])
clf = clfs[best_c]
pr = clf.predict_proba(Xte)[:, 1]
conf = np.maximum(pr, 1 - pr)
cm = confusion_matrix(y_c.loc[idx_te], (pr >= .5).astype(int))
bins = pd.cut(pr, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
calib = (pd.DataFrame({"bin": bins.astype(str), "p": pr, "y": y_c.loc[idx_te].values})
         .groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), actual_rate=("y", "mean")).round(4))
metrics["classification"] = {
    "candidates": cres, "best_model": best_c, "confusion_matrix_[[TN,FP],[FN,TP]]": cm.tolist(),
    "share_predictions_with_confidence_ge_0.9": float((conf >= 0.9).mean()),
    "mean_confidence": float(conf.mean()), "calibration": calib.reset_index().to_dict("records"),
    "positive_rate_test": float(y_c.loc[idx_te].mean()),
    "majority_class_baseline_accuracy": float(1 - y_c.loc[idx_te].mean())}

# ---------------------------------------------------------------- regression
print("Regression (annual appreciation rate)")
regs = {
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=20,
                                           max_samples=0.5, n_jobs=-1, random_state=RS),
    "Gradient Boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.1, random_state=RS),
}
rres = {}
price_te = df.loc[idx_te, "Price_in_Lakhs"].values
fut_true = df.loc[idx_te, "Future_Price_5Y_Lakhs"].values
for name, m in regs.items():
    timed(name, lambda: m.fit(Xtr, y_r.loc[idx_tr]))
    p = m.predict(Xte)
    fut = price_te * (1 + p) ** C.FORECAST_YEARS
    rres[name] = {"r2_rate": r2_score(y_r.loc[idx_te], p), "mae_rate_pp": mean_absolute_error(y_r.loc[idx_te], p) * 100,
                  "mae_future_price_lakhs": mean_absolute_error(fut_true, fut),
                  "mape_future_price_pct": float(np.mean(np.abs(fut - fut_true) / fut_true) * 100)}
    print("   ", {k: round(v, 4) for k, v in rres[name].items()})
best_r = min(rres, key=lambda k: rres[k]["mae_rate_pp"])
reg = regs[best_r]
base_rate = y_r.loc[idx_tr].mean()
base_fut = price_te * (1 + base_rate) ** C.FORECAST_YEARS
resid_pp = (y_r.loc[idx_te].values - reg.predict(Xte)) * 100
metrics["regression"] = {
    "candidates": rres, "best_model": best_r,
    "baseline_constant_rate": {"rate": float(base_rate), "mae_future_price_lakhs": float(mean_absolute_error(fut_true, base_fut)),
                               "r2_rate": 0.0},
    "residual_std_pp": float(resid_pp.std())}

# -------- negative-result check: can listing attributes explain today's price?
print("Fair-price check (should be ~0 R2 on this dataset)")
no_price = [c for c in X.columns if c not in ("Price_in_Lakhs", "Price_per_SqFt_INR", "Price_vs_City_Median", "City_Median_PPSF")]
fp = HistGradientBoostingRegressor(max_iter=150, random_state=RS).fit(Xtr[no_price], df.loc[idx_tr, "Price_in_Lakhs"])
fp_r2 = r2_score(df.loc[idx_te, "Price_in_Lakhs"], fp.predict(Xte[no_price]))
fp_mae = mean_absolute_error(df.loc[idx_te, "Price_in_Lakhs"], fp.predict(Xte[no_price]))
metrics["fair_price_check"] = {"r2_test": float(fp_r2), "mae_lakhs": float(fp_mae),
                               "mae_if_predicting_mean_lakhs": float(mean_absolute_error(df.loc[idx_te, "Price_in_Lakhs"], np.full(len(idx_te), df.loc[idx_tr, "Price_in_Lakhs"].mean())))}
print("   R2 =", round(fp_r2, 4))

# ---------------------------------------------------- permutation importance
print("Permutation importance")
samp = np.random.RandomState(RS).choice(idx_te, 6000, replace=False)
def perm(model, y, scoring):
    r = permutation_importance(model, X.loc[samp], y.loc[samp], n_repeats=3, random_state=RS, scoring=scoring, n_jobs=1)
    return pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std}).sort_values("importance", ascending=False)
fi_c = perm(clf, y_c, "f1"); fi_r = perm(reg, y_r, "r2")
fi_c.to_csv(C.MODEL_DIR / "feature_importance_classifier.csv", index=False)
fi_r.to_csv(C.MODEL_DIR / "feature_importance_regressor.csv", index=False)

# ---------------------------------------------------------------------- save
joblib.dump(clf, C.MODEL_DIR / "classifier.joblib", compress=3)
joblib.dump(reg, C.MODEL_DIR / "regressor.joblib", compress=3)
joblib.dump(list(X.columns), C.MODEL_DIR / "feature_columns.joblib")
(C.REPORT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
print(json.dumps({k: metrics[k] for k in ("classification", "regression", "fair_price_check")}, indent=1, default=float)[:3500])
print(fi_c.head(8).to_string()); print(fi_r.head(8).to_string())
