"""
STEP 3 - Modelling
  A) Classifier  : Good_Investment (0/1)               -> Logistic Regression baseline vs HistGradientBoosting
  B) Regressor   : Est_Annual_Appreciation_Pct         -> Ridge baseline vs HistGradientBoosting   (drives the price forecast)
  C) Diagnostic  : can CURRENT price be predicted from property attributes?  (honest negative-result check)
All preprocessing (scaling, encoding) lives inside sklearn Pipelines so training and the Streamlit app match.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import config as C
import features as F

df = pd.read_csv(C.CLEAN_CSV)
X = df[F.MODEL_FEATURES]
cats, nums = F.CAT_FEATURES, F.NUM_FEATURES


def linear_prep(num_cols, cat_cols):
    return ColumnTransformer([("num", StandardScaler(), num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])


def tree_prep(num_cols, cat_cols):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan)
    return ColumnTransformer([("num", "passthrough", num_cols), ("cat", enc, cat_cols)]), \
        [False] * len(num_cols) + [True] * len(cat_cols)


metrics = {}

# ----------------------------------------------------------------- A) classifier
y = df["Good_Investment"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)
prep, mask = tree_prep(nums, cats)
candidates = {
    "LogisticRegression": Pipeline([("prep", linear_prep(nums, cats)), ("m", LogisticRegression(max_iter=300, C=1.0))]),
    "HistGradientBoosting": Pipeline([("prep", prep), ("m", HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, categorical_features=mask, random_state=C.RANDOM_STATE))]),
}
cls_metrics = {}
for name, pipe in candidates.items():
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
    cls_metrics[name] = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
                         "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, p),
                         "confusion_matrix[[TN,FP],[FN,TP]]": confusion_matrix(yte, pred).tolist()}
    print(name, {k: round(v, 4) for k, v in cls_metrics[name].items() if not isinstance(v, list)})
best = max(cls_metrics, key=lambda k: cls_metrics[k]["f1"])
metrics["classifier"] = {"models": cls_metrics, "selected": best, "test_rows": len(yte),
                         "positive_rate_test": float(yte.mean()),
                         "caveat": "Label is a deterministic rule of the inputs, so high scores show the model "
                                   "reproduces the rule - they are NOT evidence of predicting real market outcomes."}
clf = candidates[best]
joblib.dump(clf, C.MODEL_DIR / "good_investment_classifier.joblib", compress=3)

# permutation importance (sample for speed)
idx = Xte.sample(6000, random_state=C.RANDOM_STATE).index
pi = permutation_importance(clf, Xte.loc[idx], yte.loc[idx], n_repeats=3, scoring="f1", random_state=C.RANDOM_STATE)
imp = pd.Series(pi.importances_mean, index=F.MODEL_FEATURES).sort_values(ascending=False)
metrics["classifier"]["permutation_importance_f1_drop_top12"] = imp.head(12).round(4).to_dict()
imp.to_csv(C.MODEL_DIR / "classifier_feature_importance.csv", header=["f1_drop"])

# ----------------------------------------------------------------- B) appreciation regressor
y2 = df["Est_Annual_Appreciation_Pct"]
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y2, test_size=0.2, random_state=C.RANDOM_STATE)
prep2, mask2 = tree_prep(nums, cats)
regs = {"Ridge": Pipeline([("prep", linear_prep(nums, cats)), ("m", Ridge(alpha=1.0))]),
        "HistGradientBoosting": Pipeline([("prep", prep2), ("m", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, categorical_features=mask2, random_state=C.RANDOM_STATE))])}
reg_metrics = {}
for name, pipe in regs.items():
    pipe.fit(Xtr2, ytr2); pr = pipe.predict(Xte2)
    reg_metrics[name] = {"r2": r2_score(yte2, pr), "mae_pct_points": mean_absolute_error(yte2, pr)}
    print("appreciation", name, {k: round(v, 4) for k, v in reg_metrics[name].items()})
best2 = max(reg_metrics, key=lambda k: reg_metrics[k]["r2"])
metrics["appreciation_regressor"] = {"models": reg_metrics, "selected": best2,
                                     "target_std_pct_points": float(yte2.std())}
joblib.dump(regs[best2], C.MODEL_DIR / "appreciation_regressor.joblib", compress=3)

# ----------------------------------------------------------------- C) diagnostic: predict current price
price_nums = [c for c in nums if c not in ("Price_in_Lakhs", "Price_per_SqFt_INR", "PPSF_vs_City")]
Xp = df[price_nums + cats]; yp = df["Price_in_Lakhs"]
Xtr3, Xte3, ytr3, yte3 = train_test_split(Xp, yp, test_size=0.2, random_state=C.RANDOM_STATE)
prep3, mask3 = tree_prep(price_nums, cats)
pm = Pipeline([("prep", prep3), ("m", HistGradientBoostingRegressor(max_iter=200, categorical_features=mask3,
                                                                  random_state=C.RANDOM_STATE))]).fit(Xtr3, ytr3)
pp = pm.predict(Xte3)
base = np.full(len(yte3), ytr3.mean())
metrics["current_price_diagnostic"] = {
    "gbm_r2": r2_score(yte3, pp), "gbm_mae_lakhs": mean_absolute_error(yte3, pp),
    "predict_mean_r2": r2_score(yte3, base), "predict_mean_mae_lakhs": mean_absolute_error(yte3, base),
    "conclusion": "Current price is not predictable from the listing attributes in this dataset (R2 ~ 0). "
                  "The app therefore benchmarks an asking price against comparable listings instead of "
                  "pretending to predict it."}
print("price diagnostic", {k: round(v, 4) for k, v in metrics["current_price_diagnostic"].items() if isinstance(v, float)})

(C.MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
print("saved models:", [p.name for p in C.MODEL_DIR.iterdir()])
