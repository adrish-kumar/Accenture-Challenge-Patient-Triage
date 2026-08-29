"""
train_triage_model.py
======================
Trains an automated Emergency Department triage-level prediction model
on the KTAS (Korean Triage and Acuity Scale) dataset.

Target      : KTAS_expert  (expert-assigned gold-standard triage level, 1-5)
                1 = most urgent (resuscitation)  ...  5 = least urgent (non-urgent)

Inputs used (only information available AT triage time):
  Numeric   : Age, Patients number per hour, NRS_pain, SBP, DBP, HR, RR, BT, Saturation
  Categorical: Group, Sex, Arrival mode, Injury, Mental, Pain
  Text      : Chief_complain   (free-text presenting complaint, via TF-IDF)

Deliberately EXCLUDED (would leak information not available at triage / not usable
in a live triage-support tool):
  - KTAS_RN            -> this is itself a human triage decision, using it as an
                           input would mean the model is just parroting a nurse,
                           not providing an independent automated assessment.
  - Diagnosis in ED     -> only known AFTER full ED work-up, long after triage.
  - Disposition         -> known only at the end of the visit.
  - Length of stay_min, KTAS duration_min -> only known after the visit ends.
  - Error_group, mistriage -> derived retrospectively by comparing KTAS_RN vs
                           KTAS_expert; not available prospectively.

Outputs:
  - triage_model.joblib   : trained scikit-learn Pipeline (preprocessing + model)
  - metrics_report.txt    : full evaluation report
  - confusion_matrix.png  : visual confusion matrix
  - feature_importance.png: top predictive features
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

from triage_common import select_text

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
DATA_PATH = "KTAS_data_cleaned.xlsx"   # cleaned file produced earlier
df = pd.read_excel(DATA_PATH)

TARGET = "KTAS_expert"

NUMERIC_FEATURES = [
    "Age", "Patients number per hour", "NRS_pain",
    "SBP", "DBP", "HR", "RR", "BT", "Saturation"
]
CATEGORICAL_FEATURES = ["Group", "Sex", "Arrival mode", "Injury", "Mental", "Pain"]
TEXT_FEATURE = "Chief_complain"

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]

df = df.dropna(subset=[TARGET]).reset_index(drop=True)
X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# Text column must be a 1-D string series for TfidfVectorizer; fill missing text
X[TEXT_FEATURE] = X[TEXT_FEATURE].fillna("").astype(str)

# Categorical columns as strings (they are numeric codes, but we treat as categories)
for c in CATEGORICAL_FEATURES:
    X[c] = X[c].astype("Int64").astype(str)

print(f"Dataset: {X.shape[0]} patients, target classes: {sorted(y.unique())}")
print(y.value_counts().sort_index())

# --------------------------------------------------------------------------
# 2. Train / test split (stratified to preserve class balance)
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# --------------------------------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------------------------------
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

text_pipe = Pipeline([
    ("select", FunctionTransformer(select_text, validate=False)),
    ("tfidf", TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), max_features=300, min_df=2
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC_FEATURES),
    ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ("text", text_pipe, [TEXT_FEATURE]),
])

# --------------------------------------------------------------------------
# 4. Candidate models
# --------------------------------------------------------------------------
candidates = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=400, max_depth=None, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

print("\n=== 5-fold cross-validation on training set (macro-F1) ===")
for name, clf in candidates.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    results[name] = scores.mean()
    print(f"{name:20s}: macro-F1 = {scores.mean():.3f}  (+/- {scores.std():.3f})")

best_name = max(results, key=results.get)
print(f"\nBest model by CV macro-F1: {best_name}")

# --------------------------------------------------------------------------
# 5. Fit best model on full training set, evaluate on held-out test set
# --------------------------------------------------------------------------
best_pipe = Pipeline([("prep", preprocessor), ("clf", candidates[best_name])])
best_pipe.fit(X_train, y_train)

y_pred = best_pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

report_txt = classification_report(y_test, y_pred, digits=3)

# --- Clinical safety metric: under-triage rate ---------------------------
# In KTAS, level 1 = most urgent, 5 = least urgent.
# "Under-triage" = model predicts a LESS urgent level (higher number) than
# the true expert level (lower number) -> the dangerous direction of error.
diff = y_pred - y_test.values
under_triage_rate = float(np.mean(diff > 0))          # predicted less urgent than truth
over_triage_rate = float(np.mean(diff < 0))           # predicted more urgent than truth
exact_match_rate = float(np.mean(diff == 0))
within_one_level = float(np.mean(np.abs(diff) <= 1))

print(f"\n=== Held-out test set performance ({best_name}) ===")
print(f"Accuracy               : {acc:.3f}")
print(f"Macro F1                : {macro_f1:.3f}")
print(f"Weighted F1             : {weighted_f1:.3f}")
print(f"Exact match rate        : {exact_match_rate:.3f}")
print(f"Within +/-1 KTAS level  : {within_one_level:.3f}")
print(f"Under-triage rate       : {under_triage_rate:.3f}  <-- most clinically important")
print(f"Over-triage rate        : {over_triage_rate:.3f}")
print("\nPer-class report:\n", report_txt)

with open("metrics_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n")
    f.write("Cross-validation macro-F1 (5-fold) by model:\n")
    for name, score in results.items():
        f.write(f"  {name}: {score:.3f}\n")
    f.write(f"\n--- Held-out test set (20% of data, n={len(y_test)}) ---\n")
    f.write(f"Accuracy               : {acc:.3f}\n")
    f.write(f"Macro F1                : {macro_f1:.3f}\n")
    f.write(f"Weighted F1             : {weighted_f1:.3f}\n")
    f.write(f"Exact match rate        : {exact_match_rate:.3f}\n")
    f.write(f"Within +/-1 KTAS level  : {within_one_level:.3f}\n")
    f.write(f"Under-triage rate       : {under_triage_rate:.3f}  (predicted LESS urgent than truth)\n")
    f.write(f"Over-triage rate        : {over_triage_rate:.3f}  (predicted MORE urgent than truth)\n\n")
    f.write("Per-class classification report:\n")
    f.write(report_txt)

# --------------------------------------------------------------------------
# 6. Confusion matrix plot
# --------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title(f"Confusion Matrix - {best_name}\n(rows=true KTAS, cols=predicted KTAS)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 7. Feature importance (only meaningful for tree-based models)
# --------------------------------------------------------------------------
if best_name in ("RandomForest", "GradientBoosting"):
    prep = best_pipe.named_steps["prep"]
    cat_names = prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
    text_names = prep.named_transformers_["text"].named_steps["tfidf"].get_feature_names_out()
    feature_names = np.concatenate([NUMERIC_FEATURES, cat_names, [f"text__{t}" for t in text_names]])
    importances = best_pipe.named_steps["clf"].feature_importances_
    top_idx = np.argsort(importances)[-20:]
    plt.figure(figsize=(8, 8))
    plt.barh(range(len(top_idx)), importances[top_idx])
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.xlabel("Importance")
    plt.title(f"Top 20 Feature Importances - {best_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

# --------------------------------------------------------------------------
# 8. Refit best model on ALL data (train+test) and save final deployable model
# --------------------------------------------------------------------------
final_pipe = Pipeline([("prep", preprocessor), ("clf", candidates[best_name])])
final_pipe.fit(X, y)

joblib.dump(
    {
        "pipeline": final_pipe,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "text_feature": TEXT_FEATURE,
        "target": TARGET,
        "model_name": best_name,
        "test_metrics": {
            "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "under_triage_rate": under_triage_rate, "over_triage_rate": over_triage_rate,
        }
    },
    "triage_model.joblib"
)

print("\nSaved: triage_model.joblib, metrics_report.txt, confusion_matrix.png"
      + (", feature_importance.png" if best_name != "LogisticRegression" else ""))
