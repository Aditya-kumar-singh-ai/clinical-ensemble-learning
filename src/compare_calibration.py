from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

print("Loading training and validation splits...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

train = ds["train"].to_pandas()
val = ds["validation"].to_pandas()

numeric = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical = ["f_5", "f_6", "f_7", "f_8", "f_9", "f_10"]
features = numeric + categorical

X_train = train[features].copy()
y_train = train["target"].copy()
X_val = val[features].copy()
y_val = val["target"].copy()

for col in categorical:
    X_train[col] = X_train[col].astype("string")
    X_val[col] = X_val[col].astype("string")

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

from xgboost import XGBClassifier

base_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )),
])

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

models = {
    "Uncalibrated": base_model,
    "Sigmoid": CalibratedClassifierCV(
        estimator=clone(base_model),
        method="sigmoid",
        cv=cv,
        ensemble=True,
    ),
    "Isotonic": CalibratedClassifierCV(
        estimator=clone(base_model),
        method="isotonic",
        cv=cv,
        ensemble=True,
    ),
}

results = []
probability_sets = {}

for name, model in models.items():
    print(f"\nFitting {name} model...")

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)[:, 1]
    probability_sets[name] = probabilities

    results.append({
        "Model": name,
        "ROC_AUC": roc_auc_score(y_val, probabilities),
        "Brier_Score": brier_score_loss(y_val, probabilities),
        "Log_Loss": log_loss(y_val, probabilities),
    })

results_df = pd.DataFrame(results)

print("\nCALIBRATION COMPARISON")
print("=" * 65)
print(results_df.round(4).to_string(index=False))

Path("reports").mkdir(exist_ok=True)
results_df.to_csv(
    "reports/calibration_comparison.csv",
    index=False,
)

# Plot all methods against the perfect-calibration diagonal.
plt.figure(figsize=(8, 7))
plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")

for name, probabilities in probability_sets.items():
    observed, predicted = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )
    plt.plot(
        predicted,
        observed,
        marker="o",
        label=name,
    )

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive rate")
plt.title("Calibration Comparison — Validation Set")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "reports/calibration_comparison.png",
    dpi=150,
)
plt.close()

print("\nSaved reports/calibration_comparison.csv")
print("Saved reports/calibration_comparison.png")
print("Calibration comparison complete.")
print("The held-out test split was not used.")
