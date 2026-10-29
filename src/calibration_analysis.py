from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

print("Loading data...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")
train = ds["train"].to_pandas()
val = ds["validation"].to_pandas()

numeric = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical = ["f_5", "f_6", "f_7", "f_8", "f_9", "f_10"]
features = numeric + categorical

X_train = train[features].copy()
y_train = train["target"]
X_val = val[features].copy()
y_val = val["target"]

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

model = Pipeline([
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

print("Training XGBoost on training split...")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_val)[:, 1]

# Ten quantile bins aim for roughly equal numbers of predictions per bin.
observed_rate, mean_probability = calibration_curve(
    y_val,
    probabilities,
    n_bins=10,
    strategy="quantile",
)

calibration_table = pd.DataFrame({
    "mean_predicted_probability": mean_probability,
    "observed_positive_rate": observed_rate,
    "calibration_gap": mean_probability - observed_rate,
})

print("\nCALIBRATION TABLE")
print(calibration_table.round(4).to_string(index=False))
print(f"\nBrier score: {brier_score_loss(y_val, probabilities):.4f}")

Path("reports").mkdir(exist_ok=True)
calibration_table.to_csv("reports/calibration_table.csv", index=False)

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.plot(
    mean_probability,
    observed_rate,
    marker="o",
    label="XGBoost",
)
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive rate")
plt.title("XGBoost Calibration Curve — Validation Set")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/calibration_curve.png", dpi=150)
plt.close()

print("\nSaved reports/calibration_table.csv")
print("Saved reports/calibration_curve.png")
print("Calibration analysis complete.")
