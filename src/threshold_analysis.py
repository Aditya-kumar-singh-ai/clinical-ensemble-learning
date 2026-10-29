from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from xgboost import XGBClassifier

print("Loading training and validation splits...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

train_df = ds["train"].to_pandas()
val_df = ds["validation"].to_pandas()

numeric = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical = ["f_5", "f_6", "f_7", "f_8", "f_9", "f_10"]
features = numeric + categorical

X_train = train_df[features].copy()
y_train = train_df["target"].copy()
X_val = val_df[features].copy()
y_val = val_df["target"].copy()

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

print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print("\nTraining model...")
model.fit(X_train, y_train)

print("Calculating validation probabilities...")
probabilities = model.predict_proba(X_val)[:, 1]

print("\nPROBABILITY QUALITY")
print("=" * 55)
print(f"ROC-AUC:          {roc_auc_score(y_val, probabilities):.4f}")
print(f"Average precision:{average_precision_score(y_val, probabilities):.4f}")
print(f"Brier score:      {brier_score_loss(y_val, probabilities):.4f}")
print(f"Log loss:         {log_loss(y_val, probabilities):.4f}")

results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else 0,
        "f1": f1_score(y_val, predictions, zero_division=0),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    })

results_df = pd.DataFrame(results)
print("\nTHRESHOLD COMPARISON")
print("=" * 110)
print(results_df.round(4).to_string(index=False))

Path("reports").mkdir(exist_ok=True)
results_df.to_csv("reports/threshold_analysis.csv", index=False)

print("\nSaved results to reports/threshold_analysis.csv")
print("Analysis complete. The held-out test split was not used.")
