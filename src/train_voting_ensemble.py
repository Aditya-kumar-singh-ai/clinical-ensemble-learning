
from datasets import load_dataset
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# 1. Load the existing dataset splits
print("Loading dataset...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

train_df = ds["train"].to_pandas()
val_df = ds["validation"].to_pandas()

numeric_features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical_features = [
    "f_5", "f_6", "f_7", "f_8", "f_9", "f_10"
]
feature_columns = numeric_features + categorical_features


def prepare_data(df):
    df = df.copy()

    for col in categorical_features:
        df[col] = df[col].astype("string")

    # Provisional limits; verify feature definitions
    # before finalizing clinical cleaning rules.
    limits = {
        "f_1": (100, 220),
        "f_2": (30, 300),
        "f_3": (50, 250),
        "f_4": (30, 150),
    }

    for col, (low, high) in limits.items():
        df.loc[
            (df[col] < low) | (df[col] > high), col
        ] = np.nan

    return df[feature_columns], df["target"]


X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)

# 2. Build shared preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 3. Define the three classifiers
logistic = LogisticRegression(max_iter=2000)

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)

xgboost = XGBClassifier(
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
)

# 4. Combine models using soft voting
voting = VotingClassifier(
    estimators=[
        ("lr", logistic),
        ("rf", random_forest),
        ("xgb", xgboost),
    ],
    voting="soft",
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("ensemble", voting),
])

# 5. Train the ensemble
print("Training soft-voting ensemble...")
model.fit(X_train, y_train)

# 6. Evaluate on validation data
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

print("\nSOFT-VOTING ENSEMBLE — VALIDATION RESULTS")
print("=========================================")
print(f"Accuracy:  {accuracy_score(y_val, predictions):.4f}")
print(f"Precision: {precision_score(y_val, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_val, predictions, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, probabilities):.4f}")

print("\nClassification Report")
print(classification_report(y_val, predictions, zero_division=0))

print("Confusion Matrix")
print(confusion_matrix(y_val, predictions))

print("\nSoft-voting ensemble training completed.")
# 7. Save the trained voting ensemble
output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "voting_ensemble_pipeline.joblib"
joblib.dump(model, model_path)

print(f"Voting ensemble saved to: {model_path}")
