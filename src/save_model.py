
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

numeric_features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical_features = [
    "f_5", "f_6", "f_7", "f_8", "f_9", "f_10"
]
feature_columns = numeric_features + categorical_features

print("Loading training data...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")
df = ds["train"].to_pandas()

for col in categorical_features:
    df[col] = df[col].astype("string")

# Provisional limits; verify feature definitions before clinical interpretation.
limits = {
    "f_1": (100, 220),
    "f_2": (30, 300),
    "f_3": (50, 250),
    "f_4": (30, 150),
}

for col, (low, high) in limits.items():
    df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

X = df[feature_columns]
y = df["target"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
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

print("Training final XGBoost model...")
model.fit(X, y)

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

joblib.dump(model, output_dir / "xgboost_pipeline.joblib")

print("Saved model to models/xgboost_pipeline.joblib")
print("Training completed.")
