
from datasets import load_dataset
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# 1. Load the existing dataset splits
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

    # Preserve encoded categorical values as categories
    for col in categorical_features:
        df[col] = df[col].astype("string")

    # Provisional limits based on our current feature mapping.
    # Verify these against the dataset documentation before
    # treating them as final clinical cleaning rules.
    limits = {
        "f_1": (100, 220),
        "f_2": (30, 300),
        "f_3": (50, 250),
        "f_4": (30, 150),
    }

    for col, (low, high) in limits.items():
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

    return df[feature_columns], df["target"]

X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)

# 2. Preprocess features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 3. Create and train Random Forest
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )),
])

print("Training Random Forest...")
model.fit(X_train, y_train)

# 4. Evaluate on validation data
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

print("\nRANDOM FOREST — VALIDATION RESULTS")
print("===================================")
print(f"Accuracy:  {accuracy_score(y_val, predictions):.4f}")
print(f"Precision: {precision_score(y_val, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_val, predictions, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, probabilities):.4f}")

print("\nClassification Report")
print(classification_report(y_val, predictions, zero_division=0))

print("Confusion Matrix")
print(confusion_matrix(y_val, predictions))

print("\nRandom Forest training completed.")
