
from datasets import load_dataset
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# 1. Load the existing splits
print("Loading cardiovascular dataset...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

train_df = ds["train"].to_pandas()
val_df = ds["validation"].to_pandas()

# 2. Define features and target
# Exclude row_idx because it appears to be an identifier.
target_col = "target"
numeric_features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical_features = [
    "f_5", "f_6", "f_7", "f_8", "f_9", "f_10"
]

feature_columns = numeric_features + categorical_features

def prepare_data(df):
    df = df.copy()

    # Convert encoded categorical strings to consistent strings.
    for col in categorical_features:
        df[col] = df[col].astype("string")

    # Mark clearly implausible measurements as missing.
    # Based on the provisional feature mapping:
    # f_1 = height, f_2 = weight,
    # f_3 = systolic BP, f_4 = diastolic BP.
    invalid_ranges = {
        "f_1": (100, 220),
        "f_2": (30, 300),
        "f_3": (50, 250),
        "f_4": (30, 150),
    }

    for col, (lower, upper) in invalid_ranges.items():
        df.loc[
            (df[col] < lower) | (df[col] > upper), col
        ] = np.nan

    X = df[feature_columns]
    y = df[target_col]

    return X, y

X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Training class counts:")
print(y_train.value_counts().sort_index())

# 3. Preprocess numerical and categorical features
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

# 4. Create the complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

# 5. Train only on training data
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# 6. Evaluate on validation data
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print("\nVALIDATION RESULTS")
print("==============================")
print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_prob):.4f}")

print("\nClassification Report")
print(classification_report(y_val, y_pred, zero_division=0))

print("Confusion Matrix")
print(confusion_matrix(y_val, y_pred))

print("\nLogistic Regression training completed.")
