from pathlib import Path

import joblib
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
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

print("Loading dataset splits...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

numeric_features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical_features = [
    "f_5", "f_6", "f_7", "f_8", "f_9", "f_10"
]
feature_columns = numeric_features + categorical_features

X_train = train_df[feature_columns].copy()
y_train = train_df["target"].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df["target"].copy()

for col in categorical_features:
    X_train[col] = X_train[col].astype("string")
    X_test[col] = X_test[col].astype("string")

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
print(f"Held-out test rows: {len(X_test)}")
print("\nTraining the frozen XGBoost pipeline...")

model.fit(X_train, y_train)

print("Evaluating on held-out test data...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nFINAL HELD-OUT TEST RESULTS")
print("=" * 55)
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

print("\nConfusion Matrix")
print("Rows = actual class; columns = predicted class")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(
    y_test,
    predictions,
    target_names=["Class 0", "Class 1"],
    zero_division=0,
))

output_path = Path("models/xgboost_pipeline.joblib")
output_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, output_path)

print(f"\nSaved evaluated pipeline to: {output_path}")
print("Final test evaluation complete.")
