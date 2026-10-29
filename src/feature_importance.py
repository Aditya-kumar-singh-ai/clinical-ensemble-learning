
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# 1. Load training data
print("Loading dataset...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")
train_df = ds["train"].to_pandas()

numeric_features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical_features = [
    "f_5", "f_6", "f_7", "f_8", "f_9", "f_10"
]
feature_columns = numeric_features + categorical_features


# 2. Prepare features using the same provisional cleaning rules
def prepare_data(df):
    df = df.copy()

    for col in categorical_features:
        df[col] = df[col].astype("string")

    # Provisional limits: verify feature meanings before final use.
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


# 3. Build preprocessing pipeline
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


# 4. Build and train XGBoost
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

print("Training XGBoost...")
model.fit(X_train, y_train)


# 5. Extract feature names and importance scores
fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()
importance_scores = fitted_classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_scores,
})

importance_df = importance_df.sort_values(
    "importance", ascending=False
).reset_index(drop=True)

importance_df.insert(
    0, "rank", range(1, len(importance_df) + 1)
)


# 6. Save full results
output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "feature_importance.csv"
importance_df.to_csv(csv_path, index=False)

print("\nTOP 15 FEATURES")
print("================")
print(importance_df.head(15).to_string(index=False))
print(f"\nFull results saved to: {csv_path}")


# 7. Plot top 15 features
top_features = importance_df.head(15).sort_values("importance")

plt.figure(figsize=(11, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("XGBoost feature importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Features")
plt.tight_layout()

chart_path = output_dir / "feature_importance.png"
plt.savefig(chart_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Chart saved to: {chart_path}")
print("\nFeature importance analysis completed.")
