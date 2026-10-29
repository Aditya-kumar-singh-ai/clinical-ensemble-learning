from datasets import load_dataset
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

print("Loading training split...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")
df = ds["train"].to_pandas()

numeric_features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical_features = ["f_5", "f_6", "f_7", "f_8", "f_9", "f_10"]
feature_columns = numeric_features + categorical_features

y = df["target"].copy()

# Keep a separate copy for each experiment.
conditions = {}

# A: Raw measurements, without manual cleaning.
conditions["A_Raw"] = df[feature_columns].copy()

# B: Your existing cleaning rules.
current = df[feature_columns].copy()

for col, (low, high) in {
    "f_1": (100, 220),
    "f_2": (30, 300),
    "f_3": (50, 250),
    "f_4": (30, 150),
}.items():
    current.loc[(current[col] < low) | (current[col] > high), col] = np.nan

conditions["B_Current_Rules"] = current

# C: Blood-pressure cleaning.
bp_clean = df[feature_columns].copy()

bad_systolic = (bp_clean["f_3"] < 50) | (bp_clean["f_3"] > 250)
bad_diastolic = (bp_clean["f_4"] < 30) | (bp_clean["f_4"] > 150)
reversed_bp = bp_clean["f_3"] < bp_clean["f_4"]

# Mark only the affected BP measurements as missing.
bp_clean.loc[bad_systolic, "f_3"] = np.nan
bp_clean.loc[bad_diastolic, "f_4"] = np.nan

# For reversed pairs, mark both BP measurements as missing.
bp_clean.loc[reversed_bp, ["f_3", "f_4"]] = np.nan

conditions["C_BP_Cleaning"] = bp_clean

# D: BP cleaning plus broader height and weight checks.
broad_clean = bp_clean.copy()

bad_height = (broad_clean["f_1"] < 120) | (broad_clean["f_1"] > 220)
bad_weight = (broad_clean["f_2"] < 30) | (broad_clean["f_2"] > 200)

broad_clean.loc[bad_height, "f_1"] = np.nan
broad_clean.loc[bad_weight, "f_2"] = np.nan

conditions["D_Broader_Cleaning"] = broad_clean

# Make categorical features consistently categorical for every condition.
for data in conditions.values():
    for col in categorical_features:
        data[col] = data[col].astype("string")

# One fixed set of folds for every condition.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

results = []

for condition_name, X in conditions.items():
    print(f"\nRunning experiment: {condition_name}")

    # Count rows with at least one manually replaced feature.
    changed_rows = int(X.isna().any(axis=1).sum())

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

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )

    row = {
        "Condition": condition_name,
        "Rows_with_missing_features_after_rules": changed_rows,
    }

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv("data_quality_experiment.csv", index=False)

print("\nDATA-QUALITY EXPERIMENT RESULTS")
print("=" * 80)
print(results_df.round(4).to_string(index=False))
print("\nSaved to data_quality_experiment.csv")
print("Experiment complete.")
