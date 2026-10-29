from datasets import load_dataset
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from xgboost import XGBClassifier

print("Loading training split...")
df = load_dataset(
    "BenchmarkDatasets/Cardiovascular-Disease-dataset"
)["train"].to_pandas()

numeric = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical = ["f_5", "f_6", "f_7", "f_8", "f_9", "f_10"]
features = numeric + categorical

X = df[features].copy()
y = df["target"].copy()

for col in categorical:
    X[col] = X[col].astype("string")

# Identical feature vectors receive the same group ID.
# row_idx and target are intentionally excluded.
groups = pd.factorize(
    pd.MultiIndex.from_frame(df[features])
)[0]

print(f"Rows: {len(df)}")
print(f"Unique feature groups: {len(np.unique(groups))}")
print(f"Repeated rows beyond first occurrences: {len(df) - len(np.unique(groups))}")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric),
    ("categorical", categorical_pipeline, categorical),
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

scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "recall": "recall",
    "roc_auc": "roc_auc",
}

cv_methods = {
    "Row_Stratifed": StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42
    ),
    "Duplicate_Grouped": StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=42
    ),
}

results = []

for name, splitter in cv_methods.items():
    print(f"\nEvaluating {name}...")

    if name == "Duplicate_Grouped":
        splits = list(splitter.split(X, y, groups=groups))

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            overlap = train_groups.intersection(val_groups)

            print(
                f"Fold {fold}: train={len(train_idx)}, "
                f"validation={len(val_idx)}, "
                f"shared feature groups={len(overlap)}"
            )

        scores = cross_validate(
            model, X, y, cv=splits, scoring=scoring,
            n_jobs=1, error_score="raise"
        )
    else:
        scores = cross_validate(
            model, X, y, cv=splitter, scoring=scoring,
            n_jobs=1, error_score="raise"
        )

    row = {"Method": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv("duplicate_aware_cv_results.csv", index=False)

print("\nDUPLICATE-AWARE CROSS-VALIDATION RESULTS")
print(results_df.round(4).to_string(index=False))
print("\nSaved to duplicate_aware_cv_results.csv")
