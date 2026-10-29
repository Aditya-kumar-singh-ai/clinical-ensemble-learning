
from datasets import load_dataset
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

# Load TRAINING data only
print("Loading training data...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")
df = ds["train"].to_pandas()

numeric_features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical_features = [
    "f_5", "f_6", "f_7", "f_8", "f_9", "f_10"
]
feature_columns = numeric_features + categorical_features

# Keep preprocessing inside the pipeline to avoid leakage.
for col in categorical_features:
    df[col] = df[col].astype("string")

# These are provisional limits based on the current feature mapping.
# Verify the feature definitions before finalizing these rules.
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

def make_pipeline(classifier):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

lr = make_pipeline(
    LogisticRegression(max_iter=2000)
)

rf = make_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
)

xgb = make_pipeline(
    XGBClassifier(
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
)

# VotingClassifier combines the three classifiers.
voting = make_pipeline(
    VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            )),
            ("xgb", XGBClassifier(
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
        ],
        voting="soft",
    )
)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb,
    "Soft Voting": voting,
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

results = []

for name, model in models.items():
    print(f"\nCross-validating {name}...")

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

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv("cross_validation_results.csv", index=False)

print("\nCROSS-VALIDATION RESULTS")
print("========================")
print(results_df.round(4).to_string(index=False))
print("\nSaved to cross_validation_results.csv")
