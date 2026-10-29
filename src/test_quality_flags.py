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
df = load_dataset(
    "BenchmarkDatasets/Cardiovascular-Disease-dataset"
)["train"].to_pandas()

numeric = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical = ["f_5", "f_6", "f_7", "f_8", "f_9", "f_10"]
features = numeric + categorical

X_raw = df[features].copy()
y = df["target"].copy()

# Identify suspicious blood-pressure measurements.
bad_sys = (df["f_3"] < 50) | (df["f_3"] > 250)
bad_dia = (df["f_4"] < 30) | (df["f_4"] > 150)
reversed_bp = df["f_3"] < df["f_4"]

flags = pd.DataFrame({
    "flag_systolic": bad_sys.astype(int),
    "flag_diastolic": bad_dia.astype(int),
    "flag_reversed_bp": reversed_bp.astype(int),
}, index=df.index)

# A: Raw features.
X_a = X_raw.copy()

# B: Raw features plus quality flags.
X_b = X_raw.join(flags)

# C: Clean suspicious BP values and retain quality flags.
X_c = X_raw.copy()
X_c.loc[bad_sys, "f_3"] = np.nan
X_c.loc[bad_dia, "f_4"] = np.nan
X_c.loc[reversed_bp, ["f_3", "f_4"]] = np.nan
X_c = X_c.join(flags)

experiments = {
    "A_Raw": (X_a, numeric, categorical),
    "B_Raw_Plus_Flags": (
        X_b, numeric + list(flags.columns), categorical
    ),
    "C_Cleaned_Plus_Flags": (
        X_c, numeric + list(flags.columns), categorical
    ),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "recall": "recall",
    "roc_auc": "roc_auc",
}

results = []

for name, (X, num_cols, cat_cols) in experiments.items():
    print(f"\nEvaluating {name}...")

    # Convert categorical fields consistently.
    X = X.copy()
    for col in cat_cols:
        X[col] = X[col].astype("string")

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
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
        model, X, y, cv=cv, scoring=scoring,
        n_jobs=1, error_score="raise"
    )

    row = {"Condition": name}
    for metric in scoring:
        vals = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = vals.mean()
        row[f"{metric}_std"] = vals.std()

    results.append(row)

result_df = pd.DataFrame(results)
result_df.to_csv("quality_flag_experiment.csv", index=False)

print("\nQUALITY-FLAG EXPERIMENT")
print(result_df.round(4).to_string(index=False))
print("\nSaved to quality_flag_experiment.csv")
