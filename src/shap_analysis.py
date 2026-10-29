from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datasets import load_dataset

MODEL_PATH = Path("models/xgboost_pipeline.joblib")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run src/final_test_evaluation.py first."
    )

print("Loading saved XGBoost pipeline...")
model = joblib.load(MODEL_PATH)

print("Loading training data for explanation...")
df = load_dataset(
    "BenchmarkDatasets/Cardiovascular-Disease-dataset"
)["train"].to_pandas()

numeric = ["f_0", "f_1", "f_2", "f_3", "f_4"]
categorical = ["f_5", "f_6", "f_7", "f_8", "f_9", "f_10"]
features = numeric + categorical

# Use a reproducible sample to keep computation manageable.
sample = df.sample(n=1000, random_state=42)[features].copy()

for col in categorical:
    sample[col] = sample[col].astype("string")

print("Transforming features using the fitted preprocessing pipeline...")
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(sample)

try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = [
        f"transformed_feature_{i}"
        for i in range(X_transformed.shape[1])
    ]

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_transformed)

# Save transformed feature names and mean absolute SHAP values.
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": abs(shap_values).mean(axis=0),
}).sort_values("mean_absolute_shap", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "shap_global_importance.csv",
    index=False,
)

print("\nTOP 15 FEATURES BY MEAN ABSOLUTE SHAP")
print(importance.head(15).round(6).to_string(index=False))

print("\nSaving SHAP summary plot...")
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    max_display=15,
    show=False,
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()

print(f"\nSaved: {OUTPUT_DIR / 'shap_global_importance.csv'}")
print(f"Saved: {OUTPUT_DIR / 'shap_summary.png'}")
print("SHAP analysis complete.")
