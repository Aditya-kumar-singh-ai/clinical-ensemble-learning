
from datasets import load_dataset
import pandas as pd

ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")
df = ds["train"].to_pandas()

# Provisional mapping; verify feature meanings against the source
df = df.rename(columns={
    "f_0": "age_days",
    "f_1": "height_cm",
    "f_2": "weight_kg",
    "f_3": "systolic_bp",
    "f_4": "diastolic_bp",
})

print("TRAINING DATA QUALITY REPORT")
print("=" * 45)

# Count invalid-looking values using screening thresholds.
# These are investigation thresholds, not medical diagnostic rules.
checks = {
    "Negative systolic BP": df["systolic_bp"] < 0,
    "Systolic BP below 50": df["systolic_bp"] < 50,
    "Systolic BP above 250": df["systolic_bp"] > 250,
    "Negative diastolic BP": df["diastolic_bp"] < 0,
    "Diastolic BP below 30": df["diastolic_bp"] < 30,
    "Diastolic BP above 150": df["diastolic_bp"] > 150,
    "Height below 100 cm": df["height_cm"] < 100,
    "Height above 220 cm": df["height_cm"] > 220,
    "Weight below 30 kg": df["weight_kg"] < 30,
    "Weight above 300 kg": df["weight_kg"] > 300,
}

for name, mask in checks.items():
    print(f"{name}: {int(mask.sum())}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nExact duplicate feature rows:")
feature_cols = [
    c for c in df.columns
    if c not in ["target", "row_idx"]
]
print(df[feature_cols].duplicated().sum())

# Show the most extreme blood-pressure records
mask = (
    (df["systolic_bp"] < 50) |
    (df["systolic_bp"] > 250) |
    (df["diastolic_bp"] < 30) |
    (df["diastolic_bp"] > 150)
)

print("\nProportion flagged:")
print(f"{mask.mean() * 100:.3f}%")

print("\nReport complete.")
