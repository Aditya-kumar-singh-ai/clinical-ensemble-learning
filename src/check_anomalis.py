
from datasets import load_dataset
import pandas as pd

ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")
df = ds["train"].to_pandas()

# Rename columns using the provisional mapping
df = df.rename(columns={
    "f_0": "age_days",
    "f_1": "height_cm",
    "f_2": "weight_kg",
    "f_3": "systolic_bp",
    "f_4": "diastolic_bp",
    "f_5": "gender",
    "f_6": "cholesterol",
    "f_7": "glucose",
    "f_8": "smoking",
    "f_9": "alcohol",
    "f_10": "active",
})

print("Age in years — summary:")
print((df["age_days"] / 365.25).describe())

print("\nBlood pressure — value counts:")
print(df["systolic_bp"].value_counts().sort_index().to_string())
print(df["diastolic_bp"].value_counts().sort_index().to_string())

print("\nHeight and weight — summary:")
print(df[["height_cm", "weight_kg"]].describe())

print("\nCategorical feature values:")
for col in [
    "gender", "cholesterol", "glucose",
    "smoking", "alcohol", "active"
]:
    print(f"{col}: {sorted(df[col].dropna().unique())}")

print("\nExtreme blood-pressure records:")
print(
    df.loc[
        (df["systolic_bp"] < 50) |
        (df["systolic_bp"] > 250) |
        (df["diastolic_bp"] < 30) |
        (df["diastolic_bp"] > 150),
        ["age_days", "systolic_bp", "diastolic_bp", "target"]
    ].head(20).to_string(index=False)
)
