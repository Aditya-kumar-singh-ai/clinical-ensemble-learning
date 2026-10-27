
from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

# Use the training split for initial inspection
df = ds["train"].to_pandas()

# Remove the row index because it is not a clinical feature
df = df.drop(columns=["row_idx"], errors="ignore")

print("\n1. DATA TYPES")
print(df.dtypes)

print("\n2. UNIQUE VALUES IN EACH FEATURE")
for col in df.columns:
    if col != "target":
        print(f"{col}: {df[col].nunique()} unique values")
        print("Sample values:", sorted(df[col].dropna().unique())[:15])

print("\n3. NUMERICAL SUMMARY")
print(df.describe().T.round(2))

print("\n4. TARGET COUNTS")
print(df["target"].value_counts().sort_index())

print("\n5. DUPLICATE FEATURE ROWS")
print("Duplicates:", df.drop(columns="target").duplicated().sum())

print("\nInspection complete.")
