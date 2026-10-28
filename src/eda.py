
from datasets import load_dataset
import pandas as pd

# Load the cardiovascular dataset
print("Loading dataset...")
ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

train_df = ds["train"].to_pandas()

# Basic information
print("\n1. DATASET SHAPE")
print(train_df.shape)

print("\n2. COLUMN NAMES")
print(train_df.columns.tolist())

print("\n3. DATA TYPES")
print(train_df.dtypes)

print("\n4. FIRST FIVE ROWS")
print(train_df.head())

print("\n5. NUMERICAL SUMMARY")
print(train_df.describe().T)

# Target distribution
if "target" in train_df.columns:
    print("\n6. TARGET DISTRIBUTION")
    print(train_df["target"].value_counts(dropna=False))

    print("\nTARGET PERCENTAGES")
    print(
        (train_df["target"].value_counts(normalize=True) * 100)
        .round(2)
    )

# Unique values help identify binary/categorical columns
print("\n7. UNIQUE VALUES PER COLUMN")
for col in train_df.columns:
    print(f"{col}: {train_df[col].nunique()} unique values")

print("\n8. MISSING VALUES")
print(train_df.isnull().sum())

print("\nEDA completed.")
