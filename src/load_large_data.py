
from datasets import load_dataset
import pandas as pd

print("Loading cardiovascular disease dataset...")

ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

print("\nDataset loaded!")
print(ds)

# Inspect the available splits
print("\nAvailable splits:", list(ds.keys()))

# Convert the training split into a DataFrame
df = ds["train"].to_pandas()

print("\nTraining data shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
