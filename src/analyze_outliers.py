
from datasets import load_dataset
import pandas as pd

print("Loading training dataset...")

df = load_dataset(
    "BenchmarkDatasets/Cardiovascular-Disease-dataset"
)["train"].to_pandas()

limits = {
    "f_1": (100, 220),
    "f_2": (30, 300),
    "f_3": (50, 250),
    "f_4": (30, 150),
}

print("\nOUTLIER ANALYSIS")
print("=" * 60)

for col, (low, high) in limits.items():

    invalid = df[(df[col] < low) | (df[col] > high)]

    print(f"\nFeature: {col}")
    print(f"Accepted range: {low} to {high}")
    print(f"Invalid records: {len(invalid)}")
    print(f"Percentage: {len(invalid) / len(df) * 100:.3f}%")

    print("\nLowest suspicious values:")
    print(sorted(invalid[col].unique())[:10])

    print("\nHighest suspicious values:")
    print(sorted(invalid[col].unique())[-10:])

    print("\nTarget distribution of invalid records:")
    print(invalid["target"].value_counts().sort_index())

print("\nOutlier analysis complete.")
