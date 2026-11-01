
from datasets import load_dataset
import pandas as pd

print("Loading dataset...")

df = load_dataset(
    "BenchmarkDatasets/Cardiovascular-Disease-dataset"
)["train"].to_pandas()

print("\n1. NUMERIC FEATURE SUMMARY")
print(df[["f_0", "f_1", "f_2", "f_3", "f_4"]].describe().round(2))

print("\n2. CATEGORICAL FEATURE VALUES")

for col in ["f_5", "f_6", "f_7", "f_8", "f_9", "f_10"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).sort_index())

print("\n3. F0 AS POSSIBLE AGE IN YEARS")
print((df["f_0"] / 365.25).describe().round(2))

print("\n4. TARGET DISTRIBUTION")
print(df["target"].value_counts(normalize=True).sort_index().round(4))

print("\n5. EXAMPLE SUSPICIOUS BLOOD PRESSURE RECORDS")
print(
    df.loc[
        (df["f_3"] > 250) |
        (df["f_4"] > 200) |
        (df["f_3"] < 50) |
        (df["f_4"] < 30),
        ["f_0", "f_1", "f_2", "f_3", "f_4", "target"]
    ].head(15).to_string(index=False)
)

print("\nFeature verification inspection complete.")
