
from datasets import load_dataset
import pandas as pd

print("Loading dataset...")

df = load_dataset(
    "BenchmarkDatasets/Cardiovascular-Disease-dataset"
)["train"].to_pandas()

# Show frequency of suspicious diastolic values
invalid_f4 = df[
    (df["f_4"] < 30) | (df["f_4"] > 150)
].copy()

print("\n1. MOST FREQUENT SUSPICIOUS f_4 VALUES")
print(invalid_f4["f_4"].value_counts().head(20))

# Inspect systolic and diastolic values together
print("\n2. SUSPICIOUS f_4 RECORDS WITH f_3")

print(
    invalid_f4[
        ["f_0", "f_1", "f_2", "f_3", "f_4", "target"]
    ].sort_values("f_4").to_string(index=False)
)

# Check whether both blood pressure features are suspicious
invalid_f3 = (
    (df["f_3"] < 50) | (df["f_3"] > 250)
)

invalid_f4_mask = (
    (df["f_4"] < 30) | (df["f_4"] > 150)
)

print("\n3. OVERLAPPING BLOOD-PRESSURE ANOMALIES")
print("Only f_3 suspicious:", (invalid_f3 & ~invalid_f4_mask).sum())
print("Only f_4 suspicious:", (~invalid_f3 & invalid_f4_mask).sum())
print("Both suspicious:", (invalid_f3 & invalid_f4_mask).sum())

# Compare target distribution
print("\n4. TARGET DISTRIBUTION")
print("All training records:")
print(df["target"].value_counts(normalize=True).round(4))

print("\nf_4 suspicious records:")
print(invalid_f4["target"].value_counts(normalize=True).round(4))

print("\nBlood-pressure anomaly analysis complete.")
