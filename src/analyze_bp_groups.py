
from datasets import load_dataset
import pandas as pd

df = load_dataset(
    "BenchmarkDatasets/Cardiovascular-Disease-dataset"
)["train"].to_pandas()

# Feature-only groups; do not use target to assign groups.
conditions = [
    df["f_4"] < 30,
    df["f_4"].between(30, 150, inclusive="both"),
    df["f_4"].between(151, 200, inclusive="both"),
    df["f_4"].between(201, 500, inclusive="both"),
    df["f_4"] > 500,
]

labels = [
    "Below 30",
    "30 to 150",
    "151 to 200",
    "201 to 500",
    "Above 500",
]

df["f4_group"] = pd.Series(
    pd.NA, index=df.index, dtype="string"
)

for condition, label in zip(conditions, labels):
    df.loc[condition, "f4_group"] = label

print("\nF4 VALUE GROUP SUMMARY")
print(df["f4_group"].value_counts().reindex(labels))

print("\nTARGET DISTRIBUTION BY F4 GROUP")
print(pd.crosstab(
    df["f4_group"],
    df["target"],
    normalize="index"
).round(4))

print("\nF3 DISTRIBUTION BY F4 GROUP")
print(
    df.groupby("f4_group", observed=False)["f_3"]
    .agg(["count", "min", "median", "max"])
)

print("\nAnalysis complete.")
