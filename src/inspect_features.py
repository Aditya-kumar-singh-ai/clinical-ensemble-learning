
from datasets import load_dataset

ds = load_dataset("BenchmarkDatasets/Cardiovascular-Disease-dataset")

print("DATASET SPLITS")
for split_name, split_data in ds.items():
    print(f"{split_name}: {split_data.shape}")

print("\nFEATURE SCHEMA")
print(ds["train"].features)

print("\nCOLUMN MAPPING TO VERIFY")
mapping = {
    "row_idx": "Possible row identifier",
    "f_0": "Possible age in days",
    "f_1": "Possible height in cm",
    "f_2": "Possible weight in kg",
    "f_3": "Possible systolic blood pressure",
    "f_4": "Possible diastolic blood pressure",
    "f_5": "Possible gender code",
    "f_6": "Possible cholesterol category",
    "f_7": "Possible glucose category",
    "f_8": "Possible smoking indicator",
    "f_9": "Possible alcohol indicator",
    "f_10": "Possible physical activity indicator",
    "target": "Target label",
}

for column, meaning in mapping.items():
    print(f"{column}: {meaning}")

print("\nSchema inspection complete.")

