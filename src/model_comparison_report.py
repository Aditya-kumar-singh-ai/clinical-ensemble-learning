
from pathlib import Path
import pandas as pd

# Project root: works whether launched from the root or src folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "cross_validation_results.csv"
OUTPUT_FILE = PROJECT_DIR / "model_comparison_report.csv"

results = pd.read_csv(INPUT_FILE)

required_columns = [
    "Model",
    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "roc_auc_mean",
    "roc_auc_std",
]

missing = [col for col in required_columns if col not in results.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

results = results.sort_values(
    by="roc_auc_mean",
    ascending=False
).reset_index(drop=True)

print("\nCLINICAL ENSEMBLE MODEL — SELECTION REPORT")
print("=" * 72)
print("Metrics are cross-validation means unless otherwise stated.")

for _, row in results.iterrows():
    print(f"\nModel: {row['Model']}")
    print(f"  Accuracy :  {row['accuracy_mean']:.4f}")
    print(f"  Precision:  {row['precision_mean']:.4f}")
    print(f"  Recall   :  {row['recall_mean']:.4f}")
    print(f"  F1-score :  {row['f1_mean']:.4f}")
    print(f"  ROC-AUC  :  {row['roc_auc_mean']:.4f}")
    print(f"  ROC-AUC SD: {row['roc_auc_std']:.4f}")

best_model = results.iloc[0]

print("\n" + "=" * 72)
print("MODEL SELECTION")
print(f"Highest mean ROC-AUC: {best_model['Model']}")
print(f"Mean ROC-AUC: {best_model['roc_auc_mean']:.4f}")

print("\nINTERPRETATION")
print(
    "The selected model has the highest mean ROC-AUC in the "
    "current cross-validation results."
)
print(
    "ROC-AUC measures ranking discrimination; it does not establish "
    "clinical safety or calibration."
)
print(
    "Recall, specificity, precision, calibration, and the consequences "
    "of false negatives must also be considered."
)
print(
    "A small ROC-AUC difference does not necessarily indicate a "
    "meaningful performance improvement."
)

results.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved report: {OUTPUT_FILE}")
