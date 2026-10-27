
from ucimlrepo import fetch_ucirepo

# Step 1: Download the UCI Heart Disease dataset
print("Downloading Heart Disease dataset...")

heart = fetch_ucirepo(id=45)

# Step 2: Separate features and target
X = heart.data.features
y = heart.data.targets

# Step 3: Display basic information
print("\nDataset loaded successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Step 4: Display the first five patient records
print("\nFirst five records:")
print(X.head())

# Step 5: Display the original target values
print("\nTarget distribution:")
print(y.iloc[:, 0].value_counts().sort_index())
