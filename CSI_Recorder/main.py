import pandas as pd

from dataset_builder import build_dataset

# Build the dataset
X, y = build_dataset("Practice_Dataset")

# Create feature column names
feature_columns = [f"Feature_{i+1}" for i in range(X.shape[1])]

# Convert features into a DataFrame
df = pd.DataFrame(X, columns=feature_columns)

# Add activity labels
df["Label"] = y

# Save to CSV
df.to_csv("feature_dataset.csv", index=False)

print("\n===================================")
print("Feature Dataset Saved Successfully!")
print("===================================")
print("File Name : feature_dataset.csv")
print("Dataset Shape :", df.shape)

print("\nFirst 5 Rows:")
print(df.head())