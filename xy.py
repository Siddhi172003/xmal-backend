import pandas as pd

print("\n=========== ANDROID MALWARE DATASET INFO ===========\n")

# -----------------------------------
# LOAD DATASET
# -----------------------------------

dataset = pd.read_csv("Drebin.csv")

print("Total Samples (Rows):", dataset.shape[0])
print("Total Columns:", dataset.shape[1])
print("Dataset Size (KB):", dataset.memory_usage(deep=True).sum() / 1024)

# -----------------------------------
# CLEAN DATASET
# -----------------------------------

print("\nCleaning dataset...")

# Replace ? with 0
dataset = dataset.replace("?", 0)

# Clean class column
dataset["class"] = dataset["class"].astype(str).str.strip()

print("\nUnique Class Labels Before Encoding:")
print(dataset["class"].unique())

# Convert labels
dataset["class"] = dataset["class"].map({"B": 0, "S": 1})

# Remove rows with invalid class
dataset = dataset.dropna(subset=["class"])
dataset["class"] = dataset["class"].astype(int)

print("\nClass Distribution:")
print(dataset["class"].value_counts())

# -----------------------------------
# FEATURE LIST
# -----------------------------------

features = dataset.columns.tolist()
print("\nTotal Features (excluding class):", len(features) - 1)

print("\nFeature Names:")
for feature in features:
    if feature != "class":
        print("-", feature)