# save_features.py
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Load dataset (same as DL code)
dataset = pd.read_csv("Drebin.csv")

# Replace ? with 0
dataset = dataset.replace("?", 0)
dataset["class"] = dataset["class"].astype(str).str.strip()
dataset["class"] = dataset["class"].map({"B":0, "S":1})
dataset = dataset.dropna(subset=["class"])
dataset["class"] = dataset["class"].astype(int)

# Features and labels
X = dataset.drop("class", axis=1)
y = dataset["class"]
X = X.apply(pd.to_numeric)

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save for RF + SVM
np.save("X_train.npy", X_resampled)
np.save("y_train.npy", y_resampled)

print("Saved X_train.npy and y_train.npy")