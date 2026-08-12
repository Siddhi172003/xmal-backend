import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("\n=========== ANDROID MALWARE MODEL TRAINING ===========\n")

# -----------------------------------
# LOAD DATASET
# -----------------------------------

dataset = pd.read_csv("Drebin.csv")

print("Total Samples:", dataset.shape[0])
print("Total Columns:", dataset.shape[1])

print("\nDataset Size (KB):",
      dataset.memory_usage(deep=True).sum()/1024)

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
dataset["class"] = dataset["class"].map({"B":0, "S":1})

# Remove rows with invalid class
dataset = dataset.dropna(subset=["class"])

dataset["class"] = dataset["class"].astype(int)

print("\nClass Distribution:")
print(dataset["class"].value_counts())

# -----------------------------------
# SAVE FEATURE LIST
# -----------------------------------

features = dataset.columns.tolist()

print("\nTotal Features:", len(features)-1)

print("\nSaving feature list for Android scanner...")

with open("features.txt", "w") as f:
    for feature in features:
        if feature != "class":
            f.write(feature + "\n")

# -----------------------------------
# SPLIT FEATURES
# -----------------------------------

X = dataset.drop("class", axis=1)
y = dataset["class"]

# Convert all features to numeric
X = X.apply(pd.to_numeric)

print("\nFeature Matrix Shape:", X.shape)

# -----------------------------------
# HANDLE CLASS IMBALANCE
# -----------------------------------

print("\nBalancing dataset using SMOTE...")

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("Balanced Dataset Shape:", X_resampled.shape)

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# BUILD MODEL
# -----------------------------------

print("\nBuilding Deep Learning Model...")

model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

# -----------------------------------
# COMPILE MODEL
# -----------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------

print("\nTraining Model...\n")

model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.2
)

import shap
import pickle
import json

print("\nGenerating SHAP explanations...")

# Convert dataframe to numpy arrays
X_sample = X_train.iloc[:500].values
X_test_sample = X_test.iloc[:100].values

# Create SHAP explainer
explainer = shap.DeepExplainer(model, X_sample)

# Generate SHAP values
shap_values = explainer.shap_values(X_test_sample)

# For binary classification
if isinstance(shap_values, list):
    shap_values = shap_values[0]

feature_names = list(X.columns)

# Mean importance
mean_importance = np.mean(np.abs(shap_values), axis=0)

feature_importance = {}

for i, feature in enumerate(feature_names):
    feature_importance[feature] = float(mean_importance[i].item())

# Save importance file
with open("shap_importance.pkl", "wb") as f:
    pickle.dump(feature_importance, f)

print("SHAP importance saved.")

# Generate readable explanations
explanations = {}

for feature, score in feature_importance.items():
    if score > 0.01:
        explanations[feature] = (
            f"{feature} significantly contributed to malware detection."
        )

with open("feature_explanations.json", "w") as f:
    json.dump(explanations, f, indent=4)

print("Feature explanations saved.")


import json

explanations = {}

for feature, score in feature_importance.items():
    if score > 0.01:
        explanations[feature] = (
            f"{feature} significantly contributed to malware detection."
        )

with open("feature_explanations.json", "w") as f:
    json.dump(explanations, f)

# -----------------------------------
# EVALUATE MODEL
# -----------------------------------

print("\nEvaluating Model...\n")

y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========= MODEL METRICS =========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nDetailed Report:\n")

print(classification_report(y_test, y_pred))

# -----------------------------------
# SAVE MODEL
# -----------------------------------

print("\nSaving Model...")

model.save("malware_model.h5")

# -----------------------------------
# CONVERT TO TFLITE
# -----------------------------------

print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("malware_model.tflite", "wb") as f:
    f.write(tflite_model)

print("\nTraining Complete!")

print("\nGenerated Files:")
print("malware_model.h5")
print("malware_model.tflite")
print("features.txt")