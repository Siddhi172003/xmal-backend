import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
X = np.load("X_train.npy")
y = np.load("y_train.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------- Random Forest ----------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

joblib.dump(rf_model, "rf_model.pkl")
print("RF model saved")


# ---------------- SVM ----------------
svm_model = make_pipeline(
    StandardScaler(),
    SVC(
        probability=True,
        kernel="rbf",
        C=10
    )
)

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

print("\nSVM Accuracy:")
print(accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

joblib.dump(svm_model, "svm_model.pkl")
print("SVM model saved")