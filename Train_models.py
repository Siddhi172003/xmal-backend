# train_models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Load Drebin features (assuming X_train, y_train are ready)
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/rf_model.pkl')
print("Random Forest saved!")

# --- SVM ---
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'models/svm_model.pkl')
print("SVM saved!")