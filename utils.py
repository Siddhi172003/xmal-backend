import joblib
import numpy as np

rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")

def predict_apk(features):
    rf_pred = rf_model.predict_proba(features)[0][1]
    svm_pred = svm_model.decision_function(features)[0]

    # normalize svm
    svm_score = 1 / (1 + np.exp(-svm_pred))

    final_score = (rf_pred + svm_score) / 2

    if final_score > 0.5:
        return "Malicious", final_score
    else:
        return "Safe", final_score