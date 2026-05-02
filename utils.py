import joblib
import numpy as np

rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")


def predict_apk(features):

    # Random Forest probability
    rf_score = rf_model.predict_proba(features)[0][1]

    # SVM raw prediction
    svm_raw = svm_model.decision_function(features)[0]

    # Convert SVM score to probability
    svm_score = 1 / (1 + np.exp(-svm_raw))

    # Final cloud score
    cloud_score = (rf_score + svm_score) / 2

    if cloud_score > 0.5:
        result = "Malicious"
    else:
        result = "Safe"

    return {
        "result": result,
        "rf_score": float(rf_score),
        "svm_score": float(svm_score),
        "cloud_score": float(cloud_score)
    }