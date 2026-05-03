import joblib

rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")


def predict_apk(features):

    rf_score = rf_model.predict_proba(features)[0][1]
    svm_score = svm_model.predict_proba(features)[0][1]

    final_score = (rf_score + svm_score) / 2

    result = "Malicious" if final_score > 0.5 else "Safe"

    return result, final_score, rf_score, svm_score