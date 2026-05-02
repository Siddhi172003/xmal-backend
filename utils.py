import joblib
import numpy as np
import shap

rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")
feature_names = joblib.load("feature_names.pkl")

explainer = shap.TreeExplainer(rf_model)

def predict_apk(features):

    rf_score = rf_model.predict_proba(features)[0][1]
    svm_score = svm_model.predict_proba(features)[0][1]

    final_score = (rf_score + svm_score) / 2

    result = "Malicious" if final_score > 0.5 else "Safe"

    # SHAP explanation
    shap_values = explainer.shap_values(features)

    feature_importance = []

    for i in range(len(features[0])):
        if features[0][i] == 1:
            impact = abs(shap_values[1][0][i])

            feature_importance.append(
                (feature_names[i], impact)
            )

    feature_importance.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_features = [
        f[0] for f in feature_importance[:5]
    ]

    return result, final_score, rf_score, svm_score, top_features