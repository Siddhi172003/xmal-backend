import joblib
import shap
import numpy as np

rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")

feature_names = joblib.load("feature_names.pkl")


# ============================================================
# SHAP EXPLAINER
# ============================================================

rf_explainer = shap.TreeExplainer(rf_model)

def predict_apk(features):

    rf_score = float(
        rf_model.predict_proba(features)[0][1]
    )
    svm_score = float(
        svm_model.predict_proba(features)[0][1]
    )


    final_score = (rf_score + svm_score) / 2

    result = "Malicious" if final_score > 0.5 else "Safe"

    return result, final_score, rf_score, svm_score

    # ============================================================
# SHAP VALUE EXTRACTION
# ============================================================

def get_shap_values(features):

    """
    Calculate SHAP values for the Random Forest model.
    """

    shap_values = rf_explainer.shap_values(
        features
    )

    return shap_values


# ============================================================
# GET SHAP VALUES FOR MALICIOUS CLASS
# ============================================================

def get_malicious_shap_values(features):

    shap_values = get_shap_values(features)


    # --------------------------------------------------------
    # SHAP has changed its return format between versions.
    # Handle both old and new formats.
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        # Binary classifier:
        # class 0 = Safe
        # class 1 = Malicious

        values = shap_values[1]

    else:

        values = np.asarray(shap_values)


        # Newer SHAP versions may return:
        #
        # (samples, features, classes)
        #
        # For example:
        # (1, 215, 2)

        if values.ndim == 3:

            values = values[:, :, 1]


    values = np.asarray(values)


    # Ensure shape is:
    #
    # (samples, features)

    if values.ndim == 1:

        values = values.reshape(1, -1)


    return values[0]


# ============================================================
# TOP SHAP FEATURES
# ============================================================

def get_top_shap_features(
        features,
        top_n=5):

    shap_values = (
        get_malicious_shap_values(
            features
        )
    )


    # --------------------------------------------------------
    # Get absolute SHAP magnitude
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        np.abs(shap_values)
    )[::-1]


    top_features = []


    for index in ranked_indices[:top_n]:

        # ------------------------------------
        # Feature name
        # ------------------------------------

        if index < len(feature_names):

            name = str(
                feature_names[index]
            )

        else:

            name = f"Feature {index}"


        # ------------------------------------
        # SHAP value
        # ------------------------------------

        value = float(
            shap_values[index]
        )


        # ------------------------------------
        # Actual feature value
        # ------------------------------------

        feature_value = float(
            features[0][index]
        )


        top_features.append({

            "feature":
                name,

            "shap_value":
                value,

            "feature_value":
                feature_value

        })


    return top_features


# ============================================================
# CREATE HUMAN-READABLE SHAP EXPLANATION
# ============================================================

def create_shap_explanation(
        features,
        rf_score,
        result,
        top_n=5):

    top_features = (
        get_top_shap_features(
            features,
            top_n
        )
    )


    lines = []


    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    if result == "Malicious":

        lines.append(
            "The Random Forest model "
            "classified this application "
            "as potentially malicious."
        )

    else:

        lines.append(
            "The Random Forest model "
            "classified this application "
            "as likely safe."
        )


    lines.append(
        f"Random Forest malware probability: "
        f"{rf_score * 100:.2f}%."
    )


    # --------------------------------------------------------
    # SHAP feature explanation
    # --------------------------------------------------------

    if top_features:

        lines.append(
            "The main factors influencing "
            "the Random Forest prediction were:"
        )


    for item in top_features:

        feature = item["feature"]
        shap_value = item["shap_value"]
        feature_value = item["feature_value"]


        if shap_value > 0:

            direction = (
                "increased the malware prediction"
            )

        elif shap_value < 0:

            direction = (
                "decreased the malware prediction"
            )

        else:

            direction = (
                "had little influence on the prediction"
            )


        lines.append(

            f"• {feature}: "
            f"SHAP value {shap_value:.4f}; "
            f"this feature {direction}."

        )


    return "\n".join(lines)