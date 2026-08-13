import joblib
import shap
import numpy as np


# ============================================================
# LOAD MODELS
# ============================================================

rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")

feature_names = joblib.load("feature_names.pkl")


# ============================================================
# SHAP EXPLAINER
# ============================================================

rf_explainer = shap.TreeExplainer(rf_model)


# ============================================================
# PREDICTION
# ============================================================

def predict_apk(features):

    rf_score = float(
        rf_model.predict_proba(features)[0][1]
    )

    svm_score = float(
        svm_model.predict_proba(features)[0][1]
    )

    final_score = (
        rf_score + svm_score
    ) / 2

    result = (
        "Malicious"
        if final_score > 0.5
        else "Safe"
    )

    return (
        result,
        final_score,
        rf_score,
        svm_score
    )


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

    shap_values = get_shap_values(
        features
    )

    # --------------------------------------------------------
    # SHAP supports different return formats
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        # Binary classifier:
        # class 0 = Safe
        # class 1 = Malicious

        values = shap_values[1]

    else:

        values = np.asarray(
            shap_values
        )

        # Newer SHAP versions may return:
        #
        # (samples, features, classes)

        if values.ndim == 3:

            values = values[:, :, 1]


    values = np.asarray(
        values
    )


    # Ensure shape is:
    #
    # (samples, features)

    if values.ndim == 1:

        values = values.reshape(
            1,
            -1
        )


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
    # Rank features by importance
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
# CONVERT TECHNICAL FEATURE NAME
# INTO USER-FRIENDLY TEXT
# ============================================================

def make_feature_friendly(feature):

    feature = str(feature).strip()

    feature_lower = feature.lower()


    # --------------------------------------------------------
    # Common Android permissions / features
    # --------------------------------------------------------

    friendly_names = {

        "read_phone_state":
            "access to phone information",

        "android.permission.read_phone_state":
            "access to phone information",

        "access_coarse_location":
            "approximate location access",

        "android.permission.access_coarse_location":
            "approximate location access",

        "access_fine_location":
            "precise location access",

        "android.permission.access_fine_location":
            "precise location access",

        "send_sms":
            "the ability to send SMS messages",

        "android.permission.send_sms":
            "the ability to send SMS messages",

        "read_sms":
            "the ability to read SMS messages",

        "android.permission.read_sms":
            "the ability to read SMS messages",

        "receive_sms":
            "the ability to receive SMS messages",

        "android.permission.receive_sms":
            "the ability to receive SMS messages",

        "write_sms":
            "the ability to modify SMS messages",

        "android.permission.write_sms":
            "the ability to modify SMS messages",

        "read_contacts":
            "access to contacts",

        "android.permission.read_contacts":
            "access to contacts",

        "write_contacts":
            "the ability to modify contacts",

        "android.permission.write_contacts":
            "the ability to modify contacts",

        "internet":
            "internet access",

        "android.permission.internet":
            "internet access",

        "getdeviceid":
            "device identification information",

        "dexclassloader":
            "dynamic code loading",

        "runtime.exec":
            "system command execution",

        "loadlibrary":
            "loading native code"

    }


    if feature_lower in friendly_names:

        return friendly_names[
            feature_lower
        ]


    # --------------------------------------------------------
    # Generic Android permission conversion
    # --------------------------------------------------------

    if feature_lower.startswith(
            "android.permission."
    ):

        clean_name = (
            feature_lower
            .replace(
                "android.permission.",
                ""
            )
            .replace(
                "_",
                " "
            )
        )

        return (
            clean_name
            + " permission"
        )


    # --------------------------------------------------------
    # Generic feature name conversion
    # --------------------------------------------------------

    clean_name = (
        feature
        .replace(
            "_",
            " "
        )
        .replace(
            ".",
            " "
        )
    )


    return clean_name.lower()


# ============================================================
# CREATE USER-FRIENDLY SHAP EXPLANATION
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


    # ========================================================
    # OVERALL RESULT
    # ========================================================

    if result == "Malicious":

        lines.append(
            "Why this app was marked as potentially harmful:"
        )

        lines.append("")

        lines.append(
            "The scan detected some behaviors "
            "that may be associated with malicious apps."
        )

    else:

        lines.append(
            "Why this app was marked as safe:"
        )

        lines.append("")

        lines.append(
            "The scan did not find strong signs "
            "of malicious behavior."
        )


    # ========================================================
    # FEATURE EXPLANATIONS
    # ========================================================

    if top_features:

        lines.append("")

        lines.append(
            "Some of the important behaviors detected were:"
        )


    displayed = 0


    for item in top_features:

        feature = item["feature"]

        shap_value = item["shap_value"]


        friendly_feature = (
            make_feature_friendly(
                feature
            )
        )


        # ----------------------------------------------------
        # Positive SHAP value
        # ----------------------------------------------------

        if shap_value > 0:

            if result == "Malicious":

                explanation = (
                    f"• The app shows "
                    f"{friendly_feature}. "
                    f"This behavior increased the "
                    f"security risk detected by the scan."
                )

            else:

                explanation = (
                    f"• The app shows "
                    f"{friendly_feature}. "
                    f"This behavior received some "
                    f"attention during the security scan."
                )


        # ----------------------------------------------------
        # Negative SHAP value
        # ----------------------------------------------------

        elif shap_value < 0:

            if result == "Malicious":

                explanation = (
                    f"• The app shows "
                    f"{friendly_feature}, "
                    f"but this behavior did not strongly "
                    f"increase the detected risk."
                )

            else:

                explanation = (
                    f"• The app shows "
                    f"{friendly_feature}. "
                    f"This behavior did not strongly "
                    f"suggest malicious activity."
                )


        # ----------------------------------------------------
        # Near-zero SHAP value
        # ----------------------------------------------------

        else:

            explanation = (
                f"• The app shows "
                f"{friendly_feature}, "
                f"but this had little effect on the "
                f"overall security assessment."
            )


        lines.append(
            explanation
        )


        displayed += 1


        if displayed >= top_n:

            break


    # ========================================================
    # FINAL USER-FRIENDLY MESSAGE
    # ========================================================

    lines.append("")


    if result == "Malicious":

        lines.append(
            "Please review the app carefully before "
            "using it, especially if it came from "
            "an untrusted source."
        )

    else:

        lines.append(
            "Overall, the detected behaviors did not "
            "strongly indicate that this app is malicious."
        )


    return "\n".join(
        lines
    )