from flask import Flask, request, jsonify
import numpy as np
from translation_service import translate_text

from utils import (
    predict_apk,
    create_shap_explanation
)

app = Flask(__name__)


@app.route("/")
def home():
    return "Android Malware Scanner API is running!"


@app.route("/scan", methods=["POST"])
def scan_apk():
    try:
        # ==================================================
        # READ JSON
        # ==================================================

        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({
                "error": "Missing features"
            }), 400

        # ==================================================
        # FEATURES
        # ==================================================

        features = np.array(
            data["features"],
            dtype=np.float32
        ).reshape(1, -1)

        print(
            "Incoming feature shape:",
            features.shape
        )

        # ==================================================
        # LANGUAGE
        # ==================================================

        target_language = data.get(
            "target_language",
            "en"
        )

        # ==================================================
        # PREDICTION
        # ==================================================

        (
            result,
            final_score,
            rf_score,
            svm_score
        ) = predict_apk(features)

        # ==================================================
        # REAL SHAP EXPLANATION
        # ==================================================

        english_explanation = create_shap_explanation(
            features,
            rf_score,
            result,
            top_n=5
        )

        # ==================================================
        # TRANSLATION
        # ==================================================

        translated_explanation = translate_text(
            english_explanation,
            target_language
        )

        # ==================================================
        # RESPONSE
        # ==================================================

        return jsonify({
            "result": result,
            "rf_score": float(rf_score),
            "svm_score": float(svm_score),
            "cloud_score": float(final_score),
            "language": target_language,
            "explanation": translated_explanation
        })

    except Exception as e:
        print("SCAN ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )