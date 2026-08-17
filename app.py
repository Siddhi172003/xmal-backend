from flask import Flask, request, jsonify
import numpy as np
import requests

from utils import predict_apk

app = Flask(__name__)

MYMEMORY_URL = "https://api.mymemory.translated.net/get"


def translate_text(text, target_language):

    if not text:
        return text

    if not target_language or target_language == "en":
        return text

    try:
        response = requests.get(
            MYMEMORY_URL,
            params={
                "q": text,
                "langpair": f"en|{target_language}"
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        translated_text = (
            data
            .get("responseData", {})
            .get("translatedText")
        )

        if translated_text:
            return translated_text

        return text

    except Exception as e:
        print("Translation error:", str(e))
        return text


@app.route("/")
def home():
    return "Android Malware Scanner API is running!"


@app.route("/scan", methods=["POST"])
def scan_apk():

    try:

        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({
                "error": "Missing features"
            }), 400

        features = np.array(
            data["features"]
        ).reshape(1, -1)

        target_language = data.get(
            "target_language",
            "en"
        )

        result, final_score, rf_score, svm_score = predict_apk(
            features
        )

        # Explanation text
        if result == "Malicious":
            explanation = (
                "This application is classified as malicious. "
                "The machine learning models detected suspicious "
                "features that may indicate malware."
            )
        else:
            explanation = (
                "This application is classified as safe. "
                "The machine learning models did not detect "
                "significant malicious behavior."
            )

        # Translate explanation
        translated_explanation = translate_text(
            explanation,
            target_language
        )

        return jsonify({

            "result": result,

            "rf_score": float(rf_score),

            "svm_score": float(svm_score),

            "cloud_score": float(final_score),

            "explanation": translated_explanation,

            "target_language": target_language

        })

    except Exception as e:

        print("SCAN ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500