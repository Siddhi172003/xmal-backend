from flask import Flask, request, jsonify
import numpy as np
from utils import predict_apk

app = Flask(__name__)

@app.route("/")
def home():
    return "Android Malware Scanner API is running!"

@app.route("/scan", methods=["POST"])
def scan_apk():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing features"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        print("Incoming feature shape:", features.shape)
        print("Incoming features:", features)

        result, final_score, rf_score, svm_score = predict_apk(features)

        return jsonify({
            "result": result,
            "rf_score": float(rf_score),
            "svm_score": float(svm_score),
            "cloud_score": float(final_score)
        })

    except Exception as e:
        print("SCAN ERROR:", str(e))
        return jsonify({
            "error": str(e)
        }), 500