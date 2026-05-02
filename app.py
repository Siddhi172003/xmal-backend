from flask import Flask, request, jsonify
import numpy as np
from utils import predict_apk  # Your trained DL + RF + SVM ensemble

app = Flask(__name__)

# Optional homepage to check server in browser
@app.route("/", methods=["GET"])
def home():
    return "Android Malware Scanner API is running!"

# Scan endpoint (POST only)
@app.route("/scan", methods=["POST"])
def scan_apk():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400

    features = np.array(data["features"]).reshape(1, -1)

    result = predict_apk(features)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)