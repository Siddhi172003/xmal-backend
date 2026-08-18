from flask import Flask, request, jsonify
import numpy as np
from translation_service import translate_text

import os
import json
import firebase_admin

from firebase_admin import credentials
from firebase_admin import messaging

from utils import (
    predict_apk,
    create_shap_explanation
)


firebase_service_account = os.environ.get(
    "FIREBASE_SERVICE_ACCOUNT"
)

if firebase_service_account:

    service_account_info = json.loads(
        firebase_service_account
    )

    cred = credentials.Certificate(
        service_account_info
    )

    firebase_admin.initialize_app(cred)


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

        print(
            "SCAN ERROR:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


@app.route(
    "/send-xmalguard-notification",
    methods=["POST"]
)
def send_xmalguard_notification():

    try:

        admin_key = request.headers.get(
            "X-Admin-Key"
        )

        expected_key = os.environ.get(
            "NOTIFICATION_ADMIN_KEY"
        )


        if admin_key != expected_key:

            return jsonify({
                "success": False,
                "error": "Unauthorized"
            }), 401


        data = request.get_json(
            silent=True
        ) or {}


        title = data.get(
            "title",
            "XMalGuard Security Alert"
        )


        body = data.get(
            "body",
            "A new security update is available."
        )


        message = messaging.Message(

            notification=messaging.Notification(

                title=title,

                body=body

            ),

            topic="xmalguard_updates"

        )


        response = messaging.send(
            message
        )


        return jsonify({

            "success": True,

            "message_id": response

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )