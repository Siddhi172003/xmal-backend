from flask import Flask, request, jsonify
import numpy as np
from translation_service import translate_text
import os
import json
import time
import requests

import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

from utils import (
    predict_apk,
    create_shap_explanation
)


# ============================================================
# FIREBASE CONFIGURATION
# ============================================================

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


# ============================================================
# EXISTING HOME ENDPOINT
# ============================================================

@app.route("/")
def home():

    return "Android Malware Scanner API is running!"


# ============================================================
# EXISTING APK SCANNER
# CODE 2 FUNCTIONALITY KEPT AS-IS
# ============================================================

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

            "result":
                result,

            "rf_score":
                float(rf_score),

            "svm_score":
                float(svm_score),

            "cloud_score":
                float(final_score),

            "language":
                target_language,

            "explanation":
                translated_explanation

        })


    except Exception as e:

        print(
            "SCAN ERROR:",
            str(e)
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# PDF / FILE SCANNER
# ADDED FROM CODE 1
# ============================================================

VIRUSTOTAL_API_KEY = os.environ.get(
    "VIRUSTOTAL_API_KEY"
)

VIRUSTOTAL_UPLOAD_URL = (
    "https://www.virustotal.com/api/v3/files"
)


@app.route("/scan-file", methods=["POST"])
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY")
VIRUSTOTAL_UPLOAD_URL = "https://www.virustotal.com/api/v3/files"


@app.route("/scan-file", methods=["POST"])
def scan_file():
    try:
        if not VIRUSTOTAL_API_KEY:
            return jsonify({
                "malicious": False,
                "message": "VirusTotal API key is not configured",
                "details": "Configure VIRUSTOTAL_API_KEY on Render."
            }), 500

        # Check whether a file was uploaded
        if "file" not in request.files:
            return jsonify({
                "malicious": False,
                "message": "No file received",
                "details": "Please upload a PDF file."
            }), 400

        uploaded_file = request.files["file"]

        # Check filename
        if not uploaded_file.filename:
            return jsonify({
                "malicious": False,
                "message": "No filename provided",
                "details": "Please select a PDF file."
            }), 400

        filename = uploaded_file.filename

        # Only allow PDF files
        if not filename.lower().endswith(".pdf"):
            return jsonify({
                "malicious": False,
                "message": "Invalid file type",
                "details": "Only PDF files are supported."
            }), 400

        # Read file
        file_bytes = uploaded_file.read()

        # Maximum file size = 32 MB
        max_size = 32 * 1024 * 1024

        if len(file_bytes) > max_size:
            return jsonify({
                "malicious": False,
                "message": "File too large",
                "details": "Maximum PDF size is 32 MB."
            }), 413

        # Empty file check
        if len(file_bytes) == 0:
            return jsonify({
                "malicious": False,
                "message": "Empty file",
                "details": "The uploaded PDF is empty."
            }), 400

        # VirusTotal headers
        headers = {
            "x-apikey": VIRUSTOTAL_API_KEY
        }

        # Prepare PDF for VirusTotal
        files = {
            "file": (
                filename,
                file_bytes,
                "application/pdf"
            )
        }

        print("Uploading PDF to VirusTotal:", filename)

        # Upload only — DO NOT wait for analysis here
        upload_response = requests.post(
            VIRUSTOTAL_UPLOAD_URL,
            headers=headers,
            files=files,
            timeout=60
        )

        print(
            "VirusTotal upload response:",
            upload_response.status_code
        )

        if upload_response.status_code not in [200, 201]:
            print(
                "VirusTotal upload error:",
                upload_response.text
            )

            return jsonify({
                "malicious": False,
                "message": "VirusTotal upload failed",
                "details": upload_response.text
            }), 502

        upload_data = upload_response.json()

        # Get VirusTotal analysis ID
        analysis_id = upload_data["data"]["id"]

        print(
            "VirusTotal analysis ID:",
            analysis_id
        )

        # Return immediately.
        # We DO NOT wait for VirusTotal here.
        return jsonify({
            "malicious": False,
            "message": "PDF uploaded successfully",
            "status": "processing",
            "analysis_id": analysis_id,
            "filename": filename,
            "details": "VirusTotal is analyzing the PDF. Use the analysis_id with /scan-file-status/ to check the result."
        }), 202

    except Exception as e:

        print(
            "FILE UPLOAD ERROR:",
            str(e)
        )

        return jsonify({
            "malicious": False,
            "message": "File upload failed",
            "details": str(e)
        }), 500


# ---------------------------------------------------------
# CHECK VIRUSTOTAL ANALYSIS STATUS
# ---------------------------------------------------------

@app.route("/scan-file-status/<analysis_id>", methods=["GET"])
def scan_file_status(analysis_id):

    try:

        if not VIRUSTOTAL_API_KEY:
            return jsonify({
                "malicious": False,
                "message": "VirusTotal API key is not configured",
                "details": "Configure VIRUSTOTAL_API_KEY on Render."
            }), 500

        headers = {
            "x-apikey": VIRUSTOTAL_API_KEY
        }

        analysis_url = (
            "https://www.virustotal.com/api/v3/analyses/"
            + analysis_id
        )

        print(
            "Checking VirusTotal analysis:",
            analysis_id
        )

        # Check ONCE only
        analysis_response = requests.get(
            analysis_url,
            headers=headers,
            timeout=30
        )

        if analysis_response.status_code != 200:

            print(
                "VirusTotal status error:",
                analysis_response.text
            )

            return jsonify({
                "malicious": False,
                "message": "Unable to check VirusTotal analysis",
                "details": analysis_response.text,
                "analysis_id": analysis_id
            }), 502

        analysis_data = analysis_response.json()

        attributes = analysis_data["data"]["attributes"]

        status = attributes.get("status")

        print(
            "VirusTotal analysis status:",
            status
        )

        # Still processing
        if status != "completed":

            return jsonify({
                "malicious": False,
                "message": "Scan still processing",
                "status": status,
                "analysis_id": analysis_id,
                "details": "VirusTotal has not completed the analysis yet."
            }), 202

        # Analysis completed
        stats = attributes.get("stats", {})

        malicious_count = stats.get(
            "malicious",
            0
        )

        suspicious_count = stats.get(
            "suspicious",
            0
        )

        total_count = sum(
            stats.values()
        )

        is_malicious = (
            malicious_count > 0
            or suspicious_count > 0
        )

        if is_malicious:

            return jsonify({
                "malicious": True,
                "message": "Malware detected in PDF",
                "status": "completed",
                "analysis_id": analysis_id,
                "details": (
                    f"Malicious detections: "
                    f"{malicious_count}\n"
                    f"Suspicious detections: "
                    f"{suspicious_count}\n"
                    f"Total engines: "
                    f"{total_count}"
                )
            })

        else:

            return jsonify({
                "malicious": False,
                "message": "No malware detected",
                "status": "completed",
                "analysis_id": analysis_id,
                "details": (
                    f"Malicious detections: "
                    f"{malicious_count}\n"
                    f"Suspicious detections: "
                    f"{suspicious_count}\n"
                    f"Total engines: "
                    f"{total_count}"
                )
            })

    except Exception as e:

        print(
            "FILE STATUS ERROR:",
            str(e)
        )

        return jsonify({
            "malicious": False,
            "message": "Unable to check scan status",
            "details": str(e),
            "analysis_id": analysis_id
        }), 500

# ============================================================
# FIREBASE NOTIFICATION
# EXISTING CODE 2 FUNCTIONALITY
# ============================================================

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

                "error":
                    "Unauthorized"

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

            "message_id":
                response

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )