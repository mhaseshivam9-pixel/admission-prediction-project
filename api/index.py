import json
import logging
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "Final_Admission_Prediction_Model.pkl"

RAW_FEATURES = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("admission_prediction")

_MODEL = None


# ============================================================
# PHASE 6 FEATURE ENGINEERING
# 7 original features + 9 engineered features = 16 features
# ============================================================

def create_engineered_features(data):
    data = data.copy()

    data["GRE_TOEFL_Combined"] = (
        data["GRE Score"] + data["TOEFL Score"]
    )

    data["GRE_TOEFL_Average"] = (
        data["GRE Score"] + data["TOEFL Score"]
    ) / 2

    data["Application_Profile_Score"] = (
        data["University Rating"]
        + data["SOP"]
        + data["LOR"]
    ) / 3

    data["Academic_Strength"] = (
        data["CGPA"]
        + data["GRE Score"] / 34
        + data["TOEFL Score"] / 12
    ) / 3

    data["Research_CGPA_Interaction"] = (
        data["Research"] * data["CGPA"]
    )

    data["CGPA_University_Interaction"] = (
        data["CGPA"] * data["University Rating"]
    )

    data["SOP_LOR_Interaction"] = (
        data["SOP"] * data["LOR"]
    )

    data["GRE_CGPA_Interaction"] = (
        data["GRE Score"] * data["CGPA"]
    )

    data["TOEFL_CGPA_Interaction"] = (
        data["TOEFL Score"] * data["CGPA"]
    )

    return data


# ============================================================
# JSON RESPONSE HELPERS
# ============================================================

def make_json_response(
    status,
    message,
    status_code=200,
    **extra
):
    payload = {
        "status": status,
        "message": message,
        **extra,
    }

    return status_code, payload


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The model file is missing from the deployed project."
        )

    logger.info("Loading model from %s", MODEL_PATH)

    # Package versions must be compatible with those used
    # when the model was saved.
    _MODEL = joblib.load(MODEL_PATH)

    logger.info(
        "Model loaded successfully: %s",
        type(_MODEL).__name__,
    )

    logger.info(
        "Model expected feature count: %s",
        getattr(_MODEL, "n_features_in_", "unknown"),
    )

    return _MODEL


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [
        feature for feature in RAW_FEATURES
        if feature not in payload
    ]

    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
        )

    values = {}

    for feature in RAW_FEATURES:
        value = payload[feature]

        if isinstance(value, bool):
            raise ValueError(
                f"{feature} must be a numeric value."
            )

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a numeric value."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        values[feature] = value

    if not 260 <= values["GRE Score"] <= 340:
        raise ValueError(
            "GRE Score must be between 260 and 340."
        )

    if not 0 <= values["TOEFL Score"] <= 120:
        raise ValueError(
            "TOEFL Score must be between 0 and 120."
        )

    if values["University Rating"] not in [1, 2, 3, 4, 5]:
        raise ValueError(
            "University Rating must be an integer from 1 to 5."
        )

    if not 1 <= values["SOP"] <= 5:
        raise ValueError("SOP must be between 1 and 5.")

    if not 1 <= values["LOR"] <= 5:
        raise ValueError("LOR must be between 1 and 5.")

    if not 0 <= values["CGPA"] <= 10:
        raise ValueError("CGPA must be between 0 and 10.")

    if values["Research"] not in [0, 1]:
        raise ValueError("Research must be 0 or 1.")

    return values


# ============================================================
# PREDICTION
# ============================================================

def make_prediction(payload):
    values = validate_input(payload)

    raw_data = pd.DataFrame(
        [values],
        columns=RAW_FEATURES,
    )

    engineered_data = create_engineered_features(raw_data)

    trained_model = load_model()

    expected_count = getattr(
        trained_model,
        "n_features_in_",
        None,
    )

    if (
        expected_count is not None
        and expected_count != engineered_data.shape[1]
    ):
        raise RuntimeError(
            f"Feature-count mismatch: model expects "
            f"{expected_count} features, but the API generated "
            f"{engineered_data.shape[1]}."
        )

    expected_names = getattr(
        trained_model,
        "feature_names_in_",
        None,
    )

    if expected_names is not None:
        expected_names = list(expected_names)
        actual_names = list(engineered_data.columns)

        if expected_names != actual_names:
            raise RuntimeError(
                "Feature names or feature order do not match "
                "the trained model."
            )

    prediction = float(
        np.asarray(
            trained_model.predict(engineered_data)
        ).reshape(-1)[0]
    )

    if not np.isfinite(prediction):
        raise RuntimeError(
            "The model returned an invalid numeric prediction."
        )

    # Return multiple compatible field names so the frontend
    # can read the result using its existing response handling.
    return {
        "status": "success",
        "message": "Prediction generated successfully.",

        "prediction": prediction,
        "predicted_chance": prediction,
        "chance_of_admit": prediction,

        "predicted_percentage": prediction * 100,

        "interpretation": (
            "Model-generated estimate only; "
            "not an admission decision."
        ),
    }


# ============================================================
# VERCEL PYTHON HANDLER
# ============================================================

class handler(BaseHTTPRequestHandler):

    def send_json(self, status_code, payload):
        body = json.dumps(
            payload,
            allow_nan=False,
        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") != "/api":
            self.send_json(404, {
                "status": "error",
                "error": "Endpoint not found.",
                "message": "Endpoint not found.",
            })
            return

        self.send_json(200, {
            "status": "success",
            "message": "Admission Prediction API is running.",
            "model_file_exists": MODEL_PATH.is_file(),
            "model_loaded": _MODEL is not None,
        })

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") != "/api":
            self.send_json(404, {
                "status": "error",
                "error": "Endpoint not found.",
                "message": "Endpoint not found.",
            })
            return

        try:
            content_length = int(
                self.headers.get("Content-Length", "0")
            )

            if content_length <= 0 or content_length > 10000:
                self.send_json(400, {
                    "status": "error",
                    "error": "Invalid request size.",
                    "message": "Invalid request size.",
                })
                return

            raw_body = self.rfile.read(content_length)

            try:
                payload = json.loads(
                    raw_body.decode("utf-8")
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                self.send_json(400, {
                    "status": "error",
                    "error": "Request body must contain valid JSON.",
                    "message": "Request body must contain valid JSON.",
                })
                return

            result = make_prediction(payload)

            self.send_json(200, result)

        except ValueError as exc:
            self.send_json(400, {
                "status": "error",
                "error": str(exc),
                "message": str(exc),
            })

        except Exception as exc:
            logger.error(
                "PREDICTION ERROR: %s: %s",
                type(exc).__name__,
                str(exc),
            )

            logger.error(traceback.format_exc())

            # Include the exception type and message to help
            # diagnose deployment/model compatibility problems.
            diagnostic = (
                f"{type(exc).__name__}: {str(exc)}"
            )

            self.send_json(500, {
                "status": "error",
                "error": diagnostic,
                "message": diagnostic,
            })
