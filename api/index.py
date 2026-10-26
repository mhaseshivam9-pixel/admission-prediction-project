import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from http.server import BaseHTTPRequestHandler


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

MODEL = None


def load_model():
    global MODEL

    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Trained model file not found."
            )

        MODEL = joblib.load(MODEL_PATH)

    return MODEL


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
        + (data["GRE Score"] / 34)
        + (data["TOEFL Score"] / 12)
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


def validate_input(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "Request body must be a JSON object."
        )

    missing = [
        name for name in RAW_FEATURES
        if name not in payload
    ]

    if missing:
        raise ValueError(
            f"Missing required fields: {missing}"
        )

    values = {}

    for name in RAW_FEATURES:
        value = payload[name]

        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be numeric."
            )

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be numeric."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be a finite number."
            )

        values[name] = value

    if not 260 <= values["GRE Score"] <= 340:
        raise ValueError("GRE Score must be between 260 and 340.")

    if not 0 <= values["TOEFL Score"] <= 120:
        raise ValueError("TOEFL Score must be between 0 and 120.")

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


class handler(BaseHTTPRequestHandler):

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/api":
            self.send_json(200, {
                "status": "success",
                "message": "Admission Prediction API is running.",
                "model_loaded": MODEL is not None
            })
        else:
            self.send_json(404, {
                "status": "error",
                "message": "Endpoint not found."
            })

    def do_POST(self):
        if self.path.rstrip("/") != "/api":
            self.send_json(404, {
                "status": "error",
                "message": "Endpoint not found."
            })
            return

        try:
            length = int(
                self.headers.get("Content-Length", "0")
            )

            if length <= 0 or length > 10000:
                self.send_json(400, {
                    "status": "error",
                    "message": "Invalid request size."
                })
                return

            payload = json.loads(
                self.rfile.read(length).decode("utf-8")
            )

            values = validate_input(payload)

            raw_data = pd.DataFrame(
                [values],
                columns=RAW_FEATURES
            )

            engineered_data = create_engineered_features(
                raw_data
            )

            trained_model = load_model()

            expected_count = getattr(
                trained_model,
                "n_features_in_",
                None
            )

            if (
                expected_count is not None
                and expected_count != engineered_data.shape[1]
            ):
                raise RuntimeError(
                    "Model feature count does not match "
                    "the engineered input."
                )

            if hasattr(
                trained_model,
                "feature_names_in_"
            ):
                expected_names = list(
                    trained_model.feature_names_in_
                )

                if expected_names != list(
                    engineered_data.columns
                ):
                    raise RuntimeError(
                        "Model feature names/order do not match."
                    )

            prediction = float(
                trained_model.predict(engineered_data)[0]
            )

            self.send_json(200, {
                "status": "success",
                "predicted_chance": prediction,
                "predicted_percentage": prediction * 100,
                "interpretation": (
                    "Model estimate only; not an admission decision."
                )
            })

        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {
                "status": "error",
                "message": str(exc)
            })

        except Exception:
            self.send_json(500, {
                "status": "error",
                "message": (
                    "Prediction failed. Check Vercel runtime logs."
                )
            })
