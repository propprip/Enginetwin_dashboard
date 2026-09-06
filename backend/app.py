"""
EngineTwin AI - Live inference backend
Loads the trained RandomForest RUL model and exposes it over a simple API,
so new sensor readings can be sent in and get a real prediction back.

Run locally:
    pip install -r requirements.txt
    python app.py
Then POST to http://localhost:5000/predict

Deploy free: Render.com, Railway.app, or Replit (see README_DEPLOY.md)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # allow the dashboard (served from a different origin) to call this API

model = joblib.load("engine_rul_model.joblib")
meta = joblib.load("model_meta.joblib")
FEATURE_COLS = meta["feature_cols"]
ENGINEERED_COLS = meta["engineered_cols"]
RUL_CAP = meta["rul_cap"]


def health_status(rul):
    if rul > 80:
        return "healthy"
    elif rul > 30:
        return "warning"
    return "critical"


def build_features(readings):
    """
    readings: list of dicts, each with keys matching FEATURE_COLS
              (op1, op2, s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15,
               s17, s20, s21 -- i.e. all sensors except the near-constant ones)
              Ordered oldest -> newest. Needs at least 1 reading; more readings
              (ideally 5+) give better rolling-window features.
    Returns a single-row DataFrame of engineered features for the LATEST reading.
    """
    df = pd.DataFrame(readings)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor fields: {missing}")

    window = min(5, len(df))
    out = {}
    for c in FEATURE_COLS:
        out[c] = df[c].iloc[-1]
        out[f"{c}_rollmean"] = df[c].tail(window).mean()
        out[f"{c}_rollstd"] = df[c].tail(window).std() if window > 1 else 0.0
    return pd.DataFrame([out])[ENGINEERED_COLS]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body: { "readings": [ {sensor readings for cycle 1}, {cycle 2}, ... ] }
    Returns the predicted RUL and health status for the most recent cycle.
    """
    body = request.get_json(force=True)
    readings = body.get("readings")
    if not readings or not isinstance(readings, list):
        return jsonify({"error": "Body must include a non-empty 'readings' list"}), 400

    try:
        X = build_features(readings)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    pred = float(model.predict(X)[0])
    pred = max(0.0, min(RUL_CAP, pred))
    return jsonify({
        "predicted_rul": round(pred, 1),
        "status": health_status(pred),
        "rul_cap": RUL_CAP
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
