"""Local Flask API serving the SuperKart pipeline.

Bound to 127.0.0.1 only: this is a local-first service, not exposed to the network.
Run the training first (`uv run python -m src.train`) so the model artifact exists.
"""

import sys
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

# Make `src` importable when launched from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import features_from_request  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "superkart_pipeline.pkl"

app = Flask(__name__)
CORS(app)

_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/")
def health():
    return jsonify(
        status="healthy",
        service="SuperKart Sales Forecasting (local)",
        model_loaded=_bundle is not None,
    )


@app.post("/predict")
def predict():
    if _bundle is None:
        return jsonify(error="model not trained; run `uv run python -m src.train`", status="failed"), 503
    try:
        X = features_from_request(request.get_json(force=True))
        pred = float(_bundle["pipeline"].predict(X)[0])
        return jsonify(predicted_sales=round(pred, 2), currency="INR", status="success")
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc), status="failed"), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
