"""
Flask web app for the MNIST ANN.
Run: python app.py   ->  http://localhost:5000
"""
import io
import os
import json
import base64
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "mnist_ann.h5")
SCALER_PATH = os.path.join(ROOT, "model", "scaler.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH) as f:
    SCALER = json.load(f)


def preprocess(b64_png: str) -> np.ndarray:
    """Decode base64 PNG (28x28-ish, white digit on black or vice-versa),
    convert to grayscale 28x28 with digit=white on black, scale to [0,1]."""
    if "," in b64_png:
        b64_png = b64_png.split(",", 1)[1]
    raw = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(raw)).convert("L")

    # If background looks white, invert (MNIST uses white digit on black bg)
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32") * SCALER["scale"] + SCALER["offset"]
    return arr.reshape(1, 28, 28)


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/static/<path:p>")
def static_files(p):
    return send_from_directory("static", p)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "missing 'image' (base64 PNG)"}), 400
    x = preprocess(img_b64)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    return jsonify({
        "prediction": pred,
        "confidence": float(probs[pred]),
        "probabilities": [float(p) for p in probs],
    })


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
