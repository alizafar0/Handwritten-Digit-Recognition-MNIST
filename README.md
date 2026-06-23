# Handwritten Digit Recognition — MNIST (ANN)

A complete Artificial Neural Network project that recognises handwritten
digits (0–9) using the classic **MNIST** dataset (70,000 grayscale 28×28
images). Includes the trained model, a Flask web app, and a fully
**standalone `index.html`** that runs the model in the browser via
TensorFlow.js — no server needed.

Test accuracy: see `model/history.json` (~0.98).

## Repo layout

```
mnist-ann/
├── train.py            # trains the ANN on MNIST and exports all artifacts
├── app.py              # Flask web app (uses model/mnist_ann.h5)
├── index.html          # static web app (uses tfjs_model/) — open directly in a browser
├── templates/
│   └── index.html      # same UI, served by Flask
├── static/             # css/js assets used by the Flask template
├── model/
│   ├── mnist_ann.h5    # trained Keras model
│   ├── scaler.json     # input scaling info (pixels/255)
│   └── history.json    # training history + final test metrics
├── saved_model/        # TensorFlow SavedModel format
├── tfjs_model/         # TensorFlow.js model (model.json + shards)
├── requirements.txt
├── runtime.txt
└── README.md
```

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — loaded automatically via
`tf.keras.datasets.mnist.load_data()`. No manual download required.

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain
The repo ships with a pre-trained model. To retrain from scratch:
```bash
python train.py
```

### 3. Run the Flask app
```bash
python app.py
# open http://localhost:5000
```

### 4. Or just open `index.html`
Because the model is converted to TensorFlow.js, the standalone
`index.html` works without any backend. Note: browsers block `fetch()`
for `file://` URLs, so serve the folder with a tiny static server:
```bash
python -m http.server 8000
# open http://localhost:8000
```

## Model architecture (ANN)

```
Input (28x28) → Flatten → Dense(256, relu) → Dropout(0.2)
              → Dense(128, relu) → Dropout(0.2)
              → Dense(64,  relu) → Dense(10, softmax)
```
Optimizer: Adam (1e-3) · Loss: sparse categorical cross-entropy ·
Early stopping on val_accuracy (patience 3).

## API

`POST /api/predict`
```json
{ "image": "data:image/png;base64,iVBORw0K..." }
```
Returns `{ "prediction": 7, "confidence": 0.998, "probabilities": [...] }`.

## License
MIT
