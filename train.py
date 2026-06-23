"""
Handwritten Digit Recognition - MNIST ANN
Trains a fully-connected Artificial Neural Network on the MNIST dataset
and exports:
  - model/mnist_ann.h5         (Keras H5 model)
  - saved_model/                (TensorFlow SavedModel)
  - tfjs_model/                 (TensorFlow.js model for the static index.html)
  - model/scaler.json           (input scaling info: divide pixels by 255)
  - model/history.json          (training history)
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(OUT_DIR, "model")
SAVED_DIR = os.path.join(OUT_DIR, "saved_model")
TFJS_DIR = os.path.join(OUT_DIR, "tfjs_model")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading MNIST dataset (via keras.datasets)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    # Scale pixels to [0,1] - this is our "scaler"
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = build_model()
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                monitor="val_accuracy"),
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=cb,
        verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}   Test loss: {loss:.4f}")

    # ---- Save artifacts ----
    h5_path = os.path.join(MODEL_DIR, "mnist_ann.h5")
    model.save(h5_path)
    print(f"Saved Keras H5  -> {h5_path}")

    model.export(SAVED_DIR)
    print(f"Saved SavedModel -> {SAVED_DIR}")

    # scaler.json (simple normalization metadata for the inference code)
    with open(os.path.join(MODEL_DIR, "scaler.json"), "w") as f:
        json.dump({"type": "min_max", "scale": 1.0 / 255.0, "offset": 0.0,
                   "input_shape": [28, 28], "num_classes": 10}, f, indent=2)

    # history
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist["test_accuracy"] = float(acc)
    hist["test_loss"] = float(loss)
    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    # Convert to TensorFlow.js so the static index.html can run in-browser
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, TFJS_DIR)
        print(f"Saved TFJS model -> {TFJS_DIR}")
    except Exception as e:
        print(f"[warn] TFJS export skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
