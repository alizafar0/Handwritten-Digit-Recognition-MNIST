"""
Streamlit app for Handwritten Digit Recognition (MNIST ANN).
Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub, then deploy on https://share.streamlit.io
"""
import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

# ---------- Page config ----------
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="✏️",
    layout="centered",
)

MODEL_PATH = os.path.join("model", "mnist_ann.h5")


# ---------- Model loader (cached) ----------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ---------- Image preprocessing ----------
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Convert any input image to the 28x28 grayscale format the ANN expects."""
    img = pil_img.convert("L")                 # grayscale
    # MNIST is white digits on black background. If the user's image is
    # dark-on-light (common for uploads / canvas), invert it.
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    img = ImageOps.fit(img, (28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 784)


def predict(pil_img: Image.Image):
    model = load_model()
    x = preprocess(pil_img)
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    return digit, confidence, probs


# ---------- UI ----------
st.title("✏️ Handwritten Digit Recognition")
st.caption("MNIST ANN · 784 → 256 → 128 → 64 → 10 · ~97.8% test accuracy")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at `{MODEL_PATH}`.\n\n"
        "Run `python train.py` first, or make sure `model/mnist_ann.h5` is "
        "committed to the repo before deploying."
    )
    st.stop()

tab_draw, tab_upload = st.tabs(["🖌️ Draw a digit", "📤 Upload an image"])

# --- Draw tab ---
with tab_draw:
    try:
        from streamlit_drawable_canvas import st_canvas
        st.write("Draw a single digit (0–9) in the box below.")
        canvas = st_canvas(
            fill_color="#000000",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas.image_data is not None and st.button("Predict drawing", type="primary"):
            img = Image.fromarray((canvas.image_data[:, :, :3]).astype("uint8"))
            digit, conf, probs = predict(img)
            st.success(f"Prediction: **{digit}**  (confidence {conf*100:.1f}%)")
            st.bar_chart({"probability": probs.tolist()})
    except ImportError:
        st.info(
            "Drawing canvas requires `streamlit-drawable-canvas`. "
            "It's already listed in `requirements.txt` — install with "
            "`pip install -r requirements.txt`."
        )

# --- Upload tab ---
with tab_upload:
    file = st.file_uploader("Upload a PNG/JPG image of a digit", type=["png", "jpg", "jpeg"])
    if file is not None:
        img = Image.open(file)
        st.image(img, caption="Input", width=200)
        if st.button("Predict upload", type="primary"):
            digit, conf, probs = predict(img)
            st.success(f"Prediction: **{digit}**  (confidence {conf*100:.1f}%)")
            st.bar_chart({"probability": probs.tolist()})

with st.expander("About this model"):
    st.markdown(
        "- Dataset: **MNIST** (60k train / 10k test)\n"
        "- Architecture: Dense ANN with dropout 0.2\n"
        "- Optimizer: Adam, early stopping\n"
        "- Test accuracy: **97.82%**\n"
    )
