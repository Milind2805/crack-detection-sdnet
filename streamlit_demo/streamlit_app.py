"""Streamlit demo for the concrete crack classifier.

Uses the same inference code as the FastAPI service (app/model.py).
Deploy on Streamlit Community Cloud with this file as the entry point.
"""
import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                       # so `import app` works
os.environ.setdefault("MODEL_PATH", str(ROOT / "weights" / "crack_effnetb0.pt"))

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from app import model as M

Image.MAX_IMAGE_PIXELS = 60_000_000                 # refuse decompression bombs
MAX_UPLOAD_MB = 15
M.BATCH_SIZE = 8                                    # smaller batches keep memory low on free hosting

DISCLAIMER = (
    "Research demo trained on SDNET2018 tiles. Not suitable for real structural assessment. "
    "Accuracy drops on surface types and imaging conditions that are not in the training data "
    "(pavements in particular)."
)
MODES = {
    "balanced": "Balanced (best F1)",
    "high_recall": "High recall (finds more cracks, more false alarms)",
}

st.set_page_config(page_title="Concrete crack detector", page_icon="🧱")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return M.load_model()


st.title("Concrete crack detector")
st.write(
    "Upload a close-up photo of a concrete surface. It is cut into 256x256 px tiles and each "
    "tile is classified as cracked or not. Flagged tiles are outlined in red."
)

mode = st.radio("Operating point", list(MODES), format_func=MODES.get)
file = st.file_uploader("Photo (JPEG or PNG)", type=["jpg", "jpeg", "png"])

if file is not None:
    if file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Image is larger than {MAX_UPLOAD_MB} MB.")
        st.stop()
    try:
        img = Image.open(file)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("Could not read the uploaded file as an image.")
        st.stop()

    with st.spinner("Analysing, this can take a few seconds on CPU..."):
        get_model()
        img, notes = M.prepare_image(img)
        probs = M.predict_tiles(img)
        threshold = M.THRESHOLDS[mode]
        overlay = M.render_overlay(img, probs, threshold)

    flagged = int((probs >= threshold).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Tiles flagged", flagged)
    c2.metric("Tiles analysed", int(probs.size))
    c3.metric("Highest tile score", f"{float(probs.max()):.2f}", help="Model output between 0 and 1. Higher means more crack-like. It is a ranking score, not a calibrated probability.")

    st.image(base64.b64decode(overlay.split(",", 1)[1]), caption="Flagged tiles outlined in red")
    for note in notes:
        st.info(note)
    with st.expander("Tile scores (rows x columns)"):
        st.dataframe(np.round(probs, 3))

st.caption(DISCLAIMER)
