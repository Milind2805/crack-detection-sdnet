"""Model loading and tile-wise inference for the concrete crack classifier.

The model was trained on 256x256 px tiles (SDNET2018), so any uploaded photo is
cut into non-overlapping 256x256 tiles and each tile is classified separately.
"""
import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models
from torchvision import transforms as T

TILE = 256
MAX_TILES = 200
PAD_MIN = 64   # leftover edge strips thinner than this are dropped; thicker ones are padded          # cap on tiles per request, keeps CPU latency reasonable
BATCH_SIZE = 32
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Operating points chosen on the validation set (see README):
#   balanced    -> threshold that maximised F1
#   high_recall -> highest threshold that still gave >= 90% recall
THRESHOLDS = {"balanced": 0.666, "high_recall": 0.127}

MODEL_PATH = Path(os.getenv("MODEL_PATH", "weights/crack_effnetb0.pt"))

_to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
_model = None


def load_model() -> nn.Module:
    """Load the EfficientNet-B0 checkpoint once and keep it in memory (CPU)."""
    global _model
    if _model is None:
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(1280, 1)
        state = torch.load(MODEL_PATH, map_location="cpu")
        net.load_state_dict(state)
        _model = net.eval()
    return _model


def prepare_image(img: Image.Image):
    """Convert to RGB and fit the image to a whole number of 256x256 tiles.

    Edge strips of at least PAD_MIN px are kept by reflect-padding them up to a full tile.
    Thinner strips are dropped, because a tile that is mostly mirrored padding adds noise.
    Returns the prepared image and a list of human-readable notes about any changes.
    """
    notes = []
    img = img.convert("RGB")
    w, h = img.size

    if min(w, h) < TILE:
        s = TILE / min(w, h)
        img = img.resize((max(TILE, round(w * s)), max(TILE, round(h * s))), Image.LANCZOS)
        notes.append("Image was upscaled so that it contains at least one 256x256 tile.")

    w, h = img.size
    n_tiles = -(-w // TILE) * -(-h // TILE)          # ceil division
    if n_tiles > MAX_TILES:
        s = (MAX_TILES / n_tiles) ** 0.5
        img = img.resize((max(TILE, int(w * s)), max(TILE, int(h * s))), Image.LANCZOS)
        notes.append(
            f"Image was downscaled to limit processing to {MAX_TILES} tiles; "
            "its scale now differs from the training data, so treat results with extra caution."
        )

    # fit width and height to a multiple of TILE (pad thick leftovers, drop thin ones)
    w, h = img.size

    def target(n):
        full, rem = divmod(n, TILE)
        return (full + 1) * TILE if rem >= PAD_MIN else full * TILE

    tw, th = target(w), target(h)
    cw, ch = min(w, tw), min(h, th)
    if (cw, ch) != (w, h):
        img = img.crop((0, 0, cw, ch))
    if (tw, th) != (cw, ch):
        arr = np.pad(np.asarray(img), ((0, th - ch), (0, tw - cw), (0, 0)), mode="reflect")
        img = Image.fromarray(arr)
        notes.append(
            "The right/bottom edge was padded with a mirrored copy of the image so that "
            "no part of the photo is skipped. Tiles on that edge contain mirrored content."
        )
    return img, notes

@torch.inference_mode()
def predict_tiles(img: Image.Image) -> np.ndarray:
    """Return a (rows, cols) array with the crack probability of every full tile."""
    net = load_model()
    w, h = img.size
    cols, rows = w // TILE, h // TILE

    tensors = []
    for r in range(rows):
        for c in range(cols):
            box = (c * TILE, r * TILE, (c + 1) * TILE, (r + 1) * TILE)
            tensors.append(_to_tensor(img.crop(box)))

    probs = []
    for i in range(0, len(tensors), BATCH_SIZE):
        batch = torch.stack(tensors[i:i + BATCH_SIZE])
        probs.append(torch.sigmoid(net(batch).squeeze(1)))
    return torch.cat(probs).numpy().reshape(rows, cols)


def render_overlay(img: Image.Image, probs: np.ndarray, threshold: float) -> str:
    """Draw red boxes on flagged tiles and return the result as a JPEG data URI."""
    rows, cols = probs.shape
    base = img.crop((0, 0, cols * TILE, rows * TILE)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for r in range(rows):
        for c in range(cols):
            p = float(probs[r, c])
            if p >= threshold:
                strength = min(1.0, (p - threshold) / max(1e-6, 1.0 - threshold))
                box = (c * TILE, r * TILE, (c + 1) * TILE - 1, (r + 1) * TILE - 1)
                draw.rectangle(box, fill=(220, 30, 30, int(60 + 120 * strength)),
                               outline=(220, 30, 30, 255), width=3)

    out = Image.alpha_composite(base, layer).convert("RGB")
    out.thumbnail((1400, 1400))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
