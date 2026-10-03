"""FastAPI service for the concrete crack classifier."""
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from . import model as M

Image.MAX_IMAGE_PIXELS = 60_000_000          # refuse decompression bombs
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"

DISCLAIMER = (
    "Research demo trained on SDNET2018 tiles. Not suitable for real structural "
    "assessment. Accuracy drops on surface types and imaging conditions that are "
    "not in the training data (pavements in particular)."
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    M.load_model()          # fail fast at startup if the weights are missing
    yield


app = FastAPI(
    title="Concrete crack detector",
    version="1.0.0",
    description="EfficientNet-B0 tile classifier fine-tuned on SDNET2018.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": M._model is not None}


@app.post("/predict")
def predict(
    file: UploadFile = File(..., description="JPEG or PNG photo of a concrete surface"),
    mode: str = Query("balanced", description="'balanced' (best F1) or 'high_recall' (>= 90% recall on validation)"),
    overlay: bool = Query(False, description="Also return an annotated image as a data URI"),
):
    if mode not in M.THRESHOLDS:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(M.THRESHOLDS)}")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 15 MB")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image")

    img, notes = M.prepare_image(img)
    probs = M.predict_tiles(img)
    threshold = M.THRESHOLDS[mode]
    flagged = int((probs >= threshold).sum())
    total = int(probs.size)

    result = {
        "mode": mode,
        "threshold": threshold,
        "image_size": list(img.size),
        "tile_size": M.TILE,
        "grid": {"rows": int(probs.shape[0]), "cols": int(probs.shape[1])},
        "tiles_total": total,
        "tiles_flagged": flagged,
        "flagged_fraction": round(flagged / total, 3),
        "max_probability": round(float(probs.max()), 3),
        "tile_probs": probs.round(3).tolist(),
        "notes": notes,
        "disclaimer": DISCLAIMER,
    }
    if overlay:
        result["overlay"] = M.render_overlay(img, probs, threshold)
    return result
