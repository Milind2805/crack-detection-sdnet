import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

WEIGHTS = Path("weights/crack_effnetb0.pt")
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="model weights not available")


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:        # entering the context runs the startup (model load)
        yield c


def _jpeg(width, height):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_single_tile(client):
    r = client.post("/predict", files={"file": ("tile.jpg", _jpeg(256, 256), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["grid"] == {"rows": 1, "cols": 1}
    assert 0.0 <= body["max_probability"] <= 1.0


def test_tile_grid(client):
    r = client.post("/predict?overlay=true",
                    files={"file": ("photo.jpg", _jpeg(600, 520), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200
    assert body["grid"] == {"rows": 2, "cols": 2}
    assert body["tiles_total"] == 4
    assert body["overlay"].startswith("data:image/jpeg;base64,")


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_rejects_unknown_mode(client):
    r = client.post("/predict?mode=nope", files={"file": ("t.jpg", _jpeg(256, 256), "image/jpeg")})
    assert r.status_code == 422
