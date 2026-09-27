"""Tests for the HTTP layer. Uses a high threshold so it doesn't need Ollama."""
from pathlib import Path

from fastapi.testclient import TestClient

import main

FIXTURE = Path(__file__).parent / "fixtures" / "sample_chest.png"
client = TestClient(main.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_returns_a_reading():
    # threshold 0.99 -> no findings -> template report, no model call, fast + deterministic
    with FIXTURE.open("rb") as f:
        r = client.post(
            "/analyze",
            files={"image": ("sample_chest.png", f, "image/png")},
            params={"threshold": 0.99},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["is_chest_xray"] is True
    assert body["findings"] == []
    assert body["heatmap_data_uri"].startswith("data:image/png;base64,")
    assert body["report"]["refused"] is False
    assert body["report"]["source"] == "template"
    assert len(body["all_scores"]) > 10


def test_analyze_rejects_junk():
    import io
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((256, 256), 128, dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/analyze", files={"image": ("flat.png", buf, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["is_chest_xray"] is False
    assert body["report"]["refused"] is True


def test_empty_upload_is_rejected():
    r = client.post("/analyze", files={"image": ("empty.png", b"", "image/png")})
    assert r.status_code == 400
