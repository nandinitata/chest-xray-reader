"""Tests for the reading pipeline: real film reads, junk gets refused."""
import io
from pathlib import Path

import numpy as np
from PIL import Image

import xray

FIXTURE = Path(__file__).parent / "fixtures" / "sample_chest.png"


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_reads_a_real_chest_xray():
    result = xray.read_xray(FIXTURE.read_bytes())
    assert result.is_chest_xray is True
    assert result.ood_reason is None
    assert result.top_label is not None
    assert result.heatmap_png and len(result.heatmap_png) > 1000
    # every known label got a probability
    assert len(result.all_scores) > 10
    assert all(0.0 <= v <= 1.0 for v in result.all_scores.values())


def test_findings_respect_threshold():
    low = xray.read_xray(FIXTURE.read_bytes(), report_threshold=0.1)
    high = xray.read_xray(FIXTURE.read_bytes(), report_threshold=0.99)
    assert len(low.findings) >= len(high.findings)
    assert high.findings == []  # nothing is that certain


def test_rejects_a_flat_image():
    flat = np.full((256, 256), 128, dtype=np.uint8)  # one solid gray, no detail
    result = xray.read_xray(_png_bytes(flat))
    assert result.is_chest_xray is False
    assert "flat" in (result.ood_reason or "").lower()


def test_rejects_a_tiny_image():
    tiny = (np.random.rand(50, 50) * 255).astype(np.uint8)
    result = xray.read_xray(_png_bytes(tiny))
    assert result.is_chest_xray is False
    assert "small" in (result.ood_reason or "").lower()
