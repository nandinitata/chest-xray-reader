"""The web API for the chest X-ray reader.

One real endpoint: POST /analyze. Send an image, get back the findings, a
heat-map, and a short report. Runs fully locally.
"""
from __future__ import annotations

import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import xray
import report

app = FastAPI(title="Chest X-ray Reader", version="0.1.0")

# The Vite dev server runs on 5173. Loosen this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FindingOut(BaseModel):
    label: str
    plain: str
    probability: float


class AnalyzeResponse(BaseModel):
    is_chest_xray: bool
    ood_reason: str | None
    top_label: str | None
    findings: list[FindingOut]
    all_scores: dict[str, float]
    warnings: list[str]
    heatmap_data_uri: str | None
    report: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(xray.DEVICE)}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    threshold: float = 0.5,
) -> AnalyzeResponse:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = xray.read_xray(raw, report_threshold=threshold)
    except Exception as exc:  # a broken upload shouldn't 500 silently
        raise HTTPException(status_code=422, detail=f"Could not read the image: {exc}")

    rep = report.write_report(result)

    heatmap_uri = None
    if result.heatmap_png:
        b64 = base64.b64encode(result.heatmap_png).decode("ascii")
        heatmap_uri = f"data:image/png;base64,{b64}"

    return AnalyzeResponse(
        is_chest_xray=result.is_chest_xray,
        ood_reason=result.ood_reason,
        top_label=result.top_label,
        findings=[FindingOut(label=f.label, plain=f.plain, probability=f.probability)
                  for f in result.findings],
        all_scores=result.all_scores,
        warnings=result.warnings,
        heatmap_data_uri=heatmap_uri,
        report=rep,
    )
