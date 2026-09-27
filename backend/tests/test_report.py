"""Tests for the report writer, including the no-model fallback and refusals."""
import report
from xray import ReadResult, Finding


def test_template_calls_a_normal_film_normal():
    text = report._template_report([])
    assert "unremarkable" in text.lower() or "normal" in text.lower()


def test_template_lists_the_findings():
    findings = [
        Finding(label="Cardiomegaly", plain="an enlarged heart", probability=0.8),
        Finding(label="Effusion", plain="fluid around the lung (effusion)", probability=0.62),
    ]
    text = report._template_report(findings)
    assert "enlarged heart" in text
    assert "effusion" in text
    assert "Impression" in text


def test_refusal_passes_through():
    result = ReadResult(
        findings=[], all_scores={}, top_label=None,
        is_chest_xray=False, ood_reason="This doesn't look like a chest X-ray.",
    )
    rep = report.write_report(result)
    assert rep["refused"] is True
    assert rep["source"] == "refusal"
    assert "chest X-ray" in rep["text"]


def test_normal_film_skips_the_model_and_uses_template():
    # No findings means there's nothing to write up, so it should never call Ollama.
    result = ReadResult(
        findings=[], all_scores={"Cardiomegaly": 0.1}, top_label="Cardiomegaly",
        is_chest_xray=True,
    )
    rep = report.write_report(result)
    assert rep["refused"] is False
    assert rep["source"] == "template"
    assert rep["disclaimer"]
