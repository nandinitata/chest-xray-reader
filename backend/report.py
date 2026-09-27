"""Turn the model's findings into a short, readable report.

The language model only ever gets the findings the image model actually
detected. It's told to write those up and nothing else, so it can't invent a
diagnosis the image model never saw. If Ollama isn't running, a plain template
writes the report instead, so the app still works.
"""
from __future__ import annotations

import textwrap

import requests

from xray import ReadResult, Finding

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

DISCLAIMER = (
    "This is a demo from an open model, not a medical diagnosis. "
    "A radiologist should read the actual film."
)


def _confidence_word(p: float) -> str:
    if p >= 0.75:
        return "likely"
    if p >= 0.6:
        return "possible"
    return "borderline"


def _findings_lines(findings: list[Finding]) -> str:
    lines = []
    for f in findings:
        lines.append(f"- {f.plain} ({_confidence_word(f.probability)}, score {f.probability:.2f})")
    return "\n".join(lines)


def _build_prompt(findings: list[Finding]) -> str:
    finding_text = _findings_lines(findings)
    return textwrap.dedent(f"""\
        You are writing a short chest X-ray report for a demo.

        Here are the findings an image model detected. These are the ONLY things
        you may mention. Do not add any finding that is not on this list. Do not
        give a diagnosis or treatment.

        {finding_text}

        Write the report in plain English, the way a doctor would jot it for a
        colleague. Two short parts:
        Findings: one or two sentences describing what was seen, using the plain
        words above and the confidence words in brackets.
        Impression: one sentence summing up.

        Keep it short. No preamble, no sign-off, just the report.
    """)


def _ollama_report(findings: list[Finding], timeout: float = 60.0) -> str | None:
    """Ask the local model to write it up. Returns None if Ollama isn't reachable."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": _build_prompt(findings),
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return text or None
    except (requests.RequestException, ValueError):
        return None


def _template_report(findings: list[Finding]) -> str:
    """Fallback writer. Plain, no model needed."""
    if not findings:
        body = "Findings: No significant abnormality detected. The film reads as unremarkable."
        impression = "Impression: Normal chest X-ray."
        return f"{body}\n{impression}"

    parts = []
    for f in findings:
        parts.append(f"{f.plain} ({_confidence_word(f.probability)})")
    if len(parts) == 1:
        findings_sentence = f"There is {parts[0]}."
    else:
        findings_sentence = "There is " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
    top = findings[0]
    impression = f"Most notable is {top.plain}."
    return f"Findings: {findings_sentence}\nImpression: {impression}"


def write_report(result: ReadResult) -> dict:
    """Build the report payload the API returns.

    Always includes a source flag so the UI can show whether the write-up came
    from the local model or the fallback template.
    """
    if not result.is_chest_xray:
        return {
            "refused": True,
            "text": (
                f"{result.ood_reason} So there's nothing to report. "
                "Upload a chest X-ray to get a reading."
            ),
            "source": "refusal",
            "disclaimer": DISCLAIMER,
        }

    findings = result.findings
    text = None
    source = "template"
    if findings:  # only bother the model when there's something to describe
        text = _ollama_report(findings)
        source = "ollama" if text else "template"
    if text is None:
        text = _template_report(findings)

    return {
        "refused": False,
        "text": text,
        "source": source,
        "disclaimer": DISCLAIMER,
    }
