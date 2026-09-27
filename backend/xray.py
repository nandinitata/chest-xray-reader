"""Chest X-ray reading: findings, a heat-map, and an "is this even a chest X-ray" check.

Everything here runs locally. The model weights come from torchxrayvision, which
is open. No paid APIs.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import torch
import torchxrayvision as xrv
import skimage
from PIL import Image

# Pick the best device we have. On the M3 that's MPS.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# A short, plain-English name for each label the model knows.
FRIENDLY = {
    "Atelectasis": "collapsed lung tissue",
    "Consolidation": "an area of consolidation",
    "Infiltration": "an infiltrate",
    "Pneumothorax": "air outside the lung (pneumothorax)",
    "Edema": "fluid in the lungs (edema)",
    "Emphysema": "emphysema",
    "Fibrosis": "scarring (fibrosis)",
    "Effusion": "fluid around the lung (effusion)",
    "Pneumonia": "signs of pneumonia",
    "Pleural_Thickening": "pleural thickening",
    "Cardiomegaly": "an enlarged heart",
    "Nodule": "a nodule",
    "Mass": "a mass",
    "Hernia": "a hernia",
    "Lung Lesion": "a lung lesion",
    "Fracture": "a fracture",
    "Lung Opacity": "a lung opacity",
    "Enlarged Cardiomediastinum": "a widened cardiomediastinum",
}


@dataclass
class Finding:
    label: str
    plain: str
    probability: float


@dataclass
class ReadResult:
    findings: list[Finding]           # everything above the report threshold
    all_scores: dict[str, float]      # every label -> probability, for the eval scripts
    top_label: str | None             # the single most likely finding, drives the heat-map
    is_chest_xray: bool               # False -> the app should refuse and not report
    ood_reason: str | None = None
    heatmap_png: bytes | None = None  # amber overlay on the image, PNG bytes
    warnings: list[str] = field(default_factory=list)


_model: xrv.models.DenseNet | None = None


def get_model() -> xrv.models.DenseNet:
    """Load the pretrained model once and keep it around."""
    global _model
    if _model is None:
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()
        model.to(DEVICE)
        _model = model
    return _model


def _to_model_tensor(gray: np.ndarray) -> torch.Tensor:
    """Take a normalised HxW float image and shape it the way the model wants."""
    img = xrv.datasets.normalize(gray, 255)  # scales to the model's [-1024, 1024] range
    img = img[None, ...]  # add a channel
    transform = xrv.datasets.XRayCenterCrop()
    img = transform(img)
    tensor = torch.from_numpy(img)[None, ...].float()  # add a batch dim
    return tensor.to(DEVICE)


def _load_gray(image_bytes: bytes) -> np.ndarray:
    """Read whatever image was uploaded and return an 8-bit grayscale array."""
    pil = Image.open(io.BytesIO(image_bytes)).convert("L")
    return np.array(pil).astype(np.float32)


def _looks_like_chest_xray(gray: np.ndarray, scores: dict[str, float]) -> tuple[bool, str | None]:
    """A light sanity check so the app doesn't confidently report on a selfie.

    Real chest films are grayscale, fairly large, and light up several of the
    model's chest labels at least a little. A photo of a cat lights up almost
    nothing. This isn't a medical guarantee, it's a guard against junk input.
    """
    h, w = gray.shape[:2]
    if min(h, w) < 200:
        return False, "The image is too small to read as an X-ray."

    # Chest films use most of the gray range and aren't nearly all one shade.
    spread = float(gray.std())
    if spread < 12:
        return False, "The image is too flat to be an X-ray."

    # If nothing the model knows about registers even weakly, it's probably not a chest film.
    strongest = max(scores.values()) if scores else 0.0
    if strongest < 0.15:
        return False, "This doesn't look like a chest X-ray."

    return True, None


def _gradcam_overlay(tensor: torch.Tensor, gray: np.ndarray, label_index: int) -> bytes:
    """Make an amber heat-map over the film for one finding.

    Uses the gradient of the chosen label against the last conv feature maps,
    the standard Grad-CAM recipe, done by hand so we don't pull an extra library.
    """
    model = get_model()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = model.features.denseblock4  # last dense block, richest features

    def fwd_hook(_m, _in, out):
        activations.append(out)

    def bwd_hook(_m, _gin, gout):
        gradients.append(gout[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        inp = tensor.clone().requires_grad_(True)
        out = model(inp)
        model.zero_grad()
        out[0, label_index].backward()

        acts = activations[0].detach()[0]         # C x h x w
        grads = gradients[0].detach()[0]          # C x h x w
        weights = grads.mean(dim=(1, 2))          # one weight per channel
        cam = torch.relu((weights[:, None, None] * acts).sum(0))
        cam = cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = skimage.transform.resize(cam, gray.shape[:2], order=1, mode="reflect")

    # Build the overlay: grayscale film tinted amber where the model looked.
    base = gray.copy()
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    rgb = np.stack([base, base, base], axis=-1)
    amber = np.array([1.0, 0.63, 0.10])  # #FFA11A-ish, matches the UI accent
    alpha = (cam ** 1.4)[..., None] * 0.6
    blended = rgb * (1 - alpha) + amber * alpha
    out_img = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()


def read_xray(image_bytes: bytes, report_threshold: float = 0.5) -> ReadResult:
    """Main entry point: bytes in, a full reading out."""
    model = get_model()
    gray = _load_gray(image_bytes)
    tensor = _to_model_tensor(gray)

    with torch.no_grad():
        probs = model(tensor)[0].cpu().numpy()

    scores = {label: float(p) for label, p in zip(model.pathologies, probs) if label}

    ok, reason = _looks_like_chest_xray(gray, scores)
    if not ok:
        return ReadResult(
            findings=[], all_scores=scores, top_label=None,
            is_chest_xray=False, ood_reason=reason,
        )

    findings = [
        Finding(label=lbl, plain=FRIENDLY.get(lbl, lbl.lower()), probability=score)
        for lbl, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if score >= report_threshold
    ]

    top_label = max(scores, key=scores.get) if scores else None
    heatmap = None
    if top_label is not None:
        idx = list(model.pathologies).index(top_label)
        heatmap = _gradcam_overlay(tensor, gray, idx)

    warnings = []
    if not findings:
        warnings.append("No finding crossed the reporting threshold. The film reads as unremarkable.")

    return ReadResult(
        findings=findings, all_scores=scores, top_label=top_label,
        is_chest_xray=True, heatmap_png=heatmap, warnings=warnings,
    )
