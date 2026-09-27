"""Measure how good the reader is, with real numbers.

Three things:
1. Per-finding AUROC on a labeled set of chest films.
2. Calibration: are the scores honest, or over/under-confident?
3. Refusal rate on images that are not chest X-rays.

It runs off a small CSV so you can point it at any labeled set. When you
download NIH ChestX-ray14, drop its labels into the same CSV shape and rerun.

CSV format (labels.csv):
    path,Cardiomegaly,Effusion,Pneumonia,...
    images/0001.png,1,0,0,...
    images/0002.png,0,1,0,...
Columns after `path` are 0/1 for each finding you want scored.

Run:
    python eval/evaluate.py --labels eval/labels.csv --images-root .
    python eval/evaluate.py --ood-dir eval/not_xrays   # refusal test only
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# import the reader from the parent folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import xray  # noqa: E402


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """AUROC without sklearn: the rank-based (Mann-Whitney) form.

    Returns None when a label has only one class present (AUROC undefined).
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[: len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def _calibration(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10):
    """Reliability table: for each score band, predicted vs actual rate."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    n = len(y_score)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & (y_score < hi if i < bins - 1 else y_score <= hi)
        if mask.sum() == 0:
            continue
        conf = float(y_score[mask].mean())
        acc = float(y_true[mask].mean())
        rows.append((lo, hi, int(mask.sum()), conf, acc))
        ece += (mask.sum() / n) * abs(conf - acc)
    return rows, float(ece)


def score_labeled_set(labels_csv: Path, images_root: Path):
    with labels_csv.open() as f:
        reader = csv.DictReader(f)
        finding_cols = [c for c in reader.fieldnames if c != "path"]
        rows = list(reader)

    print(f"Scoring {len(rows)} films on {len(finding_cols)} findings...\n")

    y_true = {c: [] for c in finding_cols}
    y_score = {c: [] for c in finding_cols}

    for row in rows:
        img_path = images_root / row["path"]
        result = xray.read_xray(img_path.read_bytes(), report_threshold=1.1)  # scores only
        for c in finding_cols:
            if c not in result.all_scores:
                continue
            y_true[c].append(int(row[c]))
            y_score[c].append(result.all_scores[c])

    print(f"{'finding':28s} {'AUROC':>7s} {'n_pos':>6s} {'n':>5s}")
    print("-" * 50)
    aurocs = []
    for c in finding_cols:
        t = np.array(y_true[c])
        s = np.array(y_score[c])
        auc = _auroc(t, s)
        aurocs.append(auc)
        auc_str = f"{auc:.3f}" if auc is not None else "  n/a"
        print(f"{c:28s} {auc_str:>7s} {int(t.sum()):>6d} {len(t):>5d}")

    valid = [a for a in aurocs if a is not None]
    if valid:
        print("-" * 50)
        print(f"{'mean AUROC':28s} {np.mean(valid):>7.3f}")

    # pooled calibration across all findings
    all_t = np.concatenate([np.array(y_true[c]) for c in finding_cols])
    all_s = np.concatenate([np.array(y_score[c]) for c in finding_cols])
    table, ece = _calibration(all_t, all_s)
    print("\nCalibration (pooled across findings):")
    print(f"{'band':>12s} {'n':>6s} {'avg score':>10s} {'actual':>8s}")
    for lo, hi, n, conf, acc in table:
        print(f"{lo:>5.1f}-{hi:<5.1f} {n:>6d} {conf:>10.3f} {acc:>8.3f}")
    print(f"\nExpected Calibration Error: {ece:.3f}  (lower is better)")


def refusal_rate(ood_dir: Path):
    imgs = [p for p in ood_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not imgs:
        print(f"No images found in {ood_dir}")
        return
    refused = 0
    for p in imgs:
        result = xray.read_xray(p.read_bytes())
        if not result.is_chest_xray:
            refused += 1
        else:
            print(f"  MISSED (reported on a non-X-ray): {p.name}")
    print(f"\nRefusal on non-X-rays: {refused}/{len(imgs)} = {refused / len(imgs):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, help="CSV of path + 0/1 finding columns")
    ap.add_argument("--images-root", type=Path, default=Path("."))
    ap.add_argument("--ood-dir", type=Path, help="folder of images that are NOT chest X-rays")
    args = ap.parse_args()

    if args.labels:
        score_labeled_set(args.labels, args.images_root)
    if args.ood_dir:
        print("\n" + "=" * 50)
        refusal_rate(args.ood_dir)
    if not args.labels and not args.ood_dir:
        ap.error("give --labels and/or --ood-dir")


if __name__ == "__main__":
    main()
