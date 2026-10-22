"""
Run the SERVED predictor over the held-out external evaluation half.

calibrate.py reports the external numbers from the model's raw logits. The
service does more than that: the learned quality gate and the anatomy gate run
first, then test-time augmentation, the site calibrator, the operating point
and the abstention band. This script pushes the external evaluation half
through exactly what build_predictor() serves, so the numbers describe what a
user of the service would actually get.

    python served_eval.py
    python served_eval.py --export-images ../site/demos/img

The evaluation half is rebuilt with splits.split_site_calibration, the same way
calibrate.py builds it, and its filename fingerprint is checked against
artifacts/evaluation.json. If they differ, the script stops: a different set of
images would not be the held-out half the model card reports on.

Reports:
  coverage, sensitivity, specificity and missed referable cases, over the
  images the service decided on its own (manual reviews excluded).
  why each manual review happened: the quality gate (never graded) or the
  abstention band (graded, too close to call).
  one representative case per outcome -- routine and correct, refer and
  correct, abstention band, quality-gate reject -- picked by a fixed rule:
  the case whose score is closest to its group's median (ties: lowest image
  number). Picking by rule rather than by eye keeps a demo from showing only
  flattering cases.

--export-images writes those cases as square crops of the retina, resized for
the web. IDRiD is CC BY 4.0: credit Porwal, Pachade and Kokare wherever the
images are shown, and say they were cropped and resized.

Writes artifacts/served_eval.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

GRADE_NAMES = ["Grade 0 (no DR)", "Grade 1 (mild)", "Grade 2 (moderate)",
               "Grade 3 (severe)", "Grade 4 (proliferative)"]


# ---------------------------------------------------------------------------
# The held-out half
# ---------------------------------------------------------------------------

def evaluation_half(cfg, evaluation_path: Path):
    """The external evaluation half, rebuilt exactly as calibrate.py does."""
    from splits import load_labels, split_site_calibration

    ecfg = copy.deepcopy(cfg)
    # The same three settings calibrate.py applies: load_labels filters on the
    # images and cache directories, so leaving them out changes the split.
    ecfg.data.csv_path = cfg.data.external_csv_path
    ecfg.data.images_dir = cfg.data.external_images_dir
    ecfg.data.cache_dir = cfg.data.external_cache_dir
    if not ecfg.data.csv_path:
        raise SystemExit("No external dataset configured (DR_EXTERNAL_CSV in dr.env).")

    edf = load_labels(ecfg.data)
    ev = split_site_calibration(edf, verbose=False)["evaluation"]
    names = "\n".join(sorted(ev["filename"].astype(str)))
    sha = hashlib.sha256(names.encode()).hexdigest()

    saved = json.loads(Path(evaluation_path).read_text())
    expected = saved.get("site_calibration", {}).get("evaluation_filenames_sha256")
    if expected is None:
        raise SystemExit(f"{evaluation_path} has no site calibration. Run: python calibrate.py --site-calibration")
    if sha != expected:
        raise SystemExit("The rebuilt evaluation half does not match the one in "
                         f"{evaluation_path} (fingerprint {sha[:12]} vs {expected[:12]}). "
                         "Re-run calibrate.py --site-calibration so the two agree.")
    return ev, Path(ecfg.data.images_dir)


# ---------------------------------------------------------------------------
# Predict, summarise, pick
# ---------------------------------------------------------------------------

def predict_all(predictor, df, images_dir: Path, log_every: int = 50) -> list[dict]:
    rows, t0 = [], time.time()
    for _, r in df.iterrows():
        img = cv2.imread(str(images_dir / r["filename"]), cv2.IMREAD_COLOR)
        if img is None:
            rows.append({"filename": str(r["filename"]), "unreadable": True})
            continue
        p = predictor.predict(img)
        rows.append({
            "filename": str(r["filename"]),
            "grade_true": int(r["grade"]),
            "referable_true": bool(r["referable"]),
            "decision": p.decision,
            "probability": p.probability,
            "grade_pred": p.grade,
            "quality": {k: v for k, v in p.quality.items()
                        if not isinstance(v, (list, dict)) or k == "reasons"},
            "reasons": p.reasons,
            "validated": p.validated,
            "model_version": p.model_version,
        })
        if log_every and len(rows) % log_every == 0:
            print(f"  {len(rows)}/{len(df)}  ({time.time() - t0:.0f}s)")
    return rows


def summarize(rows: list[dict]) -> dict:
    """Metrics over the cases the service decided itself, plus manual-review reasons."""
    rows = [r for r in rows if not r.get("unreadable")]
    auto = [r for r in rows if r["decision"] != "MANUAL_REVIEW"]
    manual = [r for r in rows if r["decision"] == "MANUAL_REVIEW"]
    tp = sum(r["decision"] == "REFER" and r["referable_true"] for r in auto)
    fn = sum(r["decision"] == "ROUTINE" and r["referable_true"] for r in auto)
    tn = sum(r["decision"] == "ROUTINE" and not r["referable_true"] for r in auto)
    fp = sum(r["decision"] == "REFER" and not r["referable_true"] for r in auto)
    ratio = lambda a, b: a / b if b else None
    return {
        "n_images": len(rows),
        "decisions": dict(Counter(r["decision"] for r in rows)),
        "coverage": ratio(len(auto), len(rows)),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "missed_referable": fn,
        "false_referrals": fp,
        "manual_review_reasons": {
            "quality_gate": sum(r["probability"] is None for r in manual),
            "abstention_band": sum(r["probability"] is not None for r in manual),
        },
    }


def _image_number(row: dict) -> int:
    m = re.search(r"(\d+)(?=\.\w+$)", row["filename"])
    return int(m.group(1)) if m else 0


def pick_median_cases(rows: list[dict]) -> dict[str, dict]:
    """One case per outcome: the one closest to its group's median score."""
    rows = [r for r in rows if not r.get("unreadable")]
    groups = {
        "routine": ([r for r in rows if r["decision"] == "ROUTINE" and not r["referable_true"]],
                    lambda r: r["probability"]),
        "refer": ([r for r in rows if r["decision"] == "REFER" and r["referable_true"]],
                  lambda r: r["probability"]),
        "abstention_band": ([r for r in rows if r["decision"] == "MANUAL_REVIEW" and r["probability"] is not None],
                            lambda r: r["probability"]),
        "quality_gate": ([r for r in rows if r["decision"] == "MANUAL_REVIEW" and r["probability"] is None],
                         lambda r: r["quality"].get("p_ungradable", 0.0)),
    }
    picked = {}
    for kind, (group, score) in groups.items():
        if not group:
            continue
        median = statistics.median(score(r) for r in group)
        best = min(group, key=lambda r: (abs(score(r) - median), _image_number(r)))
        picked[kind] = {**best, "group_size": len(group), "group_median": median,
                        "grade_name": GRADE_NAMES[best["grade_true"]]}
    return picked


def crop_retina(img: np.ndarray, size: int = 480, threshold: int = 15) -> np.ndarray:
    """Square crop centred on the retina (pixels brighter than the black surround)."""
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > threshold
    if not mask.any():
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    ys, xs = np.where(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 10
    padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    square = padded[cy:cy + 2 * half, cx:cx + 2 * half]
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evaluation", default="artifacts/evaluation.json")
    ap.add_argument("--out", default="artifacts/served_eval.json")
    ap.add_argument("--export-images", default=None, metavar="DIR",
                    help="write the picked cases as web-sized JPEGs to DIR")
    ap.add_argument("--size", type=int, default=480, help="exported image size in px")
    args = ap.parse_args(argv)

    from config import CONFIG
    from predictor import build_predictor

    ev, images_dir = evaluation_half(CONFIG, Path(args.evaluation))
    print(f"  Evaluation half: {len(ev)} images (fingerprint matches {args.evaluation})")

    predictor, note = build_predictor()
    print(f"  {note}")
    if getattr(predictor, "is_demo", False):
        raise SystemExit("build_predictor() returned the demo heuristic, not a trained model.")

    rows = predict_all(predictor, ev, images_dir)
    summary = summarize(rows)
    cases = pick_median_cases(rows)

    print(f"\n  Served on the held-out half ({summary['n_images']} images)")
    print(f"    decisions         {summary['decisions']}")
    print(f"    coverage          {summary['coverage']:.1%}")
    print(f"    sensitivity       {summary['sensitivity']:.3f}   (auto-decided cases only)")
    print(f"    specificity       {summary['specificity']:.3f}")
    print(f"    missed referable  {summary['missed_referable']}")
    print(f"    manual review     {summary['manual_review_reasons']}")
    print("\n  Representative cases (closest to each group's median):")
    for kind, c in cases.items():
        prob = "-" if c["probability"] is None else f"{c['probability']:.3f}"
        print(f"    {kind:<16} {Path(c['filename']).stem:<12} {c['grade_name']:<24} "
              f"-> {c['decision']:<13} P={prob}  ({c['group_size']} in group)")

    if args.export_images:
        out_dir = Path(args.export_images)
        out_dir.mkdir(parents=True, exist_ok=True)
        for kind, c in cases.items():
            img = cv2.imread(str(images_dir / c["filename"]), cv2.IMREAD_COLOR)
            dest = out_dir / f"{Path(c['filename']).stem.lower().replace('_', '-')}.jpg"
            cv2.imwrite(str(dest), crop_retina(img, args.size), [cv2.IMWRITE_JPEG_QUALITY, 82])
            c["exported_image"] = str(dest)
        print(f"\n  Exported {len(cases)} images to {out_dir}. IDRiD is CC BY 4.0: credit "
              "Porwal, Pachade and Kokare, and say the images were cropped and resized.")

    report = {"model_version": rows[0].get("model_version") if rows else None,
              "summary": summary, "cases": cases, "predictions": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print(f"\n  Wrote {args.out}")
    return report


if __name__ == "__main__":
    main()
