"""
Measure the quality + anatomy gate against HUMAN gradability labels.

The gate's thresholds (preprocessing.QUALITY_THRESHOLDS) were set by hand.
Until they are measured against human judgement, "ungradable images go to a
human" is a design intent, not a property. This script measures it.

Labelled data: EyeQ (Fu et al., MICCAI 2019) grades 28,792 EyePACS images as
Good (0), Usable (1) or Reject (2). Any CSV with a filename column and a
quality label column works.

    python validate_quality_gate.py --csv EyeQ/Label_EyeQ_test.csv \
        --images eyepacs/test --filename-col image --label-col quality \
        --reject-values 2

Reports:
  catch rate      share of human-REJECTED images the gate also rejects.
                  A miss here is an unusable photo the model will grade --
                  the failure mode INTENDED_USE.md calls most dangerous.
  false rejects   share of human-ACCEPTED images the gate rejects. Each one
                  costs a recapture or a manual grade, not a patient outcome.
  per-metric distributions for accepted vs rejected images, to retune the
  thresholds from evidence.

When the learned quality model (quality_model.py) is present, run_gates uses
it, so this measures the combined gate. Images the model trained or was tuned
on are left out automatically: scoring a model on its own training images
measures memory, not quality. --gate classical measures the classical checks
alone.

Writes artifacts/quality_gate_validation.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from predictor import run_gates
from preprocessing import QUALITY_THRESHOLDS
from quality_model import default_quality_gate

METRICS = ("blur_var", "fov_fraction", "mean_intensity", "fundus_score")


def evaluate(df: pd.DataFrame, images_dir: Path, filename_col: str, label_col: str,
             reject_values: set, quality_gate=None) -> dict:
    seen = getattr(quality_gate, "seen_files", None) or set()
    norm = df[filename_col].astype(str).str.replace("\\", "/", regex=False)
    excluded = int(norm.isin(seen).sum())
    df = df[~norm.isin(seen)]

    rows, unreadable = [], 0
    for _, r in df.iterrows():
        img = cv2.imread(str(images_dir / str(r[filename_col])), cv2.IMREAD_COLOR)
        if img is None:
            unreadable += 1
            continue
        q = run_gates(img, quality_gate=quality_gate)
        rows.append({"human_reject": r[label_col] in reject_values,
                     "gate_reject": not q["gradable"],
                     "reasons": q.get("reasons", []),
                     "p_ungradable": q.get("p_ungradable"),
                     **{m: q.get(m) for m in METRICS}})
    if not rows:
        raise SystemExit("  No readable images. Check --images and --filename-col.")

    res = pd.DataFrame(rows)
    rej, acc = res[res.human_reject], res[~res.human_reject]
    tp = int((rej.gate_reject).sum())
    fp = int((acc.gate_reject).sum())

    def dist(frame, m):
        v = pd.to_numeric(frame[m], errors="coerce").dropna()
        return ({p: float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)}
                if len(v) else None)

    reasons: dict[str, int] = {}
    for rs in res.loc[res.gate_reject, "reasons"]:
        for x in rs:
            key = x.split(":")[0].split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1

    auroc = None
    if quality_gate is not None and 0 < len(rej) < len(res):
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(res.human_reject, res.p_ungradable.astype(float)))

    return {
        "quality_source": "classical" if quality_gate is None else "learned",
        "quality_model": getattr(quality_gate, "version", None),
        "excluded_seen_by_model": excluded,
        "auroc_p_ungradable": auroc,
        "n_images": len(res), "unreadable": unreadable,
        "n_human_reject": len(rej), "n_human_accept": len(acc),
        "catch_rate": (tp / len(rej)) if len(rej) else None,
        "missed_rejects": len(rej) - tp,
        "false_reject_rate": (fp / len(acc)) if len(acc) else None,
        "false_rejects": fp,
        "gate_reject_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "thresholds": QUALITY_THRESHOLDS,
        "distributions": {m: {"human_accept": dist(acc, m), "human_reject": dist(rej, m)}
                          for m in METRICS},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--filename-col", default="image")
    ap.add_argument("--label-col", default="quality")
    ap.add_argument("--reject-values", nargs="+", default=["2"],
                    help="label values a human marked ungradable (EyeQ: 2)")
    ap.add_argument("--limit", type=int, help="evaluate only the first N rows")
    ap.add_argument("--gate", choices=["auto", "classical"], default="auto",
                    help="auto: the gate the service uses (learned model if present)")
    ap.add_argument("--out", default="artifacts/quality_gate_validation.json")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    for col in (args.filename_col, args.label_col):
        if col not in df.columns:
            raise SystemExit(f"  Column {col!r} not in CSV. Columns: {list(df.columns)}")
    if args.limit:
        df = df.head(args.limit)
    # Compare as strings so "2" on the command line matches 2 in the CSV.
    df[args.label_col] = df[args.label_col].astype(str)

    gate = default_quality_gate() if args.gate == "auto" else None
    rep = evaluate(df, Path(args.images), args.filename_col, args.label_col,
                   set(map(str, args.reject_values)), quality_gate=gate)

    print(f"\n  Gate: {rep['quality_source']}"
          + (f" ({rep['quality_model']})" if rep["quality_model"] else ""))
    if rep["excluded_seen_by_model"]:
        print(f"  Left out {rep['excluded_seen_by_model']} images the quality model "
              f"trained or was tuned on")
    print(f"  Images evaluated: {rep['n_images']} ({rep['unreadable']} unreadable)")
    print(f"  Human-rejected: {rep['n_human_reject']}   human-accepted: {rep['n_human_accept']}")
    if rep["catch_rate"] is not None:
        print(f"\n  Catch rate      {rep['catch_rate']:.1%}  "
              f"({rep['missed_rejects']} ungradable images would reach the model)")
    if rep["false_reject_rate"] is not None:
        print(f"  False rejects   {rep['false_reject_rate']:.1%}  "
              f"({rep['false_rejects']} usable images sent for recapture)")
    if rep["auroc_p_ungradable"] is not None:
        print(f"  AUROC of P(ungradable) {rep['auroc_p_ungradable']:.3f}")
    print("\n  Why the gate rejected:")
    for k, v in rep["gate_reject_reasons"].items():
        print(f"    {v:>6}  {k}")
    print("\n  Median per metric (accepted vs rejected by humans):")
    for m, d in rep["distributions"].items():
        a = d["human_accept"][50] if d["human_accept"] else float("nan")
        r = d["human_reject"][50] if d["human_reject"] else float("nan")
        print(f"    {m:<16} {a:>10.2f}   {r:>10.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\n  Wrote {args.out}")
    return rep


if __name__ == "__main__":
    main()
