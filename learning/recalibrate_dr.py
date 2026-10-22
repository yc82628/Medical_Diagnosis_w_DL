"""
Re-calibrate the DR model for THIS hospital from its graders' decisions.

    python learning/recalibrate_dr.py snapshots/20261022-150000
    python learning/registry.py list dr                # then a person decides

The DR model ranks cases well across sites, but its probabilities and decision
threshold do not transfer (MODEL_CARD.md, "Site calibration"). This applies
the same site-calibration procedure as calibrate.py --site-calibration to the
hospital's own cases:

  - fits the calibrator, the operating point (sensitivity >= 0.90) and the
    abstention band on the snapshot's TRAIN patients (graded Refer / Routine;
    Ungradable images are excluded -- they never reach the DR model);
  - measures that profile on the LOCKED TEST patients, which nothing is
    fitted on, and reports the currently served profile on the same patients;
  - writes a candidate evaluation.json in which this hospital's locked test
    is the external test, and registers it. The model's WEIGHTS are not
    changed, so this is fast and low-risk.

Pass / fail is the service's own acceptance gate (predictor.acceptance_failures,
config.AcceptanceConfig) applied to the candidate evaluation, so "the registry
calls it fit" and "the service will load it" cannot disagree. A recalibration
usually trades some specificity for the sensitivity it restores; that is the
point, so it is judged against the floors, not against "no number got worse".
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "learning"))


def dr_logits(model, pc, paths, device, tta: str, batch: int = 8):
    """Referable logits and grades for raw images, the way the predictor scores them."""
    import cv2
    import torch
    from model import tta_logits
    from preprocessing import preprocess_image
    from splits import enforce_monotonic, grades_from_cumulative

    logits = []
    for i in range(0, len(paths), batch):
        xs = []
        for p in paths[i:i + batch]:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(preprocess_image(img, pc), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            xs.append(((rgb - np.array(pc.norm_mean)) / np.array(pc.norm_std)).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(xs)).float().to(device)
        with torch.no_grad():                         # fp32: logits can reach -700
            logits.append(tta_logits(model, x, tta).cpu().numpy())
    lo = np.concatenate(logits)
    cum = enforce_monotonic(1 / (1 + np.exp(-np.clip(lo, -30, 30))))
    return lo[:, 1], grades_from_cumulative(cum)


def build_candidate(current: dict, cal_part: dict, test_part: dict, snapshot: str,
                    target_sensitivity: float = 0.90) -> dict:
    """
    Torch-free: from logits and labels to a candidate evaluation.json.
    cal_part / test_part: {"logit", "y", "patient", "grade_true", "grade_pred"}.
    """
    from metrics import (Calibrator, abstention_curve, full_report, screening_metrics,
                         expected_calibration_error, threshold_at_sensitivity)

    cal = Calibrator().fit(cal_part["logit"], cal_part["y"])
    c_prob = cal.predict(cal_part["logit"])
    thr = threshold_at_sensitivity(cal_part["y"], c_prob, target_sensitivity)
    t_prob = cal.predict(test_part["logit"])

    has_grade = ~np.isnan(test_part["grade_true"])
    rep = full_report(test_part["y"], t_prob, test_part["patient"], thr,
                      test_part["grade_true"][has_grade].astype(int) if has_grade.any() else None,
                      test_part["grade_pred"][has_grade] if has_grade.any() else None)
    rep["profile"] = "site"
    ab = abstention_curve(test_part["y"], t_prob, thr, band_source_prob=c_prob)

    # The profile in service today, on the same locked-test patients, for contrast.
    from predictor import apply_calibration
    cur_prob = np.array([apply_calibration(current.get("calibration") or {}, v)
                         for v in test_part["logit"]])
    served_now = screening_metrics(test_part["y"], cur_prob, float(current["operating_point"]))
    served_now["ece"] = expected_calibration_error(test_part["y"], cur_prob)

    out = copy.deepcopy(current)
    out.update({
        "operating_point": float(thr), "calibration": cal.to_dict(),
        "abstention": ab.to_dict(orient="records"),
        # This hospital becomes the external site the gate checks.
        "external_test": rep,
        "previous_external_test": current.get("external_test"),
        "site_calibration": {
            "dataset": f"local snapshot {snapshot}",
            "split_unit": "patient (pseudonymous code; locked test fixed per patient)",
            "n_calibration": int(len(cal_part["y"])), "n_evaluation": int(len(test_part["y"])),
            "calibration_positives": int(np.sum(cal_part["y"])),
            "evaluation_positives": int(np.sum(test_part["y"])),
            "locked_test_under_previous_profile": served_now,
            "previous_site_calibration": current.get("site_calibration"),
        },
        "recalibrated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-calibrate the DR model on local verdicts.")
    ap.add_argument("snapshot")
    ap.add_argument("--evaluation", default="artifacts/evaluation.json")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    import pandas as pd
    import torch
    from config import CONFIG
    from model import load_checkpoint
    from predictor import acceptance_failures, file_sha256

    snap = Path(args.snapshot)
    df = pd.read_csv(snap / "labels.csv")
    df = df[df["referable"].notna()].reset_index(drop=True)          # Refer / Routine only
    current = json.loads(Path(args.evaluation).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pc, _ = load_checkpoint(args.checkpoint, device)
    tta = current.get("tta", "none")

    parts = {}
    for name, split in (("calibration", "train"), ("evaluation", "locked_test")):
        d = df[df["split"] == split]
        logit, gpred = dr_logits(model, pc, [snap / f for f in d["filename"]], device, tta) \
            if len(d) else (np.array([]), np.array([]))
        parts[name] = {"logit": logit, "y": d["referable"].astype(int).to_numpy(),
                       "patient": d["patient"].astype(str).to_numpy(),
                       "grade_true": pd.to_numeric(d["grade"], errors="coerce").to_numpy(float),
                       "grade_pred": gpred}
        print(f"  {name:<11} {len(d):>5} graded images, {int(parts[name]['y'].sum())} referable")

    why = []
    if parts["calibration"]["y"].sum() < 30 or (1 - parts["calibration"]["y"]).sum() < 30:
        why.append("calibration patients need at least 30 referable and 30 non-referable images")
    if len(parts["evaluation"]["y"]) == 0 or parts["evaluation"]["y"].sum() == 0:
        why.append("the locked test set has no referable images yet")
    if why:
        print("  Not enough local data to re-calibrate yet:\n    - " + "\n    - ".join(why))
        return {"ok": False, "why": why}

    cand = build_candidate(current, parts["calibration"], parts["evaluation"], snap.name,
                           CONFIG.train.target_sensitivity)
    failures = acceptance_failures(cand, file_sha256(args.checkpoint), pc.fingerprint(),
                                   CONFIG.acceptance)
    comp = {"candidate_not_worse": not failures, "why": failures,
            "rule": "candidate evaluation passes config.AcceptanceConfig "
                    "with this hospital's locked test as the external test"}
    cand["learning_comparison"] = comp

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = snap / "candidates"
    out.mkdir(exist_ok=True)
    evf = out / f"dr_evaluation_{stamp}.json"
    evf.write_text(json.dumps(cand, indent=2, default=float))

    e, prev = cand["external_test"], cand["site_calibration"]["locked_test_under_previous_profile"]
    print(f"\n  Locked test ({e['n']} images, {e['tp'] + e['fn']} referable)")
    print("                  local profile   profile in service")
    for k in ("sensitivity", "specificity", "ece"):
        print(f"    {k:<14} {e[k]:<15.3f} {prev[k]:.3f}")
    print(f"\n  Passes the acceptance gate: {not failures}")
    for f in failures:
        print(f"    - {f}")

    if not args.no_register:
        import audit
        from registry import register
        v = register("dr", Path(args.checkpoint), evf, source=f"recalibrate_dr on {snap.name}",
                     summary={"locked_test": {k: e[k] for k in ("n", "sensitivity", "specificity", "ece")}},
                     comparison=comp, con=audit.connect(), actor="recalibrate_dr")
        print(f"\n  Registered candidate {v} (same weights, new site profile). NOT in service.")
        print(f"  To put it in service: python learning/registry.py promote dr {v} "
              f"--approved-by \"NAME\" --reason \"...\"")
    return {"ok": True, "evaluation": cand}


if __name__ == "__main__":
    main()
