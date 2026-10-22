"""
External evaluation of the maculopathy check on this hospital's graders' labels.

    python learning/evaluate_dme.py snapshots/20261022-150000
    python learning/registry.py list dme              # then a person decides

The maculopathy model was trained and cross-validated on one clinic (IDRiD),
so it is "experimental": advice only, never part of a referral decision. What
it needs is an EXTERNAL evaluation -- a different site -- and graders who
record "Maculopathy: present / absent" provide exactly that.

The model is not trained or re-fitted on local data here, and its threshold
stays the one frozen on IDRiD, so EVERY local image with a maculopathy label
(and a gradable verdict) can be used. Metrics are bootstrapped by patient.

If the result clears dme_model.DME_ACCEPTANCE (AUROC >= 0.85, sensitivity
>= 0.85, specificity >= 0.50, >= 30 maculopathy cases), the script writes a
candidate checkpoint -- the same weights, with this evaluation recorded as its
external test -- whose status is therefore "validated", and registers it.
Promotion is still a named decision (learning/registry.py). Once promoted,
the maculopathy result may turn a model-only ROUTINE eye into REFER.

If it does not clear them, the evaluation is still registered, with the
reasons, as a record; the check stays experimental.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "learning"))


def external_report(y, prob, suspected, patients, site: str) -> dict:
    """Torch-free: the 'external' block dme_model.dme_status() judges."""
    from metrics import bootstrap_ci
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, float)
    pred = np.asarray(suspected).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    both = 0 < y.sum() < len(y)
    return {
        "site": site, "n": int(len(y)), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "auroc": float(roc_auc_score(y, prob)) if both else float("nan"),
        "auroc_ci": bootstrap_ci(y, prob, patients, roc_auc_score) if both else None,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="External evaluation of the maculopathy check.")
    ap.add_argument("snapshot")
    ap.add_argument("--checkpoint", default="checkpoints/dme_model.pt")
    ap.add_argument("--eval", default="artifacts/dme_eval.json")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    import cv2
    import pandas as pd
    import torch
    from dme_model import DMEPredictor, dme_status, file_sha256

    snap = Path(args.snapshot)
    df = pd.read_csv(snap / "labels.csv")
    df = df[(df["gradable"] == 1) & df["dme"].notna()].reset_index(drop=True)
    print(f"  {len(df)} gradable images with a maculopathy label, "
          f"{int(df['dme'].sum())} with maculopathy, {df['patient'].nunique()} patients")
    if df.empty or df["dme"].nunique() < 2:
        print("  Not enough labels yet: graders record 'Maculopathy: present / absent' under "
              "'More details' with their decision.")
        return {"ok": False}

    dme = DMEPredictor(args.checkpoint)
    res = [dme.assess(cv2.imread(str(snap / f), cv2.IMREAD_COLOR)) for f in df["filename"]]
    ext = external_report(df["dme"].astype(int), [r["probability"] for r in res],
                          [r["suspected"] for r in res], df["patient"].astype(str).to_numpy(),
                          f"local snapshot {snap.name}")
    status, why = dme_status({"external": ext})
    print(f"\n  External: AUROC {ext['auroc']:.3f}  sensitivity {ext['sensitivity']:.3f}  "
          f"specificity {ext['specificity']:.3f}  ({ext['tp'] + ext['fn']} with maculopathy)")
    print(f"  Status this would give: {status}")
    for w in why:
        print(f"    - {w}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = snap / "candidates"
    out.mkdir(exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck["meta"] = {**(ck.get("meta") or {}), "external": ext}
    cand_ck = out / f"dme_{stamp}.pt"
    torch.save(ck, cand_ck)
    meta = json.loads(Path(args.eval).read_text()) if Path(args.eval).exists() else {}
    meta.update({"external": ext, "status": status,
                 "checkpoint_sha256": file_sha256(cand_ck)})
    cand_eval = out / f"dme_{stamp}_eval.json"
    cand_eval.write_text(json.dumps(meta, indent=2, default=float))

    if not args.no_register:
        import audit
        from registry import register
        comp = {"candidate_not_worse": status == "validated", "why": why,
                "rule": "dme_model.DME_ACCEPTANCE on this hospital's labels"}
        v = register("dme", cand_ck, cand_eval, source=f"evaluate_dme on {snap.name}",
                     summary={"external": {k: ext[k] for k in ("n", "auroc", "sensitivity", "specificity")}},
                     comparison=comp, con=audit.connect(), actor="evaluate_dme")
        print(f"\n  Registered {v} (status if promoted: {status}).")
        if status == "validated":
            print(f"  To let it affect referrals: python learning/registry.py promote dme {v} "
                  f"--approved-by \"NAME\" --reason \"...\"")
    return {"ok": True, "external": ext, "status": status}


if __name__ == "__main__":
    main()
