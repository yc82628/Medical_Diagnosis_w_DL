"""
Retrain the image-quality model with this hospital's "Ungradable" verdicts.

    python learning/retrain_quality.py snapshots/20261022-150000
    python learning/registry.py list quality          # then a person decides

Every time a grader marks an image Ungradable (or grades it), the platform
gains a quality label from THIS site's cameras and photographers -- exactly
the data the DeepDRiD-trained model has never seen. This script:

  1. trains a CANDIDATE on DeepDRiD's 2,000 human quality labels plus the
     snapshot's TRAIN patients (the snapshot's locked-test patients are never
     trained on);
  2. sets its threshold on validation (at most 5% of usable images rejected);
  3. scores the candidate AND the model currently in service on
       - the site's LOCKED TEST patients (the question that matters), and
       - DeepDRiD's held-out test patients (has anything got worse?);
  4. registers the candidate with that comparison in learning/registry.py.

It never replaces the served model. Promotion is a separate, named decision:
    python learning/registry.py promote quality <version> --approved-by ... --reason ...

A candidate counts as "not worse" only if, on the locked test set, its AUROC
and catch rate are within 0.02 of the current model's and its false-reject
rate is within 0.02 of max(current, target); and the locked test set has at
least MIN_LOCKED_REJECTS human-rejected images -- fewer cannot show anything.
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

MIN_LOCKED_REJECTS = 20
TOLERANCE = 0.02


def compare(cur: dict | None, cand: dict, target_fr: float,
            min_rejects: int = MIN_LOCKED_REJECTS) -> dict:
    """Torch-free decision rule for 'candidate not worse than current'."""
    why = []
    if cand["n_human_reject"] < min_rejects:
        why.append(f"locked test has {cand['n_human_reject']} human-rejected images "
                   f"(need {min_rejects}); collect more Ungradable verdicts first")
    if cur is not None:
        if cand["auroc"] < cur["auroc"] - TOLERANCE:
            why.append(f"AUROC {cand['auroc']:.3f} vs current {cur['auroc']:.3f}")
        if cand["catch_rate"] < cur["catch_rate"] - TOLERANCE:
            why.append(f"catch rate {cand['catch_rate']:.1%} vs current {cur['catch_rate']:.1%}")
        limit = max(cur["false_reject_rate"], target_fr) + TOLERANCE
        if cand["false_reject_rate"] > limit:
            why.append(f"false rejects {cand['false_reject_rate']:.1%} above {limit:.1%}")
    return {"candidate_not_worse": not why, "why": why, "rule": __doc__.strip().splitlines()[-4]}


def _patient_split(codes, val_fraction=0.2):
    """Train-split patients -> train / val, by code (stable)."""
    return np.array([int(c[8:16], 16) / 0xFFFFFFFF < val_fraction for c in codes])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Retrain the quality model with local verdicts.")
    ap.add_argument("snapshot")
    ap.add_argument("--base-labels", default="deepdrid/quality_labels.csv")
    ap.add_argument("--base-images", default="deepdrid/repo/regular_fundus_images")
    ap.add_argument("--base-cache", default="deepdrid/quality_cache")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-false-reject", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-locked-rejects", type=int, default=MIN_LOCKED_REJECTS)
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    import cv2
    import pandas as pd
    import torch
    import quality_model as qm
    from model import get_device
    from predictor import file_sha256

    snap = Path(args.snapshot)
    local = pd.read_csv(snap / "labels.csv")
    cfg = qm.QualityInputConfig()

    # Base: DeepDRiD, with the SAME split the served model was trained with.
    base = pd.read_csv(args.base_labels)
    bfiles = base["filename"].astype(str).tolist()
    brej = (base["quality"].astype(int) == 0).to_numpy()
    bm = qm.split_patients(bfiles, brej, seed=args.seed)
    bcache = qm.build_input_cache(bfiles, Path(args.base_images), Path(args.base_cache), cfg)

    lfiles = local["filename"].astype(str).tolist()
    lrej = (local["gradable"].astype(int) == 0).to_numpy()
    locked = (local["split"] == "locked_test").to_numpy()
    lval = ~locked & _patient_split(local["patient"].astype(str))
    lcache = qm.build_input_cache(lfiles, snap, snap / "quality_cache", cfg)

    print(f"  DeepDRiD: {len(bfiles)} images ({brej.mean():.0%} rejected)")
    print(f"  Local:    {len(lfiles)} images ({lrej.mean():.0%} rejected); "
          f"train {int((~locked & ~lval).sum())}, val {int(lval.sum())}, "
          f"locked test {int(locked.sum())} ({int(lrej[locked].sum())} rejected)")

    load = lambda paths: np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                                   for p in paths]).transpose(0, 3, 1, 2)
    x = torch.from_numpy(np.concatenate([load(bcache), load(lcache)]).copy())
    y = np.concatenate([brej, lrej]).astype(np.float32)
    nb, nl = len(bfiles), len(lfiles)
    on_base = lambda m: np.concatenate([np.asarray(m, bool), np.zeros(nl, bool)])
    on_local = lambda m: np.concatenate([np.zeros(nb, bool), np.asarray(m, bool)])
    masks = {"train": on_base(bm["train"]) | on_local(~locked & ~lval),
             "val": on_base(bm["val"]) | on_local(lval)}
    base_test, local_test = on_base(bm["test"]), on_local(locked)

    device = get_device()
    net, best = qm.train_model(x, y, masks, args.backbone, cfg, args.epochs, 3e-4, 16,
                               args.seed, device)
    p_val = qm._predict(net, x[masks["val"]], cfg, device)
    thr = qm.choose_threshold(y[masks["val"]].astype(bool), p_val, args.max_false_reject)

    def score(p_fn, mask, threshold):
        return qm.gate_metrics(y[mask].astype(bool), p_fn(mask), threshold)

    cand = {"local_locked_test": score(lambda m: qm._predict(net, x[m], cfg, device), local_test, thr),
            "deepdrid_test": score(lambda m: qm._predict(net, x[m], cfg, device), base_test, thr)}

    cur = None
    current_ck = Path(qm.QUALITY_CHECKPOINT)
    if current_ck.exists():
        gate = qm.LearnedQualityGate(current_ck, device=str(device))
        raw = [ROOT / args.base_images / f for f in bfiles] + [snap / f for f in lfiles]
        def cur_p(mask):
            return np.array([gate.p_ungradable(cv2.imread(str(raw[i]))) for i in np.where(mask)[0]])
        cur = {"local_locked_test": score(cur_p, local_test, gate.threshold),
               "deepdrid_test": score(cur_p, base_test, gate.threshold),
               "version": gate.version}

    comp = compare(cur["local_locked_test"] if cur else None, cand["local_locked_test"],
                   args.max_false_reject, args.min_locked_rejects)
    # The DeepDRiD test set guards against forgetting what the model already knew.
    if cur and cand["deepdrid_test"]["auroc"] < cur["deepdrid_test"]["auroc"] - TOLERANCE:
        comp["candidate_not_worse"] = False
        comp["why"].append(f"DeepDRiD test AUROC fell to {cand['deepdrid_test']['auroc']:.3f} "
                           f"from {cur['deepdrid_test']['auroc']:.3f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = snap / "candidates"
    out.mkdir(exist_ok=True)
    ck = out / f"quality_{stamp}.pt"
    from dataclasses import asdict
    torch.save({"state_dict": net.state_dict(), "backbone": args.backbone,
                "input": asdict(cfg), "input_fingerprint": cfg.fingerprint(),
                "threshold": thr, "max_false_reject": args.max_false_reject,
                "epoch": best["epoch"], "seed": args.seed,
                "labels": f"{args.base_labels} + {snap.name}/labels.csv",
                "seen_files": [f for f, m in zip(bfiles + lfiles, masks["train"] | masks["val"]) if m],
                "test": cand["deepdrid_test"]}, ck)
    ev = {"checkpoint_sha256": file_sha256(ck), "backbone": args.backbone,
          "input_fingerprint": cfg.fingerprint(), "image_size": cfg.image_size,
          "labels": "deepdrid/quality_labels.csv + local snapshot", "seed": args.seed,
          "max_false_reject_target": args.max_false_reject, "threshold": thr,
          "epoch": best["epoch"], "snapshot": snap.name,
          "split": {"train": {"images": int(masks["train"].sum())},
                    "val": {"images": int(masks["val"].sum())},
                    "test": {"images": int(base_test.sum())}},
          "test": cand["deepdrid_test"], "local_locked_test": cand["local_locked_test"],
          "current_model": cur, "comparison": comp}
    evf = out / f"quality_{stamp}_eval.json"
    evf.write_text(json.dumps(ev, indent=2, default=float))

    L, C = cand["local_locked_test"], (cur or {}).get("local_locked_test")
    print(f"\n  Locked test ({L['n']} images, {L['n_human_reject']} human-rejected)")
    print("                   candidate     current")
    for k, f in (("auroc", "{:.3f}"), ("catch_rate", "{:.1%}"), ("false_reject_rate", "{:.1%}")):
        print(f"    {k:<16} {f.format(L[k]):<13} {f.format(C[k]) if C else '—'}")
    print(f"\n  Not worse than current: {comp['candidate_not_worse']}")
    for w in comp["why"]:
        print(f"    - {w}")

    if not args.no_register:
        import audit
        from registry import register
        v = register("quality", ck, evf, source=f"retrain_quality on {snap.name}",
                     summary={"local_locked_test": L, "deepdrid_test": cand["deepdrid_test"]},
                     comparison=comp, con=audit.connect(), actor="retrain_quality")
        print(f"\n  Registered candidate {v}. It is NOT in service.")
        print(f"  To put it in service:  python learning/registry.py promote quality {v} "
              f"--approved-by \"NAME\" --reason \"...\"")
    return ev


if __name__ == "__main__":
    main()
