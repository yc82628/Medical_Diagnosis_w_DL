from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CONFIG, Config
from dataset import FundusDataset, build_transforms
from model import load_checkpoint
from splits import load_labels, split_by_patient
from train import predict
from metrics import (Calibrator, threshold_at_sensitivity, full_report,
                     print_report, reliability_table, abstention_curve,
                     subgroup_report)


def _loader(df, cfg, pc):
    ds = FundusDataset(df, cfg.data.cache_dir, build_transforms("val", pc), pc)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.train.num_workers, pin_memory=True)


def main(cfg: Config = CONFIG):
    tc = cfg.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pc, ck = load_checkpoint(tc.save_path, device)
    print(f"  Loaded {tc.save_path} (epoch {ck.get('epoch')}, "
          f"preprocessing {ck['preprocess_fingerprint']})")

    df = load_labels(cfg.data)
    splits = split_by_patient(df, cfg.data)

    print("\n[1/4] Fitting calibration on validation")
    val = predict(model, _loader(splits["val"], cfg, pc), device)
    cal = Calibrator().fit(val["referable_logit"], val["y_referable"])
    val_prob = cal.predict(val["referable_logit"])

    thr = threshold_at_sensitivity(val["y_referable"], val_prob, tc.target_sensitivity)
    print(f"  Operating point frozen at p >= {thr:.4f} "
          f"(target sensitivity {tc.target_sensitivity:.2f})")

    print("\n[2/4] Evaluating on the held-out test split")
    test_df = splits["test"]
    test = predict(model, _loader(test_df, cfg, pc), device)
    test_prob = cal.predict(test["referable_logit"])

    rep = full_report(test["y_referable"], test_prob, test_df["patient_id"].values,
                      thr, test["grades_true"], test["grades_pred"])
    print_report(rep, "Internal test set")

    print("\n  Reliability:")
    print(reliability_table(test["y_referable"], test_prob, 10)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n[3/4] Abstention / coverage trade-off")
    ab = abstention_curve(test["y_referable"], test_prob, thr)
    print(ab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  Read this as: 'at X% coverage the system auto-decides X% of cases "
          "at this sensitivity, and routes the rest to a human grader.'")

    print("\n[4/4] Subgroup performance at the SHARED operating point")
    sdf = test_df.copy()
    sdf["_y"], sdf["_s"] = test["y_referable"], test_prob
    sub = subgroup_report(sdf, "_y", "_s", thr, cfg.data.subgroup_cols)
    if sub.empty:
        print("  No subgroup columns present. This is a gap, not a pass: without "
              "age/sex/device metadata you cannot detect a disparity, and "
              "'we did not measure it' is not the same as 'it is not there'.")
    else:
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if sub["underpowered"].any():
            print("  ! rows flagged underpowered have too few positive cases for a "
                  "stable estimate. Report them as inconclusive, not as evidence "
                  "of fairness.")

    ext = None
    if cfg.data.external_csv_path:
        print("\n[bonus] External (different-source) validation")
        ecfg = Config(**{**cfg.__dict__})
        ecfg.data.csv_path = cfg.data.external_csv_path
        ecfg.data.images_dir = cfg.data.external_images_dir
        edf = load_labels(ecfg.data)
        e = predict(model, _loader(edf, ecfg, pc), device)
        eprob = cal.predict(e["referable_logit"])
        ext = full_report(e["y_referable"], eprob, edf["patient_id"].values, thr,
                          e["grades_true"], e["grades_pred"])
        print_report(ext, "External test set")
        drop = rep["auroc_ci"]["point"] - ext["auroc_ci"]["point"]
        print(f"\n  AUROC drop internal -> external: {drop:.3f}")
        print("  A drop of 0.05-0.15 is normal and expected. It is the honest "
              "estimate of what happens at a new site. THIS is the number to "
              "report, not the internal one.")
    else:
        print("\n  ! No external dataset configured (data.external_csv_path is None).")
        print("    Every number above is INTERNAL and therefore provisional. A random")
        print("    split from one source cannot tell you whether the model survives a")
        print("    different camera, population, or capture protocol.")

    Path("artifacts").mkdir(exist_ok=True)
    out = {
        "operating_point": thr,
        "target_sensitivity": tc.target_sensitivity,
        "calibration": cal.to_dict(),
        "internal_test": rep,
        "external_test": ext,
        "abstention": ab.to_dict(orient="records"),
        "subgroups": sub.to_dict(orient="records") if not sub.empty else [],
        "preprocess_fingerprint": pc.fingerprint(),
    }
    with open("artifacts/evaluation.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n  Wrote artifacts/evaluation.json -- this is the evidence base for "
          "your model card.")


if __name__ == "__main__":
    main()