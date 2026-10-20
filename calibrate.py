from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CONFIG, Config
from dataset import FundusDataset, build_transforms
from model import load_checkpoint
from predictor import file_sha256
from splits import (load_labels, split_by_patient, split_site_calibration,
                    SITE_SPLIT_SEED)
from train import predict
from metrics import (Calibrator, threshold_at_sensitivity, full_report,
                     print_report, reliability_table, abstention_curve,
                     subgroup_report, screening_metrics, expected_calibration_error)


def _loader(df, cfg, pc):
    ds = FundusDataset(df, cfg.data.cache_dir, build_transforms("val", pc), pc)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.train.num_workers, pin_memory=True)


def _brief(y, prob, thr) -> dict:
    """Headline numbers for a report that is shown, not gated."""
    m = screening_metrics(y, prob, thr)
    m["ece"] = expected_calibration_error(y, prob)
    return m


def main(cfg: Config = CONFIG, site_calibration: bool = False, tta: str = "none"):
    """
    site_calibration: refit calibration, operating point and abstention band
    on one half of the external site (splits.split_site_calibration) and
    report the external numbers on the OTHER half only. The site profile is
    what evaluation.json then serves. See MODEL_CARD.md, "Site calibration".

    tta: test-time augmentation views (model.TTA_VIEWS) averaged into every
    logit here; recorded in evaluation.json so the predictor does the same.
    """
    tc = cfg.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pc, ck = load_checkpoint(tc.save_path, device)
    print(f"  Loaded {tc.save_path} (epoch {ck.get('epoch')}, "
          f"preprocessing {ck['preprocess_fingerprint']}, TTA {tta})")
    if site_calibration and not cfg.data.external_csv_path:
        raise SystemExit("--site-calibration needs an external dataset "
                         "(DR_EXTERNAL_CSV / DR_EXTERNAL_CACHE).")

    df = load_labels(cfg.data)
    splits = split_by_patient(df, cfg.data)

    print("\n[1/4] Fitting calibration on validation")
    val = predict(model, _loader(splits["val"], cfg, pc), device, tta)
    cal = Calibrator().fit(val["referable_logit"], val["y_referable"])
    val_prob = cal.predict(val["referable_logit"])

    thr = threshold_at_sensitivity(val["y_referable"], val_prob, tc.target_sensitivity)
    print(f"  Operating point frozen at p >= {thr:.4f} "
          f"(target sensitivity {tc.target_sensitivity:.2f})")

    print("\n[2/4] Evaluating on the held-out test split")
    test_df = splits["test"]
    test = predict(model, _loader(test_df, cfg, pc), device, tta)
    test_prob = cal.predict(test["referable_logit"])

    rep = full_report(test["y_referable"], test_prob, test_df["patient_id"].values,
                      thr, test["grades_true"], test["grades_pred"])
    # Always the SOURCE profile (fitted on internal val), also in site mode.
    rep["profile"] = "source"
    print_report(rep, "Internal test set")

    print("\n  Reliability:")
    print(reliability_table(test["y_referable"], test_prob, 10)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n[3/4] Abstention / coverage trade-off")
    # Band edges are fitted on VALIDATION and frozen, like the operating point,
    # then applied to test. These edges are what the serving predictor uses.
    ab = abstention_curve(test["y_referable"], test_prob, thr,
                          band_source_prob=val_prob)
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

    ext, site = None, None
    if cfg.data.external_csv_path:
        print("\n[bonus] External (different-source) validation")
        if not cfg.data.external_cache_dir:
            raise SystemExit(
                "\n  data.external_csv_path is set but data.external_cache_dir is not."
                "\n  The external set needs its own preprocessed cache; reusing the"
                "\n  internal one would evaluate the wrong images. Build it with:"
                "\n      python preprocessing.py --csv <external csv> --images <external images>"
                "\n          --cache <external cache> --filename-col <col>"
                "\n  then set DR_EXTERNAL_CACHE=<external cache> in dr.env."
            )
        # Deep copy: a shallow copy shares the DataConfig object, so editing
        # ecfg.data would silently overwrite the internal config as well.
        ecfg = copy.deepcopy(cfg)
        ecfg.data.csv_path = cfg.data.external_csv_path
        ecfg.data.images_dir = cfg.data.external_images_dir
        ecfg.data.cache_dir = cfg.data.external_cache_dir
        edf = load_labels(ecfg.data)
        if not site_calibration:
            e = predict(model, _loader(edf, ecfg, pc), device, tta)
            eprob = cal.predict(e["referable_logit"])
            ext = full_report(e["y_referable"], eprob, edf["patient_id"].values, thr,
                              e["grades_true"], e["grades_pred"])
            print_report(ext, "External test set")
        else:
            site, ext, ab, thr_site, cal_site = _site_calibrate(
                model, edf, ecfg, pc, device, tta, tc.target_sensitivity,
                source=(cal, thr, ab), test=test)
            print_report(ext, "External EVALUATION half (site-calibrated)")
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
    # What the predictor serves: the site profile when one was fitted.
    if site is not None:
        thr, cal = thr_site, cal_site
    out = {
        "operating_point": thr,
        "target_sensitivity": tc.target_sensitivity,
        "calibration": cal.to_dict(),
        "tta": tta,
        "internal_test": rep,
        "external_test": ext,
        "abstention": ab.to_dict(orient="records"),
        "site_calibration": site,
        "subgroups": sub.to_dict(orient="records") if not sub.empty else [],
        "preprocess_fingerprint": pc.fingerprint(),
        # Ties this evidence to the exact weights it was measured on. The
        # service refuses an evaluation.json whose hash does not match the
        # checkpoint it is about to load.
        "checkpoint_sha256": file_sha256(tc.save_path),
        "checkpoint_epoch": ck.get("epoch"),
    }
    with open("artifacts/evaluation.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n  Wrote artifacts/evaluation.json -- this is the evidence base for "
          "your model card.")


def _site_calibrate(model, edf, ecfg, pc, device, tta, target, source, test):
    """
    Fit the site profile on the calibration half; measure on the evaluation
    half, which nothing here is fitted on. Returns
    (site metadata, external report, abstention table, threshold, calibrator).
    """
    src_cal, src_thr, src_ab = source
    parts = split_site_calibration(edf)
    cdf, edf_eval = parts["calibration"], parts["evaluation"]
    c = predict(model, _loader(cdf, ecfg, pc), device, tta)
    e = predict(model, _loader(edf_eval, ecfg, pc), device, tta)

    print("\n  Site calibration (fitted on the calibration half only)")
    cal = Calibrator().fit(c["referable_logit"], c["y_referable"])
    c_prob = cal.predict(c["referable_logit"])
    thr = threshold_at_sensitivity(c["y_referable"], c_prob, target)
    print(f"  Site operating point frozen at p >= {thr:.4f}")

    eprob = cal.predict(e["referable_logit"])
    ext = full_report(e["y_referable"], eprob, edf_eval["patient_id"].values, thr,
                      e["grades_true"], e["grades_pred"])
    ext["profile"] = "site"
    # The served band, fitted on the calibration half, applied to evaluation.
    ab = abstention_curve(e["y_referable"], eprob, thr, band_source_prob=c_prob)

    names = "\n".join(sorted(edf_eval["filename"].astype(str)))
    site = {
        "dataset": str(ecfg.data.csv_path),
        "split_seed": SITE_SPLIT_SEED,
        "n_calibration": int(len(cdf)),
        "n_evaluation": int(len(edf_eval)),
        "split_unit": "image (no patient ids)",
        "evaluation_filenames_sha256": hashlib.sha256(names.encode()).hexdigest(),
        # The same held-out half WITHOUT site calibration, for contrast.
        "evaluation_half_under_source_profile": _brief(
            e["y_referable"], src_cal.predict(e["referable_logit"]), src_thr),
        # Internal test under the site profile. Reported, not gated: this
        # profile is fitted for the external site, not for the source one.
        "internal_test_under_site_profile": _brief(
            test["y_referable"], cal.predict(test["referable_logit"]), thr),
        "source_profile": {"operating_point": src_thr,
                           "calibration": src_cal.to_dict(),
                           "abstention_on_internal_test": src_ab.to_dict(orient="records")},
    }
    return site, ext, ab, thr, cal


if __name__ == "__main__":
    from model import TTA_VIEWS
    ap = argparse.ArgumentParser(description="Calibrate, freeze the operating "
                                 "point, evaluate held-out data once.")
    ap.add_argument("--site-calibration", action="store_true",
                    help="fit the served profile on half of the external site; "
                         "report on the other half")
    ap.add_argument("--tta", choices=list(TTA_VIEWS), default="none")
    a = ap.parse_args()
    main(site_calibration=a.site_calibration, tta=a.tta)