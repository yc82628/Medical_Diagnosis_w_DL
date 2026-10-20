"""
Score candidate models for the external-site gap WITHOUT touching held-out data.

Each run is scored on:
  * DeepDRiD validation (the internal model-selection split), and
  * the CALIBRATION half of the external site (splits.split_site_calibration).

The internal test split and the external EVALUATION half are never loaded
here. They are read exactly once, by calibrate.py, for the configuration
chosen from this table. Choosing on the numbers you then report would turn
the "held-out" result into a tuned one.

Two views of the external calibration half are reported:
  source-calibrated  calibrator and operating point fitted on DeepDRiD val
                     and applied unchanged (what calibrate.py does by default)
  site-calibrated    calibrator and operating point refitted on the site,
                     estimated by 2-fold cross-fitting INSIDE the calibration
                     half, repeated over 5 shuffles (what --site-calibration
                     in calibrate.py would give, without using its eval half)

    python domain_experiments.py --name b3-strong --checkpoints checkpoints/b3_strong.pt --tta d8
    python domain_experiments.py --table
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG, Config
from dataset import FundusDataset, build_transforms
from model import TTA_VIEWS
from train import predict
from metrics import (Calibrator, threshold_at_sensitivity, screening_metrics,
                     expected_calibration_error)
from predictor import file_sha256
from splits import load_labels, split_by_patient, split_site_calibration

OUT = Path("artifacts/experiments")


def _loader(df, cache_dir, pc, cfg):
    ds = FundusDataset(df, cache_dir, build_transforms("val", pc), pc)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.train.num_workers, pin_memory=True)


def selection_sets(cfg: Config = CONFIG) -> dict:
    """DeepDRiD val and the external CALIBRATION half. Nothing held out."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_labels(cfg.data)
        val = split_by_patient(df, cfg.data)["val"]
        ecfg = copy.deepcopy(cfg)
        ecfg.data.csv_path = cfg.data.external_csv_path
        ecfg.data.cache_dir = cfg.data.external_cache_dir
        site_cal = split_site_calibration(load_labels(ecfg.data))["calibration"]
    return {"val": (val, cfg.data.cache_dir),
            "site_cal": (site_cal, cfg.data.external_cache_dir)}


def collect(checkpoint: str, tta: str, sets: dict, cfg: Config = CONFIG) -> dict:
    """Logits per set, cached on (checkpoint hash, tta) so re-scoring is free."""
    from model import load_checkpoint

    sha = file_sha256(checkpoint)[:16]
    cache = OUT / "logits" / f"{sha}_{tta}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in sets}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pc, _ = load_checkpoint(checkpoint, device)
    res = {k: predict(model, _loader(df, cdir, pc, cfg), device, tta)["logits"]
           for k, (df, cdir) in sets.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **res)
    return res


def _quiet_calibrator(logits, y):
    with contextlib.redirect_stdout(io.StringIO()):
        return Calibrator().fit(logits, y)


def score(val_logit, y_val, site_logit, y_site, target=0.90, repeats=5) -> dict:
    vm = screening_metrics(y_val, val_logit,
                           threshold_at_sensitivity(y_val, val_logit, target))
    cal = _quiet_calibrator(val_logit, y_val)
    vprob = cal.predict(val_logit)
    thr = threshold_at_sensitivity(y_val, vprob, target)
    sprob = cal.predict(site_logit)
    src = screening_metrics(y_site, sprob, thr)
    row = {"val_auroc": vm["auroc"], "val_spec_at_sens90": vm["specificity"],
           "site_auroc_raw": screening_metrics(y_site, site_logit, 0)["auroc"],
           "src_auroc": src["auroc"], "src_sens": src["sensitivity"],
           "src_spec": src["specificity"],
           "src_ece": expected_calibration_error(y_site, sprob)}

    # Site calibration, cross-fitted inside the calibration half.
    from sklearn.model_selection import StratifiedKFold
    sens, spec, ece = [], [], []
    for r in range(repeats):
        pred = np.full(len(y_site), np.nan)
        thr_of = np.full(len(y_site), np.nan)
        for fit_i, ev_i in StratifiedKFold(2, shuffle=True, random_state=r).split(site_logit, y_site):
            c = _quiet_calibrator(site_logit[fit_i], y_site[fit_i])
            t = threshold_at_sensitivity(y_site[fit_i], c.predict(site_logit[fit_i]), target)
            pred[ev_i], thr_of[ev_i] = c.predict(site_logit[ev_i]), t
        y = y_site.astype(int)
        hit = pred >= thr_of
        sens.append(hit[y == 1].mean())
        spec.append((~hit[y == 0]).mean())
        ece.append(expected_calibration_error(y, pred))
    row.update({"site_sens": float(np.mean(sens)), "site_sens_sd": float(np.std(sens)),
                "site_spec": float(np.mean(spec)),
                "site_ece": float(np.mean(ece)), "site_ece_sd": float(np.std(ece))})
    return row


def run(name: str, checkpoints: list[str], tta: str, note: str = "") -> dict:
    sets = selection_sets()
    y = {k: df["referable"].values for k, (df, _) in sets.items()}
    per = [collect(c, tta, sets) for c in checkpoints]
    # Ensemble = mean of the referable logits (index 1, P(grade >= 2)).
    lg = {k: np.mean([p[k][:, 1] for p in per], axis=0) for k in sets}
    row = {"name": name, "checkpoints": checkpoints, "tta": tta, "note": note,
           "n_val": len(y["val"]), "n_site_cal": len(y["site_cal"]),
           **score(lg["val"], y["val"], lg["site_cal"], y["site_cal"])}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    return row


def table() -> str:
    from model_card import experiments_table
    return experiments_table([json.loads(l) for l in open(OUT / "results.jsonl")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name")
    ap.add_argument("--checkpoints", nargs="+")
    ap.add_argument("--tta", choices=list(TTA_VIEWS), default="none")
    ap.add_argument("--note", default="")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        print(table())
    else:
        run(a.name, a.checkpoints, a.tta, a.note)
