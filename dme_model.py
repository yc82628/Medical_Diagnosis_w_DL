"""
Maculopathy (diabetic macular edema risk) from a colour fundus photograph.

Many screening programmes refer for maculopathy on its own: hard exudates near
the centre of vision threaten sight sooner than the DR grade suggests, and the
ophthalmologist needs to know it is the reason for the referral.

Labels: IDRiD "risk of macular edema", graded from the photograph:
    0  no hard exudates
    1  exudates, but outside one disc diameter of the macula centre
    2  exudates within one disc diameter of the macula centre  -> REFERABLE
That is the standard screening proxy (e.g. the UK programme's M1). It is NOT
an OCT diagnosis of macular oedema; see INTENDED_USE.md.

Training
    python dme_model.py                # 5-fold cross-validation, then the final model
    python dme_model.py --folds 0      # final model only (needs a previous CV run)

The network starts from the trained DR model's feature extractor (it already
responds to exudates) with a fresh 3-grade ordinal head. IDRiD is the only
open dataset here with these labels, so performance is measured by 5-fold
cross-validation within it. Every fold picks its own epoch and threshold on
an inner validation split; the out-of-fold numbers are what is reported.

Status: the model is "validated" only when an EXTERNAL, different-site
evaluation exists and passes DME_ACCEPTANCE. Cross-validation within one site
cannot show what happens at another, so until then it is "experimental": the
platform shows its result to graders as advice, and it never changes a
referral decision on its own (encounters.combine).

IDRiD has no patient ids: if a patient contributed both eyes, the two can land
in different folds, so the cross-validated numbers may be slightly optimistic.

Licence of the training data: IDRiD, CC BY 4.0 (Porwal et al., 2018).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

DME_GRADES = 3
REFERABLE_DME = 2
DME_LABELS = {0: "No exudates", 1: "Exudates away from the macula",
              2: "Exudates near the macula"}
DME_CHECKPOINT = Path("checkpoints/dme_model.pt")
DME_EVAL = Path("artifacts/dme_eval.json")
DR_CHECKPOINT = Path("checkpoints/best_model.pt")

# Floors an EXTERNAL evaluation must clear for status "validated". The same
# spirit as config.AcceptanceConfig for the DR model.
DME_ACCEPTANCE = {"min_auroc": 0.85, "min_sensitivity": 0.85,
                  "min_specificity": 0.50, "min_positives": 30}


# ---------------------------------------------------------------------------
# Status (torch-free)
# ---------------------------------------------------------------------------

def dme_status(meta: dict) -> tuple[str, list[str]]:
    """("validated" | "experimental", reasons). Reasons explain an experimental status."""
    ext = meta.get("external")
    if not ext:
        return "experimental", ["evaluated by cross-validation within a single site "
                                "(IDRiD) only; no external, different-site test yet"]
    a, why = DME_ACCEPTANCE, []
    pos = int(ext.get("tp", 0)) + int(ext.get("fn", 0))
    if pos < a["min_positives"]:
        why.append(f"external test has {pos} referable cases (need {a['min_positives']})")
    for key, floor in (("auroc", a["min_auroc"]), ("sensitivity", a["min_sensitivity"]),
                       ("specificity", a["min_specificity"])):
        v = ext.get(key)
        if v is None or not np.isfinite(v) or v < floor:
            why.append(f"external {key} {v} below {floor}")
    return ("validated", []) if not why else ("experimental", why)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

class DMEPredictor:
    """Loads checkpoints/dme_model.pt and scores one BGR image."""

    def __init__(self, path=DME_CHECKPOINT, device=None):
        import torch
        from model import load_checkpoint
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.pc, ck = load_checkpoint(path, self.device)
        if ck.get("task") != "dme":
            raise RuntimeError(f"{path} is not a maculopathy checkpoint.")
        self.threshold = float(ck["threshold"])
        self.calibration = ck.get("calibration") or {"method": "identity"}
        self.tta = ck.get("tta", "none")
        self.meta = ck.get("meta", {})
        self.status, self.status_reasons = dme_status(self.meta)
        self.version = f"{ck.get('backbone', '?')}-dme@{file_sha256(path)[:8]}"
        self.lock = threading.Lock()

    def _calibrate(self, logit: float) -> float:
        m = self.calibration.get("method", "identity")
        if m == "temperature":
            z = logit / (self.calibration.get("temperature") or 1.0)
        elif m == "isotonic":
            xs = np.asarray(self.calibration["isotonic_x"], float)
            ys = np.asarray(self.calibration["isotonic_y"], float)
            return float(np.interp(logit, xs, ys, left=ys[0], right=ys[-1]))
        else:
            z = logit
        return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))

    def assess(self, img_bgr) -> dict:
        import cv2
        from model import tta_logits
        from preprocessing import preprocess_image
        proc = preprocess_image(img_bgr, self.pc)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array(self.pc.norm_mean)) / np.array(self.pc.norm_std)
        x = self.torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        with self.lock, self.torch.no_grad():
            logits = tta_logits(self.model, x, self.tta).cpu().numpy()[0]
        cum = np.minimum.accumulate(1 / (1 + np.exp(-np.clip(logits, -30, 30))))
        grade = int((cum > 0.5).sum())
        p = self._calibrate(float(logits[REFERABLE_DME - 1]))
        return {"probability": p, "suspected": p >= self.threshold,
                "threshold": self.threshold, "grade": grade,
                "grade_label": DME_LABELS[grade], "status": self.status,
                "model": self.version}


_default = {"loaded": False, "predictor": None, "note": ""}
_default_lock = threading.Lock()


def default_dme():
    """The service's maculopathy model (DR_DME_CHECKPOINT or checkpoints/dme_model.pt), or None."""
    with _default_lock:
        if not _default["loaded"]:
            path = Path(os.environ.get("DR_DME_CHECKPOINT") or DME_CHECKPOINT)
            pred, note = None, ""
            if not path.exists():
                note = f"No maculopathy model at {path}: maculopathy is not assessed."
            else:
                try:
                    pred = DMEPredictor(path)
                    note = (f"Maculopathy model {pred.version} ({pred.status}): "
                            + ("may change the referral decision." if pred.status == "validated"
                               else "shown to graders as advice only."))
                except ImportError:
                    note = "PyTorch is not installed: maculopathy is not assessed."
            _default.update(loaded=True, predictor=pred, note=note)
        return _default["predictor"]


def default_dme_note() -> str:
    default_dme()
    return _default["note"]


def reset_default_dme() -> None:
    with _default_lock:
        _default.update(loaded=False, predictor=None, note="")


# ---------------------------------------------------------------------------
# Training (needs torch)
# ---------------------------------------------------------------------------

def _ordinal(grade: int) -> np.ndarray:
    return np.array([grade > k for k in range(DME_GRADES - 1)], dtype=np.float32)


def _dataset(df, cache_dir, transform):
    import cv2
    import torch
    from torch.utils.data import Dataset
    from preprocessing import cache_name

    class _DS(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            r = df.iloc[i]
            img = cv2.imread(str(Path(cache_dir) / cache_name(r["filename"])), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Not in cache: {r['filename']}")
            img = transform(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"]
            g = int(r["dme_risk"])
            return {"image": img, "ordinal": torch.from_numpy(_ordinal(g)), "grade": g}
    return _DS()


def _new_model(pc, device, init_from: Path | None):
    from model import DRModel, load_checkpoint
    if init_from and init_from.exists():
        dr, dr_pc, ck = load_checkpoint(init_from, "cpu")
        if dr_pc.fingerprint() != pc.fingerprint():
            raise SystemExit("  DR checkpoint preprocessing differs from the cache's.")
        m = DRModel(ck["backbone"], DME_GRADES, pretrained=False)
        m.features.load_state_dict(dr.features.state_dict())
    else:
        m = DRModel("efficientnet_b3", DME_GRADES, pretrained=True)
    return m.to(device)


def _predict(model, df, cache_dir, pc, device, tta, batch=8):
    import torch
    from torch.utils.data import DataLoader
    from dataset import build_transforms
    from model import tta_logits
    dl = DataLoader(_dataset(df, cache_dir, build_transforms("val", pc)), batch_size=batch)
    out = []
    model.eval()
    with torch.no_grad():
        for b in dl:
            out.append(tta_logits(model, b["image"].to(device), tta).cpu().numpy())
    return np.concatenate(out)


def train_one(train_df, val_df, cache_dir, pc, device, epochs, seed, init_from,
              augment="strong", log=print):
    """Train on train_df; keep the epoch with the best val AUROC. Returns (model, epoch, auroc)."""
    import copy
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from dataset import build_transforms
    from model import OrdinalLoss, param_groups

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _new_model(pc, device, init_from)
    ds = _dataset(train_df, cache_dir, build_transforms("train", pc, augment))
    dl = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    y = train_df["dme_risk"].to_numpy()
    pos = np.array([(y > k).mean() for k in range(DME_GRADES - 1)])
    pw = torch.tensor((1 - pos) / np.clip(pos, 1e-3, None), dtype=torch.float32, device=device)
    loss_fn = OrdinalLoss(pos_weight=pw,
                          threshold_weight=torch.tensor([1.0, 2.0], device=device)).to(device)
    opt = torch.optim.AdamW(param_groups(model, 3e-4, 0.1, 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    yv = (val_df["dme_risk"].to_numpy() >= REFERABLE_DME).astype(int)
    best = (-1.0, 0, None)
    for ep in range(1, epochs + 1):
        model.train()
        for b in dl:
            x, t = b["image"].to(device), b["ordinal"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                loss = loss_fn(model(x), t)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
        lv = _predict(model, val_df, cache_dir, pc, device, "none")[:, REFERABLE_DME - 1]
        auc = roc_auc_score(yv, lv) if 0 < yv.sum() < len(yv) else float("nan")
        log(f"    epoch {ep:>2}/{epochs}  val AUROC {auc:.3f}")
        if auc > best[0]:
            best = (auc, ep, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    return model, best[1], best[0]


def cross_validate(df, cache_dir, pc, device, folds, epochs, seed, init_from, tta, log=print):
    """Out-of-fold P(referable maculopathy), with each fold's own frozen threshold."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from metrics import threshold_at_sensitivity

    oof_logit = np.full(len(df), np.nan)
    oof_thr = np.full(len(df), np.nan)
    fold_of = np.full(len(df), -1)
    epochs_used = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(skf.split(df, df["dme_risk"])):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        fit_df, val_df = train_test_split(tr_df, test_size=0.15, random_state=seed + f,
                                          stratify=tr_df["dme_risk"])
        log(f"  fold {f + 1}/{folds}: train {len(fit_df)}, inner val {len(val_df)}, test {len(te_df)}")
        model, ep, auc = train_one(fit_df, val_df, cache_dir, pc, device, epochs, seed + f,
                                   init_from, log=log)
        epochs_used.append(ep)
        v = _predict(model, val_df, cache_dir, pc, device, tta)[:, REFERABLE_DME - 1]
        thr = threshold_at_sensitivity((val_df["dme_risk"] >= REFERABLE_DME).astype(int), v, 0.90)
        t = _predict(model, te_df, cache_dir, pc, device, tta)
        oof_logit[te] = t[:, REFERABLE_DME - 1]
        oof_thr[te] = thr
        fold_of[te] = f
        log(f"    fold {f + 1}: best epoch {ep} (inner AUROC {auc:.3f}), threshold logit {thr:.3f}")
    return oof_logit, oof_thr, fold_of, epochs_used


def report(df, oof_logit, oof_thr) -> dict:
    from metrics import bootstrap_ci, expected_calibration_error
    from sklearn.metrics import roc_auc_score
    y = (df["dme_risk"].to_numpy() >= REFERABLE_DME).astype(int)
    pred = (oof_logit >= oof_thr).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    ids = df["patient_id"].to_numpy()
    auc = bootstrap_ci(y, oof_logit, ids, roc_auc_score)
    sens = bootstrap_ci(y, (oof_logit >= oof_thr).astype(float), ids,
                        lambda a, b: ((b == 1) & (a == 1)).sum() / max((a == 1).sum(), 1))
    spec = bootstrap_ci(y, (oof_logit >= oof_thr).astype(float), ids,
                        lambda a, b: ((b == 0) & (a == 0)).sum() / max((a == 0).sum(), 1))
    p = 1 / (1 + np.exp(-np.clip(oof_logit, -30, 30)))
    return {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, oof_logit)), "auroc_ci": auc,
            "sensitivity": tp / max(tp + fn, 1), "sensitivity_ci": sens,
            "specificity": tn / max(tn + fp, 1), "specificity_ci": spec,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "ece_uncalibrated": float(expected_calibration_error(y, p))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the maculopathy model on IDRiD.")
    ap.add_argument("--labels", default="idrid/labels.csv")
    ap.add_argument("--cache", default="idrid/cache_512")
    ap.add_argument("--folds", type=int, default=5, help="0 = skip cross-validation")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--tta", default="d8")
    ap.add_argument("--init-from", default=str(DR_CHECKPOINT))
    args = ap.parse_args(argv)

    import pandas as pd
    import torch
    from metrics import Calibrator, threshold_at_sensitivity
    from model import load_checkpoint, save_checkpoint

    df = pd.read_csv(args.labels).dropna(subset=["dme_risk"]).reset_index(drop=True)
    df["dme_risk"] = df["dme_risk"].astype(int)
    q = Path(args.cache) / "quality_report.csv"
    if q.exists():
        bad = set(pd.read_csv(q).query("gradable == False")["filename"])
        df = df[~df["filename"].isin(bad)].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init = Path(args.init_from)
    _, pc, _ = load_checkpoint(init, "cpu")          # same preprocessing as the DR model
    print(f"  {len(df)} images; referable maculopathy {(df.dme_risk >= 2).mean():.0%}; "
          f"device {device}; initialised from {init}")

    meta_path = DME_EVAL
    if args.folds:
        oof, thr, fold_of, eps = cross_validate(df, args.cache, pc, device, args.folds,
                                                args.epochs, args.seed, init, args.tta)
        rep = report(df, oof, thr)
        cal = Calibrator().fit(oof, (df["dme_risk"] >= REFERABLE_DME).astype(int))
        cv = {"folds": args.folds, "epochs_chosen": eps, "tta": args.tta,
              "seed": args.seed, "oof": rep, "calibration": cal.to_dict()}
        print("\n  Cross-validated (out-of-fold, each fold's own threshold):")
        print(f"    AUROC {rep['auroc']:.3f} ({rep['auroc_ci']['lo']:.3f}-{rep['auroc_ci']['hi']:.3f})"
              f"   sensitivity {rep['sensitivity']:.3f}   specificity {rep['specificity']:.3f}"
              f"   missed {rep['fn']}/{rep['positives']}")
        np.save(Path("artifacts/dme_oof.npy"), np.stack([oof, thr, fold_of]))
    else:
        cv = json.loads(meta_path.read_text())["cross_validation"]
        oof, thr, _ = np.load(Path("artifacts/dme_oof.npy"))
        cal = Calibrator().fit(oof, (df["dme_risk"] >= REFERABLE_DME).astype(int))

    # Final model: 85% to train, 15% to pick the epoch (up to the folds' median).
    epochs = int(np.median(cv["epochs_chosen"]))
    print(f"\n  Final model: {len(df)} images (85% train, 15% to choose the epoch), "
          f"up to {epochs} epochs")
    from sklearn.model_selection import train_test_split
    fit_df, val_df = train_test_split(df, test_size=0.15, random_state=args.seed,
                                      stratify=df["dme_risk"])
    model, _, _ = train_one(fit_df, val_df, args.cache, pc, device, epochs, args.seed, init)
    # Threshold on the calibrated out-of-fold scores: sensitivity >= 0.90.
    y = (df["dme_risk"] >= REFERABLE_DME).astype(int)
    threshold = threshold_at_sensitivity(y, cal.predict(oof), 0.90)
    meta = {"cross_validation": cv, "external": None,
            "trained_on": "idrid/labels.csv", "n_train": int(len(df))}
    DME_CHECKPOINT.parent.mkdir(exist_ok=True)
    save_checkpoint(DME_CHECKPOINT, model, pc,
                    extra={"task": "dme", "threshold": float(threshold),
                           "calibration": cal.to_dict(), "tta": args.tta, "meta": meta})
    meta["checkpoint_sha256"] = file_sha256(DME_CHECKPOINT)
    meta["threshold"] = float(threshold)
    meta["status"] = dme_status(meta)[0]
    Path("artifacts").mkdir(exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, default=float))
    print(f"  Saved {DME_CHECKPOINT} (threshold {threshold:.3f} on calibrated P, "
          f"status {meta['status']}) and {meta_path}")


if __name__ == "__main__":
    main()
