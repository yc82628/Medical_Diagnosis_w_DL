"""
Learned image-quality (gradability) model.

The classical gate in preprocessing.assess_quality (sharpness, field of view,
exposure) was measured against DeepDRiD's 2,000 human quality ratings and
carries almost no signal (AUROC 0.51 for sharpness). This module trains a small
CNN on those same ratings and, when its checkpoint exists, predictor.run_gates
uses it for the gradability decision. The classical checks stay as a backstop
for gross failures, and are the whole gate when no checkpoint exists.

    python quality_model.py --labels deepdrid/quality_labels.csv \
        --images deepdrid/repo/regular_fundus_images

Split by PATIENT (the id prefix of the image name, "347_l1" -> 347), never by
image: both eyes and both captures of a patient look alike, and an image-level
split would let the model recognise patients instead of quality.

The threshold is chosen on validation so that at most --max-false-reject of
human-ACCEPTED images are rejected; test is touched once, at the end.

Writes:
  checkpoints/quality_model.pt         weights, threshold, input fingerprint, split
  artifacts/quality_model_eval.json    validation + held-out test metrics, and the
                                       classical gate on the same test patients

Everything above the training section is torch-free, so the serving wiring and
the threshold logic are unit-tested without torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from preprocessing import _pad_to_square, circle_crop

QUALITY_CHECKPOINT = "checkpoints/quality_model.pt"
QUALITY_EVAL = "artifacts/quality_model_eval.json"


# ---------------------------------------------------------------------------
# Input pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityInputConfig:
    """
    Image -> tensor pipeline for the quality model. Frozen and hashed into the
    checkpoint, like config.PreprocessConfig for the DR model.

    Deliberately NOT the DR pipeline: Ben Graham normalisation removes uneven
    illumination, haze and exposure faults -- exactly what a grader rejects an
    image for. The quality model sees the raw colour image, cropped and resized.
    """
    image_size: int = 448
    circle_crop: bool = True
    norm_mean: tuple = (0.485, 0.456, 0.406)      # ImageNet: raw RGB input
    norm_std: tuple = (0.229, 0.224, 0.225)

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def prepare_input(img_bgr: np.ndarray, cfg: QualityInputConfig) -> np.ndarray:
    """Crop to the retinal disc, pad square, resize. Returns uint8 BGR."""
    img = circle_crop(img_bgr) if cfg.circle_crop else img_bgr
    img = _pad_to_square(img)
    return cv2.resize(img, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)


def to_chw_float(img_bgr: np.ndarray, cfg: QualityInputConfig) -> np.ndarray:
    """Prepared uint8 BGR -> normalised float32 RGB, C x H x W."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(cfg.norm_mean, np.float32)) / np.array(cfg.norm_std, np.float32)
    return rgb.transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# Splits, threshold, metrics (torch-free)
# ---------------------------------------------------------------------------

def patient_of(filename: str) -> str:
    """'regular-fundus-training/Images/347/347_l1.jpg' -> '347'."""
    return Path(str(filename).replace("\\", "/")).stem.split("_")[0]


def split_patients(filenames, rejected, seed: int = 0, n_folds: int = 5) -> dict:
    """
    Patient-grouped train/val/test split, stratified on the reject label.

    One fold is test, one is validation, the rest train (60/20/20 at 5 folds).
    Returns {"train"|"val"|"test": boolean mask over the input rows}.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.array([patient_of(f) for f in filenames])
    y = np.asarray(rejected).astype(int)
    folds = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                 .split(np.zeros(len(y)), y, groups))
    test = np.zeros(len(y), bool)
    val = np.zeros(len(y), bool)
    test[folds[0][1]] = True
    val[folds[1][1]] = True
    masks = {"train": ~(test | val), "val": val, "test": test}
    for a in masks:
        for b in masks:
            if a < b and set(groups[masks[a]]) & set(groups[masks[b]]):
                raise RuntimeError(f"patients shared between {a} and {b}")
    return masks


def choose_threshold(rejected, p_ungradable, max_false_reject: float = 0.05) -> float:
    """
    Lowest threshold on P(ungradable) that rejects at most `max_false_reject`
    of human-ACCEPTED images. An image is rejected when p >= threshold.

    The constraint is on false rejects because each one costs a recapture and
    erodes trust in the gate; within it, the lowest threshold catches the most
    human-rejected images. Chosen on validation, then frozen.
    """
    y = np.asarray(rejected).astype(bool)
    p = np.asarray(p_ungradable, dtype=float)
    acc = np.sort(p[~y])[::-1]                       # human-accepted, descending
    if acc.size == 0:
        return 0.5
    k = int(np.floor(max_false_reject * acc.size))   # false rejects allowed
    if k >= acc.size:
        return float(acc[-1])
    # Strictly above the (k+1)-th highest accepted score: at most k reach it.
    return float(np.nextafter(acc[k], np.inf))


def gate_metrics(rejected, p_ungradable, threshold: float) -> dict:
    """Catch rate, false-reject rate and AUROC, with human REJECT as positive."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(rejected).astype(bool)
    p = np.asarray(p_ungradable, dtype=float)
    pred = p >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    n_rej, n_acc = int(y.sum()), int((~y).sum())
    return {
        "n": int(len(y)), "n_human_reject": n_rej, "n_human_accept": n_acc,
        "threshold": float(threshold),
        "catch_rate": tp / n_rej if n_rej else float("nan"),
        "missed_rejects": n_rej - tp,
        "false_reject_rate": fp / n_acc if n_acc else float("nan"),
        "false_rejects": fp,
        "auroc": float(roc_auc_score(y, p)) if 0 < n_rej < len(y) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

class LearnedQualityGate:
    """
    The trained quality model, as run_gates uses it.

    The input fingerprint is verified at load: a mismatch is a hard error, for
    the same reason as model.load_checkpoint -- silent train/serve skew.
    """

    def __init__(self, path=QUALITY_CHECKPOINT, device=None):
        import torch

        self.torch = torch
        self.path = str(path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(self.path, map_location=self.device, weights_only=False)
        self.cfg = QualityInputConfig(**{k: tuple(v) if isinstance(v, list) else v
                                         for k, v in ck["input"].items()})
        if self.cfg.fingerprint() != ck.get("input_fingerprint"):
            raise RuntimeError(
                f"Quality checkpoint {self.path}: input fingerprint does not match its "
                f"own stored config. The checkpoint is corrupt or was written by a "
                f"different version of quality_model.py.")
        self.threshold = float(ck["threshold"])
        self.backbone = ck["backbone"]
        self.version = f"{self.backbone}-quality@{self.cfg.fingerprint()}"
        # Images the model trained or was tuned on: validators must leave them out.
        self.seen_files = set(ck.get("seen_files") or [])
        self.net = build_net(self.backbone, pretrained=False)
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(self.device).eval()
        self.lock = threading.Lock()

    def p_ungradable(self, img_bgr: np.ndarray) -> float:
        torch = self.torch
        x = to_chw_float(prepare_input(img_bgr, self.cfg), self.cfg)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with self.lock, torch.no_grad():
            logit = self.net(x).float().cpu().numpy().ravel()[0]
        return float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))

    def assess(self, img_bgr: np.ndarray) -> dict:
        p = self.p_ungradable(img_bgr)
        ok = p < self.threshold
        return {"gradable": ok, "p_ungradable": p, "threshold": self.threshold,
                "model": self.version,
                "reasons": [] if ok else [
                    # No nested brackets: the web page strips "(...)" from reasons.
                    f"learned quality model judges the image too poor to grade "
                    f"(probability ungradable {p:.2f}, threshold {self.threshold:.2f})"]}


_default = {"loaded": False, "gate": None, "note": ""}
_default_lock = threading.Lock()


def default_quality_gate():
    """
    The gate the service uses: the checkpoint at DR_QUALITY_CHECKPOINT (default
    checkpoints/quality_model.pt) if it exists, else None (classical only).
    Loaded once. A corrupt checkpoint raises rather than silently falling back.
    """
    with _default_lock:
        if not _default["loaded"]:
            path = Path(os.environ.get("DR_QUALITY_CHECKPOINT") or QUALITY_CHECKPOINT)
            gate, note = None, ""
            if not path.exists():
                note = (f"No learned quality model at {path}: the classical image-quality "
                        f"checks decide gradability on their own, and they barely "
                        f"separate gradable from ungradable images.")
            else:
                try:
                    gate = LearnedQualityGate(path)
                    note = (f"Learned quality model {gate.version} decides gradability "
                            f"(threshold {gate.threshold:.3f}); classical checks are a "
                            f"backstop.")
                except ImportError:
                    note = (f"PyTorch is not installed, so the learned quality model at "
                            f"{path} cannot be loaded: classical image-quality checks only.")
            _default.update(loaded=True, gate=gate, note=note)
        return _default["gate"]


def default_quality_note() -> str:
    default_quality_gate()
    return _default["note"]


def reset_default_quality_gate() -> None:
    """Forget the loaded gate (tests, or after replacing the checkpoint)."""
    with _default_lock:
        _default.update(loaded=False, gate=None, note="")


# ---------------------------------------------------------------------------
# Training (needs torch)
# ---------------------------------------------------------------------------

def build_net(backbone: str, pretrained: bool = True):
    """Pretrained torchvision backbone with a single P(ungradable) logit."""
    import torch.nn as nn
    from torchvision import models

    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, 1)
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"Unsupported quality backbone {backbone!r}")
    return net


def _cache_one(args):
    src, dst, cfg = args
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return str(src), "unreadable"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), prepare_input(img, cfg), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return str(src), None


def build_input_cache(filenames, images_dir: Path, cache_dir: Path,
                      cfg: QualityInputConfig, workers: int = 8) -> list[Path]:
    """Prepare every image once (decoding full-size JPEGs every epoch is the bottleneck)."""
    from concurrent.futures import ProcessPoolExecutor

    from preprocessing import cache_name

    cache_dir = cache_dir / cfg.fingerprint()
    out = [cache_dir / cache_name(f) for f in filenames]
    jobs = [(images_dir / f, d, cfg) for f, d in zip(filenames, out) if not d.exists()]
    if jobs:
        print(f"  Preparing {len(jobs)} images -> {cache_dir}")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            bad = [s for s, err in ex.map(_cache_one, jobs, chunksize=8) if err]
        if bad:
            raise SystemExit(f"  {len(bad)} unreadable images, e.g. {bad[:3]}")
    return out


def _predict(net, x_uint8, cfg, device, batch=32):
    import torch

    mean = torch.tensor(cfg.norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.norm_std, device=device).view(1, 3, 1, 1)
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for i in range(0, len(x_uint8), batch):
            xb = x_uint8[i:i + batch].to(device).float().div_(255).sub_(mean).div_(std)
            # Average with the horizontal flip: cheap, and steadies the threshold.
            lo = (net(xb).float() + net(xb.flip(3)).float()) / 2
            out.append(torch.sigmoid(lo).squeeze(1).cpu())
    return torch.cat(out).numpy()


def train_model(x, y, masks, backbone, cfg, epochs, lr, batch, seed, device):
    """Train on masks['train']; keep the epoch with the best validation AUROC."""
    import copy

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from torchvision.transforms import v2

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_net(backbone).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    tr = np.where(masks["train"])[0]
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # Geometry only. Brightness, contrast or blur augmentation would change the
    # very property the labels describe.
    aug = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                      v2.RandomAffine(degrees=25, scale=(0.9, 1.1))])
    mean = torch.tensor(cfg.norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.norm_std, device=device).view(1, 3, 1, 1)
    yt = torch.from_numpy(np.asarray(y, np.float32))
    pos_weight = torch.tensor([(1 - y[tr].mean()) / max(y[tr].mean(), 1e-6)], device=device)

    best = {"auroc": -1.0}
    for ep in range(1, epochs + 1):
        net.train()
        perm = np.random.permutation(tr)
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = torch.stack([aug(x[j]) for j in idx]).to(device)
            xb = xb.float().div_(255).sub_(mean).div_(std).contiguous(
                memory_format=torch.channels_last)
            yb = yt[idx].to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(
                    net(xb).squeeze(1).float(), yb, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        pv = _predict(net, x[masks["val"]], cfg, device)
        auc = roc_auc_score(y[masks["val"]], pv)
        print(f"  [{backbone}] epoch {ep:>2}/{epochs}  loss {total / len(tr):.4f}  "
              f"val AUROC {auc:.4f}")
        if auc > best["auroc"]:
            best = {"auroc": float(auc), "epoch": ep,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in
                                            net.state_dict().items()})}
    net.load_state_dict(best["state"])
    return net, best


def _patient_ci(y, p, files, fn, n_boot=1000):
    from metrics import bootstrap_ci
    return bootstrap_ci(y, p, [patient_of(f) for f in files], fn, n_boot=n_boot)


def classical_on(files, images_dir: Path, rejected) -> dict:
    """The classical gate (run_gates without a learned model) on the same images."""
    from sklearn.metrics import roc_auc_score

    from predictor import run_gates

    gate_rej, blur = [], []
    for f in files:
        q = run_gates(cv2.imread(str(images_dir / f), cv2.IMREAD_COLOR), quality_gate=None)
        gate_rej.append(not q["gradable"])
        blur.append(q["blur_var"])
    y = np.asarray(rejected).astype(bool)
    g = np.asarray(gate_rej)
    return {"catch_rate": float((g & y).sum() / y.sum()),
            "false_reject_rate": float((g & ~y).sum() / (~y).sum()),
            # Sharpness is the classical gate's main signal; low = reject.
            "auroc_blur_var": float(roc_auc_score(y, -np.asarray(blur)))}


def main(argv=None):
    import pandas as pd
    import torch

    from predictor import file_sha256

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", default="deepdrid/quality_labels.csv")
    ap.add_argument("--images", default="deepdrid/repo/regular_fundus_images")
    ap.add_argument("--cache", default="deepdrid/quality_cache")
    ap.add_argument("--filename-col", default="filename")
    ap.add_argument("--label-col", default="quality",
                    help="1 = gradable / diagnosable, 0 = not")
    ap.add_argument("--backbones", nargs="+", default=["efficientnet_b0", "resnet18"],
                    help="trained in turn; the best on validation AUROC is kept")
    ap.add_argument("--image-size", type=int, default=448)
    # efficientnet_b0 peaked at epoch 3 (val AUROC 0.82) and overfit after;
    # 2,000 images do not support long schedules.
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-false-reject", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=QUALITY_CHECKPOINT)
    ap.add_argument("--eval-out", default=QUALITY_EVAL)
    args = ap.parse_args(argv)

    cfg = QualityInputConfig(image_size=args.image_size)
    df = pd.read_csv(args.labels)
    files = df[args.filename_col].astype(str).tolist()
    rejected = (df[args.label_col].astype(int) == 0).to_numpy()
    masks = split_patients(files, rejected, seed=args.seed)
    for k, m in masks.items():
        print(f"  {k:<5} {m.sum():>5} images  "
              f"{len({patient_of(f) for f, b in zip(files, m) if b}):>4} patients  "
              f"{rejected[m].mean():.1%} human-rejected")

    images_dir = Path(args.images)
    cached = build_input_cache(files, images_dir, Path(args.cache), cfg)
    x = torch.from_numpy(np.stack([cv2.cvtColor(cv2.imread(str(c)), cv2.COLOR_BGR2RGB)
                                   for c in cached]).transpose(0, 3, 1, 2).copy())
    y = rejected.astype(np.float32)

    from model import get_device
    device = get_device()
    runs = [train_model(x, y, masks, b, cfg, args.epochs, args.lr, args.batch,
                        args.seed, device) + (b,) for b in args.backbones]
    net, best, backbone = max(runs, key=lambda r: r[1]["auroc"])
    print(f"\n  Kept {backbone} (epoch {best['epoch']}, val AUROC {best['auroc']:.4f})")

    p_val = _predict(net, x[masks["val"]], cfg, device)
    thr = choose_threshold(rejected[masks["val"]], p_val, args.max_false_reject)
    val = gate_metrics(rejected[masks["val"]], p_val, thr)

    # Test: touched once, with the frozen threshold.
    test_files = [f for f, b in zip(files, masks["test"]) if b]
    y_test, p_test = rejected[masks["test"]], _predict(net, x[masks["test"]], cfg, device)
    test = gate_metrics(y_test, p_test, thr)
    test["auroc_ci"] = _patient_ci(y_test, p_test, test_files, gate_metrics_fn("auroc"))
    test["catch_rate_ci"] = _patient_ci(y_test, p_test, test_files,
                                        gate_metrics_fn("catch_rate", thr))
    test["false_reject_rate_ci"] = _patient_ci(y_test, p_test, test_files,
                                               gate_metrics_fn("false_reject_rate", thr))
    print("  Running the classical gate on the test images for comparison...")
    classical = classical_on(test_files, images_dir, y_test)

    seen = [f for f, b in zip(files, masks["train"] | masks["val"]) if b]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(), "backbone": backbone,
        "input": asdict(cfg), "input_fingerprint": cfg.fingerprint(),
        "threshold": thr, "max_false_reject": args.max_false_reject,
        "epoch": best["epoch"], "seed": args.seed, "labels": args.labels,
        "seen_files": seen, "val": val, "test": test,
    }, args.out)

    rep = {
        "checkpoint_sha256": file_sha256(args.out), "backbone": backbone,
        "input_fingerprint": cfg.fingerprint(), "image_size": cfg.image_size,
        "labels": Path(args.labels).as_posix(), "seed": args.seed,
        "max_false_reject_target": args.max_false_reject, "threshold": thr,
        "epoch": best["epoch"],
        "split": {k: {"images": int(m.sum()),
                      "patients": len({patient_of(f) for f, b in zip(files, m) if b})}
                  for k, m in masks.items()},
        "validation": val, "test": test, "classical_test": classical,
        "backbones_tried": {r[2]: {"val_auroc": r[1]["auroc"], "epoch": r[1]["epoch"]}
                            for r in runs},
    }
    Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.eval_out).write_text(json.dumps(rep, indent=2))

    print(f"\n  Threshold P(ungradable) >= {thr:.4f} (val false rejects "
          f"{val['false_reject_rate']:.1%}, catch {val['catch_rate']:.1%})")
    print(f"\n  Held-out test ({test['n']} images, {rep['split']['test']['patients']} patients)")
    print(f"                    learned        classical")
    print(f"    AUROC           {test['auroc']:.3f}          {classical['auroc_blur_var']:.3f}"
          f" (sharpness)")
    print(f"    catch rate      {test['catch_rate']:.1%}          {classical['catch_rate']:.1%}")
    print(f"    false rejects   {test['false_reject_rate']:.1%}           "
          f"{classical['false_reject_rate']:.1%}")
    print(f"\n  Wrote {args.out} and {args.eval_out}")
    return rep


def gate_metrics_fn(key: str, threshold: float = 0.5):
    """metric_fn(y, p) for metrics.bootstrap_ci."""
    def _f(y, p):
        return gate_metrics(y, p, threshold)[key]
    return _f


if __name__ == "__main__":
    main()
