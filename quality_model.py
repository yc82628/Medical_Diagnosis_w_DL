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

Recapture reasons: when the label file has DeepDRiD's sub-scores (clarity,
field definition, artefact; prepare_deepdrid.py writes them), the same backbone
also learns them as ordinal heads (multi-task). They never change the decision,
which stays P(ungradable) >= threshold; they say what to fix when it rejects.

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
# Sub-scores: WHY an image is ungradable (torch-free)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubScore:
    """
    One of DeepDRiD's graded quality sub-scores, learned as a recapture reason.

    `levels` are the raw label values ordered WORST first, so a higher level
    index is always a better image. The first `problem_levels` levels count as
    the problem the photographer should fix. Each sub-score gets len(levels)-1
    cumulative logits on the shared backbone: logit k models P(level index > k),
    an ordinal head (Frank & Hall 2001; Cao et al., CORAL, 2020) that respects
    the order of the grades, which plain softmax classification ignores.
    """
    key: str             # column in quality_labels.csv
    levels: tuple        # raw values, worst first
    problem_levels: int  # how many of the worst levels are "a problem"
    problem: str         # what is wrong, as the reason states it
    fix: str             # what to do about it

    @property
    def n_logits(self) -> int:
        return len(self.levels) - 1


# Meanings from DeepDRiD's Readme (Liu et al., Patterns 2022); see
# prepare_deepdrid.SUBSCORE_COLUMNS. Problem cut-offs:
#   clarity <= 6           level-3 vascular arch visible but not most lesions
#   field definition <= 6  disc and macula not within 2 disc diameters of the centre
#   artifact >= 6          artefacts cover more than a quarter of the image
# Reasons: no brackets in `problem` or `fix` (the web page strips "(...)").
SUBSCORES = (
    SubScore("clarity", (1, 4, 6, 8, 10), 3, "out of focus or hazy",
             "refocus, check the lens is clean, and retake"),
    SubScore("field_definition", (1, 4, 6, 8, 10), 3, "eye off-centre",
             "re-centre on the macula with the optic disc in view, and retake"),
    SubScore("artifact", (10, 8, 6, 4, 1, 0), 3,
             "artefact such as an eyelash, eyelid or reflection",
             "ask the patient to open wide, realign to remove reflections, and retake"),
)
# What "problem" means for each, in the graders' terms (model card, eval file).
PROBLEM_DEFINITION = {
    "clarity": "clarity 1-6 of 10: most lesions could not be identified",
    "field_definition": "field definition 1-6 of 10: optic disc and macula not both "
                        "within 2 disc diameters of the centre",
    "artifact": "artefact 6-10 of 10: artefacts cover more than a quarter of the image",
}
NO_SINGLE_CAUSE = ("no single cause stood out: retake, and consider dilating the pupil "
                   "if it is small or the lens is cloudy")


def subscore_targets(values, spec: SubScore) -> np.ndarray:
    """Raw label values -> level index (0 = worst); -1 where missing or unknown."""
    pos = {float(v): i for i, v in enumerate(spec.levels)}
    out = []
    for v in values:
        try:
            out.append(pos.get(float(v), -1))
        except (TypeError, ValueError):
            out.append(-1)
    return np.array(out, dtype=np.int64)


def split_sub_logits(logits, specs=SUBSCORES) -> dict:
    """(N, sum of n_logits) -> {key: (N, n_logits)}, in spec order."""
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    out, i = {}, 0
    for s in specs:
        out[s.key] = logits[:, i:i + s.n_logits]
        i += s.n_logits
    if i != logits.shape[1]:
        raise ValueError(f"expected {i} sub-score logits, got {logits.shape[1]}")
    return out


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, dtype=float), -30, 30)))


def problem_probability(logits, spec: SubScore) -> np.ndarray:
    """P(level index < problem_levels) = 1 - P(index > problem_levels - 1)."""
    return 1 - _sigmoid(np.atleast_2d(logits)[:, spec.problem_levels - 1])


def predicted_level(logits, spec: SubScore) -> np.ndarray:
    """Level index: how many cumulative 'better than level k' calls are yes."""
    return (_sigmoid(np.atleast_2d(logits)) > 0.5).sum(axis=1)


def choose_problem_threshold(has_problem, p_problem) -> float:
    """Threshold maximising sensitivity + specificity (Youden's J), on validation."""
    from sklearn.metrics import roc_curve

    y = np.asarray(has_problem).astype(bool)
    if y.all() or not y.any():
        return 0.5
    fpr, tpr, thr = roc_curve(y, np.asarray(p_problem, dtype=float))
    k = int(np.argmax(tpr - fpr))
    return float(min(thr[k], 1.0))


def subscore_metrics(level_true, logits, spec: SubScore, threshold: float) -> dict:
    """
    Held-out performance of one sub-score head on images that carry the label.
    Problem detection (the reason the photographer reads): AUROC, sensitivity,
    specificity at the frozen threshold. The full ordinal grade: quadratic
    weighted kappa and exact agreement of the predicted level.
    """
    from sklearn.metrics import cohen_kappa_score, roc_auc_score

    t = np.asarray(level_true)
    keep = t >= 0
    t, lg = t[keep], np.atleast_2d(logits)[keep]
    y = t < spec.problem_levels
    p = problem_probability(lg, spec)
    pred = p >= threshold
    lvl = predicted_level(lg, spec)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return {
        "n": int(len(t)), "n_problem": n_pos, "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if n_pos and n_neg else float("nan"),
        "sensitivity": float((pred & y).sum() / n_pos) if n_pos else float("nan"),
        "specificity": float((~pred & ~y).sum() / n_neg) if n_neg else float("nan"),
        "weighted_kappa": float(cohen_kappa_score(t, lvl, weights="quadratic",
                                                  labels=list(range(len(spec.levels))))),
        "exact_agreement": float((lvl == t).mean()),
    }


def describe_subscores(sub_logits, specs, thresholds: dict) -> dict:
    """One image's sub-score logits -> {key: p_problem, threshold, flagged, level, problem, fix}."""
    parts = split_sub_logits(sub_logits, specs)
    out = {}
    for s in specs:
        p = float(problem_probability(parts[s.key], s)[0])
        thr = float(thresholds[s.key])
        out[s.key] = {"p_problem": p, "threshold": thr, "flagged": p >= thr,
                      "level": s.levels[int(predicted_level(parts[s.key], s)[0])],
                      "problem": s.problem, "fix": s.fix}
    return out


def recapture_reasons(subscores: dict, specs=SUBSCORES) -> list:
    """
    What the photographer should fix, most likely first. Only problems whose
    probability reaches their own validation-chosen threshold are named; when
    none does, the reason says so rather than guessing.
    """
    by_key = {s.key: s for s in specs}
    hits = sorted((k for k, v in subscores.items() if v["flagged"] and k in by_key),
                  key=lambda k: -subscores[k]["p_problem"])
    if not hits:
        return [NO_SINGLE_CAUSE]
    # Single brackets only: the web page strips "(...)" for its plain view.
    return [f"{by_key[k].problem}: {by_key[k].fix} "
            f"(probability {subscores[k]['p_problem']:.2f})" for k in hits]


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
        # Sub-score heads (why an image is ungradable). Checkpoints written
        # before them, or retrained without sub-score labels, have none.
        self.subscores, self.sub_thresholds = subscores_from_checkpoint(ck)
        self.net = build_net(self.backbone, pretrained=False,
                             n_out=1 + sum(s.n_logits for s in self.subscores))
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(self.device).eval()
        self.lock = threading.Lock()

    def logits(self, img_bgr: np.ndarray) -> np.ndarray:
        """All outputs: [overall ungradable logit, sub-score logits...]."""
        torch = self.torch
        x = to_chw_float(prepare_input(img_bgr, self.cfg), self.cfg)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with self.lock, torch.no_grad():
            return self.net(x).float().cpu().numpy().ravel()

    def p_ungradable(self, img_bgr: np.ndarray) -> float:
        return float(_sigmoid(self.logits(img_bgr)[0]))

    def assess(self, img_bgr: np.ndarray) -> dict:
        out = self.logits(img_bgr)
        p = float(_sigmoid(out[0]))
        subs = (describe_subscores(out[1:], self.subscores, self.sub_thresholds)
                if self.subscores else {})
        return assessment(p, self.threshold, self.version, subs, self.subscores)


def assessment(p: float, threshold: float, model: str, subscores: dict,
               specs=SUBSCORES) -> dict:
    """
    The gate's verdict for one image. The overall P(ungradable) alone decides;
    sub-scores only explain a rejection, so they never reject or pass anything.
    """
    ok = p < threshold
    reasons, recapture = [], []
    if not ok:
        recapture = recapture_reasons(subscores, specs) if subscores else []
        # No nested brackets: the web page strips "(...)" from reasons.
        reasons = [f"learned quality model judges the image too poor to grade "
                   f"(probability ungradable {p:.2f}, threshold {threshold:.2f})"] + recapture
    return {"gradable": ok, "p_ungradable": p, "threshold": threshold, "model": model,
            "subscores": subscores, "recapture": recapture, "reasons": reasons}


def subscores_from_checkpoint(ck: dict):
    """(specs, thresholds) stored in a checkpoint; ((), {}) when it has none."""
    known = {s.key: s for s in SUBSCORES}
    specs, thr = [], {}
    for d in ck.get("subscores") or []:
        base = known.get(d["key"])
        # Heads are defined by the stored levels; wording comes from this file
        # so it can be improved without retraining.
        specs.append(SubScore(d["key"], tuple(d["levels"]), int(d["problem_levels"]),
                              base.problem if base else d.get("problem", d["key"]),
                              base.fix if base else d.get("fix", "retake")))
        thr[d["key"]] = float(d["threshold"])
    return tuple(specs), thr


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
                    if gate.subscores:
                        note += (" Rejections name a likely cause: "
                                 + ", ".join(s.key.replace("_", " ") for s in gate.subscores)
                                 + ".")
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

def build_net(backbone: str, pretrained: bool = True, n_out: int = 1):
    """
    Pretrained torchvision backbone. Output 0 is the P(ungradable) logit; any
    further outputs are the sub-score heads' cumulative logits (SUBSCORES order).
    """
    import torch.nn as nn
    from torchvision import models

    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, n_out)
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, n_out)
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


def _predict(net, x_uint8, cfg, device, batch=32, all_outputs=False):
    """
    P(ungradable) per image, flip-averaged. With all_outputs, also the
    sub-score logits (flip-averaged, before the sigmoid) as an (N, k) array.
    """
    import torch

    mean = torch.tensor(cfg.norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.norm_std, device=device).view(1, 3, 1, 1)
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for i in range(0, len(x_uint8), batch):
            xb = x_uint8[i:i + batch].to(device).float().div_(255).sub_(mean).div_(std)
            # Average with the horizontal flip: cheap, and steadies the threshold.
            out.append(((net(xb).float() + net(xb.flip(3)).float()) / 2).cpu())
    lo = torch.cat(out).numpy()
    p = _sigmoid(lo[:, 0])
    return (p, lo[:, 1:]) if all_outputs else p


def _ordinal_loss(sub_logits, sub_targets, specs):
    """
    Mean binary cross-entropy over every sub-score's cumulative 'better than
    level k' logits, ignoring images without that label (target -1).
    """
    import torch
    import torch.nn.functional as F

    total, count, i = sub_logits.new_zeros(()), 0, 0
    for j, s in enumerate(specs):
        lg = sub_logits[:, i:i + s.n_logits]
        i += s.n_logits
        t = sub_targets[:, j]
        keep = t >= 0
        if keep.any():
            ks = torch.arange(s.n_logits, device=lg.device)
            target = (t[keep].unsqueeze(1) > ks).float()
            total = total + F.binary_cross_entropy_with_logits(lg[keep], target)
            count += 1
    return total / max(count, 1)


def train_model(x, y, masks, backbone, cfg, epochs, lr, batch, seed, device,
                sub=None, specs=SUBSCORES, sub_weight=0.5):
    """
    Train on masks['train']; keep the epoch with the best validation AUROC of
    the OVERALL ungradable output (the one that decides).

    `sub` (N x len(specs) level indices, -1 = missing) adds the sub-score heads
    as auxiliary tasks on the same backbone, weighted by `sub_weight`. Without
    it the net has the single ungradable output, as before.
    """
    import copy

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from torchvision.transforms import v2

    torch.manual_seed(seed)
    np.random.seed(seed)
    specs = tuple(specs) if sub is not None else ()
    net = build_net(backbone, n_out=1 + sum(s.n_logits for s in specs))
    net = net.to(device).to(memory_format=torch.channels_last)
    st = torch.from_numpy(np.asarray(sub, np.int64)) if specs else None
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
                out = net(xb).float()
                loss = F.binary_cross_entropy_with_logits(out[:, 0], yb, pos_weight=pos_weight)
                if specs:
                    loss = loss + sub_weight * _ordinal_loss(out[:, 1:], st[idx].to(device), specs)
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
    ap.add_argument("--sub-weight", type=float, default=0.5,
                    help="weight of the sub-score (recapture reason) loss")
    ap.add_argument("--no-subscores", action="store_true",
                    help="train the single ungradable output only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=QUALITY_CHECKPOINT)
    ap.add_argument("--eval-out", default=QUALITY_EVAL)
    args = ap.parse_args(argv)

    cfg = QualityInputConfig(image_size=args.image_size)
    df = pd.read_csv(args.labels)
    files = df[args.filename_col].astype(str).tolist()
    rejected = (df[args.label_col].astype(int) == 0).to_numpy()
    # Same split as before the sub-scores existed: it depends only on the
    # filenames, the overall label and the seed.
    masks = split_patients(files, rejected, seed=args.seed)
    specs = () if args.no_subscores else tuple(s for s in SUBSCORES if s.key in df.columns)
    if not specs and not args.no_subscores:
        print("  No sub-score columns in the labels: re-run prepare_deepdrid.py for "
              "recapture reasons. Training the ungradable output only.")
    sub = (np.stack([subscore_targets(df[s.key], s) for s in specs], axis=1)
           if specs else None)
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
                        args.seed, device, sub=sub, specs=specs or SUBSCORES,
                        sub_weight=args.sub_weight) + (b,) for b in args.backbones]
    net, best, backbone = max(runs, key=lambda r: r[1]["auroc"])
    print(f"\n  Kept {backbone} (epoch {best['epoch']}, val AUROC {best['auroc']:.4f})")

    p_val, sl_val = _predict(net, x[masks["val"]], cfg, device, all_outputs=True)
    thr = choose_threshold(rejected[masks["val"]], p_val, args.max_false_reject)
    val = gate_metrics(rejected[masks["val"]], p_val, thr)
    # Sub-score problem thresholds: also chosen on validation, then frozen.
    sub_thr, sub_rep = {}, {}
    if specs:
        parts = split_sub_logits(sl_val, specs)
        for j, s in enumerate(specs):
            t = sub[masks["val"], j]
            k = t >= 0
            sub_thr[s.key] = choose_problem_threshold(
                t[k] < s.problem_levels, problem_probability(parts[s.key][k], s))
            sub_rep[s.key] = {"levels_worst_first": list(s.levels),
                              "problem_levels": [s.levels[i] for i in range(s.problem_levels)],
                              "problem": s.problem, "fix": s.fix,
                              "problem_definition": PROBLEM_DEFINITION.get(s.key),
                              "validation": subscore_metrics(t, parts[s.key], s, sub_thr[s.key])}

    # Test: touched once, with the frozen thresholds.
    test_files = [f for f, b in zip(files, masks["test"]) if b]
    p_test, sl_test = _predict(net, x[masks["test"]], cfg, device, all_outputs=True)
    y_test = rejected[masks["test"]]
    test = gate_metrics(y_test, p_test, thr)
    test["auroc_ci"] = _patient_ci(y_test, p_test, test_files, gate_metrics_fn("auroc"))
    test["catch_rate_ci"] = _patient_ci(y_test, p_test, test_files,
                                        gate_metrics_fn("catch_rate", thr))
    test["false_reject_rate_ci"] = _patient_ci(y_test, p_test, test_files,
                                               gate_metrics_fn("false_reject_rate", thr))
    reasons_test = None
    if specs:
        from metrics import bootstrap_ci
        parts = split_sub_logits(sl_test, specs)
        groups = np.asarray([patient_of(f) for f in test_files])
        for j, s in enumerate(specs):
            t = sub[masks["test"], j]
            m = subscore_metrics(t, parts[s.key], s, sub_thr[s.key])
            k = t >= 0
            m["auroc_ci"] = bootstrap_ci(
                t[k] < s.problem_levels, problem_probability(parts[s.key][k], s),
                groups[k], _auroc)
            sub_rep[s.key]["test"] = m
        reasons_test = reason_agreement(p_test >= thr, sl_test, sub[masks["test"]],
                                        specs, sub_thr)
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
        "subscores": [{**asdict(s), "threshold": sub_thr[s.key]} for s in specs],
        "sub_weight": args.sub_weight if specs else None,
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
        "sub_weight": args.sub_weight if specs else None,
        "subscores": sub_rep, "recapture_reasons_test": reasons_test,
        "backbones_tried": {r[2]: {"val_auroc": r[1]["auroc"], "epoch": r[1]["epoch"]}
                            for r in runs},
    }
    Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.eval_out).write_text(json.dumps(rep, indent=2))

    print(f"\n  Threshold P(ungradable) >= {thr:.4f} (val false rejects "
          f"{val['false_reject_rate']:.1%}, catch {val['catch_rate']:.1%})")
    print(f"\n  Held-out test ({test['n']} images, {rep['split']['test']['patients']} patients)")
    print("                    learned        classical")
    print(f"    AUROC           {test['auroc']:.3f}          {classical['auroc_blur_var']:.3f}"
          f" (sharpness)")
    print(f"    catch rate      {test['catch_rate']:.1%}          {classical['catch_rate']:.1%}")
    print(f"    false rejects   {test['false_reject_rate']:.1%}           "
          f"{classical['false_reject_rate']:.1%}")
    if specs:
        print("\n  Recapture reasons, held-out test (problem = worst levels of each sub-score)")
        print("    sub-score          n   problem  AUROC  sens   spec   weighted kappa")
        for k, r in sub_rep.items():
            m = r["test"]
            print(f"    {k:<17} {m['n']:>4}  {m['n_problem']:>6}   {m['auroc']:.3f}  "
                  f"{m['sensitivity']:.1%}  {m['specificity']:.1%}  {m['weighted_kappa']:.3f}")
        rt = reasons_test
        print(f"    Of {rt['n_gate_rejected']} images the gate rejected: "
              f"{rt['named_a_cause']:.0%} given a cause; the top cause matches the "
              f"graders' sub-scores for {rt['top_cause_confirmed']:.0%} of those")
    print(f"\n  Wrote {args.out} and {args.eval_out}")
    return rep


def _auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def reason_agreement(gate_rejected, sub_logits, sub_levels, specs, thresholds) -> dict:
    """
    For the images the gate rejected: how often a cause was named, and how
    often the named cause is a problem the human graders also scored. Only
    images with every sub-score label count toward the 'confirmed' rates.
    """
    rej = np.where(np.asarray(gate_rejected, bool))[0]
    parts = split_sub_logits(sub_logits, specs)
    sub_levels = np.asarray(sub_levels)
    named = top_ok = any_ok = n_named_lab = human_any = 0
    per = {s.key: {"named": 0, "confirmed": 0} for s in specs}
    for i in rej:
        p = {s.key: float(problem_probability(parts[s.key][i:i + 1], s)[0]) for s in specs}
        hits = sorted((k for k in p if p[k] >= thresholds[k]), key=lambda k: -p[k])
        labelled = bool((sub_levels[i] >= 0).all())
        human = {s.key: bool(0 <= sub_levels[i, j] < s.problem_levels)
                 for j, s in enumerate(specs)}
        human_any += labelled and any(human.values())
        if hits:
            named += 1
            if labelled:
                n_named_lab += 1
                top_ok += human[hits[0]]
                any_ok += any(human[k] for k in hits)
                for k in hits:
                    per[k]["named"] += 1
                    per[k]["confirmed"] += human[k]
    n = len(rej)
    nan = float("nan")
    return {
        "n_gate_rejected": n,
        "named_a_cause": named / n if n else nan,
        "top_cause_confirmed": top_ok / n_named_lab if n_named_lab else nan,
        "any_named_cause_confirmed": any_ok / n_named_lab if n_named_lab else nan,
        "human_saw_a_subscore_problem": human_any / n if n else nan,
        "per_cause_named": {k: v["named"] for k, v in per.items()},
        "per_cause_precision": {k: (v["confirmed"] / v["named"] if v["named"] else nan)
                                for k, v in per.items()},
    }


def gate_metrics_fn(key: str, threshold: float = 0.5):
    """metric_fn(y, p) for metrics.bootstrap_ci."""
    def _f(y, p):
        return gate_metrics(y, p, threshold)[key]
    return _f


if __name__ == "__main__":
    main()
