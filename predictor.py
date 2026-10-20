from __future__ import annotations
 
import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
 
import cv2
import numpy as np
 
from config import (AcceptanceConfig, ICDR_GRADES, PreprocessConfig,
                    REFERABLE_THRESHOLD, TEMPERATURE_RANGE)
from preprocessing import assess_quality, preprocess_image, verify_is_fundus
 
 
# ---------------------------------------------------------------------------
 
DECISION_REFER = "REFER"
DECISION_ROUTINE = "ROUTINE"
DECISION_MANUAL = "MANUAL_REVIEW"
 
 
@dataclass
class Prediction:
    decision: str
    probability: float | None            # calibrated P(referable)
    grade: int | None                    # ICDR 0-4, indicative only
    grade_label: str | None
    quality: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    operating_point: float | None = None
    abstain_low: float | None = None
    abstain_high: float | None = None
    model_version: str = "unknown"
    preprocess_fingerprint: str = "unknown"
    is_demo: bool = False
    # False for the demo heuristic and for a model loaded with the acceptance
    # gate overridden. Stored with every logged prediction.
    validated: bool = False
    latency_ms: float = 0.0
    image_sha256: str = ""
 
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
 
 
# ---------------------------------------------------------------------------
 
def run_gates(img_bgr: np.ndarray) -> dict:
    """
    Quality gate + anatomy gate, exactly as the serving path applies them.
    Shared with validate_quality_gate.py so the validator measures the real
    gate, not a re-implementation of it.
    """
    quality = assess_quality(img_bgr)

    # Anatomy gate. Something that is not a retina (a document, a wall, an
    # X-ray) must never reach the model: it would return a confident number
    # that no offline metric can catch. A non-fundus input is treated as
    # ungradable, so it follows the same MANUAL_REVIEW path, is logged with
    # gradable=0, and gets no Grad-CAM.
    fundus = verify_is_fundus(img_bgr)
    quality["is_fundus"] = bool(fundus["is_fundus"])
    quality["fundus_score"] = float(fundus["score"])
    if not fundus["is_fundus"]:
        quality["gradable"] = False
        quality["reasons"] = list(quality["reasons"]) + [
            "does not look like a retinal photograph: "
            + "; ".join(fundus["reasons"] or ["anatomy check failed"])
        ]
    return quality


# ---------------------------------------------------------------------------

class BasePredictor:
    """Shared decision logic. Subclasses supply `_score(image) -> (logit, cum_probs)`."""
 
    def __init__(self, preprocess: PreprocessConfig, operating_point: float = 0.5,
                 abstain_low: float | None = None, abstain_high: float | None = None,
                 model_version: str = "unknown", is_demo: bool = False):
        self.pc = preprocess
        self.operating_point = operating_point
        self.abstain_low = abstain_low
        self.abstain_high = abstain_high
        self.model_version = model_version
        self.is_demo = is_demo
        self.validated = False      # set by build_predictor once the gate passes
        self.acceptance_failures: list[str] = []
        # Serialises model use. Grad-CAM registers hooks on the shared model; a
        # prediction running concurrently would fire them mid-explanation.
        self.lock = threading.Lock()
 
    # -- to be implemented -------------------------------------------------
    def _score(self, img_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        raise NotImplementedError
 
    # ----------------------------------------------------------------------
    def predict(self, img_bgr: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        sha = hashlib.sha256(img_bgr.tobytes()).hexdigest()[:16]
 
        quality = run_gates(img_bgr)
        base = dict(
            quality=quality, operating_point=self.operating_point,
            abstain_low=self.abstain_low, abstain_high=self.abstain_high,
            model_version=self.model_version,
            preprocess_fingerprint=self.pc.fingerprint(),
            is_demo=self.is_demo, validated=self.validated, image_sha256=sha,
        )
 
        # Gate FIRST. The model never sees an ungradable image.
        if not quality["gradable"]:
            return Prediction(
                decision=DECISION_MANUAL, probability=None, grade=None,
                grade_label=None,
                reasons=["Image not gradable: " + "; ".join(quality["reasons"]),
                         "Recapture recommended. This is not a negative result."],
                latency_ms=(time.perf_counter() - t0) * 1000, **base,
            )
 
        with self.lock:
            prob, cum = self._score(img_bgr)
        grade = int((np.minimum.accumulate(cum) > 0.5).sum())
 
        reasons = []
        if (self.abstain_low is not None and self.abstain_high is not None
                and self.abstain_low < prob < self.abstain_high):
            decision = DECISION_MANUAL
            reasons.append(
                f"Model confidence falls inside the abstention band "
                f"({self.abstain_low:.2f}-{self.abstain_high:.2f}). "
                f"Routed to a human grader by design."
            )
        elif prob >= self.operating_point:
            decision = DECISION_REFER
            reasons.append(
                f"Estimated probability of referable DR ({prob:.2f}) is at or above "
                f"the operating point ({self.operating_point:.2f})."
            )
        else:
            decision = DECISION_ROUTINE
            reasons.append(
                f"Estimated probability of referable DR ({prob:.2f}) is below the "
                f"operating point ({self.operating_point:.2f})."
            )
            reasons.append(
                "Detects referable diabetic retinopathy only. Does not assess "
                "glaucoma, AMD, or other pathology."
            )
 
        return Prediction(
            decision=decision, probability=float(prob), grade=grade,
            grade_label=ICDR_GRADES.get(grade), reasons=reasons,
            latency_ms=(time.perf_counter() - t0) * 1000, **base,
        )
 
 
# ---------------------------------------------------------------------------
 
class TorchPredictor(BasePredictor):
    """Real model. Preprocessing comes from the checkpoint, not from config.py."""
 
    def __init__(self, checkpoint_path, operating_point=None, calibration=None,
                 abstain_low=None, abstain_high=None, device=None, tta="none"):
        import torch
        from model import load_checkpoint
 
        self.torch = torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.model, pc, ck = load_checkpoint(checkpoint_path, self.device)
 
        op = operating_point if operating_point is not None else checkpoint_operating_point(ck)
        version = f"{ck.get('backbone','?')}@epoch{ck.get('epoch','?')}"
        super().__init__(pc, op, abstain_low, abstain_high, version, is_demo=False)
 
        self.calibration = calibration or {"method": "identity", "temperature": 1.0}
        # Calibration was fitted on logits averaged over these views
        # (evaluation.json "tta"), so serving must average the same ones.
        self.tta = tta or "none"
 
    def _apply_calibration(self, logit: float) -> float:
        m = self.calibration.get("method", "identity")
        if m == "temperature":
            T = self.calibration.get("temperature") or 1.0
            return float(1 / (1 + np.exp(-np.clip(logit / T, -30, 30))))
        if m == "isotonic":
            xs = np.asarray(self.calibration["isotonic_x"], dtype=float)
            ys = np.asarray(self.calibration["isotonic_y"], dtype=float)
            return float(np.interp(logit, xs, ys, left=ys[0], right=ys[-1]))
        return float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))
 
    def _score(self, img_bgr):
        torch = self.torch
        proc = preprocess_image(img_bgr, self.pc)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array(self.pc.norm_mean)) / np.array(self.pc.norm_std)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
 
        with torch.no_grad():
            from model import tta_logits
            logits = tta_logits(self.model, x, self.tta).cpu().numpy()[0]
 
        cum = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        prob = self._apply_calibration(float(logits[REFERABLE_THRESHOLD - 1]))
        return prob, cum
 
 
# ---------------------------------------------------------------------------
 
class DemoPredictor(BasePredictor):
    """
    NOT A MODEL. A deterministic image-statistics heuristic so the platform can
    be demonstrated end to end before training finishes.
 
    It counts small dark blobs after local colour normalisation -- crudely
    lesion-like -- and maps that to a score. It is responsive to the image, which
    makes the demo feel real, and it is clinically worthless, which is why every
    response it produces carries is_demo=True and the UI shows a banner.
 
    Do not let this reach anyone who might mistake it for a result.
    """
 
    def __init__(self, operating_point=0.35, abstain_low=0.22, abstain_high=0.52):
        super().__init__(PreprocessConfig(), operating_point, abstain_low,
                         abstain_high, model_version="demo-heuristic-v1", is_demo=True)
 
    def _score(self, img_bgr):
        proc = preprocess_image(img_bgr, self.pc)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
 
        h, w = gray.shape
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), int(0.46 * min(h, w)), 255, -1)
 
        dark = ((gray < 96) & (mask > 0)).astype(np.uint8)
        bright = ((gray > 168) & (mask > 0)).astype(np.uint8)
 
        def blob_count(binary, lo, hi):
            n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
            return int(((areas >= lo) & (areas <= hi)).sum())
 
        score = 0.020 * blob_count(dark, 4, 90) + 0.030 * blob_count(bright, 6, 260)
        prob = float(np.clip(1 / (1 + np.exp(-(score - 1.1) * 2.2)), 0.01, 0.99))
 
        cum = np.clip([prob * 1.25, prob, prob * 0.55, prob * 0.25], 0.001, 0.999)
        return prob, np.asarray(cum)
 
 
# ---------------------------------------------------------------------------
 

def file_sha256(path) -> str:
    """Hash of the checkpoint file -- identifies the exact weights, not just the config."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-np.clip(x, -30, 30))))


def checkpoint_operating_point(ck: dict) -> float:
    """
    Operating point on the PROBABILITY scale, from a checkpoint alone.

    train.py selects the threshold on raw logits. Comparing that number with a
    probability is a units error: a logit threshold of -2 refers every case, and
    one of +1.5 refers none (no probability exceeds 1). Newer checkpoints store
    both scales under explicit names; older ones stored the logit as
    `val_threshold`, which is converted here.

    Only used when evaluation.json is absent, i.e. calibration is identity, so
    sigmoid(logit) is exactly the probability the predictor compares against.
    """
    if ck.get("val_threshold_prob") is not None:
        return float(ck["val_threshold_prob"])
    for key in ("val_threshold_logit", "val_threshold"):
        if ck.get(key) is not None:
            return _sigmoid(float(ck[key]))
    return 0.5


def _band_for(evaluation: dict, target_coverage: float):
    for row in evaluation.get("abstention", []) or []:
        if abs(row.get("target_coverage", 0) - target_coverage) < 1e-6:
            return row.get("band_low"), row.get("band_high")
    return None, None


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _check_report(rep: dict, label: str, acc: AcceptanceConfig) -> list[str]:
    """Floors on one evaluation report. NaN fails every check: unmeasured is not passed."""
    fails = []
    tp, fn = _num(rep.get("tp")), _num(rep.get("fn"))
    positives = int(tp + fn) if np.isfinite(tp) and np.isfinite(fn) else 0
    if positives < acc.min_test_positives:
        fails.append(f"{label}: only {positives} referable cases (need >= "
                     f"{acc.min_test_positives} for a usable sensitivity estimate)")
    checks = [("AUROC", rep.get("auroc"), acc.min_auroc, ">="),
              ("sensitivity", rep.get("sensitivity"), acc.min_sensitivity, ">="),
              ("specificity", rep.get("specificity"), acc.min_specificity, ">="),
              ("ECE", rep.get("ece"), acc.max_ece, "<=")]
    for name, value, limit, op in checks:
        v = _num(value)
        ok = np.isfinite(v) and (v >= limit if op == ">=" else v <= limit)
        if not ok:
            shown = f"{v:.3f}" if np.isfinite(v) else "not measured"
            fails.append(f"{label}: {name} {shown} (required {op} {limit})")
    return fails


def acceptance_failures(evaluation: dict | None, checkpoint_sha: str,
                        preprocess_fp: str, acc: AcceptanceConfig | None = None,
                        target_coverage: float = 0.85) -> list[str]:
    """
    Every reason this checkpoint must not be served. Empty list = accepted.

    Torch-free, so it is unit-tested directly. The criteria live in
    config.AcceptanceConfig.
    """
    acc = acc or AcceptanceConfig()
    if evaluation is None:
        return ["no artifacts/evaluation.json, so the model has no measured "
                "evidence (run calibrate.py)"]

    fails = []
    sha = evaluation.get("checkpoint_sha256")
    if sha is None:
        fails.append("evaluation.json does not record which checkpoint it measured "
                     "(written by an older calibrate.py; re-run it)")
    elif sha != checkpoint_sha:
        fails.append("evaluation.json was measured on a different checkpoint than "
                     "the one being loaded (re-run calibrate.py)")
    if evaluation.get("preprocess_fingerprint") != preprocess_fp:
        fails.append("evaluation.json preprocessing fingerprint does not match the "
                     "checkpoint")

    op = _num(evaluation.get("operating_point"))
    lo, hi = acc.operating_point_range
    if not (np.isfinite(op) and lo < op < hi):
        fails.append(f"operating point {op:.4g} is outside ({lo}, {hi}): the model "
                     f"is referring everyone or no-one rather than deciding")

    cal = evaluation.get("calibration") or {}
    if not cal.get("method"):
        fails.append("no calibration recorded")
    elif cal["method"] == "temperature":
        t = _num(cal.get("temperature"))
        tlo, thi = TEMPERATURE_RANGE
        if not np.isfinite(t) or t <= tlo * 1.0001 or t >= thi * 0.9999:
            fails.append(f"calibration temperature {t:.3g} is pinned at the search "
                         f"bound {TEMPERATURE_RANGE}: the logits carry no usable signal")

    fails += _check_report(evaluation.get("internal_test") or {}, "internal test", acc)

    ext = evaluation.get("external_test")
    if ext:
        fails += _check_report(ext, "external test", acc)
    elif acc.require_external_validation:
        fails.append("no external (different-source) validation, which "
                     "AcceptanceConfig.require_external_validation requires")

    if acc.require_abstention_band:
        blo, bhi = _band_for(evaluation, target_coverage)
        if blo is None or bhi is None:
            fails.append(f"no abstention band for {target_coverage:.0%} coverage, so "
                         f"borderline cases would not reach a human (re-run calibrate.py)")
    return fails


def build_predictor(checkpoint_path="checkpoints/best_model.pt",
                    evaluation_path="artifacts/evaluation.json",
                    target_coverage: float = 0.85,
                    acceptance: AcceptanceConfig | None = None,
                    allow_unvalidated: bool | None = None):
    """
    Load the trained model if it exists AND passes the acceptance gate;
    otherwise fall back to demo mode.

    Returns (predictor, note). The note says which path was taken and, when a
    checkpoint was rejected, every criterion it failed -- the API surfaces it so
    nobody has to guess whether they are looking at a real model.

    `allow_unvalidated` (or DR_ALLOW_UNVALIDATED_MODEL=1) serves a failing model
    anyway, for research debugging such as checking Grad-CAM on demo data. The
    predictor is then marked validated=False, which the API reports as mode
    UNVALIDATED and stores on every logged prediction.
    """
    import json
    import os

    acc = acceptance or AcceptanceConfig()
    if allow_unvalidated is None:
        allow_unvalidated = os.environ.get("DR_ALLOW_UNVALIDATED_MODEL") == "1"

    ck = Path(checkpoint_path)
    if not ck.exists():
        p = DemoPredictor()
        return p, (
            f"No checkpoint at {ck}. Running in DEMO mode: outputs come from an "
            f"image-statistics heuristic, not a trained model, and are clinically "
            f"meaningless. Train a model to replace it."
        )

    ev = Path(evaluation_path)
    evaluation = json.loads(ev.read_text()) if ev.exists() else None
    op, cal, lo, hi, tta = None, None, None, None, "none"
    if evaluation is not None:
        op = evaluation.get("operating_point")
        cal = evaluation.get("calibration")
        lo, hi = _band_for(evaluation, target_coverage)
        tta = evaluation.get("tta") or "none"

    try:
        p = TorchPredictor(checkpoint_path, op, cal, lo, hi, tta=tta)
    except ImportError:
        return DemoPredictor(), (
            "PyTorch is not installed, so the trained checkpoint cannot be loaded. "
            "Running in DEMO mode. Install with: pip install -r requirements.txt"
        )

    failures = acceptance_failures(evaluation, file_sha256(ck), p.pc.fingerprint(),
                                   acc, target_coverage)
    listed = "; ".join(failures)

    if failures and not allow_unvalidated:
        d = DemoPredictor()
        d.acceptance_failures = failures
        return d, (
            f"Checkpoint {ck} was NOT loaded because it failed the acceptance gate "
            f"(config.AcceptanceConfig): {listed}. Running in DEMO mode. Retrain or "
            f"re-run calibrate.py; for research debugging only, set "
            f"DR_ALLOW_UNVALIDATED_MODEL=1."
        )

    p.acceptance_failures = failures
    p.validated = not failures
    if failures:
        note = (f"UNVALIDATED MODEL {p.model_version}: acceptance gate overridden by "
                f"DR_ALLOW_UNVALIDATED_MODEL. Failed: {listed}. Outputs must not "
                f"inform any decision.")
    else:
        note = (f"Loaded model {p.model_version} (operating point "
                f"{p.operating_point:.3f}); passed the acceptance gate.")
    if lo is not None and hi is not None:
        note += (f" Abstention band {lo:.3f}-{hi:.3f} "
                 f"({target_coverage:.0%} coverage) is active.")
    elif evaluation is None:
        note += (" No evaluation.json, so calibration and the abstention band are "
                 "NOT applied.")
    else:
        note += (f" No abstention band for {target_coverage:.0%} coverage, so "
                 f"borderline cases are NOT routed to a human.")
    return p, note
