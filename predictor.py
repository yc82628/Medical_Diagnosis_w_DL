from __future__ import annotations
 
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
 
import cv2
import numpy as np
 
from config import PreprocessConfig, REFERABLE_THRESHOLD, ICDR_GRADES
from preprocessing import assess_quality, preprocess_image
 
 
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
    latency_ms: float = 0.0
    image_sha256: str = ""
 
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
 
 
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
 
    # -- to be implemented -------------------------------------------------
    def _score(self, img_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        raise NotImplementedError
 
    # ----------------------------------------------------------------------
    def predict(self, img_bgr: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        sha = hashlib.sha256(img_bgr.tobytes()).hexdigest()[:16]
 
        quality = assess_quality(img_bgr)
        base = dict(
            quality=quality, operating_point=self.operating_point,
            abstain_low=self.abstain_low, abstain_high=self.abstain_high,
            model_version=self.model_version,
            preprocess_fingerprint=self.pc.fingerprint(),
            is_demo=self.is_demo, image_sha256=sha,
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
                 abstain_low=None, abstain_high=None, device=None):
        import torch
        from model import load_checkpoint
 
        self.torch = torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.model, pc, ck = load_checkpoint(checkpoint_path, self.device)
 
        op = operating_point if operating_point is not None else ck.get("val_threshold", 0.5)
        version = f"{ck.get('backbone','?')}@epoch{ck.get('epoch','?')}"
        super().__init__(pc, op, abstain_low, abstain_high, version, is_demo=False)
 
        self.calibration = calibration or {"method": "identity", "temperature": 1.0}
 
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
            logits = self.model(x).float().cpu().numpy()[0]
 
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
 
def build_predictor(checkpoint_path="checkpoints/best_model.pt",
                    evaluation_path="artifacts/evaluation.json",
                    target_coverage: float = 0.85):
    """
    Load the trained model if available; otherwise fall back to demo mode.
 
    Returns (predictor, note) where note explains which path was taken -- the API
    surfaces it so nobody has to guess whether they are looking at a real model.
    """
    import json
 
    ck = Path(checkpoint_path)
    if not ck.exists():
        return DemoPredictor(), (
            f"No checkpoint at {ck}. Running in DEMO mode: outputs come from an "
            f"image-statistics heuristic, not a trained model, and are clinically "
            f"meaningless. Train a model to replace it."
        )
 
    op, cal, lo, hi = None, None, None, None
    ev = Path(evaluation_path)
    if ev.exists():
        data = json.loads(ev.read_text())
        op = data.get("operating_point")
        cal = data.get("calibration")
        for row in data.get("abstention", []):
            if abs(row.get("target_coverage", 0) - target_coverage) < 1e-6:
                lo, hi = row.get("band_low"), row.get("band_high")
                break
 
    try:
        p = TorchPredictor(checkpoint_path, op, cal, lo, hi)
    except ImportError:
        return DemoPredictor(), (
            "PyTorch is not installed, so the trained checkpoint cannot be loaded. "
            "Running in DEMO mode. Install with: pip install -r requirements.txt"
        )
 
    note = f"Loaded model {p.model_version} (operating point {p.operating_point:.3f})."
    if not ev.exists():
        note += (" No artifacts/evaluation.json found, so calibration and the "
                 "abstention band are NOT applied. Run calibrate.py.")
    return p, note