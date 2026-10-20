"""
Decision-path tests. Torch-free: they exercise BasePredictor directly with a
stub score, so they run anywhere the rest of the suite runs.
"""
import json

import cv2
import numpy as np

from metrics import abstention_curve
from predictor import (BasePredictor, DECISION_MANUAL, DECISION_REFER,
                       DECISION_ROUTINE, build_predictor)
from config import PreprocessConfig


class _Stub(BasePredictor):
    """Returns a fixed probability, so only the gates and the band decide."""

    def __init__(self, prob, **kw):
        super().__init__(PreprocessConfig(), **kw)
        self.prob, self.called = prob, False

    def _score(self, img):
        self.called = True
        return self.prob, np.array([self.prob] * 4)


def _fake_fundus(size=600, seed=0):
    """Red-orange disc on black, bright optic disc, some vessel-like texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(size * 0.45), (40, 90, 190), -1)
    noise = rng.integers(0, 40, (size, size, 3), dtype=np.uint8)
    img = cv2.add(img, cv2.bitwise_and(noise, noise, mask=(img[:, :, 2] > 0).astype(np.uint8)))
    cv2.circle(img, (int(c * 1.35), c), size // 18, (170, 230, 250), -1)
    for k in range(12):
        a = rng.uniform(0, 2 * np.pi)
        p2 = (int(c * 1.35 + np.cos(a) * size * 0.35), int(c + np.sin(a) * size * 0.35))
        cv2.line(img, (int(c * 1.35), c), p2, (20, 40, 110), 2)
    return img


def test_non_fundus_never_reaches_model():
    # Grey, sharp, well-exposed, fills the frame: passes the QUALITY gate,
    # but is plainly not a retina (a document scan or X-ray looks like this).
    rng = np.random.default_rng(1)
    grey = rng.integers(60, 200, (600, 600), dtype=np.uint8)
    img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    p = _Stub(0.01, operating_point=0.5)
    out = p.predict(img)
    assert out.decision == DECISION_MANUAL
    assert out.quality["is_fundus"] is False
    assert out.quality["gradable"] is False
    assert not p.called, "the model must not run on a non-retinal image"
    assert any("retinal photograph" in r for r in out.reasons)


def test_fundus_like_image_passes_gate():
    p = _Stub(0.9, operating_point=0.5)
    out = p.predict(_fake_fundus())
    assert out.quality["is_fundus"] is True
    assert out.decision == DECISION_REFER and p.called


def test_band_routes_borderline_to_human():
    img = _fake_fundus()
    kw = dict(operating_point=0.5, abstain_low=0.3, abstain_high=0.7)
    assert _Stub(0.45, **kw).predict(img).decision == DECISION_MANUAL
    assert _Stub(0.10, **kw).predict(img).decision == DECISION_ROUTINE
    assert _Stub(0.90, **kw).predict(img).decision == DECISION_REFER


def test_abstention_curve_records_band_edges_fitted_on_source():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    val_p = np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0.01, 0.99)
    test_p = np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0.01, 0.99)
    df = abstention_curve(y, test_p, 0.5, band_source_prob=val_p)
    assert {"band_low", "band_high"} <= set(df.columns)
    row = df[df["target_coverage"] == 0.85].iloc[0]
    assert row["band_low"] < 0.5 < row["band_high"]
    # Edges come from validation, so they must differ from a test-fitted band.
    own = abstention_curve(y, test_p, 0.5)
    assert not np.isclose(row["band_low"],
                          own[own["target_coverage"] == 0.85].iloc[0]["band_low"])


def test_build_predictor_reads_band_from_evaluation(tmp_path, monkeypatch):
    """The serving side must find the band calibrate.py writes."""
    import predictor as pr

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0.01, 0.99)
    ab = abstention_curve(y, p, 0.5).to_dict(orient="records")
    ev = tmp_path / "evaluation.json"
    ev.write_text(json.dumps({"operating_point": 0.5, "calibration": None,
                              "abstention": ab, "tta": "d8"}))
    ck = tmp_path / "best_model.pt"
    ck.write_bytes(b"x")

    captured = {}

    class _FakeTorch(BasePredictor):
        def __init__(self, path, op, cal, lo, hi, tta="none"):
            captured.update(lo=lo, hi=hi, tta=tta)
            super().__init__(PreprocessConfig(), op, lo, hi, "fake")

    monkeypatch.setattr(pr, "TorchPredictor", _FakeTorch)
    # This evaluation has no test metrics, so it cannot pass the acceptance
    # gate; the override isolates band wiring (the gate has its own tests).
    pred, note = build_predictor(str(ck), str(ev), allow_unvalidated=True)
    assert captured["lo"] is not None and captured["hi"] is not None
    # Calibration was fitted on TTA-averaged logits; serving must average too.
    assert captured["tta"] == "d8"
    assert "is active" in note
