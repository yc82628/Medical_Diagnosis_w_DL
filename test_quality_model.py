"""
Learned quality gate: wiring into run_gates, the classical fallback, and the
threshold rule. Torch-free: a stub stands in for the trained model.
"""
import cv2
import numpy as np

import quality_model as qm
import validate_quality_gate as vq
from predictor import DECISION_MANUAL, DECISION_REFER, run_gates
from test_predictor import _fake_fundus, _Stub


class _Gate:
    """Stands in for LearnedQualityGate: a fixed P(ungradable)."""

    def __init__(self, p, threshold=0.5, seen_files=()):
        self.p, self.threshold, self.calls = p, threshold, 0
        self.seen_files = set(seen_files)

    def assess(self, img):
        self.calls += 1
        ok = self.p < self.threshold
        return {"gradable": ok, "p_ungradable": self.p, "threshold": self.threshold,
                "model": "stub", "reasons": [] if ok else ["learned: too poor"]}


def _use_default(monkeypatch, gate):
    monkeypatch.setitem(qm._default, "loaded", True)
    monkeypatch.setitem(qm._default, "gate", gate)


def test_learned_gate_rejects_an_image_the_classical_checks_pass():
    q = run_gates(_fake_fundus(), quality_gate=_Gate(0.9))
    assert q["quality_source"] == "learned"
    assert q["gradable"] is False and "learned: too poor" in q["reasons"]
    assert q["p_ungradable"] == 0.9 and q["quality_threshold"] == 0.5


def test_classical_backstop_still_rejects_when_learned_gate_passes():
    dark = (_fake_fundus() * 0.03).astype(np.uint8)
    q = run_gates(dark, quality_gate=_Gate(0.01))
    assert q["quality_source"] == "learned"
    assert q["gradable"] is False
    assert any("underexposed" in r for r in q["reasons"])


def test_learned_gate_passes_a_good_image():
    q = run_gates(_fake_fundus(), quality_gate=_Gate(0.1))
    assert q["gradable"] is True and q["reasons"] == []


def test_no_checkpoint_falls_back_to_classical():
    # conftest points DR_QUALITY_CHECKPOINT at a file that does not exist.
    assert qm.default_quality_gate() is None
    assert "classical" in qm.default_quality_note()
    q = run_gates(_fake_fundus())
    assert q["quality_source"] == "classical" and q["gradable"] is True
    assert "p_ungradable" not in q


def test_service_path_uses_learned_gate_when_present(monkeypatch):
    gate = _Gate(0.95)
    _use_default(monkeypatch, gate)
    p = _Stub(0.9, operating_point=0.5)
    out = p.predict(_fake_fundus())
    assert gate.calls == 1
    assert out.decision == DECISION_MANUAL and not p.called
    assert out.quality["quality_source"] == "learned"

    _use_default(monkeypatch, _Gate(0.05))
    assert _Stub(0.9, operating_point=0.5).predict(_fake_fundus()).decision == DECISION_REFER


def test_threshold_caps_false_rejects_on_accepted_images():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200, bool), np.ones(200, bool)]
    p = np.r_[rng.uniform(0, 0.7, 200), rng.uniform(0.3, 1, 200)]
    t = qm.choose_threshold(y, p, max_false_reject=0.05)
    m = qm.gate_metrics(y, p, t)
    assert m["false_rejects"] <= 10 and m["false_reject_rate"] <= 0.05
    # Lowest such threshold: one step lower would exceed the cap.
    below = p[~y][p[~y] < t].max()
    assert (p[~y] >= below).sum() > 10
    assert 0 < m["catch_rate"] < 1 and m["auroc"] > 0.5


def test_split_is_by_patient():
    files = [f"Images/{pid}/{pid}_{e}{k}.jpg" for pid in range(60)
             for e in "lr" for k in (1, 2)]
    y = np.arange(len(files)) % 3 == 0
    masks = qm.split_patients(files, y, seed=1)
    pats = {k: {qm.patient_of(f) for f, b in zip(files, m) if b} for k, m in masks.items()}
    assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"]
                or pats["val"] & pats["test"])
    assert sum(m.sum() for m in masks.values()) == len(files)
    assert qm.patient_of("regular-fundus-training/Images/347/347_l1.jpg") == "347"


def test_validator_excludes_images_the_learned_gate_trained_on(tmp_path, monkeypatch):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for n in ("a.png", "b.png", "c.png"):
        cv2.imwrite(str(imgs / n), _fake_fundus(seed=len(n)))
    csv = tmp_path / "labels.csv"
    csv.write_text("image,quality\na.png,0\nb.png,1\nc.png,1\n")
    _use_default(monkeypatch, _Gate(0.9, seen_files={"a.png"}))

    rep = vq.main(["--csv", str(csv), "--images", str(imgs), "--reject-values", "0",
                   "--out", str(tmp_path / "out.json")])
    assert rep["quality_source"] == "learned"
    assert rep["excluded_seen_by_model"] == 1 and rep["n_images"] == 2
    assert rep["false_reject_rate"] == 1.0          # the stub rejects everything

    rep = vq.main(["--csv", str(csv), "--images", str(imgs), "--reject-values", "0",
                   "--gate", "classical", "--out", str(tmp_path / "out2.json")])
    assert rep["quality_source"] == "classical" and rep["n_images"] == 3
