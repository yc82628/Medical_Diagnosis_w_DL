"""
Operating-point units and the acceptance gate. Torch-free: TorchPredictor is
replaced with a stub, so these run anywhere the rest of the suite runs.
"""
import hashlib
import json

import numpy as np
import pytest

import predictor as pr
from config import AcceptanceConfig, PreprocessConfig
from predictor import (BasePredictor, acceptance_failures, build_predictor,
                       checkpoint_operating_point)


# ---------------------------------------------------------------------------
# Units: the checkpoint threshold is a logit, the predictor compares a probability

def test_legacy_logit_threshold_is_converted_to_probability():
    # A healthy model's logit threshold can easily exceed 1. Read as a
    # probability, no case could ever reach it and every eye would be ROUTINE.
    op = checkpoint_operating_point({"val_threshold": 1.5})
    assert 0 < op < 1
    assert op == pytest.approx(1 / (1 + np.exp(-1.5)))


def test_negative_logit_threshold_is_not_refer_everyone():
    # Read as a probability, -2.0 would make every case REFER.
    assert checkpoint_operating_point({"val_threshold": -2.0}) == pytest.approx(
        1 / (1 + np.exp(2.0)))


def test_explicit_probability_threshold_wins():
    ck = {"val_threshold_prob": 0.31, "val_threshold_logit": 5.0, "val_threshold": 9.0}
    assert checkpoint_operating_point(ck) == pytest.approx(0.31)


def test_missing_threshold_defaults_to_half():
    assert checkpoint_operating_point({}) == 0.5


# ---------------------------------------------------------------------------
# Acceptance gate

FP = PreprocessConfig().fingerprint()
SHA = hashlib.sha256(b"weights").hexdigest()


def _good_evaluation(**over):
    ev = {
        "operating_point": 0.32,
        "calibration": {"method": "isotonic", "temperature": 1.4},
        "internal_test": {"tp": 54, "fn": 4, "auroc": 0.93, "sensitivity": 0.93,
                          "specificity": 0.81, "ece": 0.04},
        "external_test": None,
        "abstention": [{"target_coverage": 0.85, "band_low": 0.2, "band_high": 0.45}],
        "preprocess_fingerprint": FP,
        "checkpoint_sha256": SHA,
    }
    ev.update(over)
    return ev


def test_good_evaluation_passes():
    assert acceptance_failures(_good_evaluation(), SHA, FP) == []


def test_missing_evaluation_fails():
    assert acceptance_failures(None, SHA, FP)


@pytest.mark.parametrize("over, expect", [
    ({"operating_point": 9e-14}, "operating point"),
    ({"operating_point": 1.0}, "operating point"),
    ({"calibration": {"method": "temperature", "temperature": 10.0}}, "pinned"),
    ({"calibration": {"method": "temperature", "temperature": 0.05}}, "pinned"),
    ({"calibration": None}, "no calibration"),
    ({"abstention": []}, "abstention band"),
    ({"checkpoint_sha256": None}, "does not record which checkpoint"),
    ({"checkpoint_sha256": "0" * 64}, "different checkpoint"),
    ({"preprocess_fingerprint": "deadbeef"}, "fingerprint"),
])
def test_each_failure_is_named(over, expect):
    fails = acceptance_failures(_good_evaluation(**over), SHA, FP)
    assert any(expect in f for f in fails), fails


@pytest.mark.parametrize("field, value, expect", [
    ("auroc", 0.70, "AUROC"),
    ("sensitivity", 0.80, "sensitivity"),
    ("specificity", 0.0, "specificity"),
    ("ece", 0.17, "ECE"),
    ("auroc", float("nan"), "not measured"),
])
def test_performance_floors(field, value, expect):
    ev = _good_evaluation()
    ev["internal_test"][field] = value
    fails = acceptance_failures(ev, SHA, FP)
    assert any(expect in f for f in fails), fails


def test_too_few_positives_fails():
    ev = _good_evaluation()
    ev["internal_test"].update(tp=4, fn=1)
    assert any("referable cases" in f for f in acceptance_failures(ev, SHA, FP))


def test_external_set_is_held_to_the_same_floors():
    ev = _good_evaluation(external_test={"tp": 40, "fn": 12, "auroc": 0.80,
                                         "sensitivity": 0.77, "specificity": 0.7,
                                         "ece": 0.06})
    fails = acceptance_failures(ev, SHA, FP)
    assert any(f.startswith("external test: AUROC") for f in fails)
    assert any(f.startswith("external test: sensitivity") for f in fails)


def test_external_validation_can_be_required():
    acc = AcceptanceConfig(require_external_validation=True)
    assert any("external" in f for f in acceptance_failures(_good_evaluation(), SHA, FP, acc))


# ---------------------------------------------------------------------------
# build_predictor wiring

class _FakeTorch(BasePredictor):
    def __init__(self, path, op, cal, lo, hi, tta="none"):
        super().__init__(PreprocessConfig(), op if op is not None else 0.5, lo, hi, "fake")


def _write(tmp_path, evaluation):
    ck = tmp_path / "best_model.pt"
    ck.write_bytes(b"weights")
    ev = tmp_path / "evaluation.json"
    if evaluation is not None:
        ev.write_text(json.dumps(evaluation))
    return str(ck), str(ev)


def test_failing_model_is_not_served(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "TorchPredictor", _FakeTorch)
    monkeypatch.delenv("DR_ALLOW_UNVALIDATED_MODEL", raising=False)
    ck, ev = _write(tmp_path, _good_evaluation(operating_point=0.0))
    p, note = build_predictor(ck, ev)
    assert p.is_demo and not p.validated
    assert "NOT loaded" in note and "operating point" in note
    assert p.acceptance_failures


def test_passing_model_is_served_and_marked_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "TorchPredictor", _FakeTorch)
    ck, ev = _write(tmp_path, _good_evaluation())
    p, note = build_predictor(ck, ev)
    assert not p.is_demo and p.validated
    assert "passed the acceptance gate" in note and "is active" in note
    assert p.predict  # and every prediction it makes carries validated=True


def test_override_serves_but_marks_unvalidated(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "TorchPredictor", _FakeTorch)
    ck, ev = _write(tmp_path, None)
    p, note = build_predictor(ck, ev, allow_unvalidated=True)
    assert not p.is_demo and not p.validated
    assert note.startswith("UNVALIDATED MODEL")


def test_override_from_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "TorchPredictor", _FakeTorch)
    monkeypatch.setenv("DR_ALLOW_UNVALIDATED_MODEL", "1")
    ck, ev = _write(tmp_path, None)
    p, _ = build_predictor(ck, ev)
    assert not p.is_demo and not p.validated
