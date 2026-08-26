"""Tests for the screening evaluation harness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from metrics import (threshold_at_sensitivity, screening_metrics, bootstrap_ci,
                     expected_calibration_error, Calibrator, abstention_curve,
                     quadratic_weighted_kappa, subgroup_report)
from sklearn.metrics import roc_auc_score


def synth(seed=1, n_patients=500):
    rng = np.random.default_rng(seed)
    pid = np.repeat([f"P{i}" for i in range(n_patients)], 2)
    y = np.repeat((rng.random(n_patients) < 0.08).astype(int), 2)
    pe = np.repeat(rng.normal(0, 0.9, n_patients), 2)
    logit = (-1.6 + 1.9 * y + pe + rng.normal(0, 0.8, len(y))) * 2.6
    return y, logit, pid


def test_threshold_honours_sensitivity_constraint():
    y, logit, _ = synth()
    for target in (0.85, 0.90, 0.95, 0.99):
        t = threshold_at_sensitivity(y, logit, target)
        assert screening_metrics(y, logit, t)["sensitivity"] >= target - 1e-9


def test_majority_baseline_scores_chance_auroc():
    """If the harness cannot tell a majority baseline from a model, it is broken."""
    y, logit, _ = synth()
    assert abs(screening_metrics(y, np.zeros(len(y)), 0.5)["auroc"] - 0.5) < 1e-9
    assert screening_metrics(y, logit, 0.0)["auroc"] > 0.7


def test_patient_bootstrap_is_wider_than_image_bootstrap():
    """Images within a patient are not independent draws."""
    y, logit, pid = synth()
    f = lambda a, b: roc_auc_score(a, b)
    cp = bootstrap_ci(y, logit, pid, f, n_boot=400, seed=3)
    ci = bootstrap_ci(y, logit, np.arange(len(y)), f, n_boot=400, seed=3)
    assert (cp["hi"] - cp["lo"]) > (ci["hi"] - ci["lo"])


def test_calibration_improves_ece_on_held_out_data():
    y, logit, _ = synth()
    rng = np.random.default_rng(0)
    val = rng.random(len(y)) < 0.5
    cal = Calibrator().fit(logit[val], y[val])
    before = expected_calibration_error(y[~val], 1 / (1 + np.exp(-logit[~val])))
    after = expected_calibration_error(y[~val], cal.predict(logit[~val]))
    assert after <= before + 1e-6


def test_abstention_coverage_is_monotone_and_exact():
    y, logit, _ = synth()
    p = 1 / (1 + np.exp(-logit))
    t = threshold_at_sensitivity(y, p, 0.90)
    df = abstention_curve(y, p, t)
    assert np.allclose(df["actual_coverage"], df["target_coverage"], atol=0.02)
    assert df["n_retained"].is_monotonic_decreasing


def test_qwk_penalises_distant_errors_more():
    gt = np.array([0, 1, 2, 3, 4] * 20)
    perfect = quadratic_weighted_kappa(gt, gt)
    near = quadratic_weighted_kappa(gt, np.clip(gt + 1, 0, 4))
    far = quadratic_weighted_kappa(gt, np.where(gt == 4, 0, gt))
    assert perfect > near > far


def test_subgroup_report_detects_degraded_group():
    y, logit, _ = synth()
    rng = np.random.default_rng(5)
    dev = rng.choice(["A", "B"], len(y))
    deg = logit.copy()
    m = dev == "B"
    deg[m] = logit[m] * 0.25 + rng.normal(0, 2.0, m.sum())
    p = 1 / (1 + np.exp(-deg))
    t = threshold_at_sensitivity(y, p, 0.90)
    r = subgroup_report(pd.DataFrame({"y": y, "s": p, "device": dev}),
                        "y", "s", t, ["device"])
    a = r.loc[r["group"] == "A", "auroc"].iloc[0]
    b = r.loc[r["group"] == "B", "auroc"].iloc[0]
    assert a - b > 0.1


def test_underpowered_groups_are_flagged():
    y, logit, _ = synth()
    p = 1 / (1 + np.exp(-logit))
    grp = np.array(["big"] * (len(y) - 15) + ["tiny"] * 15)
    r = subgroup_report(pd.DataFrame({"y": y, "s": p, "g": grp}), "y", "s", 0.5, ["g"])
    assert bool(r.loc[r["group"] == "tiny", "underpowered"].iloc[0])