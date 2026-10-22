"""served_eval.py: metrics, the case-picking rule and the retina crop. No model or data needed."""
import numpy as np

import served_eval as se
from test_predictor import _fake_fundus


def _row(n, decision, prob, referable, grade=None, p_ungradable=0.1):
    return {"filename": f"x/IDRiD_{n:03d}.jpg", "decision": decision, "probability": prob,
            "referable_true": referable, "grade_true": grade if grade is not None else (3 if referable else 0),
            "quality": {"p_ungradable": p_ungradable}}


ROWS = [
    _row(1, "ROUTINE", 0.05, False), _row(2, "ROUTINE", 0.12, False), _row(3, "ROUTINE", 0.20, False),
    _row(4, "ROUTINE", 0.28, True),                       # a missed referable eye
    _row(5, "REFER", 1.00, True), _row(6, "REFER", 0.95, True), _row(7, "REFER", 1.00, True),
    _row(8, "REFER", 0.93, False),                        # a false referral
    _row(9, "MANUAL_REVIEW", 0.70, False), _row(10, "MANUAL_REVIEW", 0.50, True),
    _row(11, "MANUAL_REVIEW", None, True, p_ungradable=0.96),
    _row(12, "MANUAL_REVIEW", None, False, p_ungradable=0.99),
]


def test_summary_counts_only_auto_decided_cases():
    s = se.summarize(ROWS)
    assert s["n_images"] == 12
    assert s["coverage"] == 8 / 12
    assert s["sensitivity"] == 3 / 4          # 3 referred, 1 missed
    assert s["specificity"] == 3 / 4          # 3 cleared, 1 false referral
    assert s["missed_referable"] == 1 and s["false_referrals"] == 1
    assert s["manual_review_reasons"] == {"quality_gate": 2, "abstention_band": 2}


def test_unreadable_images_are_left_out():
    s = se.summarize(ROWS + [{"filename": "x/IDRiD_099.jpg", "unreadable": True}])
    assert s["n_images"] == 12


def test_picks_the_case_closest_to_each_groups_median():
    cases = se.pick_median_cases(ROWS)
    assert set(cases) == {"routine", "refer", "abstention_band", "quality_gate"}
    # Routine & truly healthy: 0.05, 0.12, 0.20 -> median 0.12. The missed eye (4) is not eligible.
    assert cases["routine"]["filename"].endswith("IDRiD_002.jpg")
    # Refer & truly referable: 0.95, 1.00, 1.00 -> median 1.00; tie goes to the lowest number. 8 is not eligible.
    assert cases["refer"]["filename"].endswith("IDRiD_005.jpg")
    # Band: 0.70 and 0.50 -> median 0.60; equally close, so the lowest number wins.
    assert cases["abstention_band"]["filename"].endswith("IDRiD_009.jpg")
    assert cases["quality_gate"]["group_size"] == 2
    assert cases["routine"]["grade_name"] == "Grade 0 (no DR)"


def test_empty_groups_are_skipped():
    cases = se.pick_median_cases([r for r in ROWS if r["decision"] != "MANUAL_REVIEW"])
    assert set(cases) == {"routine", "refer"}


def test_crop_is_square_and_keeps_the_retina():
    img = _fake_fundus(seed=3)
    wide = np.zeros((img.shape[0], img.shape[1] * 2, 3), np.uint8)   # retina on the left of a wide frame
    wide[:, :img.shape[1]] = img
    out = se.crop_retina(wide, size=200)
    assert out.shape == (200, 200, 3)
    assert out.mean() > wide.mean()            # mostly retina, not the black surround
