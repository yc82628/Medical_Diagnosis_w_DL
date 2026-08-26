"""
Tests for the leakage-critical logic. Torch-free so they run anywhere.

The leakage test is the one that matters: it asserts that the detector FIRES on
a naive row-level split. A guard nobody has watched fail is not a guard.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from config import DataConfig
from splits import (prepare_labels, split_by_patient, verify_no_leakage,
                    ordinal_targets, grades_from_cumulative, enforce_monotonic,
                    collapse_rare_strata)


def make_df(n_patients=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        g = rng.choice([0, 1, 2, 3, 4], p=[.73, .15, .07, .03, .02])
        for e in range(rng.integers(1, 4)):
            rows.append({"id_code": f"{p}_{e}.png",
                         "diagnosis": int(max(0, g - rng.integers(0, 2))),
                         "patient_id": f"P{p:04d}"})
    return pd.DataFrame(rows)


DC = DataConfig(filename_col="id_code", grade_col="diagnosis",
                patient_col="patient_id", patient_id_source="column",
                cache_dir="/tmp/_no_cache")


def test_split_has_no_patient_leakage():
    d = prepare_labels(make_df(), DC, verbose=False)
    sp = split_by_patient(d, DC, verbose=False)
    ids = {k: set(v["patient_id"]) for k, v in sp.items()}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    assert sum(len(v) for v in sp.values()) == len(d)


def test_leakage_detector_fires_on_row_level_split():
    """The naive split your original code used must be caught."""
    from sklearn.model_selection import train_test_split
    d = prepare_labels(make_df(), DC, verbose=False)
    a, b = train_test_split(d, test_size=0.3, random_state=0)
    with pytest.raises(RuntimeError, match="PATIENT LEAKAGE"):
        verify_no_leakage({"train": a, "test": b})


def test_split_preserves_referable_prevalence():
    d = prepare_labels(make_df(800), DC, verbose=False)
    sp = split_by_patient(d, DC, verbose=False)
    overall = d["referable"].mean()
    for k, v in sp.items():
        assert abs(v["referable"].mean() - overall) < 0.06, f"{k} prevalence drifted"


def test_split_is_deterministic():
    d = prepare_labels(make_df(), DC, verbose=False)
    a = split_by_patient(d, DC, verbose=False)
    b = split_by_patient(d, DC, verbose=False)
    for k in a:
        assert set(a[k]["patient_id"]) == set(b[k]["patient_id"])


def test_survives_extremely_rare_severe_grades():
    """Grade 4 with 2 patients used to crash the second split stage."""
    d = prepare_labels(make_df(), DC, verbose=False)
    d.loc[d["grade"] == 4, "grade"] = 3
    idx = d.index[:2]
    d.loc[idx, "grade"] = 4
    d["referable"] = (d["grade"] >= 2).astype(int)
    sp = split_by_patient(d, DC, verbose=False)
    assert all(len(v) > 0 for v in sp.values())


def test_ordinal_round_trip():
    for g in range(5):
        t = ordinal_targets(g).astype(float)[None, :]
        assert grades_from_cumulative(t)[0] == g


def test_monotonicity_repair():
    noisy = np.array([[0.9, 0.2, 0.7, 0.1]])       # P(y>2) > P(y>1): impossible
    fixed = enforce_monotonic(noisy)[0]
    assert all(fixed[i] >= fixed[i + 1] for i in range(len(fixed) - 1))
    assert grades_from_cumulative(noisy)[0] == 1


def test_referable_threshold_is_grade_2():
    d = prepare_labels(make_df(), DC, verbose=False)
    assert (d.loc[d["grade"] >= 2, "referable"] == 1).all()
    assert (d.loc[d["grade"] < 2, "referable"] == 0).all()


def test_rare_strata_merge_into_adjacent_grades():
    """Grade 4 must merge with 3, not into a bucket with grade 0."""
    v = pd.Series([0]*100 + [1]*40 + [2]*20 + [3]*4 + [4]*2)
    labels, merges = collapse_rare_strata(v, min_count=7)
    assert labels[v == 4].iloc[0] == labels[v == 3].iloc[0]
    assert labels[v == 0].iloc[0] != labels[v == 4].iloc[0]


def test_missing_patient_column_raises_rather_than_guessing():
    df = make_df().drop(columns=["patient_id"])
    with pytest.raises(ValueError, match="patient_id_source"):
        prepare_labels(df, DC, verbose=False)


def test_filename_prefix_mode():
    df = pd.DataFrame({"id_code": ["10_left.jpeg", "10_right.jpeg",
                                   "11_left.jpeg", "11_right.jpeg"],
                       "diagnosis": [2, 2, 0, 1]})
    dc = DataConfig(filename_col="id_code", grade_col="diagnosis",
                    patient_id_source="filename_prefix", cache_dir="/tmp/_no_cache")
    d = prepare_labels(df, dc, verbose=False)
    assert d["patient_id"].tolist() == ["10", "10", "11", "11"]


def test_invalid_grades_rejected():
    df = make_df(); df.loc[0, "diagnosis"] = 7
    with pytest.raises(ValueError, match="outside"):
        prepare_labels(df, DC, verbose=False)