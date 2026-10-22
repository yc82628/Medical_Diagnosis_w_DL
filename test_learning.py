"""Learning loop: snapshot export, the locked test set, the model registry. Torch-free."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).parent / "learning"))

import audit  # noqa: E402
import registry as reg  # noqa: E402
import snapshot as snap  # noqa: E402
from predictor import DemoPredictor  # noqa: E402
from retrain_quality import compare  # noqa: E402
from storage import ImageStore  # noqa: E402
from test_predictor import _fake_fundus  # noqa: E402

KEY = b"0" * 64


# -- pseudonymisation and the locked test set ----------------------------------

def test_codes_are_stable_keyed_and_not_the_id():
    a = snap.code(KEY, "patient", "P0001")
    assert a == snap.code(KEY, "patient", "P0001") and "P0001" not in a
    assert a != snap.code(b"1" * 64, "patient", "P0001")       # another key, another code
    assert a != snap.code(KEY, "visit", "P0001")               # kinds are separated


def test_locked_test_assignment_is_permanent_and_about_a_fifth():
    codes = [snap.code(KEY, "patient", f"P{i:05d}") for i in range(4000)]
    first = [snap.split_of(c) for c in codes]
    assert first == [snap.split_of(c) for c in codes]          # same every time
    share = first.count("locked_test") / len(first)
    assert 0.17 < share < 0.23


def _graded_db(tmp_path, key):
    con = audit.connect(tmp_path / "a.db")
    store = ImageStore(tmp_path / "img", key)
    p = DemoPredictor()
    ok, png = cv2.imencode(".png", _fake_fundus())
    cases = []
    for i, (pid, eye, verdict) in enumerate([("MRN-111", "left", "REFER"),
                                             ("MRN-111", "right", "UNGRADABLE"),
                                             ("MRN-222", "left", "ROUTINE"),
                                             ("MRN-333", "left", None)]):     # never graded
        cid = audit.log_prediction(con, p.predict(_fake_fundus()), patient_id=pid, eye=eye,
                                   encounter_id=f"{pid}:2025-03-14")
        store.put(cid, "raw", png.tobytes())
        if verdict:
            audit.log_verdict(con, cid, verdict, grade=3 if verdict == "REFER" else None,
                              grader="dr.a", dme=True if verdict == "REFER" else None,
                              findings=["glaucoma_suspect"] if verdict == "ROUTINE" else None)
        cases.append(cid)
    return con, store


def test_snapshot_exports_only_graded_and_no_identifiers(tmp_path):
    key = Fernet.generate_key()
    con, store = _graded_db(tmp_path, key)
    card = snap.export(con, store, tmp_path / "s1", KEY, actor="tester")
    df = pd.read_csv(tmp_path / "s1" / "labels.csv")
    assert card["images"] == len(df) == 3                      # the ungraded one is left out
    assert set(df["verdict"]) == {"REFER", "UNGRADABLE", "ROUTINE"}
    row = df[df.verdict == "REFER"].iloc[0]
    assert (row.grade, row.dme, row.referable, row.gradable) == (3, 1, 1, 1)
    assert df[df.verdict == "UNGRADABLE"].iloc[0].gradable == 0
    assert df[df.verdict == "ROUTINE"].iloc[0].findings == "glaucoma_suspect"
    text = (tmp_path / "s1" / "labels.csv").read_text() + json.dumps(card)
    files = " ".join(p.name for p in (tmp_path / "s1" / "images").iterdir())
    for secret in ("MRN-111", "MRN-222", "2025-03-14", "dr.a"):
        assert secret not in text and secret not in files
    assert len(list((tmp_path / "s1" / "images").glob("*.png"))) == 3
    assert con.execute("SELECT COUNT(*) FROM audit_log WHERE action='export_snapshot'").fetchone()[0] == 1
    assert audit.verify_chain(con)["ok"]


def test_patients_keep_their_split_across_snapshots(tmp_path):
    key = Fernet.generate_key()
    con, store = _graded_db(tmp_path, key)
    snap.export(con, store, tmp_path / "s1", KEY)
    snap.export(con, store, tmp_path / "s2", KEY)
    a = pd.read_csv(tmp_path / "s1" / "labels.csv").set_index("patient")["split"]
    b = pd.read_csv(tmp_path / "s2" / "labels.csv").set_index("patient")["split"]
    assert a.to_dict() == b.to_dict()


# -- comparison rule --------------------------------------------------------------

def _m(auroc, catch, fr, rej=40):
    return {"auroc": auroc, "catch_rate": catch, "false_reject_rate": fr, "n_human_reject": rej}


def test_candidate_must_not_be_worse_on_the_locked_test():
    assert compare(_m(.80, .40, .04), _m(.85, .45, .04), .05)["candidate_not_worse"]
    worse = compare(_m(.85, .45, .04), _m(.80, .45, .04), .05)
    assert not worse["candidate_not_worse"] and "AUROC" in worse["why"][0]
    assert not compare(_m(.85, .45, .04), _m(.86, .45, .09), .05)["candidate_not_worse"]
    small = compare(_m(.80, .40, .04), _m(.90, .50, .03, rej=5), .05)
    assert not small["candidate_not_worse"] and "collect more" in small["why"][0]


# -- registry ------------------------------------------------------------------------

def _served(tmp_path):
    return {"checkpoint": tmp_path / "served" / "quality_model.pt",
            "eval": tmp_path / "served" / "quality_model_eval.json"}


def _ck(tmp_path, name, content):
    p = tmp_path / name
    p.write_bytes(content)
    e = tmp_path / f"{name}.json"
    e.write_text(json.dumps({"for": name}))
    return p, e


def test_register_promote_and_roll_back(tmp_path):
    root, served, con = tmp_path / "registry", _served(tmp_path), audit.connect(tmp_path / "a.db")
    served["checkpoint"].parent.mkdir()
    served["checkpoint"].write_bytes(b"original model")       # the model in service today
    ck, ev = _ck(tmp_path, "cand.pt", b"candidate model")
    v = reg.register("quality", ck, ev, source="test", root=root, con=con,
                     comparison={"candidate_not_worse": True, "why": []})
    assert reg.versions("quality", root)[0]["status"] == "candidate"
    assert served["checkpoint"].read_bytes() == b"original model", "registering must not deploy"

    r = reg.promote("quality", v, "Dr A. Example", "better on locked test",
                    root=root, con=con, served=served)
    assert served["checkpoint"].read_bytes() == b"candidate model"
    assert json.loads(served["eval"].read_text()) == {"for": "cand.pt"}
    statuses = {x["version"]: x["status"] for x in reg.versions("quality", root)}
    assert statuses[v] == "active" and statuses[r["replaced"]] == "retired"

    reg.rollback("quality", "Dr A. Example", "unexpected rejects", root=root, con=con, served=served)
    assert served["checkpoint"].read_bytes() == b"original model"
    actions = [a for (a,) in con.execute("SELECT action FROM audit_log ORDER BY seq")]
    assert actions.count("model_promoted") == 2 and "model_rolled_back" in actions
    assert audit.verify_chain(con)["ok"]


def test_promotion_refuses_a_worse_candidate_and_needs_a_name(tmp_path):
    root, served = tmp_path / "registry", _served(tmp_path)
    ck, ev = _ck(tmp_path, "cand.pt", b"worse model")
    v = reg.register("quality", ck, ev, source="test", root=root,
                     comparison={"candidate_not_worse": False, "why": ["AUROC 0.70 vs 0.80"]})
    with pytest.raises(SystemExit, match="WORSE"):
        reg.promote("quality", v, "Dr A", "try", root=root, served=served)
    with pytest.raises(SystemExit, match="approved-by"):
        reg.promote("quality", v, " ", "try", root=root, served=served, override_reason="x")
    reg.promote("quality", v, "Dr A", "accept for a trial", override_reason="ethics-approved trial",
                root=root, served=served)
    hist = reg.versions("quality", root)[0]["history"][-1]
    assert hist["override"] == "ethics-approved trial"


# -- research sample size -------------------------------------------------------------

def test_sample_size_is_sane():
    sys.path.insert(0, str(Path(__file__).parent / "research"))
    import sample_size as ss
    n = ss.n_for_precision(0.90, 0.05)
    assert 130 <= n <= 175, n          # normal approximation gives 139; exact is wider
    lo, hi = ss.clopper_pearson(90, 100)
    assert 0.82 < lo < 0.83 and 0.94 < hi < 0.96
    totals = [ss.plan(prevalence=p)["total_images"] for p in (0.08, 0.15, 0.25)]
    assert totals[0] > totals[1] > totals[2]


# -- DR site recalibration ------------------------------------------------------------

def _part(rng, n, pos, sep, prefix, sd=3.0):
    # Gaussian classes at +-sep with spread sd are calibrated at T = sd^2 / (2 sep):
    # sep 4.5, sd 3 -> T 1.0 and AUROC ~0.98, like a real, informative model.
    y = (rng.random(n) < pos).astype(int)
    logit = rng.normal(0, sd, n) + sep * (2 * y - 1)
    return {"logit": logit, "y": y, "patient": [f"{prefix}{i // 2}" for i in range(n)],
            "grade_true": np.where(y == 1, 3.0, 0.0), "grade_pred": np.where(logit > 0, 3, 0)}


def test_recalibration_candidate_uses_the_local_locked_test(monkeypatch):
    import numpy as np  # noqa: F811
    from recalibrate_dr import build_candidate
    from predictor import acceptance_failures
    from test_acceptance import FP, SHA, _good_evaluation

    rng = np.random.default_rng(0)
    current = _good_evaluation(calibration={"method": "temperature", "temperature": 1.4},
                               external_test={"tp": 40, "fn": 12, "auroc": 0.88,
                                              "sensitivity": 0.77, "specificity": 0.85, "ece": 0.15})
    cand = build_candidate(current, _part(rng, 400, 0.4, 4.5, "c"), _part(rng, 300, 0.4, 4.5, "t"), "snapX")
    e = cand["external_test"]
    assert e["n"] == 300 and e["profile"] == "site"
    assert cand["previous_external_test"]["sensitivity"] == 0.77
    assert cand["site_calibration"]["dataset"] == "local snapshot snapX"
    assert "locked_test_under_previous_profile" in cand["site_calibration"]
    assert any(abs(r["target_coverage"] - 0.85) < 1e-9 for r in cand["abstention"])
    assert e["sensitivity"] >= 0.85
    assert acceptance_failures(cand, SHA, FP) == []

    weak = build_candidate(current, _part(rng, 400, 0.4, 0.45, "c", sd=0.95),
                           _part(rng, 300, 0.4, 0.45, "t", sd=0.95), "snapY")
    assert acceptance_failures(weak, SHA, FP), "uninformative scores must fail the gate"


# -- external maculopathy evaluation ----------------------------------------------------

def test_external_dme_report_decides_the_status():
    from dme_model import dme_status
    from evaluate_dme import external_report

    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.3).astype(int)
    good_p = np.clip(0.75 * y + rng.normal(0.15, 0.12, 300), 0, 1)
    rep = external_report(y, good_p, good_p >= 0.5, [f"p{i // 2}" for i in range(300)], "site")
    assert rep["tp"] + rep["fn"] == y.sum() and rep["n"] == 300
    assert dme_status({"external": rep})[0] == "validated"

    noise = rng.random(300)
    weak = external_report(y, noise, noise >= 0.5, [f"p{i}" for i in range(300)], "site")
    status, why = dme_status({"external": weak})
    assert status == "experimental" and any("auroc" in w for w in why)

    few = external_report(y[:40], good_p[:40], good_p[:40] >= 0.5, [f"p{i}" for i in range(40)], "s")
    assert dme_status({"external": few})[0] == "experimental"     # < 30 cases
