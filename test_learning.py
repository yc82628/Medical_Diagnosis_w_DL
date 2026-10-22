"""Learning loop: snapshot export, the locked test set, the model registry. Torch-free."""
import json
import sys
from pathlib import Path

import cv2
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
