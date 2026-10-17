"""
Serving layer: sign-in and roles, patient linkage, encrypted storage, the
tamper-evident audit chain, per-patient decisions, FHIR, DICOM, monitoring.

Torch-free: the app is built with the demo predictor and temporary paths.
"""
import io
import json
import sqlite3

import cv2
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import audit
import auth
import encounters
import fhir
from predictor import DemoPredictor
from service import create_app
from storage import ImageStore
from test_predictor import _fake_fundus

PW = "correct horse battery"


@pytest.fixture
def env(tmp_path):
    key = Fernet.generate_key()
    ev = tmp_path / "evaluation.json"
    ev.write_text('{"internal_test": {"npv": NaN}}')     # NaN is real: see _json_safe
    app = create_app(db_path=tmp_path / "audit.db", image_root=tmp_path / "images",
                     evaluation_path=ev, data_key=key, predictor=DemoPredictor(), load_note="test")
    con = app.state.con
    for name, role in (("ada", "admin"), ("gus", "grader"), ("pia", "photographer")):
        auth.create_user(con, name, PW, role, iterations=1000)
    return app, con, tmp_path, key


def client_for(app, user):
    c = TestClient(app)
    r = c.post("/api/login", json={"username": user, "password": PW})
    assert r.status_code == 200, r.text
    return c


def _png(img=None):
    ok, buf = cv2.imencode(".png", _fake_fundus() if img is None else img)
    return buf.tobytes()


def upload(c, patient="P001", eye="left", encounter="V1", data=None, name="a.png"):
    return c.post("/api/predict",
                  files={"file": (name, data or _png(), "image/png")},
                  data={k: v for k, v in (("patient_id", patient), ("eye", eye),
                                          ("encounter_id", encounter)) if v is not None})


# -- authentication & roles -----------------------------------------------------

def test_everything_but_signin_requires_an_account(env):
    app, *_ = env
    c = TestClient(app)
    assert c.get("/").status_code == 200
    for path in ("/api/system", "/api/cases", "/api/summary", "/api/me"):
        assert c.get(path).status_code == 401, path
    assert upload(c).status_code == 401


def test_wrong_password_is_refused_and_logged(env):
    app, con, *_ = env
    r = TestClient(app).post("/api/login", json={"username": "gus", "password": "nope"})
    assert r.status_code == 401
    n = con.execute("SELECT COUNT(*) FROM audit_log WHERE action='login_failed'").fetchone()[0]
    assert n == 1


def test_lockout_after_repeated_failures(env):
    app, *_ = env
    c = TestClient(app)
    for _ in range(auth.MAX_FAILED_ATTEMPTS):
        c.post("/api/login", json={"username": "gus", "password": "nope"})
    r = c.post("/api/login", json={"username": "gus", "password": PW})
    assert r.status_code == 401 and "locked" in r.json()["detail"]


def test_disabled_account_loses_its_session(env):
    app, con, *_ = env
    c = client_for(app, "gus")
    assert c.get("/api/me").status_code == 200
    auth.set_active(con, "gus", False)
    assert c.get("/api/me").status_code == 401


def test_logout_revokes_session(env):
    app, *_ = env
    c = client_for(app, "gus")
    token = c.cookies.get("dr_session")
    c.post("/api/logout")
    c2 = TestClient(app)
    c2.cookies.set("dr_session", token)
    assert c2.get("/api/me").status_code == 401


def test_photographer_cannot_record_verdicts(env):
    app, *_ = env
    c = client_for(app, "pia")
    case_id = upload(c).json()["case_id"]
    r = c.post(f"/api/cases/{case_id}/verdict", json={"verdict": "REFER"})
    assert r.status_code == 403
    assert c.get("/api/audit/verify").status_code == 403


def test_verdict_is_attributed_to_the_signed_in_grader(env):
    app, *_ = env
    case_id = upload(client_for(app, "pia")).json()["case_id"]
    g = client_for(app, "gus")
    # A `grader` field in the body is ignored: identity comes from the session.
    r = g.post(f"/api/cases/{case_id}/verdict", json={"verdict": "REFER", "grader": "mallory"})
    assert r.status_code == 200 and r.json()["grader"] == "gus"
    assert g.get(f"/api/cases/{case_id}").json()["verdicts"][0]["grader"] == "gus"


def test_sso_header_maps_to_local_account(tmp_path):
    app = create_app(db_path=tmp_path / "a.db", image_root=tmp_path / "i",
                     data_key=Fernet.generate_key(), predictor=DemoPredictor(),
                     sso_header="X-Forwarded-User")
    auth.create_user(app.state.con, "gus", PW, "grader", iterations=1000)
    c = TestClient(app)
    assert c.get("/api/me", headers={"X-Forwarded-User": "gus"}).json()["role"] == "grader"
    assert c.get("/api/me", headers={"X-Forwarded-User": "eve"}).status_code == 403


def test_password_policy():
    con = audit.connect(":memory:")
    auth.init_schema(con)
    with pytest.raises(ValueError):
        auth.create_user(con, "x", "short", "grader", iterations=1000)
    with pytest.raises(ValueError):
        auth.create_user(con, "x", PW, "superuser", iterations=1000)


# -- patient linkage ------------------------------------------------------------

def test_patient_and_eye_are_required(env):
    app, *_ = env
    c = client_for(app, "pia")
    assert upload(c, patient=None).status_code == 400
    assert upload(c, eye=None).status_code == 400
    assert upload(c, eye="middle").status_code == 400
    r = upload(c, eye="OD")
    assert r.status_code == 200 and r.json()["eye"] == "right"


def test_default_encounter_is_patient_and_date(env):
    app, *_ = env
    r = upload(client_for(app, "pia"), encounter=None).json()
    assert r["encounter_id"].startswith("P001:")


def test_upload_size_limit(env, monkeypatch):
    import service
    monkeypatch.setattr(service, "MAX_UPLOAD_MB", 0.001)
    app, *_ = env
    assert upload(client_for(app, "pia")).status_code == 413


def test_undecodable_upload(env):
    app, *_ = env
    assert upload(client_for(app, "pia"), data=b"not an image").status_code == 400


# -- encrypted storage ----------------------------------------------------------

def test_no_plaintext_pixels_on_disk(env):
    app, _, tmp, key = env
    c = client_for(app, "pia")
    case_id = upload(c).json()["case_id"]
    files = list((tmp / "images").rglob("*"))
    blobs = [f.read_bytes() for f in files if f.is_file()]
    assert blobs, "images should be stored"
    for b in blobs:
        assert not b.startswith(b"\x89PNG") and not b.startswith(b"\xff\xd8")
    # ...and the API decrypts them for an authorised user.
    r = c.get(f"/api/cases/{case_id}/image")
    assert r.status_code == 200 and r.content.startswith(b"\xff\xd8")


def test_wrong_key_cannot_read(tmp_path):
    s = ImageStore(tmp_path, Fernet.generate_key())
    s.put("abc123", "raw", b"pixels")
    with pytest.raises(RuntimeError):
        ImageStore(tmp_path, Fernet.generate_key()).get("abc123", "raw")


def test_store_rejects_path_like_ids(tmp_path):
    s = ImageStore(tmp_path, Fernet.generate_key())
    with pytest.raises(ValueError):
        s.put("../../etc", "raw", b"x")


def test_encrypt_legacy_and_purge(tmp_path):
    prev, raw = tmp_path / "previews", tmp_path / "raw"
    prev.mkdir(); raw.mkdir()
    (prev / "abc123.jpg").write_bytes(b"\xff\xd8jpeg")
    (prev / "abc123_gradcam.jpg").write_bytes(b"\xff\xd8cam")
    (raw / "abc123.png").write_bytes(b"\x89PNGpng")
    s = ImageStore(tmp_path / "store", Fernet.generate_key())
    assert s.encrypt_legacy(prev, raw) == 3
    assert not list(prev.iterdir()) and not list(raw.iterdir())
    assert s.get("abc123", "gradcam") == b"\xff\xd8cam"
    assert s.purge_older_than(-1) == 3 and s.get("abc123", "raw") is None


# -- audit chain ----------------------------------------------------------------

def test_chain_verifies_and_is_append_only(env):
    app, con, *_ = env
    case_id = upload(client_for(app, "pia")).json()["case_id"]
    client_for(app, "gus").post(f"/api/cases/{case_id}/verdict", json={"verdict": "ROUTINE"})
    res = client_for(app, "ada").get("/api/audit/verify").json()
    assert res["ok"], res["problems"]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE predictions SET decision='ROUTINE'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM audit_log")


def test_direct_tampering_is_detected(env):
    app, con, *_ = env
    case_id = upload(client_for(app, "pia")).json()["case_id"]
    # Someone with file access drops the trigger and edits the decision.
    con.execute("DROP TRIGGER predictions_no_update")
    con.execute("UPDATE predictions SET decision='ROUTINE' WHERE case_id=?", (case_id,))
    con.commit()
    res = audit.verify_chain(con)
    assert not res["ok"] and any("altered" in p for p in res["problems"])


def test_views_and_exports_are_logged(env):
    app, con, *_ = env
    g = client_for(app, "gus")
    case_id = upload(g).json()["case_id"]
    g.get(f"/api/cases/{case_id}")
    g.get("/api/encounters/V1/fhir")
    actions = {r[0] for r in con.execute("SELECT action FROM audit_log")}
    assert {"login", "prediction", "view_case", "export_fhir"} <= actions


def test_migration_adds_columns_to_old_database(tmp_path):
    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.executescript("""CREATE TABLE predictions (case_id TEXT PRIMARY KEY,
        created_utc TEXT NOT NULL, image_sha256 TEXT NOT NULL, filename TEXT,
        decision TEXT NOT NULL, probability REAL, grade INTEGER, gradable INTEGER NOT NULL,
        quality_reasons TEXT, operating_point REAL, abstain_low REAL, abstain_high REAL,
        model_version TEXT NOT NULL, preprocess_fp TEXT NOT NULL, is_demo INTEGER NOT NULL,
        latency_ms REAL, payload_json TEXT NOT NULL);
        INSERT INTO predictions VALUES ('aaa','2026-01-01T00:00:00+00:00','h',NULL,'ROUTINE',
        0.1,0,1,'',0.5,NULL,NULL,'m','fp',1,10,'{}');""")
    old.commit(); old.close()
    con = audit.connect(db)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(predictions)")}
    assert {"patient_id", "eye", "encounter_id", "validated"} <= cols
    assert audit.case_detail(con, "aaa")["decision"] == "ROUTINE"


# -- summary statistics -----------------------------------------------------------

def test_agreement_counts_each_case_once_by_latest_verdict(env):
    app, con, *_ = env
    g = client_for(app, "gus")
    # Pin a known model decision so agreement is deterministic.
    p = app.state.predictor
    p.operating_point, p.abstain_low, p.abstain_high = 0.0, None, None   # always REFER
    case_id = upload(g).json()["case_id"]
    for v in ("ROUTINE", "ROUTINE", "REFER"):                            # revised twice
        g.post(f"/api/cases/{case_id}/verdict", json={"verdict": v})
    s = g.get("/api/summary").json()
    assert s["agreement_rate"] == 1.0     # latest verdict REFER agrees; earlier ones do not count


def test_median_latency_is_a_median(tmp_path):
    con = audit.connect(tmp_path / "a.db")
    p = DemoPredictor()
    img = _fake_fundus()
    for ms in (1.0, 2.0, 100.0):
        pred = p.predict(img)
        pred.latency_ms = ms
        audit.log_prediction(con, pred, patient_id="P", eye="left")
    assert audit.summary(con)["median_latency_ms"] == 2.0


# -- patient-level decision -----------------------------------------------------

def _c(eye, decision, verdict=None, cid=None):
    return {"case_id": cid or f"{eye}{decision}", "eye": eye, "decision": decision,
            "latest_verdict": verdict}


@pytest.mark.parametrize("cases, expect", [
    ([_c("left", "ROUTINE"), _c("right", "REFER")], "REFER"),
    ([_c("left", "ROUTINE"), _c("right", "MANUAL_REVIEW")], "MANUAL_REVIEW"),
    ([_c("left", "ROUTINE"), _c("right", "ROUTINE")], "ROUTINE"),
    ([_c("left", "ROUTINE")], "INCOMPLETE"),
    ([_c("left", "ROUTINE"), _c("right", "REFER", verdict="ROUTINE")], "ROUTINE"),
    ([_c("left", "ROUTINE"), _c("right", "ROUTINE", verdict="UNGRADABLE")], "MANUAL_REVIEW"),
    # Two images of one eye: the more severe wins.
    ([_c("left", "ROUTINE", cid="a"), _c("left", "REFER", cid="b"),
      _c("right", "ROUTINE")], "REFER"),
])
def test_visit_decision(cases, expect):
    assert encounters.combine(cases)["decision"] == expect


def test_visit_final_only_when_every_eye_human_reviewed():
    assert not encounters.combine([_c("left", "ROUTINE", "ROUTINE"),
                                   _c("right", "ROUTINE")])["final"]
    assert encounters.combine([_c("left", "ROUTINE", "ROUTINE"),
                               _c("right", "REFER", "REFER")])["final"]


def test_encounter_endpoint_combines_both_eyes(env):
    app, *_ = env
    c = client_for(app, "gus")
    upload(c, eye="left")
    upload(c, eye="right")
    e = c.get("/api/encounters/V1").json()
    assert set(e["eyes"]) == {"left", "right"} and e["patient_id"] == "P001"
    assert e["decision"] in ("REFER", "ROUTINE", "MANUAL_REVIEW")


# -- FHIR -----------------------------------------------------------------------

def test_fhir_report_shape(env):
    app, *_ = env
    c = client_for(app, "gus")
    ids = [upload(c, eye=e).json()["case_id"] for e in ("left", "right")]
    r = c.get("/api/encounters/V1/fhir")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/fhir+json")
    rep = r.json()
    assert rep["resourceType"] == "DiagnosticReport" and rep["status"] == "preliminary"
    assert rep["subject"]["identifier"]["value"] == "P001"
    assert len(rep["contained"]) == 2 and len(rep["result"]) == 2
    sites = {o["bodySite"]["coding"][0]["code"] for o in rep["contained"]}
    assert sites == {"8966001", "18944008"}
    for cid in ids:
        c.post(f"/api/cases/{cid}/verdict", json={"verdict": "ROUTINE"})
    assert c.get("/api/encounters/V1/fhir").json()["status"] == "final"


def test_fhir_refuses_mixed_patients():
    combined = encounters.combine([_c("left", "ROUTINE")])
    rows = [dict(_c("left", "ROUTINE"), patient_id="A"),
            dict(_c("right", "ROUTINE", cid="z"), patient_id="B")]
    with pytest.raises(ValueError):
        fhir.encounter_report("V", rows, combined)


# -- DICOM ----------------------------------------------------------------------

def _dicom_bytes(patient="D-77", laterality="R"):
    pytest.importorskip("pydicom")
    from pydicom.dataset import FileMetaDataset, Dataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    rgb = cv2.cvtColor(_fake_fundus(256), cv2.COLOR_BGR2RGB)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"  # Ophthalmic Photography 8 Bit
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID, ds.PatientName = patient, "Never^Stored"
    ds.StudyInstanceUID = "1.2.3.4.5"
    ds.ImageLaterality = laterality
    ds.Manufacturer, ds.ManufacturerModelName = "Acme", "Fundus 3000"
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 3, "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelData = rgb.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_dicom_upload_reads_identifiers_from_tags(env):
    app, con, *_ = env
    c = client_for(app, "pia")
    r = c.post("/api/predict", files={"file": ("x.dcm", _dicom_bytes(), "application/dicom")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["patient_id"], j["eye"], j["encounter_id"], j["device"]) == \
        ("D-77", "right", "1.2.3.4.5", "Acme Fundus 3000")
    row = audit.case_detail(con, j["case_id"])
    assert row["source_format"] == "dicom"
    assert "Never" not in json.dumps(row), "patient name must not be stored"


def test_dicom_patient_mismatch_is_refused(env):
    app, *_ = env
    r = client_for(app, "pia").post(
        "/api/predict", files={"file": ("x.dcm", _dicom_bytes(), "application/dicom")},
        data={"patient_id": "SOMEONE-ELSE"})
    assert r.status_code == 400 and "does not match" in r.json()["detail"]


# -- monitoring -----------------------------------------------------------------

def test_monitoring_flags_falling_agreement(tmp_path):
    con = audit.connect(tmp_path / "m.db")
    p = DemoPredictor()
    p.operating_point, p.abstain_low, p.abstain_high = 0.0, None, None   # always REFER
    img = _fake_fundus()
    for i in range(25):
        cid = audit.log_prediction(con, p.predict(img), patient_id=f"P{i}", eye="left",
                                   device="CamA")
        audit.log_verdict(con, cid, "ROUTINE", grader="gus")          # all disagree
    m = audit.monitoring(con)
    assert m["recent"]["agreement_rate"] == 0.0
    assert any("agreement" in a.lower() for a in m["alerts"])
    assert m["weekly"] and m["by_device"][0]["device"] == "CamA"


def test_monitoring_endpoint_is_clinical_only(env):
    app, *_ = env
    assert client_for(app, "pia").get("/api/monitoring").status_code == 403
    assert client_for(app, "gus").get("/api/monitoring").status_code == 200


def test_system_reports_demo_mode(env):
    app, *_ = env
    s = client_for(app, "pia").get("/api/system").json()
    assert s["mode"] == "DEMO" and s["validated"] is False
    assert s["evaluation"]["internal_test"]["npv"] is None


def test_security_headers(env):
    app, *_ = env
    r = TestClient(app).get("/")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"


def test_health_is_public_and_says_nothing(env):
    app, *_ = env
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
