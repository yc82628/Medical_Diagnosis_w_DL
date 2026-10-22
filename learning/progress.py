"""
How far the learning loop has got, counted live from graders' decisions.

Used by the service's Monitoring view (GET /api/learning). For each
retraining step it reports progress towards the minimum data that step needs
before its result can mean anything -- the same minimums the scripts enforce:

  quality model   >= 20 Ungradable images among LOCKED-TEST patients
                  (learning/retrain_quality.py, MIN_LOCKED_REJECTS)
  DR calibration  >= 30 referable and >= 30 non-referable images among TRAIN
                  patients, and >= 30 referable in the locked test
                  (learning/recalibrate_dr.py; config.AcceptanceConfig)
  maculopathy     >= 30 images graders marked "maculopathy present"
                  (learning/evaluate_dme.py; dme_model.DME_ACCEPTANCE)

Train / locked-test membership uses the same pseudonymous codes and rule as
snapshot.py, so these numbers match what an export would contain.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from learning.registry import KINDS, REGISTRY, versions
from learning.snapshot import code, load_pseudonym_key, split_of

QUALITY_LOCKED_REJECTS = 20
DR_MIN_EACH = 30
DME_MIN_PRESENT = 30


def _step(name, have, need, what, script):
    return {"name": name, "have": int(have), "need": int(need), "ready": have >= need,
            "what": what, "script": script}


def progress(con, key: bytes | None = None, registry_root: Path = REGISTRY) -> dict:
    key = key or load_pseudonym_key()
    with con.lock:
        rows = [dict(r) for r in con.execute(
            """SELECT p.patient_id, v.verdict, v.grade, v.dme, v.findings
                 FROM predictions p
                 JOIN verdicts v ON v.verdict_id = (
                      SELECT v2.verdict_id FROM verdicts v2 WHERE v2.case_id = p.case_id
                       ORDER BY v2.created_utc DESC, v2.rowid DESC LIMIT 1)
                WHERE p.patient_id IS NOT NULL""")]
    for r in rows:
        r["split"] = split_of(code(key, "patient", r["patient_id"]))
    count = lambda pred: sum(1 for r in rows if pred(r))
    locked = lambda r: r["split"] == "locked_test"
    train = lambda r: r["split"] == "train"

    findings = Counter()
    for r in rows:
        for f in json.loads(r["findings"]) if r["findings"] else []:
            findings[f] += 1

    dr_train_ref = count(lambda r: train(r) and r["verdict"] == "REFER")
    dr_train_non = count(lambda r: train(r) and r["verdict"] == "ROUTINE")
    steps = [
        _step("Image-quality model", count(lambda r: locked(r) and r["verdict"] == "UNGRADABLE"),
              QUALITY_LOCKED_REJECTS, "Ungradable images among locked-test patients",
              "learning/retrain_quality.py"),
        _step("DR calibration for this hospital", min(dr_train_ref, dr_train_non), DR_MIN_EACH,
              "Refer and Routine images among training patients (the smaller of the two)",
              "learning/recalibrate_dr.py"),
        _step("DR calibration: locked test", count(lambda r: locked(r) and r["verdict"] == "REFER"),
              DR_MIN_EACH, "Refer images among locked-test patients",
              "learning/recalibrate_dr.py"),
        _step("Maculopathy check: external validation", count(lambda r: r["dme"] == 1),
              DME_MIN_PRESENT, "images marked 'maculopathy present'", "learning/evaluate_dme.py"),
    ]
    pending = [{"kind": k, "version": v["version"], "source": v.get("source"),
                "passes": (v.get("comparison") or {}).get("candidate_not_worse")}
               for k in KINDS for v in versions(k, registry_root) if v["status"] == "candidate"]
    active = {k: next((v["version"] for v in versions(k, registry_root)
                       if v["status"] == "active"), None) for k in KINDS}
    return {
        "graded_images": len(rows),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "with_grade": count(lambda r: r["grade"] is not None),
        "with_maculopathy_label": count(lambda r: r["dme"] is not None),
        "locked_test_images": count(locked),
        "findings": dict(findings),
        "steps": steps,
        "pending_candidates": pending,
        "active_versions": active,
    }
