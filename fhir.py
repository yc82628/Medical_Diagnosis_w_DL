"""
HL7 FHIR R4 output for one screening visit.

A DiagnosticReport is what an EHR or a referral system ingests; a JSON blob
shaped for this app's UI is not. The report carries one contained Observation
per image, with laterality as a SNOMED CT body site.

Status is "preliminary" until every eye has a human verdict, then "final".
Codes are deliberately plain text plus SNOMED laterality: mapping the decision
itself to a LOINC/SNOMED code is a terminology decision for the receiving
hospital's integration team, and an invented code is worse than none.

Set DR_PATIENT_ID_SYSTEM to the hospital's patient-identifier namespace (its
MRN system URI) so the receiving system can match the patient.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

_BODY_SITE = {
    "right": ("18944008", "Right eye structure"),
    "left": ("8966001", "Left eye structure"),
}

_DECISION_TEXT = {
    "REFER": "Referable diabetic retinopathy suspected - refer to ophthalmology",
    "ROUTINE": "No referable diabetic retinopathy detected - routine rescreening",
    "MANUAL_REVIEW": "Requires review by a human grader",
    "INCOMPLETE": "Screening incomplete - both eyes not yet assessed",
    "UNGRADABLE": "Image ungradable - recapture or examine",
}

DISCLAIMER = ("Produced by research software that is not a medical device. "
              "Detects referable diabetic retinopathy only; does not assess "
              "glaucoma, AMD or other pathology.")


def _patient_system() -> str:
    return os.environ.get("DR_PATIENT_ID_SYSTEM", "urn:dr-screening:patient-id")


def _observation(case: dict, outcome: str, source: str) -> dict:
    obs = {
        "resourceType": "Observation",
        "id": f"obs-{case['case_id']}",
        "status": "final" if source == "grader" else "preliminary",
        "code": {"text": "Referable diabetic retinopathy triage (ICDR grade >= 2)"},
        "valueCodeableConcept": {"text": _DECISION_TEXT.get(outcome, outcome)},
        "method": {"text": ("Human grader verdict" if source == "grader"
                            else f"Automated triage, model {case.get('model_version')}")},
        "note": [{"text": DISCLAIMER}],
    }
    eye = case.get("eye")
    if eye in _BODY_SITE:
        code, display = _BODY_SITE[eye]
        obs["bodySite"] = {"coding": [{"system": "http://snomed.info/sct",
                                       "code": code, "display": display}],
                           "text": f"{eye} eye"}
    if case.get("probability") is not None:
        obs["component"] = [{
            "code": {"text": "Calibrated probability of referable DR"},
            "valueQuantity": {"value": round(float(case["probability"]), 4),
                              "unit": "probability"},
        }]
    if case.get("latest_grader") and source == "grader":
        obs["performer"] = [{"display": case["latest_grader"]}]
    return obs


def encounter_report(encounter_id: str, cases: list[dict], combined: dict) -> dict:
    patient_ids = {c.get("patient_id") for c in cases if c.get("patient_id")}
    if len(patient_ids) > 1:
        raise ValueError(f"Encounter {encounter_id} mixes patient identifiers; "
                         f"refusing to build a report for more than one patient.")
    patient_id = next(iter(patient_ids), None)

    observations = []
    for c in cases:
        if c.get("latest_verdict"):
            outcome, source = c["latest_verdict"], "grader"
        else:
            outcome, source = c["decision"], "model"
        observations.append(_observation(c, outcome, source))

    report = {
        "resourceType": "DiagnosticReport",
        "id": f"dr-screening-{encounter_id}",
        "status": "final" if combined["final"] else "preliminary",
        "code": {"text": "Diabetic retinopathy screening"},
        "issued": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identifier": [{"system": "urn:dr-screening:encounter", "value": encounter_id}],
        "contained": observations,
        "result": [{"reference": f"#{o['id']}"} for o in observations],
        "conclusion": " ".join([_DECISION_TEXT.get(combined["decision"],
                                                   combined["decision"]) + "."]
                               + combined["reasons"] + [DISCLAIMER]),
        "conclusionCode": [{"text": _DECISION_TEXT.get(combined["decision"],
                                                       combined["decision"])}],
    }
    if patient_id:
        report["subject"] = {"identifier": {"system": _patient_system(),
                                            "value": patient_id}}
    return report
