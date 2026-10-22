"""
Patient-level decision for one screening visit.

Screening refers PATIENTS, not images. A grader looks at both eyes and refers
if either is referable; scoring images independently and reporting each one
alone leaves that step to whoever reads the output. This module makes it
explicit and deterministic.

Per eye, the effective outcome is the latest human verdict if there is one,
otherwise the model decision. A grader's verdict always overrides the model.
Several images of one eye: the most severe outcome wins.

Visit outcome:
    any eye REFER                               -> REFER
    else any eye MANUAL_REVIEW / UNGRADABLE     -> MANUAL_REVIEW
    else an eye missing (only one eye imaged)   -> INCOMPLETE
    else (both eyes ROUTINE)                    -> ROUTINE

Maculopathy (dme_model): a VALIDATED model's finding turns a model-only
ROUTINE eye into REFER, with its own reason; an EXPERIMENTAL model's finding
only adds advice for the grader. A grader's verdict always stands.

`final` is True only when every eye's outcome comes from a human verdict.
"""
from __future__ import annotations

EYES = ("left", "right")
REFER, ROUTINE, MANUAL, INCOMPLETE = "REFER", "ROUTINE", "MANUAL_REVIEW", "INCOMPLETE"

# Higher = more severe. UNGRADABLE (a human verdict) means recapture/refer for
# examination, so it ranks with MANUAL_REVIEW, never with ROUTINE.
_SEVERITY = {ROUTINE: 0, MANUAL: 1, "UNGRADABLE": 1, REFER: 2}


def normalise_eye(value: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    return {"l": "left", "os": "left", "left": "left",
            "r": "right", "od": "right", "right": "right"}.get(v)


def eye_outcome(case: dict) -> tuple[str, str]:
    """(outcome, source) for one image: source is 'grader' or 'model'."""
    if case.get("latest_verdict"):
        v = case["latest_verdict"]
        return (MANUAL if v == "UNGRADABLE" else v), "grader"
    return case["decision"], "model"


def combine(cases: list[dict]) -> dict:
    per_eye: dict[str, list] = {}
    for c in cases:
        eye = normalise_eye(c.get("eye"))
        if eye is not None:
            per_eye.setdefault(eye, []).append((*eye_outcome(c), c["case_id"]))

    eyes: dict[str, dict] = {}
    for eye, results in per_eye.items():
        outcome, source, case_id = max(results, key=lambda r: _SEVERITY[r[0]])
        eyes[eye] = {"outcome": outcome, "source": source, "case_id": case_id,
                     "all_human": all(r[1] == "grader" for r in results)}

    # Maculopathy, per eye: suspected if any image of that eye is. A VALIDATED
    # maculopathy model turns a model-only ROUTINE eye into REFER; an
    # EXPERIMENTAL one is advice for the grader and changes nothing. A grader's
    # verdict always stands.
    dme_eyes: dict[str, dict] = {}
    for c in cases:
        eye, d = normalise_eye(c.get("eye")), c.get("dme")
        if eye is None or not d:
            continue
        rec = dme_eyes.setdefault(eye, {"suspected": False, "probability": 0.0,
                                        "status": d.get("status", "experimental")})
        rec["suspected"] = rec["suspected"] or bool(d.get("suspected"))
        rec["probability"] = max(rec["probability"], float(d.get("probability") or 0.0))
    dme_referred = []
    for eye, info in dme_eyes.items():
        if eye not in eyes:
            continue
        eyes[eye]["dme"] = info
        if (info["suspected"] and info["status"] == "validated"
                and eyes[eye]["source"] == "model" and eyes[eye]["outcome"] == ROUTINE):
            eyes[eye]["outcome"] = REFER
            dme_referred.append(eye)

    reasons = []
    outcomes = {e: d["outcome"] for e, d in eyes.items()}

    def which(sides) -> str:
        """'both eyes' / 'the right eye' -- read by clinicians, so plain English."""
        sides = [s for s in EYES if s in sides]
        return "both eyes" if len(sides) == 2 else f"the {sides[0]} eye"

    if REFER in outcomes.values():
        decision = REFER
        dr_eyes = [e for e, o in outcomes.items() if o == REFER and e not in dme_referred]
        if dr_eyes:
            reasons.append("Referable diabetic retinopathy suspected in " + which(dr_eyes) + ".")
        if dme_referred:
            reasons.append("Maculopathy suspected in " + which(dme_referred)
                           + " (exudates near the centre of vision).")
    elif MANUAL in outcomes.values():
        decision = MANUAL
        reasons.append("A grader needs to decide on "
                       + which([e for e, o in outcomes.items() if o == MANUAL]) + ".")
    elif set(outcomes) != set(EYES):
        decision = INCOMPLETE
        missing = [e for e in EYES if e not in outcomes]
        reasons.append(f"No result yet for {which(missing)}. A visit is complete "
                       f"only when both eyes are screened.")
    else:
        decision = ROUTINE
        reasons.append("No referable diabetic retinopathy detected in either eye. "
                       "This does not exclude other eye disease.")

    validated_dme = [e for e, i in dme_eyes.items()
                     if i["suspected"] and i["status"] == "validated" and e not in dme_referred]
    if validated_dme:
        reasons.append("Maculopathy also suspected in " + which(validated_dme) + ".")
    advice = [e for e, i in dme_eyes.items() if i["suspected"] and i["status"] != "validated"]
    if advice:
        reasons.append("Maculopathy check (experimental, advice only): exudates near the "
                       "macula may be present in " + which(advice) + ". Please assess.")

    unlabelled = sum(1 for c in cases if normalise_eye(c.get("eye")) is None)
    if unlabelled:
        reasons.append(f"{unlabelled} image(s) have no eye recorded and were not "
                       f"used for the visit decision.")

    final = (decision != INCOMPLETE and set(eyes) == set(EYES)
             and all(d["all_human"] for d in eyes.values()))
    return {"decision": decision, "final": final, "eyes": eyes, "reasons": reasons}
