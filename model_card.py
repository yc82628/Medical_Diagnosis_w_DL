from __future__ import annotations
 
import json
from datetime import date
from pathlib import Path
 
 
def _ci(d):
    if not d or d.get("point") is None:
        return "—"
    return f"{d['point']:.3f} (95% CI {d['lo']:.3f}–{d['hi']:.3f})"
 
 
def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"
 
 
# Known datasets, matched against the CSV paths in the config. Licence and
# citation are what the card must carry when a model is trained or evaluated
# on them; "notes" are caveats that change how the numbers should be read.
DATASETS = {
    "deepdrid": {
        "name": "DeepDRiD (ISBI 2020), regular fundus images",
        "source": "https://github.com/deepdrdoc/DeepDRiD",
        "licence": "CC BY-SA 4.0 (share-alike: whether a trained model must be shared "
                   "under the same licence is legally unsettled; take advice before "
                   "commercial use)",
        "cite": "Liu R. et al. DeepDRiD: Diabetic Retinopathy -- Grading and Image "
                "Quality Estimation Challenge. Patterns 3(6), 2022.",
        "notes": ["Sites in China (Nicheng, Shanghai, national programme); 2 images per "
                  "eye, patient ids available, so the split and bootstrap are "
                  "patient-level.",
                  "Graders rated about half the images not diagnosable; graded ones were "
                  "kept, as in the challenge."],
    },
    "idrid": {
        "name": "IDRiD, disease grading",
        "source": "https://zenodo.org/records/17219542 (original: IEEE DataPort, "
                  "doi:10.21227/H25W98)",
        "licence": "CC BY 4.0",
        "cite": "Porwal P. et al. Indian Diabetic Retinopathy Image Dataset (IDRiD). "
                "IEEE DataPort, 2018.",
        "notes": ["Single clinic in India, different cameras from DeepDRiD.",
                  "No patient ids: each image is treated as its own patient, so if some "
                  "patients contributed both eyes the confidence intervals are somewhat "
                  "too narrow.",
                  "Referable prevalence ~63%, far above a screening population (8-25%), "
                  "so PPV/NPV here do not transfer; sensitivity and specificity do."],
    },
    "aptos": {
        "name": "APTOS 2019 Blindness Detection",
        "source": "https://www.kaggle.com/competitions/aptos2019-blindness-detection",
        "licence": "Kaggle competition rules: NON-COMMERCIAL use only, no redistribution",
        "cite": "APTOS 2019 Blindness Detection, Kaggle.",
        "notes": [],
    },
    "demo_data": {
        "name": "Synthetic demo data (make_demo_data.py)",
        "source": "generated locally", "licence": "n/a", "cite": "",
        "notes": ["Synthetic. Numbers measured on it mean nothing clinically."],
    },
}


def _dataset_for(csv_path):
    if not csv_path:
        return None
    low = str(csv_path).lower().replace("\\", "/")
    for key, meta in DATASETS.items():
        if key in low:
            return meta
    return {"name": str(csv_path), "source": "unknown", "licence": "unknown -- check "
            "before use", "cite": "", "notes": []}


def _grade34_share(csv_path, grade_col):
    """Share of ICDR 3-4 in the training labels, or None if unavailable."""
    try:
        import pandas as pd
        g = pd.read_csv(csv_path)[grade_col]
        return float((g >= 3).mean())
    except Exception:                                   # noqa: BLE001
        return None


def _gate(d):
    """(failures, checked) from the same acceptance gate the service applies."""
    try:
        from config import CONFIG
        from predictor import acceptance_failures, file_sha256
        ck = Path(CONFIG.train.save_path)
        sha = file_sha256(ck) if ck.exists() else ""
        return acceptance_failures(d, sha, d.get("preprocess_fingerprint"),
                                   CONFIG.acceptance), True
    except Exception as e:                              # noqa: BLE001
        return [f"could not run the gate: {e}"], False


def _pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def _ci_pct(d):
    if not d or d.get("point") is None:
        return "—"
    return f"{100 * d['point']:.1f}% ({100 * d['lo']:.1f}–{100 * d['hi']:.1f})"


def _quality_section(model_eval="artifacts/quality_model_eval.json",
                     classical_eval="artifacts/quality_gate_validation.json",
                     checkpoint=None) -> list[str]:
    """The image-quality gate: learned model vs classical checks, as measured."""
    from quality_model import QUALITY_CHECKPOINT

    me, ce = Path(model_eval), Path(classical_eval)
    q = json.loads(ce.read_text()) if ce.exists() else None
    if q and q.get("quality_source", "classical") != "classical":
        q = None                    # that file measured the learned gate, not this row
    if not me.exists() and not q:
        return []
    L = ["## Image-quality gate", ""]

    if me.exists():
        m = json.loads(me.read_text())
        t, c, s = m.get("test") or {}, m.get("classical_test") or {}, m.get("split") or {}
        ts = s.get("test") or {}
        L += [f"A learned quality model ({m.get('backbone')}, {m.get('image_size')} px, "
              f"`quality_model.py`) decides gradability, trained on human quality "
              f"ratings (`{m.get('labels')}`). Patients were split "
              f"{(s.get('train') or {}).get('patients', '—')} train / "
              f"{(s.get('val') or {}).get('patients', '—')} validation / "
              f"{ts.get('patients', '—')} test; no patient is in more than one.", "",
              f"Its threshold (P(ungradable) ≥ **{_fmt(m.get('threshold'), 3)}**) was "
              f"chosen on validation so that at most "
              f"{_pct(m.get('max_false_reject_target'))} of human-accepted images are "
              f"rejected, then frozen. Held-out test, {t.get('n', '—')} images from "
              f"{ts.get('patients', '—')} patients ({t.get('n_human_reject', '—')} "
              f"human-rejected):", "",
              "| | Learned model | Classical checks |", "|---|---|---|",
              f"| AUROC (human reject = positive) | {_ci(t.get('auroc_ci'))} | "
              f"{_fmt(c.get('auroc_blur_var'))} (sharpness) |",
              f"| Human-rejected images caught | {_ci_pct(t.get('catch_rate_ci'))} | "
              f"{_pct(c.get('catch_rate'))} |",
              f"| Human-accepted images rejected | "
              f"{_ci_pct(t.get('false_reject_rate_ci'))} | "
              f"{_pct(c.get('false_reject_rate'))} |", "",
              "Intervals bootstrap patients. The classical checks (field of view, gross "
              "blur, exposure) still run as a backstop: the learned model can reject an "
              "image they pass, never pass one they reject.", ""]
        ck = Path(checkpoint or QUALITY_CHECKPOINT)
        if ck.exists():
            from predictor import file_sha256
            if file_sha256(ck) != m.get("checkpoint_sha256"):
                L += [f"> **Stale:** `{ck}` is not the checkpoint these numbers were "
                      f"measured on. Re-run `quality_model.py`.", ""]
        missed = None if t.get("catch_rate") is None else 1 - t["catch_rate"]
        L += [f"{_pct(missed)} of human-rejected test images still reach the DR model "
              "at this false-reject budget. The gate lowers that number; it does not remove "
              "the need for the abstention band and for human review of every "
              "output. Trained and tested on one source (DeepDRiD); not yet measured "
              "on other cameras.", ""]
    else:
        L += ["No learned quality model: the classical checks decide gradability on "
              "their own.", ""]

    if q:
        L += [f"Classical checks alone, against human gradability labels on "
              f"{q.get('n_images')} images (`validate_quality_gate.py --gate classical`):",
              "", "| | |", "|---|---|",
              f"| Human-rejected images the gate also rejects (catch rate) | "
              f"{_fmt(q.get('catch_rate'))} |",
              f"| Human-accepted images the gate rejects (false rejects) | "
              f"{_fmt(q.get('false_reject_rate'))} |",
              f"| Blur threshold in force when measured | "
              f"{(q.get('thresholds') or {}).get('min_blur_var', '—')} |", "",
              "They barely separate human-accepted from human-rejected images, so they "
              "are set to remove gross failures only.", ""]
    return L


def build(eval_path="artifacts/evaluation.json", out_path="MODEL_CARD.md") -> str:
    p = Path(eval_path)
    if not p.exists():
        raise SystemExit(
            f"\n{p} not found.\n"
            f"Run the pipeline first:\n"
            f"    python train.py\n"
            f"    python calibrate.py\n"
            f"That writes the evaluation artifacts this card is generated from."
        )
    d = json.loads(p.read_text())
    internal = d.get("internal_test") or {}
    external = d.get("external_test")
    L = []
 
    L += [
        "# Model Card — Diabetic Retinopathy Referral Triage",
        "",
        f"*Generated {date.today().isoformat()} from `{eval_path}`. "
        "Every number below is measured, not asserted.*",
        "",
        "> **RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.** No regulatory clearance "
        "has been sought or obtained. Not for clinical use.",
        "",
        "## Intended use",
        "",
        "| | |",
        "|---|---|",
        "| Decision supported | Referral triage in a DR screening programme |",
        "| Target condition | Referable DR — ICDR grade 2 or worse |",
        "| Intended user | Trained grader or clinician within a screening programme |",
        "| Autonomy | None. Every output requires human review. |",
        "| Out of scope | Diagnosis, treatment planning, non-DR pathology "
        "(glaucoma, AMD, retinal detachment), diabetic macular oedema, "
        "non-fundus modalities, paediatric patients |",
        "",
        "See `INTENDED_USE.md` for the full scope statement.",
        "",
        "## Operating point",
        "",
        f"- Decision threshold: **{_fmt(d.get('operating_point'), 4)}** on calibrated "
        f"P(referable)",
        f"- Chosen to achieve sensitivity ≥ "
        f"**{_fmt(d.get('target_sensitivity'), 2)}** on the validation split, then frozen",
        f"- Calibration method: **{(d.get('calibration') or {}).get('method', 'none')}** "
        f"(selected empirically on held-out validation)",
        "",
        "The threshold was selected on validation and never on test. Sensitivity is "
        "treated as a constraint rather than a quantity to trade off: a missed "
        "referable case risks irreversible sight loss, while a false positive costs "
        "one appointment.",
        "",
        "## Performance — internal test set",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cases | {internal.get('n', '—')} |",
        f"| Referable prevalence | {_fmt(internal.get('prevalence'))} |",
        f"| AUROC | {_ci(internal.get('auroc_ci'))} |",
        f"| Sensitivity | {_fmt(internal.get('sensitivity'))} |",
        f"| Specificity | {_fmt(internal.get('specificity'))} |",
        f"| Specificity @ sensitivity 0.90 | {_ci(internal.get('spec_at_sens_ci'))} |",
        f"| PPV / NPV | {_fmt(internal.get('ppv'))} / {_fmt(internal.get('npv'))} |",
        f"| Expected calibration error | {_fmt(internal.get('ece'), 4)} |",
        f"| Quadratic weighted kappa (grade 0–4) | {_fmt(internal.get('qwk'))} |",
        f"| Referable cases missed | **{internal.get('fn', '—')}** |",
        "",
        "Confidence intervals bootstrap **patients**, not images. Resampling images "
        "from a dataset where patients contribute two eyes treats correlated "
        "observations as independent and produces intervals that are too narrow.",
        "",
        "Accuracy is deliberately not reported. At this prevalence, always "
        "predicting 'not referable' would score "
        f"{100*(1-internal.get('prevalence', 0)):.0f}% while being clinically useless.",
        "",
    ]
 
    # ---- external ----------------------------------------------------------
    L.append("## Performance — external validation")
    L.append("")
    if external:
        L += [
            "Evaluated on a **different data source** — different camera and "
            "population from training.",
            "",
            "| Metric | Internal | External |",
            "|---|---|---|",
            f"| AUROC | {_ci(internal.get('auroc_ci'))} | {_ci(external.get('auroc_ci'))} |",
            f"| Sensitivity | {_fmt(internal.get('sensitivity'))} | "
            f"{_fmt(external.get('sensitivity'))} |",
            f"| Specificity | {_fmt(internal.get('specificity'))} | "
            f"{_fmt(external.get('specificity'))} |",
            f"| PPV / NPV | {_fmt(internal.get('ppv'))} / {_fmt(internal.get('npv'))} | "
            f"{_fmt(external.get('ppv'))} / {_fmt(external.get('npv'))} |",
            f"| Expected calibration error | {_fmt(internal.get('ece'), 4)} | "
            f"{_fmt(external.get('ece'), 4)} |",
            f"| Quadratic weighted kappa | {_fmt(internal.get('qwk'))} | "
            f"{_fmt(external.get('qwk'))} |",
            f"| Cases / referable prevalence | {internal.get('n','—')} / "
            f"{_fmt(internal.get('prevalence'))} | {external.get('n','—')} / "
            f"{_fmt(external.get('prevalence'))} |",
            f"| Missed referable | {internal.get('fn','—')} | {external.get('fn','—')} |",
            "",
            "Same frozen operating point and calibration on both: nothing was refitted "
            "on the external set.",
            "",
        ]
        a = (internal.get("auroc_ci") or {}).get("point")
        b = (external.get("auroc_ci") or {}).get("point")
        if a is not None and b is not None:
            L += [f"AUROC change internal → external: **{b - a:+.3f}**.", ""]
        L += ["**The external figures are the ones to rely on.** Internal test "
              "performance on a random split of a single source systematically "
              "overstates what happens at a new site.", ""]
    else:
        L += [
            "> **None performed.** This model has never been evaluated on data from "
            "a source other than its training set.",
            "",
            "This is the most important limitation on this card. A random split of a "
            "single dataset shares camera hardware, capture protocol, operator "
            "technique, and patient population with the training data, so it cannot "
            "detect the failure mode that matters most: degradation at a new site. "
            "Published DR models commonly lose 0.05–0.15 AUROC on external data.",
            "",
            "**Every figure above should be treated as provisional and optimistic "
            "until external validation is run.**",
            "",
        ]

    # ---- acceptance gate ---------------------------------------------------
    fails, checked = _gate(d)
    L += ["## Acceptance gate", ""]
    if checked and not fails:
        L += ["**PASSED** every criterion in `config.AcceptanceConfig`. The service "
              "will load this model.", ""]
    else:
        L += ["**NOT PASSED.** The service will not load this model; it runs in "
              "labelled DEMO mode instead. Failed criteria:", ""]
        L += [f"- {f}" for f in fails]
        L += ["", "These are floors that catch an unready model, not a clinical claim. "
              "Do not lower them to make a model pass.", ""]

    # ---- data --------------------------------------------------------------
    try:
        from config import CONFIG
        dc = CONFIG.data
        sets = [("Training / validation / internal test", dc.csv_path)]
        if dc.external_csv_path:
            sets.append(("External test", dc.external_csv_path))
    except Exception:                                   # noqa: BLE001
        sets = []
    if sets:
        L += ["## Data", "", "| Role | Dataset | Licence |", "|---|---|---|"]
        metas = [(role, _dataset_for(path)) for role, path in sets]
        for role, meta in metas:
            L.append(f"| {role} | {meta['name']} ({meta['source']}) | {meta['licence']} |")
        L.append("")
        for role, meta in metas:
            if meta["cite"]:
                L.append(f"- **Cite ({role.lower()}):** {meta['cite']}")
            L += [f"- {n}" for n in meta["notes"]]
        L.append("")

    # ---- quality gate validation ------------------------------------------
    L += _quality_section()

 
    # ---- abstention --------------------------------------------------------
    ab = d.get("abstention") or []
    if ab:
        L += ["## Abstention and coverage", "",
              "The system can decline to decide and route a case to a human grader.",
              "",
              "| Coverage | Cases auto-decided | Sent to human | Sensitivity | Specificity |",
              "|---|---|---|---|---|"]
        for r in ab:
            L.append(
                f"| {r.get('target_coverage', 0)*100:.0f}% | {r.get('n_retained','—')} | "
                f"{r.get('n_referred_to_human','—')} | {_fmt(r.get('sensitivity'))} | "
                f"{_fmt(r.get('specificity'))} |")
        L += ["",
              "Retained-set performance must always be read alongside coverage. "
              "Performance on retained cases alone is trivially inflated by "
              "abstaining on everything difficult.", ""]
 
    # ---- subgroups ---------------------------------------------------------
    L += ["## Subgroup performance", ""]
    subs = d.get("subgroups") or []
    if subs:
        L += ["Measured at the **shared** operating point. Using a per-subgroup "
              "threshold would conceal precisely the disparity being looked for.",
              "",
              "| Attribute | Group | n | Positives | Sensitivity | Specificity | AUROC | |",
              "|---|---|---|---|---|---|---|---|"]
        for s in subs:
            flag = "⚠ inconclusive" if s.get("underpowered") else ""
            L.append(
                f"| {s.get('attribute')} | {s.get('group')} | {s.get('n')} | "
                f"{s.get('n_positive')} | {_fmt(s.get('sensitivity'))} | "
                f"{_fmt(s.get('specificity'))} | {_fmt(s.get('auroc'))} | {flag} |")
        if any(s.get("underpowered") for s in subs):
            L += ["", "Rows marked inconclusive have too few positive cases for a "
                  "stable estimate. They are reported as unmeasured, not as evidence "
                  "of fairness."]
        L.append("")
    else:
        L += ["> **Not measured.** The dataset carried no age, sex, ethnicity, or "
              "device metadata, so performance disparities cannot be detected.",
              "",
              "Absence of measurement is not evidence of fairness. Documented "
              "disparities in retinal and dermatological imaging models make this a "
              "material gap, not a formality.", ""]
 
    def _severe_line():
        try:
            from config import CONFIG
            share = _grade34_share(CONFIG.data.csv_path, CONFIG.data.grade_col)
        except Exception:                               # noqa: BLE001
            share = None
        if share is None:
            return ("- **Severe grades.** ICDR 3 and 4 are rare in screening "
                    "populations, so estimates at the severe end have wide intervals.")
        return (f"- **Grade mix.** ICDR 3-4 make up {share:.0%} of the training labels "
                f"-- far more than in a screening population, where they are often "
                f"under 3%. Performance on a real screening mix must be measured, "
                f"not assumed.")

    # ---- limitations -------------------------------------------------------
    L += [
        "## Limitations",
        "",
        "- **Label ceiling.** Performance cannot exceed the quality of the reference "
        "grades. Human inter-grader agreement on ICDR is typically κ 0.6–0.8, which "
        "bounds any achievable metric.",
        _severe_line(),
        "- **Ungradable images.** Routed to human review, never scored as negative. "
        "In real programmes 5–20% of captures are ungradable.",
        "- **Saliency maps are debugging tools, not explanations.** Any saliency "
        "shown in an interface must be labelled as such.",
        "- **Single-image model.** The model scores each image alone, without prior "
        "images or patient history. The platform combines both eyes into a visit "
        "decision (refer if either eye is referable), but the model never sees the "
        "fellow eye.",
        "- **No maculopathy.** Diabetic macular oedema is not assessed, although many "
        "screening programmes refer on it.",
        "",
        "## Provenance",
        "",
        "| | |",
        "|---|---|",
        f"| Preprocessing fingerprint | `{d.get('preprocess_fingerprint','—')}` |",
        f"| Calibration | `{json.dumps((d.get('calibration') or {}).get('method'))}` |",
        f"| Evaluation artifact | `{eval_path}` |",
        "",
        "The preprocessing fingerprint is stored inside the model checkpoint and "
        "verified at load time. A mismatch is a hard error, because silent "
        "train/serve skew degrades a model without producing any error message.",
        "",
        "## Regulatory status",
        "",
        "None. Software informing a clinical decision is a medical device — EU MDR "
        "(likely Class IIa+), plus EU AI Act high-risk obligations, or FDA 510(k) in "
        "the US. This model has no clearance, no ISO 13485 quality system, and no "
        "clinical evaluation. The research-use boundary is a real constraint, not a "
        "disclaimer.",
    ]
 
    text = "\n".join(L)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(text.splitlines())} lines)")
    if not external:
        print("\n  NOTE: no external validation found. The card states this "
              "prominently, as it should.")
    return text
 
 
if __name__ == "__main__":
    build()
 