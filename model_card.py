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
            f"| Missed referable | {internal.get('fn','—')} | {external.get('fn','—')} |",
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
 
    # ---- limitations -------------------------------------------------------
    L += [
        "## Limitations",
        "",
        "- **Label ceiling.** Performance cannot exceed the quality of the reference "
        "grades. Human inter-grader agreement on ICDR is typically κ 0.6–0.8, which "
        "bounds any achievable metric.",
        "- **Rare severe grades.** ICDR 3 and 4 are often under 3% combined, so "
        "estimates at the severe end have wide intervals regardless of dataset size.",
        "- **Ungradable images.** Routed to human review, never scored as negative. "
        "In real programmes 5–20% of captures are ungradable.",
        "- **Saliency maps are debugging tools, not explanations.** Any saliency "
        "shown in an interface must be labelled as such.",
        "- **Single-image decisions.** No use of prior images, patient history, or "
        "the fellow eye, all of which a human grader uses.",
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
 