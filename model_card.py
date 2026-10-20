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


def _site_section(site: dict, external: dict) -> list[str]:
    src = site.get("evaluation_half_under_source_profile") or {}
    its = site.get("internal_test_under_site_profile") or {}
    return [
        "### Site calibration",
        "",
        f"The external set was split once, by image and stratified on grade (seed "
        f"{site.get('split_seed')}), into a **calibration half** "
        f"({site.get('n_calibration')} images) and an **evaluation half** "
        f"({site.get('n_evaluation')} images). The calibrator, operating point and "
        f"abstention band served for this site were refitted on the calibration half "
        f"only. The evaluation half was read once, after the configuration was "
        f"frozen; model selection used DeepDRiD validation and the calibration half "
        f"(see *Domain-shift experiments*).",
        "",
        "| Same held-out evaluation half | Sensitivity | Specificity | ECE |",
        "|---|---|---|---|",
        f"| Source profile (no site step) | {_fmt(src.get('sensitivity'))} | "
        f"{_fmt(src.get('specificity'))} | {_fmt(src.get('ece'), 4)} |",
        f"| **Site profile (served)** | {_fmt(external.get('sensitivity'))} | "
        f"{_fmt(external.get('specificity'))} | {_fmt(external.get('ece'), 4)} |",
        "",
        "What this means for deployment:",
        "",
        "- **The model is not plug-and-play at a new site.** Its ranking of cases "
        "transfers; its probabilities and threshold do not. Every new site needs "
        "its own labelled calibration sample, graded to the reference standard, "
        "before results are used — and a held-out sample to confirm, as here.",
        f"- The site profile is specific to this site. On the internal test it gives "
        f"sensitivity {_fmt(its.get('sensitivity'))}, specificity "
        f"{_fmt(its.get('specificity'))}, ECE {_fmt(its.get('ece'), 4)} (reported, not "
        f"gated); the internal-test row above uses the source profile.",
        "- IDRiD has no patient ids, so the split is by image: if one patient "
        "contributed both eyes, they may sit on opposite halves, which flatters the "
        "evaluation half slightly.",
        _sens_interval_line(external),
        "",
    ]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def _sens_interval_line(external: dict) -> str:
    tp, fn = external.get("tp"), external.get("fn")
    if tp is None or fn is None:
        return "- The evaluation half is small, so its sensitivity is imprecise."
    lo, hi = _wilson(int(tp), int(tp) + int(fn))
    return (f"- The evaluation half is small: {int(tp) + int(fn)} referable images, so "
            f"its sensitivity has a 95% interval of {lo:.3f}–{hi:.3f} (Wilson). The "
            f"point estimate clears the gate; the interval does not rule out a true "
            f"sensitivity below the 0.85 floor. Confirm on a larger sample from the "
            f"deploying site before any use.")


def _experiments_section(path="artifacts/experiments/results.jsonl") -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        return []
    return [
        "## Domain-shift experiments",
        "",
        "Every configuration tried to close the external gap, failures included "
        "(`domain_experiments.py`). Scored on DeepDRiD validation and the external "
        "**calibration half** only — never on the internal test or the external "
        "evaluation half. *src cal*: calibrator and threshold from DeepDRiD validation, "
        "applied unchanged. *site cal*: refitted on the site, estimated by 2-fold "
        "cross-fitting inside the calibration half (mean ± sd over 5 shuffles). Each "
        "configuration was trained once, with no replicate run, so differences of "
        "about 0.02 AUROC or less should not be read as real.",
        "",
        *experiments_table(rows).splitlines(),
        "",
        "Configurations started or planned but not completed have no result and are "
        "not in the table; `README.md` (*Current results*) lists them.",
        "",
    ]


def experiments_table(rows: list[dict]) -> str:
    head = ("| Experiment | TTA | Val AUROC | Val spec@sens90 | Site AUROC | "
            "Sens (src cal) | ECE (src cal) | Sens (site cal) | Spec (site cal) | "
            "ECE (site cal) |")
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['tta']} | {r['val_auroc']:.3f} | "
            f"{r['val_spec_at_sens90']:.3f} | {r['site_auroc_raw']:.3f} | "
            f"{r['src_sens']:.3f} | {r['src_ece']:.3f} | "
            f"{r['site_sens']:.3f} ± {r['site_sens_sd']:.3f} | {r['site_spec']:.3f} | "
            f"{r['site_ece']:.3f} ± {r['site_ece_sd']:.3f} |")
    return "\n".join(lines)


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
    site = d.get("site_calibration") or {}
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
        (f"- Chosen to achieve sensitivity ≥ **{_fmt(d.get('target_sensitivity'), 2)}** "
         f"on the **calibration half of the external site** ({site.get('n_calibration')} "
         f"images), then frozen — see *Site calibration* below"
         if site else
         f"- Chosen to achieve sensitivity ≥ "
         f"**{_fmt(d.get('target_sensitivity'), 2)}** on the validation split, then frozen"),
        f"- Calibration method: **{(d.get('calibration') or {}).get('method', 'none')}** "
        f"(selected empirically on held-out "
        f"{'site-calibration' if site else 'validation'} data)",
        f"- Test-time augmentation: **{d.get('tta') or 'none'}** (the served predictor "
        f"averages the same views)",
        "",
        ("The threshold was fitted on data that every reported number excludes. "
         if site else
         "The threshold was selected on validation and never on test. ") +
        "Sensitivity is "
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
        ]
        if site:
            L += [
                f"Internal: the source profile (calibration and operating point fitted "
                f"on DeepDRiD validation). External: the **site profile**, measured on "
                f"the {site.get('n_evaluation')} held-out images of the external set "
                f"that no calibration, threshold, band or model choice was fitted on.",
                "",
            ]
        else:
            L += ["Same frozen operating point and calibration on both: nothing was "
                  "refitted on the external set.", ""]
        a = (internal.get("auroc_ci") or {}).get("point")
        b = (external.get("auroc_ci") or {}).get("point")
        if a is not None and b is not None:
            L += [f"AUROC change internal → external: **{b - a:+.3f}**.", ""]
        L += ["**The external figures are the ones to rely on.** Internal test "
              "performance on a random split of a single source systematically "
              "overstates what happens at a new site.", ""]
        if site:
            L += _site_section(site, external)
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

    L += _experiments_section()

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
    qv = Path("artifacts/quality_gate_validation.json")
    if qv.exists():
        q = json.loads(qv.read_text())
        L += ["## Image-quality gate", "",
              f"Measured against human gradability labels on {q.get('n_images')} images "
              f"(`validate_quality_gate.py`):", "",
              "| | |", "|---|---|",
              f"| Human-rejected images the gate also rejects (catch rate) | "
              f"{_fmt(q.get('catch_rate'))} |",
              f"| Human-accepted images the gate rejects (false rejects) | "
              f"{_fmt(q.get('false_reject_rate'))} |",
              f"| Blur threshold in force when measured | "
              f"{(q.get('thresholds') or {}).get('min_blur_var', '—')} |", "",
              "The classical gate (sharpness, field of view, exposure) barely separates "
              "human-accepted from human-rejected images on real data, so it is set to "
              "remove gross failures only. It is a backstop, not a gradability "
              "assessment: safety rests on the abstention band and on human review of "
              "every output. A learned quality model trained on these labels is the "
              "intended replacement.", ""]

 
    # ---- abstention --------------------------------------------------------
    ab = d.get("abstention") or []
    if ab:
        L += ["## Abstention and coverage", "",
              "The system can decline to decide and route a case to a human grader. "
              + ("The band is fitted on the site's calibration half and measured on its "
                 "held-out evaluation half." if site else
                 "The band is fitted on validation and measured on the internal test set."),
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
        f"| Calibration | `{json.dumps((d.get('calibration') or {}).get('method'))}`"
        f"{' (site profile)' if site else ''} |",
        f"| Test-time augmentation | `{d.get('tta') or 'none'}` |",
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
 