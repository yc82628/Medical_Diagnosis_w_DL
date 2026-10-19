# Model Card — Diabetic Retinopathy Referral Triage

*Generated 2026-10-19 from `artifacts/evaluation.json`. Every number below is measured, not asserted.*

> **RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.** No regulatory clearance has been sought or obtained. Not for clinical use.

## Intended use

| | |
|---|---|
| Decision supported | Referral triage in a DR screening programme |
| Target condition | Referable DR — ICDR grade 2 or worse |
| Intended user | Trained grader or clinician within a screening programme |
| Autonomy | None. Every output requires human review. |
| Out of scope | Diagnosis, treatment planning, non-DR pathology (glaucoma, AMD, retinal detachment), diabetic macular oedema, non-fundus modalities, paediatric patients |

See `INTENDED_USE.md` for the full scope statement.

## Operating point

- Decision threshold: **0.4000** on calibrated P(referable)
- Chosen to achieve sensitivity ≥ **0.90** on the validation split, then frozen
- Calibration method: **isotonic** (selected empirically on held-out validation)

The threshold was selected on validation and never on test. Sensitivity is treated as a constraint rather than a quantity to trade off: a missed referable case risks irreversible sight loss, while a false positive costs one appointment.

## Performance — internal test set

| Metric | Value |
|---|---|
| Cases | 294 |
| Referable prevalence | 0.418 |
| AUROC | 0.946 (95% CI 0.910–0.973) |
| Sensitivity | 0.902 |
| Specificity | 0.906 |
| Specificity @ sensitivity 0.90 | 0.906 (95% CI 0.609–0.951) |
| PPV / NPV | 0.874 / 0.928 |
| Expected calibration error | 0.0334 |
| Quadratic weighted kappa (grade 0–4) | 0.732 |
| Referable cases missed | **12** |

Confidence intervals bootstrap **patients**, not images. Resampling images from a dataset where patients contribute two eyes treats correlated observations as independent and produces intervals that are too narrow.

Accuracy is deliberately not reported. At this prevalence, always predicting 'not referable' would score 58% while being clinically useless.

## Performance — external validation

Evaluated on a **different data source** — different camera and population from training.

| Metric | Internal | External |
|---|---|---|
| AUROC | 0.946 (95% CI 0.910–0.973) | 0.875 (95% CI 0.842–0.903) |
| Sensitivity | 0.902 | 0.752 |
| Specificity | 0.906 | 0.854 |
| PPV / NPV | 0.874 / 0.928 | 0.896 / 0.672 |
| Expected calibration error | 0.0334 | 0.1478 |
| Quadratic weighted kappa | 0.732 | 0.589 |
| Cases / referable prevalence | 294 / 0.418 | 514 / 0.626 |
| Missed referable | 12 | 80 |

Same frozen operating point and calibration on both: nothing was refitted on the external set.

AUROC change internal → external: **-0.071**.

**The external figures are the ones to rely on.** Internal test performance on a random split of a single source systematically overstates what happens at a new site.

## Acceptance gate

**NOT PASSED.** The service will not load this model; it runs in labelled DEMO mode instead. Failed criteria:

- external test: sensitivity 0.752 (required >= 0.85)
- external test: ECE 0.148 (required <= 0.1)

These are floors that catch an unready model, not a clinical claim. Do not lower them to make a model pass.

## Data

| Role | Dataset | Licence |
|---|---|---|
| Training / validation / internal test | DeepDRiD (ISBI 2020), regular fundus images (https://github.com/deepdrdoc/DeepDRiD) | CC BY-SA 4.0 (share-alike: whether a trained model must be shared under the same licence is legally unsettled; take advice before commercial use) |
| External test | IDRiD, disease grading (https://zenodo.org/records/17219542 (original: IEEE DataPort, doi:10.21227/H25W98)) | CC BY 4.0 |

- **Cite (training / validation / internal test):** Liu R. et al. DeepDRiD: Diabetic Retinopathy -- Grading and Image Quality Estimation Challenge. Patterns 3(6), 2022.
- Sites in China (Nicheng, Shanghai, national programme); 2 images per eye, patient ids available, so the split and bootstrap are patient-level.
- Graders rated about half the images not diagnosable; graded ones were kept, as in the challenge.
- **Cite (external test):** Porwal P. et al. Indian Diabetic Retinopathy Image Dataset (IDRiD). IEEE DataPort, 2018.
- Single clinic in India, different cameras from DeepDRiD.
- No patient ids: each image is treated as its own patient, so if some patients contributed both eyes the confidence intervals are somewhat too narrow.
- Referable prevalence ~63%, far above a screening population (8-25%), so PPV/NPV here do not transfer; sensitivity and specificity do.

## Image-quality gate

Measured against human gradability labels on 2000 images (`validate_quality_gate.py`):

| | |
|---|---|
| Human-rejected images the gate also rejects (catch rate) | 0.049 |
| Human-accepted images the gate rejects (false rejects) | 0.025 |
| Blur threshold in force when measured | 12.0 |

The classical gate (sharpness, field of view, exposure) barely separates human-accepted from human-rejected images on real data, so it is set to remove gross failures only. It is a backstop, not a gradability assessment: safety rests on the abstention band and on human review of every output. A learned quality model trained on these labels is the intended replacement.

## Abstention and coverage

The system can decline to decide and route a case to a human grader.

| Coverage | Cases auto-decided | Sent to human | Sensitivity | Specificity |
|---|---|---|---|---|
| 100% | 294 | 0 | 0.902 | 0.906 |
| 95% | 285 | 9 | 0.898 | 0.916 |
| 90% | 285 | 9 | 0.898 | 0.916 |
| 85% | 263 | 31 | 0.921 | 0.906 |
| 80% | 263 | 31 | 0.921 | 0.906 |
| 70% | 230 | 64 | 0.971 | 0.896 |
| 60% | 230 | 64 | 0.971 | 0.896 |

Retained-set performance must always be read alongside coverage. Performance on retained cases alone is trivially inflated by abstaining on everything difficult.

## Subgroup performance

> **Not measured.** The dataset carried no age, sex, ethnicity, or device metadata, so performance disparities cannot be detected.

Absence of measurement is not evidence of fairness. Documented disparities in retinal and dermatological imaging models make this a material gap, not a formality.

## Limitations

- **Label ceiling.** Performance cannot exceed the quality of the reference grades. Human inter-grader agreement on ICDR is typically κ 0.6–0.8, which bounds any achievable metric.
- **Grade mix.** ICDR 3-4 make up 23% of the training labels -- far more than in a screening population, where they are often under 3%. Performance on a real screening mix must be measured, not assumed.
- **Ungradable images.** Routed to human review, never scored as negative. In real programmes 5–20% of captures are ungradable.
- **Saliency maps are debugging tools, not explanations.** Any saliency shown in an interface must be labelled as such.
- **Single-image model.** The model scores each image alone, without prior images or patient history. The platform combines both eyes into a visit decision (refer if either eye is referable), but the model never sees the fellow eye.
- **No maculopathy.** Diabetic macular oedema is not assessed, although many screening programmes refer on it.

## Provenance

| | |
|---|---|
| Preprocessing fingerprint | `325d1be426e8c5e1` |
| Calibration | `"isotonic"` |
| Evaluation artifact | `artifacts/evaluation.json` |

The preprocessing fingerprint is stored inside the model checkpoint and verified at load time. A mismatch is a hard error, because silent train/serve skew degrades a model without producing any error message.

## Regulatory status

None. Software informing a clinical decision is a medical device — EU MDR (likely Class IIa+), plus EU AI Act high-risk obligations, or FDA 510(k) in the US. This model has no clearance, no ISO 13485 quality system, and no clinical evaluation. The research-use boundary is a real constraint, not a disclaimer.