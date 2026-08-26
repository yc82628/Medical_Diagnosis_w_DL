# Model Card — Diabetic Retinopathy Referral Triage

*Generated 2026-08-11 from `artifacts/evaluation.json`. Every number below is measured, not asserted.*

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

- Decision threshold: **0.0000** on calibrated P(referable)
- Chosen to achieve sensitivity ≥ **0.90** on the validation split, then frozen
- Calibration method: **temperature** (selected empirically on held-out validation)

The threshold was selected on validation and never on test. Sensitivity is treated as a constraint rather than a quantity to trade off: a missed referable case risks irreversible sight loss, while a false positive costs one appointment.

## Performance — internal test set

| Metric | Value |
|---|---|
| Cases | 24 |
| Referable prevalence | 0.208 |
| AUROC | 0.647 (95% CI 0.378–0.905) |
| Sensitivity | 1.000 |
| Specificity | 0.000 |
| Specificity @ sensitivity 0.90 | 0.000 (95% CI 0.000–0.848) |
| PPV / NPV | 0.208 / nan |
| Expected calibration error | 0.1747 |
| Quadratic weighted kappa (grade 0–4) | 0.000 |
| Referable cases missed | **0** |

Confidence intervals bootstrap **patients**, not images. Resampling images from a dataset where patients contribute two eyes treats correlated observations as independent and produces intervals that are too narrow.

Accuracy is deliberately not reported. At this prevalence, always predicting 'not referable' would score 79% while being clinically useless.

## Performance — external validation

> **None performed.** This model has never been evaluated on data from a source other than its training set.

This is the most important limitation on this card. A random split of a single dataset shares camera hardware, capture protocol, operator technique, and patient population with the training data, so it cannot detect the failure mode that matters most: degradation at a new site. Published DR models commonly lose 0.05–0.15 AUROC on external data.

**Every figure above should be treated as provisional and optimistic until external validation is run.**

## Abstention and coverage

The system can decline to decide and route a case to a human grader.

| Coverage | Cases auto-decided | Sent to human | Sensitivity | Specificity |
|---|---|---|---|---|
| 100% | 24 | 0 | 1.000 | 0.000 |
| 95% | 24 | 0 | 1.000 | 0.000 |
| 90% | 24 | 0 | 1.000 | 0.000 |
| 85% | 24 | 0 | 1.000 | 0.000 |
| 80% | 24 | 0 | 1.000 | 0.000 |
| 70% | 24 | 0 | 1.000 | 0.000 |
| 60% | 24 | 0 | 1.000 | 0.000 |

Retained-set performance must always be read alongside coverage. Performance on retained cases alone is trivially inflated by abstaining on everything difficult.

## Subgroup performance

Measured at the **shared** operating point. Using a per-subgroup threshold would conceal precisely the disparity being looked for.

| Attribute | Group | n | Positives | Sensitivity | Specificity | AUROC | |
|---|---|---|---|---|---|---|---|
| age_group | 50-65 | 12 | 4 | 1.000 | 0.000 | 0.625 | ⚠ inconclusive |
| age_group | <50 | 9 | 1 | 1.000 | 0.000 | 0.875 | ⚠ inconclusive |
| age_group | >65 | 3 | 0 | nan | 0.000 | nan | ⚠ inconclusive |
| sex | F | 11 | 2 | 1.000 | 0.000 | 0.778 | ⚠ inconclusive |
| sex | M | 13 | 3 | 1.000 | 0.000 | 0.567 | ⚠ inconclusive |
| device | CamA | 9 | 3 | 1.000 | 0.000 | 0.389 | ⚠ inconclusive |
| device | CamB | 10 | 2 | 1.000 | 0.000 | 0.812 | ⚠ inconclusive |
| device | CamC | 5 | 0 | nan | 0.000 | nan | ⚠ inconclusive |

Rows marked inconclusive have too few positive cases for a stable estimate. They are reported as unmeasured, not as evidence of fairness.

## Limitations

- **Label ceiling.** Performance cannot exceed the quality of the reference grades. Human inter-grader agreement on ICDR is typically κ 0.6–0.8, which bounds any achievable metric.
- **Rare severe grades.** ICDR 3 and 4 are often under 3% combined, so estimates at the severe end have wide intervals regardless of dataset size.
- **Ungradable images.** Routed to human review, never scored as negative. In real programmes 5–20% of captures are ungradable.
- **Saliency maps are debugging tools, not explanations.** Any saliency shown in an interface must be labelled as such.
- **Single-image decisions.** No use of prior images, patient history, or the fellow eye, all of which a human grader uses.

## Provenance

| | |
|---|---|
| Preprocessing fingerprint | `325d1be426e8c5e1` |
| Calibration | `"temperature"` |
| Evaluation artifact | `artifacts/evaluation.json` |

The preprocessing fingerprint is stored inside the model checkpoint and verified at load time. A mismatch is a hard error, because silent train/serve skew degrades a model without producing any error message.

## Regulatory status

None. Software informing a clinical decision is a medical device — EU MDR (likely Class IIa+), plus EU AI Act high-risk obligations, or FDA 510(k) in the US. This model has no clearance, no ISO 13485 quality system, and no clinical evaluation. The research-use boundary is a real constraint, not a disclaimer.