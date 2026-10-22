# Model Card — Diabetic Retinopathy Referral Triage

*Generated 2026-10-22 from `artifacts/evaluation.json`. Every number below is measured, not asserted.*

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

- Decision threshold: **0.6667** on calibrated P(referable)
- Chosen to achieve sensitivity ≥ **0.90** on the **calibration half of the external site** (257 images), then frozen — see *Site calibration* below
- Calibration method: **isotonic** (selected empirically on held-out site-calibration data)
- Test-time augmentation: **d8** (the served predictor averages the same views)

The threshold was fitted on data that every reported number excludes. Sensitivity is treated as a constraint rather than a quantity to trade off: a missed referable case risks irreversible sight loss, while a false positive costs one appointment.

## Performance — internal test set

| Metric | Value |
|---|---|
| Cases | 294 |
| Referable prevalence | 0.418 |
| AUROC | 0.976 (95% CI 0.949–0.994) |
| Sensitivity | 0.902 |
| Specificity | 0.953 |
| Specificity @ sensitivity 0.90 | 0.953 (95% CI 0.813–0.984) |
| PPV / NPV | 0.933 / 0.931 |
| Expected calibration error | 0.0382 |
| Quadratic weighted kappa (grade 0–4) | 0.763 |
| Referable cases missed | **12** |

Confidence intervals bootstrap **patients**, not images. Resampling images from a dataset where patients contribute two eyes treats correlated observations as independent and produces intervals that are too narrow.

Accuracy is deliberately not reported. At this prevalence, always predicting 'not referable' would score 58% while being clinically useless.

## Performance — external validation

Evaluated on a **different data source** — different camera and population from training.

| Metric | Internal | External |
|---|---|---|
| AUROC | 0.976 (95% CI 0.949–0.994) | 0.938 (95% CI 0.908–0.963) |
| Sensitivity | 0.902 | 0.882 |
| Specificity | 0.953 | 0.854 |
| PPV / NPV | 0.933 / 0.931 | 0.910 / 0.812 |
| Expected calibration error | 0.0382 | 0.0522 |
| Quadratic weighted kappa | 0.763 | 0.664 |
| Cases / referable prevalence | 294 / 0.418 | 257 / 0.626 |
| Missed referable | 12 | 19 |

Internal: the source profile (calibration and operating point fitted on DeepDRiD validation). External: the **site profile**, measured on the 257 held-out images of the external set that no calibration, threshold, band or model choice was fitted on.

AUROC change internal → external: **-0.038**.

**The external figures are the ones to rely on.** Internal test performance on a random split of a single source systematically overstates what happens at a new site.

### Site calibration

The external set was split once, by image and stratified on grade (seed 20261019), into a **calibration half** (257 images) and an **evaluation half** (257 images). The calibrator, operating point and abstention band served for this site were refitted on the calibration half only. The evaluation half was read once, after the configuration was frozen; model selection used DeepDRiD validation and the calibration half (see *Domain-shift experiments*).

| Same held-out evaluation half | Sensitivity | Specificity | ECE |
|---|---|---|---|
| Source profile (no site step) | 0.839 | 0.896 | 0.1162 |
| **Site profile (served)** | 0.882 | 0.854 | 0.0522 |

What this means for deployment:

- **The model is not plug-and-play at a new site.** Its ranking of cases transfers; its probabilities and threshold do not. Every new site needs its own labelled calibration sample, graded to the reference standard, before results are used — and a held-out sample to confirm, as here.
- The site profile is specific to this site. On the internal test it gives sensitivity 0.959, specificity 0.901, ECE 0.0708 (reported, not gated); the internal-test row above uses the source profile.
- IDRiD has no patient ids, so the split is by image: if one patient contributed both eyes, they may sit on opposite halves, which flatters the evaluation half slightly.
- The evaluation half is small: 161 referable images, so its sensitivity has a 95% interval of 0.823–0.923 (Wilson). The point estimate clears the gate; the interval does not rule out a true sensitivity below the 0.85 floor. Confirm on a larger sample from the deploying site before any use.

## Acceptance gate

**PASSED** every criterion in `config.AcceptanceConfig`. The service will load this model.

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

A learned quality model (efficientnet_b0, 448 px, `quality_model.py`) decides gradability, trained on human quality ratings (`deepdrid/quality_labels.csv`). Patients were split 300 train / 100 validation / 100 test; no patient is in more than one.

Its threshold (P(ungradable) ≥ **0.915**) was chosen on validation so that at most 5.0% of human-accepted images are rejected, then frozen. Held-out test, 400 images from 100 patients (214 human-rejected):

| | Learned model | Classical checks |
|---|---|---|
| AUROC (human reject = positive) | 0.811 (95% CI 0.761–0.864) | 0.499 (sharpness) |
| Human-rejected images caught | 33.6% (26.0–42.1) | 3.3% |
| Human-accepted images rejected | 2.7% (0.0–6.4) | 1.1% |

Intervals bootstrap patients. The classical checks (field of view, gross blur, exposure) still run as a backstop: the learned model can reject an image they pass, never pass one they reject.

As served (learned model plus backstop), on the 400 held-out images (`validate_quality_gate.py`, which leaves out every image the model saw): catches **34.6%** of human-rejected images and rejects **3.2%** of human-accepted ones (6 of 186).

66.4% of human-rejected test images still reach the DR model at this false-reject budget. The gate lowers that number; it does not remove the need for the abstention band and for human review of every output. Trained and tested on one source (DeepDRiD); not yet measured on other cameras.

Classical checks alone, against human gradability labels on 2000 images (`validate_quality_gate.py --gate classical`):

| | |
|---|---|
| Human-rejected images the gate also rejects (catch rate) | 0.049 |
| Human-accepted images the gate rejects (false rejects) | 0.025 |
| Blur threshold in force when measured | 12.0 |

They barely separate human-accepted from human-rejected images, so they are set to remove gross failures only.

## Maculopathy (experimental)

Exudates within one disc diameter of the macula centre: IDRiD's *risk of macular edema* grade 2, the usual screening proxy. Not an OCT diagnosis of macular oedema.

Trained on IDRiD (514 gradable images, CC BY 4.0), starting from the DR model's feature extractor. Measured by 5-fold cross-validation: each fold chose its epoch and threshold on its own inner validation split, then scored its held-out fold (test-time augmentation: d8).

| Out-of-fold metric | Value |
|---|---|
| Images / referable maculopathy | 514 / 243 (0.473) |
| AUROC | 0.963 (95% CI 0.947–0.978) |
| Sensitivity | 0.918 (95% CI 0.881–0.950) |
| Specificity | 0.878 (95% CI 0.837–0.916) |
| Missed referable maculopathy | **20** of 243 |

**Status: experimental.** Cross-validation within one clinic cannot show what happens at another, and no second dataset with these labels is available here. Until an external, different-site evaluation passes, the platform shows the result to graders as advice only; it never changes a referral decision.

In IDRiD every eye with referable maculopathy also has DR grade 2 or worse, so the DR model already refers them: here the maculopathy result adds the *reason* (central vision at risk, often more urgent) rather than extra referrals. Eyes with maculopathy but mild DR are exactly where it would add referrals, and this data has none, so that case is untested.

No patient identifiers in IDRiD: if a patient contributed both eyes, they may sit in different folds, making the cross-validated figures slightly optimistic.

## Abstention and coverage

The system can decline to decide and route a case to a human grader. The band is fitted on the site's calibration half and measured on its held-out evaluation half.

| Coverage | Cases auto-decided | Sent to human | Sensitivity | Specificity |
|---|---|---|---|---|
| 100% | 257 | 0 | 0.882 | 0.854 |
| 95% | 257 | 0 | 0.882 | 0.854 |
| 90% | 225 | 32 | 0.862 | 0.931 |
| 85% | 224 | 33 | 0.869 | 0.931 |
| 80% | 210 | 47 | 0.908 | 0.924 |
| 70% | 164 | 93 | 0.920 | 0.984 |
| 60% | 164 | 93 | 0.920 | 0.984 |

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
| Calibration | `"isotonic"` (site profile) |
| Test-time augmentation | `d8` |
| Evaluation artifact | `artifacts/evaluation.json` |

The preprocessing fingerprint is stored inside the model checkpoint and verified at load time. A mismatch is a hard error, because silent train/serve skew degrades a model without producing any error message.

## Regulatory status

None. Software informing a clinical decision is a medical device — EU MDR (likely Class IIa+), plus EU AI Act high-risk obligations, or FDA 510(k) in the US. This model has no clearance, no ISO 13485 quality system, and no clinical evaluation. The research-use boundary is a real constraint, not a disclaimer.