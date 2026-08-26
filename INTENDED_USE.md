# Intended Use Statement

**Status: RESEARCH SOFTWARE. NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE.**

This document is written first and changed rarely. Everything downstream — the
metric, the threshold, the abstention policy, the interface — is derived from it.
If a proposed change to the system cannot be justified against this document,
the change is out of scope.

---

## 1. What decision does this support?

**Referral triage in a diabetic retinopathy screening programme.**

Given a colour fundus photograph, the system estimates whether the eye shows
**referable diabetic retinopathy** — ICDR grade ≥ 2 (moderate NPDR or worse) —
and outputs one of three states:

| Output | Meaning |
|---|---|
| `REFER` | Route to an ophthalmologist for assessment |
| `ROUTINE` | No referable disease detected; continue routine screening interval |
| `MANUAL REVIEW` | Model abstains, or image is ungradable; route to a human grader |

The system supports **prioritisation of a human grader's queue**. It does not
diagnose, does not stage disease for treatment planning, and does not discharge
patients from screening.

## 2. Who is the intended user?

A trained grader or clinician within an established screening programme. **Not**
patients, and not clinicians without a screening workflow to place it in.

Every output is reviewed by a human before it reaches a patient. The system is
assistive; there is no autonomous mode.

## 3. What is explicitly out of scope?

- Any use on a patient without a diabetes diagnosis
- Grading for treatment decisions (anti-VEGF, photocoagulation, vitrectomy)
- Detection of non-DR pathology — glaucoma, AMD, retinal detachment, tumours.
  **The model has never been trained to see these and will not flag them.**
  A `ROUTINE` output means "no referable DR detected", never "this eye is healthy."
- Diabetic macular oedema, which frequently requires OCT and cannot be reliably
  determined from a single fundus photograph
- Any imaging modality other than colour fundus photography
- Paediatric patients
- Images from camera types not represented in the validation data

## 4. How is performance measured, and why?

**Primary metric: specificity at fixed sensitivity ≥ 0.90**, with a 95%
confidence interval bootstrapped at the **patient** level.

Rationale: in screening, a missed referable case can progress to irreversible
sight loss, while a false positive costs one unnecessary appointment. These are
not symmetric, so sensitivity is a **constraint**, not something to trade off.
Specificity is what we optimise subject to that constraint.

**Accuracy is not reported anywhere in this project.** At the ~8–25% referable
prevalence typical of screening populations, a model that outputs `ROUTINE`
unconditionally scores 75–92% accuracy while being clinically worthless.

Also reported: AUROC (patient-level CI), quadratic weighted kappa for the full
0–4 grade, expected calibration error, coverage/sensitivity trade-off under
abstention, and per-subgroup metrics at the **shared** operating point.

## 5. What happens when it is wrong?

| Failure | Consequence | Mitigation |
|---|---|---|
| False negative | Referable disease missed; possible progression to sight loss | Sensitivity constrained ≥ 0.90; abstention band routes borderline cases to humans; every output is human-reviewed |
| False positive | Unnecessary referral; patient anxiety; wasted clinic capacity | Specificity reported at the fixed operating point so service load is predictable |
| Ungradable image scored as `ROUTINE` | **Most dangerous mode.** A blurred or dark image is silently called healthy | Quality gate runs *before* the model; ungradable → `MANUAL REVIEW`, never `ROUTINE` |
| Out-of-distribution input (non-fundus, other pathology) | Confident nonsense | Quality gate; OOD detection; explicit scope limits above |
| Silent degradation after deployment | Undetected drift in performance | Input/prediction drift monitoring; periodic re-validation against human grades |

## 6. Known limitations

- **Single-source training data.** Until external validation on a different
  site/camera is run, every reported number is provisional.
- **Label ceiling.** Performance cannot exceed the quality of the reference
  grades. Human inter-grader agreement on ICDR is itself imperfect (κ typically
  0.6–0.8), which bounds any achievable metric.
- **Subgroup coverage.** Without age, sex, ethnicity, and camera metadata,
  disparities cannot be detected. Absence of measurement is not evidence of
  fairness.
- **Grade 3 and 4 are rare** (often <3% combined), so per-grade estimates at the
  severe end have wide intervals regardless of dataset size.
- **Saliency maps are a debugging tool, not an explanation.** Grad-CAM output
  shown in any interface must be labelled as such.

## 7. Regulatory position

Software that informs a clinical decision is a medical device. In the EU this
falls under **MDR**, likely Class IIa or higher, requiring a notified body, an
ISO 13485 quality management system, IEC 62304 lifecycle documentation, clinical
evaluation, and post-market surveillance. The **EU AI Act** adds high-risk
obligations. In the US the route is typically FDA **510(k)**.

**This project has none of these.** It is therefore research software, and the
"not for clinical use" boundary is a real constraint, not a disclaimer. This
document is the artefact that would become the root of a technical file if the
work were ever taken down that path.

## 8. Change log

| Date | Change | Rationale |
|---|---|---|
| _(initial)_ | Scope defined: referral triage, ICDR ≥ 2 | Standard screening threshold |
