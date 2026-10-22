# Study protocol (template)

**Retrospective diagnostic-accuracy study of an AI triage tool for referable
diabetic retinopathy, with site calibration, at [HOSPITAL]**

*Template. Text in [brackets] is for the hospital to complete. The design,
statistics and data handling below describe what this software actually does
(version: git commit [HASH], models by checkpoint SHA-256: see section 6).
Review it with the principal investigator, the data protection officer and the
ethics committee before submission. This is not legal advice.*

| | |
|---|---|
| Sponsor / principal investigator | [Name, department, hospital] |
| Technical lead | [Name] |
| Ethics committee | [Committee], reference [number] |
| Data protection officer | [Name]; DPIA reference [number] (see `DATA_PROTECTION.md`) |
| Registration | [e.g. DRKS / ClinicalTrials.gov number, if required] |
| Reporting guideline | STARD 2015 (diagnostic accuracy), informed by TRIPOD+AI |
| Version / date | [v0.1, date] |

## 1. Background

Diabetic retinopathy (DR) screening programmes grade large numbers of retinal
photographs. An automated triage tool could prioritise the grader's worklist
and flag images that cannot be graded. The tool under study, *Retinal
Screening*, is **research software, not a medical device**. Its performance
was measured on open datasets: DeepDRiD (China) for development, and IDRiD
(India) as an external test. On the external site, the model ranked cases
well but needed **re-calibration** before its decision threshold met the
pre-specified floors (see `MODEL_CARD.md`). Whether it does so at [HOSPITAL],
with [HOSPITAL]'s cameras and population, is unknown. This study answers that
question **without any output reaching patient care**.

## 2. Objectives

**Primary.** Estimate the sensitivity and specificity of the tool for
**referable DR** (ICDR grade 2 or worse) against the study reference standard,
on [HOSPITAL]'s images, after site calibration.

**Secondary.**
1. AUROC for referable DR, overall and per camera model.
2. Agreement of the indicative ICDR grade with the reference standard
   (quadratic weighted kappa).
3. Image-quality check: share of reference-standard *ungradable* images it
   catches, and share of gradable images it wrongly rejects.
4. Maculopathy (exudates near the macula, experimental component):
   sensitivity and specificity against the reference standard. This is the
   external evaluation that could change its status from "experimental".
5. Proportion of images the tool defers to a human (abstention band), and
   accuracy on the rest.
6. Subgroups at the shared operating point: age band, sex, camera, and
   [other locally recorded factors].

## 3. Design

Retrospective, single-centre diagnostic-accuracy study on existing screening
photographs. **No patient is contacted, and no result is returned to the
clinical record or used for care.**

Two phases on separate patients, assigned permanently by pseudonymous code
(see section 7):

| Phase | Patients | Purpose |
|---|---|---|
| A. Site calibration | [~40]% | Fit the calibration and operating point for [HOSPITAL]. The model's weights are **not** changed |
| B. Evaluation | [~60]% | All reported accuracy figures. Opened **once**, after phase A is frozen and its version recorded |

The sensitivity target (≥ 0.90 on phase A) and every analysis below are fixed
in this protocol **before** phase B is unblinded.

## 4. Participants and images

- **Source:** consecutive diabetic-retinopathy screening visits at [HOSPITAL]
  between [start] and [end].
- **Inclusion:** adults (≥ 18) with diabetes; colour fundus photographs of
  either eye, [macula-centred / two-field] protocol, from cameras [models].
- **Exclusion:** ultra-widefield images; images without a reference-standard
  grade; patients who objected to secondary use of their data [if applicable
  under local law].
- **Unit of analysis:** the eye, with confidence intervals bootstrapped by
  **patient** (both eyes of one patient are correlated). The patient-level
  decision (refer if either eye is referable) is a secondary analysis.

## 5. Reference standard

Each image is graded by **two independent graders**, [qualification, e.g.
certified screening graders or ophthalmologists], blind to the tool's output
and to each other, using the platform's grading interface in a study
installation where model outputs are hidden [or: a separate grading tool].
Each grader records:
- referable / not referable / ungradable
- the ICDR grade (0–4)
- maculopathy present / absent
- other findings

**Disagreements are adjudicated** by [a senior ophthalmologist], whose
decision is final. Inter-grader agreement (kappa) is reported, because it
bounds the accuracy any tool can show.

## 6. Index test

The tool, **frozen** for the study. Its version is identified by the git
commit and by the SHA-256 of every model checkpoint (DR model, image-quality
model, maculopathy model). The platform shows these on every result, and the
audit log records them. Settings:
- the DR operating point and the abstention band fitted in phase A
- the image-quality threshold as released
- the maculopathy threshold as released

No change to the software or models during phase B. Any change restarts phase B.

## 7. Data handling

Details in `DATA_PROTECTION.md`. In short:
- Data stays on a hospital server.
- Images are encrypted at rest.
- Patient identifiers are replaced by keyed pseudonymous codes
  (`learning/snapshot.py`), and the code key is held by [role].
- Phase assignment is a fixed function of the code, so it can't be altered
  after the fact.
- Every access, export and model change is recorded in a hash-chained audit
  log.

## 8. Sample size

Computed with `research/sample_size.py` (exact binomial intervals).

| Assumption | Value |
|---|---|
| Expected sensitivity / specificity | 0.90 / 0.85 |
| Success floor (lower 95% bound must exceed) | 0.80 / 0.75 |
| Power | 80% |
| 95% CI half-width at the expected values | ≤ 0.05 |
| Referable prevalence at [HOSPITAL] | [15%, from programme statistics] |
| Ungradable images | [10%] |

This gives **155 referable and 214 non-referable gradable eyes** in phase B:
about **1,150 images** at 15% prevalence, **2,150** at 8% and **690** at 25%.
Phase A adds [~40%] on top. [Recompute with local figures and paste the
output here.]

## 9. Statistical analysis (pre-specified)

1. **Primary:** sensitivity and specificity on phase B at the frozen operating
   point, with exact 95% CIs and patient-level bootstrap CIs. **Success** means
   both lower bounds clear their floors: 0.80 for sensitivity and 0.75 for
   specificity.
2. Images the tool calls ungradable are **not** counted as negatives. They are
   reported separately, and a sensitivity analysis counts them as referrals.
3. Images in the abstention band are analysed both ways: as deferred to a
   human, and as forced to the nearer decision.
4. **AUROC** with a patient-level bootstrap CI.
5. **Quadratic weighted kappa** for the ICDR grade.
6. **Subgroups** at the **shared** operating point (never per-subgroup
   thresholds). Groups with fewer than 30 referable cases are reported as
   inconclusive.
7. **Missing data:** images without an adjudicated reference grade are
   excluded and counted. No imputation.
8. **No interim look at phase B.**

## 10. Optional phase C: shadow mode (separate approval)

The tool runs alongside routine grading. Graders record their decision
**first**, then see the tool's output, and the platform records both. This
measures the effect on grading time and agreement, and it builds the labelled
data for the learning loop (`learning/README.md`). Its outputs still never
reach the patient record.

## 11. Ethics and governance

- The tool is not a CE-marked medical device. This study does not use it for
  care decisions, which [the ethics committee / national authority] should
  confirm places the study outside the medical-device clinical-investigation
  rules (EU MDR Article 62 onwards).
- Legal basis for using the data: [e.g. GDPR Article 9(2)(j) with national
  research provisions; in Germany possibly the GDNG / state hospital law;
  consent or a waiver, as the committee decides].
- No incidental findings are fed back to patients, because the tool's
  outputs are unvalidated. [Or: describe the process if the committee
  requires one.]
- The training datasets' licences permit this research use. DeepDRiD is
  CC BY-SA 4.0 and IDRiD CC BY 4.0, and both are cited in `MODEL_CARD.md`.

## 12. Reporting and outputs

A STARD 2015 checklist and flow diagram (images included, excluded, ungradable
and deferred), the results tables above, the model card regenerated with
[HOSPITAL] as an external site, and the frozen software version. If the
maculopathy component passes its floors on phase B, the result is recorded as
its external evaluation; changing its status is a separate, approved release.

## 13. Timeline and roles

| Step | Who | When |
|---|---|---|
| Ethics and data-protection approval | PI, DPO | [ ] |
| Install the server on hospital network (`DEPLOY.md`) | Hospital IT, technical lead | [ ] |
| Select images, export to the study installation | [ ] | [ ] |
| Reference grading and adjudication | Graders, adjudicator | [ ] |
| Phase A calibration, version frozen | Technical lead | [ ] |
| Phase B evaluation, analysis | Statistician, PI | [ ] |
| Report | PI | [ ] |
