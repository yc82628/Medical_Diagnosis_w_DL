# Research study pack

Material for a hospital that wants to evaluate Retinal Screening on its own
patients' images, before anyone considers clinical use.

| File | For | What it is |
|---|---|---|
| [`STUDY_PROTOCOL.md`](STUDY_PROTOCOL.md) | Principal investigator, ethics committee | Retrospective diagnostic-accuracy study: objectives, design (site calibration, then a locked evaluation), reference standard, pre-specified statistics, ethics |
| [`DATA_PROTECTION.md`](DATA_PROTECTION.md) | Data protection officer | Input for the DPIA: data, storage, access, pseudonymisation, retention, residual risks |
| [`sample_size.py`](sample_size.py) | Statistician | How many images the study needs (exact binomial); about 1,150 at 15% referable prevalence |
| [`../DEPLOY.md`](../DEPLOY.md) | Hospital IT | Installing the server on the staff network |
| [`../learning/README.md`](../learning/README.md) | Technical lead | How graders' decisions improve the models, under change control |
| [`../MODEL_CARD.md`](../MODEL_CARD.md) | Everyone | What the models do, how well, on what data, and their limits |
| [`../INTENDED_USE.md`](../INTENDED_USE.md) | Everyone | Scope, what it is not for, regulatory position |

## What the hospital provides

- An ethics approval and a data-protection assessment.
- Past screening photographs, about 1,150 or more (see the sample size), from
  the cameras the hospital actually uses.
- Two graders and an adjudicator for the reference standard.
- A server: one GPU laptop or PC on the staff network.

## What the hospital gets

- Sensitivity, specificity and AUROC **on its own patients and cameras**,
  with confidence intervals, against its own graders.
- The image-quality check's catch and false-reject rates on its cameras.
- The first external evaluation of the maculopathy check.
- A site-calibrated version of the model and a model card naming the
  hospital as an external site.
- No change to patient care during the study.
