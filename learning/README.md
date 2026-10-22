# The learning loop: improving the models from doctors' decisions

Every time a grader records a decision, the platform gains a labelled
example from **this hospital's** cameras and patients. That's what the
open-data-trained models are missing. This folder turns those decisions into
better models, **without the model ever changing itself**.

```
 graders decide          export             retrain offline         a person decides
┌──────────────┐    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ verdict,     │──▶ │ snapshot.py  │ ──▶ │ retrain_*.py     │ ──▶ │ registry.py      │
│ grade,       │    │ pseudonymised│     │ candidate model, │     │ promote (named,  │
│ maculopathy, │    │ locked test  │     │ compared on the  │     │ reasoned, logged)│
│ findings     │    │ set fixed    │     │ LOCKED TEST set  │     │ or roll back     │
└──────────────┘    └──────────────┘     └──────────────────┘     └──────────────────┘
        ▲                                                                  │
        └─────────────── restart: the new version serves ◀────────────────┘
```

## Why it never updates itself

A model that silently learned from its own users could learn one person's
mistakes, or drift without anyone noticing. Medical-device rules expect
every change to be a **controlled, versioned release**. In the EU that means
MDR change control; in the US, the FDA's *predetermined change control plan*
for AI. So:
- **Retraining makes a *candidate*, which serves no one.**
- **The candidate is compared with the current model on locked-test patients**
  it has never seen.
- **A named person promotes it**, with a reason, recorded in the
  tamper-evident audit log. A worse candidate is refused unless an override
  reason is recorded.
- **The previous version is archived**, so `rollback` restores it at once.

## Step by step

```bat
REM 1. Export what graders decided (stays on hospital storage)
.venv\Scripts\python learning\snapshot.py

REM 2. Retrain the image-quality model with the hospital's "Ungradable" verdicts
.venv\Scripts\python learning\retrain_quality.py snapshots\20261022-150000

REM    and re-calibrate the DR model for this hospital from its Refer / Routine verdicts
.venv\Scripts\python learning\recalibrate_dr.py snapshots\20261022-150000

REM 3. Look, then decide
.venv\Scripts\python learning\registry.py list quality
.venv\Scripts\python learning\registry.py promote quality <version> --approved-by "Dr ..." --reason "..."
REM    restart the server; if anything looks wrong:
.venv\Scripts\python learning\registry.py rollback quality --approved-by "Dr ..." --reason "..."
```

## What each label teaches

| Grader records | Used for |
|---|---|
| **Ungradable** | Image-quality model: this site's cameras and capture faults (`retrain_quality.py`) |
| Refer / Routine | DR model: **site calibration** on local cases (`recalibrate_dr.py`; the study protocol's phase A) |
| ICDR grade 0–4 | DR model: grade-level retraining |
| Maculopathy present / absent | The **external evaluation** the experimental maculopathy check needs before it may affect decisions |
| Other findings (glaucoma suspected, AMD, vein occlusion, …) | Nothing yet. They build the labelled dataset a future model for that condition would need, and count trends |

## New or unexpected diseases

The models can't learn a new disease from a handful of cases, and the
platform doesn't pretend to. A doctor who sees something new tags it under
**Other findings**, with a note. Those tags:
- collect the cases (with their images) in every snapshot, ready for a
  future, separately validated model
- show the doctors' own counts over time, so a cluster gets noticed by
  people, not by a model guessing

## The locked test set

`snapshot.py` assigns each patient **permanently**, by their pseudonymous
code, to training (80%) or the locked test set (20%). A patient in the locked
test set is never trained on, in any snapshot, ever. That's what makes "the
new version is better on our own patients" an honest claim. Don't change
`--locked-fraction` between snapshots of one site.

## Requirements before a candidate can show anything

**DR re-calibration** (`recalibrate_dr.py`) needs at least 30 referable and
30 non-referable graded images among the *training* patients, and referable
images in the locked test set. It keeps the model's weights, re-fits only the
calibration, threshold and uncertainty band, and passes only if the result
clears the service's own acceptance gate, with this hospital's locked test as
the external test.

**Image quality**

`retrain_quality.py` won't call a candidate "not worse" until the locked test
set holds **at least 20 human-rejected images**. With fewer, no comparison is
meaningful. At a typical 5–20% ungradable rate, that's roughly 100–400
graded images in the locked test set, or 500–2,000 graded images in total.
