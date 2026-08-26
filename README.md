# Diabetic Retinopathy Screening — foundation

**Research software. Not a medical device.** Hitting errors? Go to `RUNBOOK.md`
— every error, cause and fix, plus the exact command sequence. New here? `SETUP.md`
for a step-by-step walkthrough. Read `INTENDED_USE.md` first — it
defines the scope that every design decision here follows from.

## Run order

Full step-by-step with troubleshooting: **`RUNBOOK.md`**.

```bash
pip install -r requirements.txt -r requirements-dev.txt -r requirements-serve.txt
# Windows GPU, or a specific CUDA version: pip install -r requirements-gpu.txt FIRST
# Slim CPU-only (CI, laptops):             pip install -r requirements-cpu.txt FIRST

# A CPU fallback fails silently, so check explicitly.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 1. Generate synthetic data. Also writes dr.env, which points the pipeline at
#    it and persists across terminal sessions -- no `set`/`export` needed.
python make_demo_data.py --out demo_data --patients 80

# 2. Build the preprocessed cache (once). Ben Graham normalisation is expensive;
#    doing it per-epoch in the DataLoader wastes enormous CPU.
#    --filename-col is REQUIRED: the default is `filename`, this CSV uses `id_code`.
python preprocessing.py --csv demo_data/labels.csv --images demo_data/images 
    --cache demo_data/cache_512 --filename-col id_code --size 512

# 3. Diagnose the environment and config. Run this whenever anything breaks.
python check_setup.py

# 4. Verify the harness BEFORE trusting a model. No GPU, no torch, no data needed.
python -m pytest tests -q

# 5. Train. Selects on specificity@sensitivity-90, not accuracy.
python train.py

# 6. Fit calibration on val, freeze the operating point, evaluate test ONCE.
python calibrate.py
python model_card.py

# 7. Run the grading platform. Works with or without a trained checkpoint.
python service.py            # http://127.0.0.1:8000
```

For real data, run `prepare_aptos.py` before step 2 — it repairs missing file
extensions, removes duplicate images, and rewrites `dr.env` to point at the
cleaned labels. See `RUNBOOK.md` Part 5.

## CSV schema

| Column | Required | Notes |
|---|---|---|
| `id_code` / `filename` | yes | image filename, set via `DataConfig.filename_col` |
| `diagnosis` | yes | ICDR grade, integer 0–4 |
| `patient_id` | **strongly** | without it you cannot split safely — see below |
| `age_group`, `sex`, `device` | optional | needed for subgroup fairness reporting |

### On `patient_id`

`DataConfig.patient_id_source` must be set deliberately:

- `"column"` — a real patient identifier (best)
- `"filename_prefix"` — EyePACS convention, `10_left.jpeg` → patient `10`
- `"unique_per_row"` — **only** if the dataset is genuinely one image per
  patient (APTOS 2019 is; EyePACS is not)

The loader refuses to guess. A wrong choice here inflates every metric you
report, silently. In testing, a naive row-level split of 400 patients put **142
of them on both sides of the split**.

## Datasets

| Dataset | Size | Role |
|---|---|---|
| APTOS 2019 | ~3.6k | prototyping; one image per patient |
| EyePACS / Kaggle DR | ~88k | scale; `patient_left/right` filenames |
| Messidor-2 | ~1.7k | **external test set** — different source, do not merge |

Set `DataConfig.external_csv_path` to Messidor-2. Until you do, every number is
internal and provisional.

## Module map

| File | Role |
|---|---|
| `config.py` | Single source of truth. `PreprocessConfig` is hashed into checkpoints. |
| `preprocessing.py` | FOV crop, Ben Graham normalisation, quality gate, cache builder |
| `splits.py` | **Torch-free.** Patient-grouped stratified splits, ordinal encoding. Unit-tested. |
| `dataset.py` | Torch layer only — augmentation and tensor loading |
| `model.py` | Ordinal CORAL head, backbone registry, skew-proof checkpointing |
| `metrics.py` | Screening harness: spec@sens, patient-level bootstrap, calibration, abstention, subgroups |
| `train.py` | Training loop; selects on the clinical metric |
| `calibrate.py` | Fits calibration + freezes threshold on val, evaluates test once |
| `gradcam.py` | Grad-CAM + randomised-weights sanity check + shortcut-learning stats |
| `prepare_aptos.py` | Cleans APTOS: fixes extensions, removes duplicate images, adds patient_id |
| `pick_test_images.py` | Curated known-ground-truth batch for testing |
| `check_setup.py` | Environment + config doctor; run when anything breaks |
| `check_setup.py` | **Run this when anything breaks.** Diagnoses env + config, prints fixes |
| `make_demo_data.py` | Synthetic fundus dataset for a known-good end-to-end run |
| `requirements*.txt` | `.txt` core, `-dev` tests/lint, `-cpu` slim install, `-gpu` Windows/specific CUDA |

## Dependency notes

**`opencv-python-headless`, not `opencv-python`.** Albumentations depends on the
headless build; installing both puts two `cv2` distributions in the same path
where they overwrite each other, and which one you import depends on install
order. The headless build also avoids the `ImportError: libGL.so.1` that the GUI
build triggers in slim Docker images. This project makes no cv2 GUI calls.

**`albumentations>=2.0` is required, not preferred.** `dataset.py` uses the 2.0
signatures `RandomResizedCrop(size=...)` and `Affine(fill=...)`; the 1.x
`height=`/`width=`/`cval=` forms were removed, so 1.x raises `TypeError`.

**Linux does not need the PyTorch index URL.** The default PyPI wheel already
bundles CUDA. Windows is the opposite — its default wheel is CPU-only.

## Design decisions worth knowing

**512px, not 224.** A 10px microaneurysm becomes 3.0 pixels at 512 and **1.3
pixels at 224** — indistinguishable from sensor noise. Measured, not assumed.

**Ordinal head, not 5-way softmax.** Cross-entropy prices a 4↔0 confusion the
same as 3↔4. Measured quadratic kappa: perfect 1.00, off-by-one 0.80, 4↔0 0.20.
Output index 1 is exactly P(grade ≥ 2) — the screening decision as one
calibratable scalar.

**Patient-level bootstrap.** Measured 15% wider CIs than image-level resampling.
Image-level is falsely precise when patients contribute two eyes.

**Calibration is selected, not assumed.** Temperature scaling is one parameter
and only fixes temperature-shaped miscalibration; in testing it made ECE *worse*
(0.102 → 0.116) while isotonic fixed it (0.102 → 0.031). The `Calibrator` fits
both and keeps whichever wins on held-out validation. Note isotonic creates ties
and can move AUROC slightly, so report AUROC on calibrated scores.

**Mild colour augmentation.** Hue is signal here — haemorrhages are dark red,
exudates yellow-white. Full rotation and both flips are safe (a fundus has no
canonical up); this would be wrong for chest X-ray, where a horizontal flip
manufactures dextrocardia.

**Ungradable ≠ negative.** The quality gate runs before the model. In a real
programme 5–20% of captures are ungradable, and calling a blurred image
"healthy" is the failure mode that actually harms patients.

## The platform

A grader review application, runnable before a model exists.

```bash
pip install -r requirements-serve.txt
python service.py            # http://127.0.0.1:8000
```

Without a checkpoint it starts in **demo mode**, using an image-statistics
heuristic that is labelled as such in every API response and across the top of
the interface. A demo that can be mistaken for a working diagnostic is worse
than no demo.

| Piece | What it does |
|---|---|
| `predictor.py` | Locks the decision path: quality gate → preprocess → model → calibrate → abstain. An ungradable image never reaches the model. |
| `audit.py` | Append-only SQLite log. Every prediction and grader verdict, with model version and preprocessing fingerprint. No pixels stored, only hashes. |
| `service.py` | FastAPI: predict, worklist, verdict capture, provenance. |
| `webapp/index.html` | Grader review interface. Worklist ordered by clinical priority, not upload time. |
| `model_card.py` | Generates `MODEL_CARD.md` from measured evaluation artifacts. |

**The decision axis** is the interface's central element: the operating point,
the abstention band, and the case's calibrated probability drawn on one line. It
shows at a glance not just what the model concluded but how close to the
boundary it was, and whether the system chose to defer.

**The verdict capture is the feedback loop.** Grader agreement rate is the
signal that detects post-deployment drift — the failure mode offline metrics
cannot see.

## Anatomy & quality gates

Before the model runs, two classical (non-learned, so trustworthy on unseen
input) checks guard the input:

- **Quality gate** (`assess_quality`) — rejects blurred, dark, or low-field-of-view
  captures. Computed at a fixed working resolution so the verdict does not depend
  on the camera's stored image size. An ungradable image becomes MANUAL_REVIEW,
  never a negative.
- **Anatomy gate** (`verify_is_fundus`) — colour, circular field of view, and
  optic-disc cues confirm the upload is a retina, not a wall, a document, or an
  X-ray. Known limitation, documented in the code: a smooth warm-toned circular
  image can still pass. A learned OOD detector is the long-term fix and needs a
  trained model first.

## Explainability

`gradcam.py` produces a Grad-CAM attention map over the referable-DR logit,
surfaced in the review UI as a toggle. It is framed as a **debugging aid, not an
explanation** — it marks where, never why. Two safeguards:

- `sanity_check` re-runs the map under cascading weight randomisation. If the map
  survives (correlation > 0.6), it is an edge detector, not an explanation — do
  not show it. Run once per trained model.
- `cam_statistics` reports `border_mass`; attention on the black surround means
  the model is keying on the camera, not the retina — the shortcut-learning
  signature.

Degenerate maps (e.g. an untrained backbone) raise an error rather than
rendering a flat overlay that looks like "nothing suspicious."

## Testing on real data before you trust it

`pick_test_images.py` assembles a batch spanning the severity range plus
ungradable cases, and prints what each SHOULD produce, so you check output
against ground truth instead of guessing. Works on demo and cleaned-APTOS data
without arguments.

## Running the platform safely

`service.py` binds to `127.0.0.1` (this machine only) by default. Binding to a
public address requires an explicit `--i-understand-this-is-not-a-medical-device`
flag, because exposing it changes the legal and data-protection picture — see
`RUNBOOK.md`.

## Not built yet

- **DICOM ingest / structured output** for real PACS integration
- **Learned OOD detector** to replace the heuristic anatomy gate
- **`border_mass` badge** surfaced in the UI as an automatic shortcut-learning warning
- **External validation wiring** beyond the single hook in `calibrate.py`