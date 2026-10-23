# Diabetic Retinopathy Screening — foundation

A screening tool that grades retina photos for diabetic retinopathy, built to fail safely: bad photos
and borderline cases go to a human instead of getting a confident wrong answer.

> **▶ Try the demo:** [the safety checks and decision rule, interactive](https://yc82628.github.io/medical-imaging.html) on my portfolio.
>
> **My role:** solo project. I built everything: the image quality and anatomy checks, the model and
> its training, calibration, the grading web app and the audit log.

**Research software. Not a medical device.** New here, or hitting errors? `SETUP.md`
has the step-by-step walkthrough plus a "Common errors" section with causes and fixes. Read `INTENDED_USE.md` first — it
defines the scope that every design decision here follows from.

## Run order

Full step-by-step with troubleshooting: **`SETUP.md`**.

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
python -m pytest -q          # expect 124 passed (fewer without torch: some skip)

# 5. Train. Selects on specificity@sensitivity-90, not accuracy.
python train.py

# 6. Fit calibration on val, freeze the operating point, evaluate test ONCE.
python calibrate.py
python model_card.py

# 7. Run the grading platform. Create an account first; sign in at
#    http://127.0.0.1:8000. A model that fails the acceptance gate is not
#    served -- the platform runs in labelled DEMO mode instead.
python auth.py create-user --username you --role admin
python service.py
```

For real data, run `prepare_aptos.py` before step 2 — it repairs missing file
extensions, removes duplicate images, and rewrites `dr.env` to point at the
cleaned labels. See "Switching to real data (APTOS 2019)" in `SETUP.md`.

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

The current model is trained on **DeepDRiD** and validated externally on
**IDRiD**. Both are open, need no account, and allow commercial use with
attribution — unlike the Kaggle sets, whose rules restrict them.

| Dataset | Size | Role | Licence | Prepare with |
|---|---|---|---|---|
| [DeepDRiD](https://github.com/deepdrdoc/DeepDRiD) | 2,000 images, 500 patients | **training** / validation / internal test; human image-quality labels | CC BY-SA 4.0 | `prepare_deepdrid.py` |
| [IDRiD](https://zenodo.org/records/17219542) | 516 images | **external test** — India, different cameras | CC BY 4.0 | `prepare_idrid.py` |
| APTOS 2019 | ~3.6k | prototyping | Kaggle rules: **non-commercial only** | `prepare_aptos.py` |
| EyePACS / Kaggle DR | ~88k | scale | Kaggle rules | — |
| Messidor-2 | ~1.7k | external test | licence agreement | — |

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/deepdrdoc/DeepDRiD.git deepdrid/repo
git -C deepdrid/repo sparse-checkout set regular_fundus_images
# IDRiD: download "B. Disease Grading.zip" from the Zenodo record into idrid/ and extract
python prepare_idrid.py
python prepare_deepdrid.py         # writes dr.env: DeepDRiD for training, IDRiD external
python preprocessing.py --csv deepdrid/labels.csv --images deepdrid/repo/regular_fundus_images --cache deepdrid/cache_512 --filename-col filename
python preprocessing.py --csv idrid/labels.csv --images "idrid/B. Disease Grading" --cache idrid/cache_512 --filename-col filename
# The current model: strong augmentation, 8-view TTA, site calibration on half of IDRiD
DR_AUGMENT=strong DR_SEED=1 python train.py
python calibrate.py --site-calibration --tta d8 && python model_card.py
```

Cite both datasets in anything you publish: Liu et al., *Patterns* 3(6), 2022
(DeepDRiD); Porwal et al., IEEE DataPort, 2018 (IDRiD). The dataset folders
are gitignored.

**Try it with real photos:** `test_scans/` holds 12 IDRiD photos from the
held-out half (CC BY 4.0, Porwal et al.; renamed, otherwise unmodified) —
healthy, referable, too-close-to-call and one the model misses — with an
answer key giving each specialist grade and the expected result.

### Current results

From `MODEL_CARD.md` (EfficientNet-B3, 512 px, strong augmentation, 8-view
test-time augmentation, trained on an RTX 5070 laptop GPU):

| | Internal test (DeepDRiD) | External test (IDRiD, held-out half) |
|---|---|---|
| AUROC | 0.976 (0.949–0.994) | 0.938 (0.908–0.963) |
| Sensitivity / specificity | 0.902 / 0.953 | 0.882 / 0.854 |
| Calibration error | 0.038 | 0.052 |

The model **passes the acceptance gate, with site calibration** and with no
floor lowered. The first model did not: at IDRiD it missed one referable eye in
four (sensitivity 0.752, ECE 0.148). Two changes closed the gap:

1. **Augmentation aimed at the camera shift.** IDRiD's camera crops the field of
   view flat at top and bottom and its images are sharper; DeepDRiD's are full
   circles. The `strong` profile in `dataset.py` simulates that (plus colour
   balance, blur/sharpen and compression), keeping the background black. It
   raised IDRiD AUROC from 0.888 to 0.951 on the development half.
2. **Site calibration** (`calibrate.py --site-calibration`). IDRiD is split once,
   by image and stratified on grade, into a calibration half and a held-out
   evaluation half. The calibrator, operating point and abstention band are
   refitted on the calibration half; every external number above comes from the
   other half, read once after the configuration was frozen. Without this step
   the same model scores sensitivity 0.839 and ECE 0.116 on that half, still
   failing. **The model's ranking transfers to a new site; its probabilities and
   threshold do not.** Each deploying site needs its own labelled calibration
   sample, and a held-out one to confirm.

Caveats, all on the card: one training run per configuration (no replicate
seed); the 161 referable images in the held-out half give a sensitivity
interval of 0.823–0.923, which does not exclude falling below the 0.85 floor;
IDRiD has no patient ids, so the split is by image.

Model selection never touched the internal test or the held-out half:
`domain_experiments.py` scores candidates on DeepDRiD validation and the IDRiD
calibration half only, and the card lists every result. Planned but not
completed when the experiment budget was closed: EfficientNet-V2-S (stopped at
epoch 3 of 30), ConvNeXt-Tiny, a second seed of the chosen configuration, a
mild-augmentation control with the same seed, and 768 px input.

## Module map

| File | Role |
|---|---|
| `config.py` | Single source of truth. `PreprocessConfig` is hashed into checkpoints; `AcceptanceConfig` sets the bar a model must clear to be served. |
| `preprocessing.py` | FOV crop, Ben Graham normalisation, quality gate, cache builder |
| `splits.py` | **Torch-free.** Patient-grouped stratified splits, ordinal encoding. Unit-tested. |
| `dataset.py` | Torch layer only — augmentation and tensor loading |
| `model.py` | Ordinal CORAL head, backbone registry, skew-proof checkpointing |
| `metrics.py` | Screening harness: spec@sens, patient-level bootstrap, calibration, abstention, subgroups |
| `train.py` | Training loop; selects on the clinical metric |
| `calibrate.py` | Fits calibration + freezes threshold on val, evaluates test once; `--site-calibration` refits on half of the external site |
| `domain_experiments.py` | Scores candidate models for the external gap without touching held-out data |
| `gradcam.py` | Grad-CAM + randomised-weights sanity check + shortcut-learning stats |
| `prepare_aptos.py` | Cleans APTOS: fixes extensions, removes duplicate images, adds patient_id |
| `pick_test_images.py` | Curated known-ground-truth batch for testing |
| `check_setup.py` | **Run this when anything breaks.** Diagnoses env + config, prints fixes |
| `validate_quality_gate.py` | Measures the quality/anatomy gate against human gradability labels (EyeQ) |
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

**Augmentation that moves everything except hue.** Hue is signal here —
haemorrhages are dark red, exudates yellow-white — so the `strong` profile
shifts hue by at most ±4°, while brightness, gamma, saturation, sharpness,
compression and the field-of-view shape vary widely: those are what change
between cameras. Full rotation and both flips are safe (a fundus has no
canonical up); this would be wrong for chest X-ray, where a horizontal flip
manufactures dextrocardia. The same symmetry justifies 8-view test-time
augmentation, which the serving predictor applies exactly as evaluated.

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
| `predictor.py` | Locks the decision path: quality gate → preprocess → model → calibrate → abstain. An ungradable image never reaches the model. `build_predictor` refuses a checkpoint that fails the acceptance gate. |
| `audit.py` | Append-only (triggers) and hash-chained SQLite log: predictions, verdicts, sign-ins, case views, exports. `verify_chain` detects direct edits. Also the drift-monitoring rollup. No pixels here, only hashes. |
| `storage.py` | Encrypted image store (Fernet). Originals, previews and Grad-CAM overlays are never written in plaintext. Retention purge; one-off encryption of legacy files. |
| `auth.py` | Accounts with roles (photographer / grader / admin), PBKDF2 passwords, sessions, lockout, optional hospital SSO header. |
| `encounters.py` | Patient-level visit decision from both eyes; grader verdicts override the model. |
| `fhir.py` | HL7 FHIR R4 `DiagnosticReport` per visit, with SNOMED laterality. |
| `dicom_io.py` | DICOM ingest: pixels plus PatientID, laterality, study and camera from tags. Never reads the patient name. |
| `service.py` | FastAPI: sign-in, predict, worklist, verdicts, visits, FHIR, monitoring, audit verification. |
| `webapp/index.html` | Grader review interface. Worklist ordered by clinical priority, not upload time. |
| `Dockerfile`, `docker-compose.yml` | On-premise packaging behind the hospital TLS proxy. No data or keys in image layers. |
| `model_card.py` | Generates `MODEL_CARD.md` from measured evaluation artifacts. |

**The decision axis** is the interface's central element: the operating point,
the abstention band, and the case's calibrated probability drawn on one line. It
shows at a glance not just what the model concluded but how close to the
boundary it was, and whether the system chose to defer.

**The verdict capture is the feedback loop.** Grader agreement rate is the
signal that detects post-deployment drift — the failure mode offline metrics
cannot see. The Monitoring view tracks it per week and per camera, alongside
ungradable and referral rates, and raises alerts when they move.

**Patients, not images.** Every upload carries a patient ID and eye (typed in,
or read from DICOM tags). Images from one visit are combined: refer if either
eye is referable; a visit with only one eye screened is INCOMPLETE, never
ROUTINE. The visit prints as a one-page report and exports as a FHIR
`DiagnosticReport`, `preliminary` until a grader has reviewed every eye.

**The model must earn its place.** `AcceptanceConfig` sets floors on AUROC,
sensitivity, specificity, calibration error and the number of referable test
cases, and requires an abstention band and an `evaluation.json` tied to the
exact checkpoint by hash. A model that fails is not loaded; the banner lists
why. `DR_ALLOW_UNVALIDATED_MODEL=1` overrides this for research only, and every
prediction it makes is stored with `validated=0`.

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

**To serve it to staff on a hospital network** (one GPU laptop or PC, HTTPS,
staff Wi-Fi only), follow **`DEPLOY.md`**: `deploy/setup.py` does the first-run
setup, `deploy/start_server.bat` starts it, `deploy/backup.py` backs it up.

`service.py` binds to `127.0.0.1` (this machine only) by default. Binding to
any other address requires both the
`--i-understand-this-is-not-a-medical-device` flag — exposing it changes the
legal and data-protection picture, see `INTENDED_USE.md` section 7 — and TLS
(`--ssl-certfile`/`--ssl-keyfile`, or `--behind-tls-proxy` when the hospital
reverse proxy terminates TLS, as in `docker-compose.yml`).

What protects patient data:

- Sign-in required for everything except the page itself; roles limit who can
  record verdicts and export reports. Verdicts are attributed to the account.
- Images encrypted at rest under `DR_DATA_KEY` (or `secrets/data.key`, which
  `.gitignore` excludes). Back the key up; schedule
  `python storage.py purge --days N` for your retention period.
- Every view, export and sign-in is in the hash-chained audit log.
- Strict Content-Security-Policy; no third-party scripts or fonts.

SQLite (WAL, one locked connection) is sized for one site running one
instance. Running several instances behind a load balancer would need a
server database such as PostgreSQL; that port is not done.

## Not built yet — the path to clinical use

Done in code: acceptance gate, accounts and roles, encryption at rest,
tamper-evident audit, patient/visit linkage, FHIR output, DICOM input, drift
monitoring, Grad-CAM border warning, quality-gate validator, Docker packaging.

Still needed, none of which code alone can supply:

- **A model that generalises.** Trained on DeepDRiD it passes every internal
  criterion but fails external sensitivity (0.75) and calibration on IDRiD.
  Next: train on more sources together (DeepDRiD + IDRiD's own split, or
  licensed hospital data), stronger colour/illumination augmentation, and
  a site-calibration step on held-out local data before any pilot.
- **External validation on the partner hospital's own cameras** before any
  pilot. Then set `AcceptanceConfig.require_external_validation = True`.
- **A learned image-quality model.** On DeepDRiD's human quality labels the
  classical gate is at chance (sharpness AUROC 0.51), so its blur threshold
  was lowered from 40 to 12 to remove gross failures only. Train a quality
  classifier on those 2,000 labels to replace it.
- **Maculopathy.** Many programmes refer for exudates near the fovea; this
  model does not assess it (see `INTENDED_USE.md`).
- **Learned OOD detector** to replace the heuristic anatomy gate (needs a
  trained model first).
- **PostgreSQL** for multi-instance deployments.
- **Regulatory.** Clinical use makes this a medical device (EU MDR, likely
  Class IIa; plus GDPR DPIA and EU AI Act). That needs a quality management
  system (ISO 13485), lifecycle and risk files (IEC 62304, ISO 14971),
  clinical evaluation and a notified body. The realistic first step is a
  retrospective research study with a hospital partner under ethics approval,
  with no output used for patient care.