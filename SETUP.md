# Step-by-step setup (Windows Command Prompt)

Run these in order. **After every step, if something looks wrong, run
`python check_setup.py`** — it diagnoses the environment and prints a fix for
each problem rather than making you decode a traceback.

The strategy here matters: steps 3–7 run the whole pipeline on **synthetic
data** first. Get a clean end-to-end run on data you control, *then* swap in
APTOS. If it works on demo data and breaks on real data, the problem is your
CSV columns or paths — not the code. That halves the debugging surface.

---

## Step 0 — Open Command Prompt in the project folder

Press `Win + R`, type `cmd`, press Enter. Then:

```bat
cd C:\path\to\dr_screening
dir
```

You should see `train.py`, `config.py`, `requirements.txt`. If not, you are in
the wrong folder.

> Tip: in File Explorer, open the project folder, click the address bar, type
> `cmd`, and press Enter. That opens Command Prompt already in the right place.

## Step 1 — Create a virtual environment

Skipping this is the single biggest cause of "it worked yesterday" problems.

```bat
python -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`. If `python` is not recognised,
Python is not on your PATH — reinstall from python.org with **"Add Python to
PATH"** ticked.

## Step 2 — Install dependencies

**If you have an NVIDIA GPU**, install PyTorch first. On Windows the default
PyPI wheel is CPU-only, unlike Linux:

```bat
pip install -r requirements-gpu.txt
```

Then, everyone:

```bat
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Verify the GPU is actually visible. A CPU fallback fails **silently** — nothing
errors, training is just 50× slower:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

No GPU? That's fine for the demo run below. For real training use Google Colab.

## Step 3 — Run the doctor

```bat
python check_setup.py
```

Fix any `[FAIL]` lines (each prints a `>` suggestion), then re-run until only
the "labels CSV not found" failure remains — Step 4 fixes that.

## Step 4 — Generate synthetic data

```bat
python make_demo_data.py --out demo_data --patients 80
```

This writes 160 fake fundus images and a `labels.csv`. They are procedural
drawings, **not** medical images — a model will learn them easily and that means
nothing clinically. This is a plumbing test.

## Step 5 — Point the config at the demo data

```bat
set DR_CSV=demo_data\labels.csv
set DR_IMAGES=demo_data\images
set DR_CACHE=demo_data\cache_512
set DR_FILENAME_COL=id_code
set DR_GRADE_COL=diagnosis
set DR_PATIENT_COL=patient_id
set DR_PATIENT_SOURCE=column
set DR_NUM_WORKERS=0
set DR_EPOCHS=3
```

Two Windows-specific notes:

- **`set` only lasts for this Command Prompt window.** Close it and you must
  re-run these. Put them in a `.bat` file to save retyping.
- **In PowerShell the syntax is different**: `$env:DR_CSV="demo_data\labels.csv"`.
  If you see `set` "not recognized", you are in PowerShell, not cmd.
- `DR_NUM_WORKERS=0` matters on Windows. DataLoader workers use *spawn* rather
  than *fork*, which is slow and a frequent source of confusing crashes.

Confirm they took effect:

```bat
python check_setup.py
```

You want `[ OK ]` on the labels CSV, images directory, and all three columns.

## Step 6 — Build the preprocessed cache

```bat
python preprocessing.py --csv demo_data\labels.csv --images demo_data\images --cache demo_data\cache_512 --filename-col id_code --size 512
```

Expect a report like `ungradable / failed: 12 / 160`. That is the quality gate
working — those images are deliberately blurred or dark. **They are excluded
from training and become "manual review" at serving time, never "healthy".**

## Step 7 — Run the tests

```bat
python -m pytest tests\ -q
```

Expect `20 passed`. These need no GPU and no torch — they cover the
leakage-critical splitting and the evaluation harness, which is where results
actually get invalidated.

## Step 8 — Train

```bat
python train.py
```

You should see per-epoch lines with `spec@sens90`, `QWK` and `missed`. On demo
data the numbers will look great and mean nothing.

If you get `CUDA out of memory`, lower the batch size:

```bat
set DR_BATCH_SIZE=8
```

## Step 9 — Calibrate and evaluate

```bat
python calibrate.py
```

This fits calibration and freezes the operating point on **validation**, then
touches test exactly once. Results land in `artifacts\evaluation.json`.

---

## Switching to real data (APTOS 2019)

Download from Kaggle, then:

```bat
set DR_CSV=aptos\train.csv
set DR_IMAGES=aptos\train_images
set DR_CACHE=aptos\cache_512
set DR_FILENAME_COL=id_code
set DR_GRADE_COL=diagnosis
set DR_PATIENT_SOURCE=unique_per_row
set DR_EPOCHS=30
python check_setup.py
```

**`unique_per_row` is correct for APTOS specifically**, because it genuinely has
one image per patient. It is **wrong** for EyePACS, where each patient has two
eyes — there, use `set DR_PATIENT_SOURCE=filename_prefix`. Getting this wrong
inflates every metric you report, with no error message.

APTOS filenames in the CSV have no extension, while the files are `.png`. The
cache lookup uses `Path(filename).stem`, so this resolves — but if you hit
"Cached image unreadable", that is the first thing to check.

---

## Common errors

| What you see | What it means | Fix |
|---|---|---|
| `'python' is not recognized` | Python not on PATH | Reinstall from python.org, tick "Add Python to PATH" |
| `'set' is not recognized` | You are in PowerShell | Use `$env:DR_CSV="..."`, or type `cmd` first |
| `ModuleNotFoundError: No module named 'config'` | Running from the wrong folder | `cd` into the project folder |
| `ModuleNotFoundError: albumentations` | venv not activated, or install skipped | `.venv\Scripts\activate` then `pip install -r requirements.txt` |
| `TypeError: __init__() got an unexpected keyword argument 'size'` | albumentations 1.x | `pip install "albumentations>=2.0" --upgrade` |
| `ImportError: libGL.so.1` | GUI OpenCV build | `pip uninstall opencv-python` then `pip install opencv-python-headless` |
| `FileNotFoundError: CSV not found` | Path wrong or `set` not applied | `python check_setup.py` |
| `ValueError: Column 'diagnosis' not in CSV` | Column names differ | `set DR_GRADE_COL=<your column>` |
| `ValueError: patient_id_source='column' but ...` | No patient column | Choose deliberately — see Step 5 and the note above |
| `RuntimeError: Cache fingerprint mismatch` | Preprocessing settings changed | Delete the cache folder and rebuild (Step 6) |
| `RuntimeError: PATIENT LEAKAGE` | The guard caught a real bug | Do not bypass this. Your split is invalid. |
| `CUDA out of memory` | Batch too large for the card | `set DR_BATCH_SIZE=8` |
| `torch.cuda.is_available()` → `False` | CPU-only wheel on Windows | `pip install -r requirements-gpu.txt` |
| Training crashes with worker/pickle errors | Windows spawn semantics | `set DR_NUM_WORKERS=0` |

## If you are still stuck

Run this and send the full output — it captures almost everything needed to
diagnose a problem:

```bat
python check_setup.py > diagnostic.txt 2>&1
```

Along with the **complete** traceback of the failing command. The last line of a
traceback names the error, but the lines above it say where it came from, which
is usually the part that identifies the actual cause.
