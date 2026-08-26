from __future__ import annotations
 
import importlib
import platform
import sys
from pathlib import Path
 
OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
_problems, _warnings = [], []
 
 
def report(status, label, detail="", fix=""):
    print(f"  {status} {label}" + (f" -- {detail}" if detail else ""))
    if fix:
        for line in fix.strip().split("\n"):
            print(f"         > {line}")
    if label.startswith("skipped"):
        return                       # not a finding; keeps the summary readable
    if status == FAIL and label not in _problems:
        _problems.append(label)
    elif status == WARN and label not in _warnings:
        _warnings.append(label)
 
 
def section(title):
    print(f"\n{title}\n" + "-" * len(title))
 
 
def check_python():
    section("Python")
    v = sys.version_info
    if v < (3, 10):
        report(FAIL, f"Python {v.major}.{v.minor}", "3.10+ required",
               "Install a newer Python from python.org, then recreate your venv.")
    else:
        report(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    report(OK, f"Platform: {platform.system()} {platform.release()}")
 
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        report(OK, "Virtual environment active", sys.prefix)
    else:
        report(WARN, "Not in a virtual environment",
               "packages are installing globally",
               "python -m venv .venv\n"
               ".venv\\Scripts\\activate      (Windows)\n"
               "source .venv/bin/activate    (macOS/Linux)")
 
 
REQUIRED = [
    ("numpy", "1.26"), ("pandas", "2.0"), ("sklearn", "1.3"),
    ("cv2", "4.10"), ("albumentations", "2.0"), ("tqdm", "4.65"),
    ("torch", "2.4"), ("torchvision", "0.19"),
]
 
 
def _ver_tuple(s):
    out = []
    for part in str(s).split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)
 
 
def check_packages():
    section("Packages")
    for name, minver in REQUIRED:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            report(FAIL, name, "not installed",
                   "pip install -r requirements.txt")
            continue
        got = getattr(mod, "__version__", "?")
        if got != "?" and _ver_tuple(got) < _ver_tuple(minver):
            fix = "pip install -r requirements.txt --upgrade"
            if name == "albumentations":
                fix += ("\nalbumentations 1.x is NOT compatible: dataset.py uses the "
                        "2.0 signatures RandomResizedCrop(size=) and Affine(fill=).")
            report(FAIL, name, f"{got} installed, need >= {minver}", fix)
        else:
            report(OK, name, got)
 
 
def check_opencv_conflict():
    section("OpenCV install sanity")
    try:
        from importlib.metadata import distributions
        installed = {d.metadata["Name"].lower() for d in distributions()
                     if d.metadata and d.metadata.get("Name")}
    except Exception:
        report(WARN, "could not inspect installed distributions")
        return
 
    variants = [v for v in ("opencv-python", "opencv-python-headless",
                            "opencv-contrib-python", "opencv-contrib-python-headless")
                if v in installed]
    if len(variants) > 1:
        report(WARN, "Multiple OpenCV builds installed", ", ".join(variants),
               "They share the same site-packages/cv2/ directory and overwrite each\n"
               "other; which one you get depends on install order. Keep only headless:\n"
               "pip uninstall -y " + " ".join(v for v in variants
                                              if v != "opencv-python-headless") + "\n"
               "pip install opencv-python-headless")
    elif variants:
        report(OK, "Single OpenCV build", variants[0])
 
 
def check_torch():
    section("PyTorch / accelerator")
    try:
        import torch
    except ImportError:
        report(FAIL, "torch", "not installed",
               "Linux/macOS: pip install -r requirements.txt\n"
               "Windows GPU: pip install -r requirements-gpu.txt  (FIRST)")
        return
 
    report(OK, "torch", torch.__version__)
    if torch.cuda.is_available():
        report(OK, "CUDA GPU", torch.cuda.get_device_name(0))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        report(OK, "Apple Silicon MPS available")
    else:
        is_win = platform.system() == "Windows"
        report(WARN, "No GPU detected -- training will fall back to CPU",
               "a 512px epoch takes hours instead of minutes",
               ("On Windows the default PyPI torch wheel is CPU-ONLY. Install the\n"
                "CUDA build:  pip install -r requirements-gpu.txt\n"
                "Then re-check: python -c \"import torch; print(torch.cuda.is_available())\""
                if is_win else
                "If you have an NVIDIA GPU, check drivers with: nvidia-smi\n"
                "Otherwise use Google Colab (free GPU) for training."))
 
    if platform.system() == "Windows":
        from config import CONFIG
        if CONFIG.train.num_workers > 0:
            report(WARN, f"num_workers={CONFIG.train.num_workers} on Windows",
                   "DataLoader workers use spawn, not fork; this is slow and a "
                   "common source of confusing crashes",
                   "set DR_NUM_WORKERS=0")
 
 
def check_config_and_data():
    section("Data configuration")
    try:
        from config import CONFIG
    except Exception as e:
        report(FAIL, "config.py failed to import", f"{type(e).__name__}: {e}")
        return None
    dc = CONFIG.data
 
    csv = Path(dc.csv_path)
    if not csv.exists():
        report(FAIL, "labels CSV not found", str(csv.resolve()),
               "Generate a synthetic dataset to test the pipeline first:\n"
               "  python make_demo_data.py --out demo_data --patients 80\n"
               "then point the config at it (Windows cmd):\n"
               "  set DR_CSV=demo_data\\labels.csv\n"
               "  set DR_IMAGES=demo_data\\images\n"
               "  set DR_CACHE=demo_data\\cache_512\n"
               "  set DR_FILENAME_COL=id_code")
        return None
    report(OK, "labels CSV", str(csv))
 
    imgs = Path(dc.images_dir)
    if not imgs.exists():
        report(FAIL, "images directory not found", str(imgs.resolve()),
               "set DR_IMAGES=<path to your image folder>")
    else:
        n = sum(1 for p in imgs.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        if n == 0:
            report(FAIL, "images directory is empty", str(imgs))
        else:
            report(OK, "images directory", f"{n} image files")
 
    # columns
    try:
        import pandas as pd
        df = pd.read_csv(csv, nrows=200)
    except Exception as e:
        report(FAIL, "could not read CSV", f"{type(e).__name__}: {e}")
        return None
 
    cols = list(df.columns)
    for label, col, env in (("filename column", dc.filename_col, "DR_FILENAME_COL"),
                            ("grade column", dc.grade_col, "DR_GRADE_COL")):
        if col not in cols:
            report(FAIL, f"{label} '{col}' missing", f"CSV has: {cols}",
                   f"set {env}=<correct column name>")
        else:
            report(OK, f"{label} '{col}'")
 
    # patient identity -- the leakage-critical setting
    src = dc.patient_id_source
    if src == "column":
        if dc.patient_col in cols:
            n_pat = df[dc.patient_col].nunique()
            report(OK, f"patient column '{dc.patient_col}'", f"{n_pat} unique in sample")
        else:
            report(FAIL, f"patient_id_source='column' but '{dc.patient_col}' missing",
                   f"CSV has: {cols}",
                   "Pick ONE deliberately -- this decides whether your metrics are real:\n"
                   "  set DR_PATIENT_COL=<your column>\n"
                   "  set DR_PATIENT_SOURCE=filename_prefix   (EyePACS: 10_left.jpeg)\n"
                   "  set DR_PATIENT_SOURCE=unique_per_row    (APTOS: 1 image/patient)")
    elif src == "unique_per_row":
        report(WARN, "patient_id_source='unique_per_row'",
               "assuming one image per patient",
               "Correct for APTOS 2019. WRONG for EyePACS/Messidor, where it will\n"
               "inflate every metric you report, silently.")
    else:
        report(OK, f"patient_id_source='{src}'")
 
    # grade sanity
    if dc.grade_col in cols:
        try:
            g = df[dc.grade_col].dropna().astype(int)
            lo, hi = int(g.min()), int(g.max())
            if lo < 0 or hi > 4:
                report(FAIL, "grades outside 0-4", f"found {lo}-{hi}",
                       "This project expects the ICDR 0-4 scale.")
            else:
                report(OK, "grade range", f"{lo}-{hi}")
        except (ValueError, TypeError):
            report(FAIL, f"grade column '{dc.grade_col}' is not integer-valued",
                   "check you selected the right column")
    return CONFIG
 
 
def check_cache(cfg):
    section("Preprocessed cache")
    if cfg is None:
        report(WARN, "skipped (data config unresolved)")
        return
    cache = Path(cfg.data.cache_dir)
    if not cache.exists():
        report(WARN, "cache not built yet", str(cache),
               "python preprocessing.py --csv <csv> --images <dir> "
               "--cache <dir> --filename-col <col>")
        return
 
    n = sum(1 for p in cache.glob("*.png"))
    report(OK, "cache directory", f"{n} preprocessed images")
 
    fp = cache / ".fingerprint"
    want = cfg.preprocess.fingerprint()
    if not fp.exists():
        report(WARN, "no fingerprint file", "cache may predate this config")
    elif fp.read_text().strip() != want:
        report(FAIL, "cache fingerprint mismatch",
               f"cache={fp.read_text().strip()} config={want}",
               "Preprocessing settings changed since the cache was built.\n"
               "Delete the cache directory and rebuild it. Training on a stale\n"
               "cache is a train/serve skew bug that produces no error message.")
    else:
        report(OK, "fingerprint matches config", want)
 
 
def check_pipeline_smoke(cfg):
    section("Pipeline smoke test")
    if cfg is None:
        report(WARN, "skipped (data config unresolved)")
        return
    try:
        from splits import load_labels, split_by_patient
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df = load_labels(cfg.data, verbose=False)
            sp = split_by_patient(df, cfg.data, verbose=False)
        report(OK, "labels load + patient-grouped split",
               f"train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}")
 
        for name, s in sp.items():
            if len(s) == 0:
                report(FAIL, f"'{name}' split is empty",
                       "too few patients for the configured ratios",
                       "Use more data, or adjust train/val/test ratios in config.py")
            elif s["referable"].sum() == 0:
                report(WARN, f"'{name}' split has no referable cases",
                       "screening metrics will be undefined",
                       "You need more positive cases -- this dataset is too small.")
    except Exception as e:
        report(FAIL, "pipeline smoke test failed", f"{type(e).__name__}: {e}",
               "Full traceback:  python -c \"from config import CONFIG; "
               "from splits import load_labels; load_labels(CONFIG.data)\"")
 
 
def main():
    print("=" * 70)
    print("  DR screening -- setup doctor")
    print("=" * 70)
    check_python()
    check_packages()
    check_opencv_conflict()
    check_torch()
    cfg = check_config_and_data()
    check_cache(cfg)
    check_pipeline_smoke(cfg)
 
    section("Summary")
    if _problems:
        print(f"  {len(_problems)} blocking problem(s): {', '.join(_problems)}")
        print("  Fix the [FAIL] items above (each has a '>' suggestion), then re-run.")
    if _warnings:
        print(f"  {len(_warnings)} warning(s): {', '.join(_warnings)}")
    if not _problems:
        print("  No blocking problems. You are ready to run:")
        print("    python preprocessing.py ...   (if the cache is not built)")
        print("    python train.py")
    return 1 if _problems else 0
 
 
if __name__ == "__main__":
    sys.exit(main())