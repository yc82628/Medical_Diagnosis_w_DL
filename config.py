from __future__ import annotations
 
import hashlib
import json
from dataclasses import dataclass, asdict, field
 
 
# ---------------------------------------------------------------------------
# Clinical task definition  (see INTENDED_USE.md)
# ---------------------------------------------------------------------------
 
# International Clinical Diabetic Retinopathy (ICDR) severity scale.
ICDR_GRADES = {
    0: "No apparent retinopathy",
    1: "Mild non-proliferative DR",
    2: "Moderate non-proliferative DR",
    3: "Severe non-proliferative DR",
    4: "Proliferative DR",
}
 
# "Referable DR" is the decision this system supports: grade >= 2 warrants
# referral to an ophthalmologist. This threshold is the standard used in
# screening literature. It is NOT a tunable hyperparameter -- changing it
# changes the clinical claim.
REFERABLE_THRESHOLD = 2
 
NUM_GRADES = 5
 
 
@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic image -> tensor pipeline. Frozen and hashed into checkpoints."""
 
    image_size: int = 512
    # Microaneurysms -- the earliest referable lesion -- are ~5-10 px across at
    # native fundus resolution. At 224 px they are gone. 512 is the practical
    # floor; 768 is better if you have the VRAM.
 
    circle_crop: bool = True          # crop to the retinal field of view
    ben_graham: bool = True           # local colour normalisation (illumination fix)
    ben_graham_sigma_divisor: float = 30.0   # sigma = image_size / divisor
    mask_border_frac: float = 0.97    # zero out the outermost rim after normalisation
 
    # CLAHE, applied to the L channel of LAB so hue is preserved.
    # OFF by default because it overlaps with Ben Graham: both normalise local
    # contrast and illumination. Running both is not obviously better than
    # either alone. Treat it as an A/B test -- flip it, retrain, compare
    # spec@sens90 -- rather than stacking techniques because each sounds good.
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
 
    # Replicate the green channel across RGB. OFF by default: green carries the
    # best vessel contrast, but hue separates hard exudates (yellow-white) from
    # haemorrhages (dark red), and that distinction is what grade 2+ turns on.
    green_channel: bool = False
 
    # Normalisation. Ben Graham output is centred near 128/255, so ImageNet
    # statistics are wrong for it; we use 0.5/0.5 and let the network adapt.
    norm_mean: tuple = (0.5, 0.5, 0.5)
    norm_std: tuple = (0.5, 0.5, 0.5)
 
    def sigma(self) -> float:
        return self.image_size / self.ben_graham_sigma_divisor
 
    def fingerprint(self) -> str:
        """Stable hash. Compared at load time; mismatch is a hard error."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
 
 
@dataclass
class DataConfig:
    csv_path: str = "labels.csv"
    images_dir: str = "images"
    cache_dir: str = "cache_512"        # preprocessed images land here
 
    filename_col: str = "filename"
    grade_col: str = "diagnosis"        # ICDR grade 0-4
    patient_col: str | None = "patient_id"
    eye_col: str | None = None          # "left"/"right" if available
 
    # How to obtain a patient identifier. Options:
    #   "column"          -- use patient_col (preferred)
    #   "filename_prefix" -- EyePACS style: 10_left.jpeg -> patient "10"
    #   "unique_per_row"  -- ONLY valid if the dataset is genuinely one image
    #                        per patient (e.g. APTOS 2019). Must be set
    #                        deliberately; the loader will not guess.
    patient_id_source: str = "column"
 
    # Optional metadata columns used for subgroup fairness reporting.
    subgroup_cols: list = field(default_factory=lambda: ["age_group", "sex", "device"])
 
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
 
    # A held-out *different-source* dataset. Not a re-split of the above.
    # Leave as None during development; results without it are provisional.
    external_csv_path: str | None = None
    external_images_dir: str | None = None
 
 
@dataclass
class TrainConfig:
    backbone: str = "efficientnet_b3"
    dropout: float = 0.3
 
    batch_size: int = 16                # 512px is memory-hungry
    grad_accum_steps: int = 2           # effective batch 32
    num_epochs: int = 30
    learning_rate: float = 3e-4
    backbone_lr_mult: float = 0.1       # lower LR for pretrained weights
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    num_workers: int = 4
    amp: bool = True
 
    # Model selection. NOT accuracy, and NOT val loss.
    # See INTENDED_USE.md section 4: at ~73% grade-0 prevalence, accuracy
    # selects a model that is excellent at confirming healthy eyes.
    selection_metric: str = "spec_at_sens"
    target_sensitivity: float = 0.90
    patience: int = 7
 
    save_path: str = "checkpoints/best_model.pt"
 
 
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
 
 
# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------
# So you can point at a different dataset without editing this file (and
# without accidentally committing a local path). On Windows cmd:
#     set DR_CSV=demo_data\labels.csv
# On PowerShell:
#     $env:DR_CSV="demo_data\labels.csv"
# On macOS/Linux:
#     export DR_CSV=demo_data/labels.csv
 
_ENV_MAP = {
    "DR_CSV":            ("data", "csv_path", str),
    "DR_IMAGES":         ("data", "images_dir", str),
    "DR_CACHE":          ("data", "cache_dir", str),
    "DR_FILENAME_COL":   ("data", "filename_col", str),
    "DR_GRADE_COL":      ("data", "grade_col", str),
    "DR_PATIENT_COL":    ("data", "patient_col", str),
    "DR_PATIENT_SOURCE": ("data", "patient_id_source", str),
    "DR_EXTERNAL_CSV":   ("data", "external_csv_path", str),
    "DR_EXTERNAL_IMAGES": ("data", "external_images_dir", str),
    "DR_IMAGE_SIZE":     ("preprocess", "image_size", int),
    "DR_BACKBONE":       ("train", "backbone", str),
    "DR_BATCH_SIZE":     ("train", "batch_size", int),
    "DR_EPOCHS":         ("train", "num_epochs", int),
    "DR_NUM_WORKERS":    ("train", "num_workers", int),
    "DR_SAVE_PATH":      ("train", "save_path", str),
}
 
 
def _load_env_file(path: str = "dr.env") -> dict:
    """
    Read persistent settings from a `dr.env` file next to the project.
 
    This exists because `set` on Windows (and `export` on Unix) only lasts for
    the current shell. Re-running a batch file in every new terminal is a trap:
    forget once and you silently train on the wrong dataset with no error.
 
    Real environment variables take precedence over the file, so a one-off
    override still works without editing anything.
 
    Format -- KEY=VALUE, one per line, `#` starts a comment:
 
        DR_CSV=demo_data\\labels.csv
        DR_IMAGES=demo_data\\images
    """
    import os
 
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(p):
        return {}
 
    values = {}
    with open(p, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v:
                values[k] = v
    return values
 
 
def _apply_env(cfg: Config) -> Config:
    import os
    from dataclasses import replace
 
    from_file = _load_env_file()
    applied, sources = [], []
 
    for env, (section, attr, cast) in _ENV_MAP.items():
        # Shell environment wins over the file.
        raw = os.environ.get(env) or from_file.get(env)
        if raw is None or raw == "":
            continue
        origin = "env" if os.environ.get(env) else "dr.env"
        value = cast(raw)
        if section == "preprocess":
            # PreprocessConfig is frozen (it is hashed into checkpoints), so it
            # is rebuilt rather than mutated.
            cfg.preprocess = replace(cfg.preprocess, **{attr: value})
        else:
            setattr(getattr(cfg, section), attr, value)
        applied.append(f"{env}={value}")
        sources.append(origin)
 
    if applied:
        where = "dr.env" if all(s == "dr.env" for s in sources) else "dr.env + shell"
        print(f"  Config from {where}: {', '.join(applied)}")
    return cfg
 
 
CONFIG = _apply_env(Config())
 