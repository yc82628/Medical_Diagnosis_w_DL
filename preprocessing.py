from __future__ import annotations
 
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
 
from config import PreprocessConfig
 
 
# ---------------------------------------------------------------------------
# Field-of-view detection
# ---------------------------------------------------------------------------
 
def _fov_mask(img_bgr: np.ndarray, tol: int = 7) -> np.ndarray:
    """Boolean mask of the illuminated retinal disc."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Blur first so that dark retinal regions (or a dark choroid) don't punch
    # holes in the mask.
    gray = cv2.GaussianBlur(gray, (0, 0), 5)
    return gray > tol
 
 
def circle_crop(img_bgr: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop to the bounding box of the retinal disc."""
    mask = _fov_mask(img_bgr, tol)
    if mask.sum() == 0:                      # entirely black image
        return img_bgr
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    cropped = img_bgr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    if cropped.size == 0:
        return img_bgr
    return cropped
 
 
def _pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad the short side so resizing does not distort lesion geometry."""
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
 
 
# ---------------------------------------------------------------------------
# Colour normalisation
# ---------------------------------------------------------------------------
 
def ben_graham(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    """img*4 - blur(img)*4 + 128. Highlights local structure, kills illumination gradient."""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    out = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    return out
 
 
def apply_circular_mask(img_bgr: np.ndarray, frac: float) -> np.ndarray:
    """Zero the rim. Ben Graham produces a bright halo at the FOV boundary."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = int(frac * min(h, w) / 2)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, thickness=-1)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
 
 
# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------
 
def apply_clahe(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """
    CLAHE on the L channel of LAB, so contrast is enhanced without shifting hue.
 
    Applying CLAHE per-BGR-channel independently is a common mistake: it changes
    the colour balance, and colour is diagnostic here (hard exudates are
    yellow-white, haemorrhages dark red).
 
    Note this overlaps with Ben Graham -- both normalise local contrast and
    illumination. Running both is not obviously better than either alone, so
    treat it as an A/B choice to measure, not a stack to pile up. See
    PreprocessConfig for the flags.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
 
 
def green_channel(img_bgr: np.ndarray) -> np.ndarray:
    """
    Replicate the green channel across all three channels.
 
    Green does carry the highest vessel/lesion contrast, and this was the
    standard choice in pre-deep-learning fundus work.
 
    But it discards information a CNN can use. Hard exudates (yellow-white),
    haemorrhages (dark red) and cotton-wool spots differ in HUE, not only in
    green-channel intensity, and grade 2+ depends on telling them apart. It also
    wastes two thirds of an ImageNet-pretrained backbone's first-layer filters,
    which are tuned for colour input.
 
    Kept configurable rather than default, because this is an empirical question
    for your dataset: run it both ways and compare spec@sens90.
    """
    g = img_bgr[:, :, 1]
    return cv2.merge([g, g, g])
 
 
def preprocess_image(img_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Deterministic. The same function runs at train, eval, and serve time."""
    if cfg.circle_crop:
        img_bgr = circle_crop(img_bgr)
    img_bgr = _pad_to_square(img_bgr)          # pad, never squash: squashing
                                               # distorts lesion geometry
    img_bgr = cv2.resize(img_bgr, (cfg.image_size, cfg.image_size),
                         interpolation=cv2.INTER_AREA)
    if cfg.clahe:
        img_bgr = apply_clahe(img_bgr, cfg.clahe_clip, cfg.clahe_grid)
    if cfg.ben_graham:
        img_bgr = ben_graham(img_bgr, cfg.sigma())
    if cfg.green_channel:
        img_bgr = green_channel(img_bgr)
    if cfg.ben_graham or cfg.clahe:
        img_bgr = apply_circular_mask(img_bgr, cfg.mask_border_frac)
    return img_bgr
 
 
# ---------------------------------------------------------------------------
# Anatomy verification -- is this actually a retina?
# ---------------------------------------------------------------------------
 
def detect_optic_disc(img_bgr: np.ndarray) -> dict:
    """
    Locate the optic disc: the brightest roughly-circular structure in a fundus.
 
    Deliberately classical rather than learned. This gate has to be trustworthy
    on inputs the model has never seen -- that is its whole purpose -- so a
    second neural network with its own unknown failure modes would defeat it.
 
    Returns location, a normalised radius, and a plausibility score.
    """
    work = _to_work_size(img_bgr, 512)
    mask = _fov_mask(work)
    if mask.sum() < 0.05 * mask.size:
        return {"found": False, "score": 0.0, "reason": "no retinal field of view"}
 
    # The disc is bright in red AND green; specular flash artefacts usually are
    # not, so requiring both suppresses the commonest false positive.
    b, g, r = cv2.split(work.astype(np.float32))
    bright = cv2.GaussianBlur(np.minimum(r, g * 1.6), (0, 0), 5)
    bright[~mask] = 0
 
    inside = bright[mask]
    thresh = np.percentile(inside, 99.0)
    if thresh <= 1:
        return {"found": False, "score": 0.0, "reason": "no bright structure"}
 
    binary = ((bright >= thresh) & mask).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    if n < 2:
        return {"found": False, "score": 0.0, "reason": "no bright structure"}
 
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[i, cv2.CC_STAT_AREA])
    w, h = float(stats[i, cv2.CC_STAT_WIDTH]), float(stats[i, cv2.CC_STAT_HEIGHT])
    cx, cy = cents[i]
 
    retina_area = float(mask.sum())
    area_frac = area / retina_area
    aspect = min(w, h) / max(w, h, 1e-6)
    fill = area / max(w * h, 1e-6)            # circular blob fills ~0.79 of its box
 
    # A real optic disc is roughly 1-6% of retinal area, near-circular, and
    # solid. Score each property rather than hard-thresholding, so one marginal
    # measurement does not by itself reject a valid image.
    s_area = 1.0 if 0.005 <= area_frac <= 0.08 else 0.0
    s_aspect = max(0.0, min(1.0, (aspect - 0.4) / 0.35))
    s_fill = max(0.0, min(1.0, (fill - 0.4) / 0.35))
    score = float(0.4 * s_area + 0.3 * s_aspect + 0.3 * s_fill)
 
    return {
        "found": score >= 0.5,
        "score": score,
        "center": (float(cx / work.shape[1]), float(cy / work.shape[0])),
        "area_fraction": area_frac,
        "aspect": float(aspect),
        "fill": float(fill),
        "reason": "" if score >= 0.5 else "no plausible optic disc",
    }
 
 
def vessel_energy(img_bgr: np.ndarray) -> float:
    """
    Fraction of the retina occupied by dark, elongated, branching structures.
 
    Every retina has a vessel tree. A smooth warm-toned disc -- a sepia scan, a
    lens-flare photo, a rendered circle -- does not, and colour and shape alone
    cannot tell those apart from a fundus. This is the cue that does.
 
    Black-hat morphology with elongated kernels at four orientations responds to
    dark linear structures and ignores smooth gradients.
    """
    work = _to_work_size(img_bgr, 512)
    mask = _fov_mask(work)
    if mask.sum() < 0.05 * mask.size:
        return 0.0
 
    green = cv2.split(work)[1]
    green = cv2.GaussianBlur(green, (0, 0), 1.2)
 
    response = np.zeros_like(green, dtype=np.uint8)
    for angle in (0, 45, 90, 135):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        if angle:
            M = cv2.getRotationMatrix2D((7, 0), angle, 1.0)
            k = cv2.warpAffine(k.astype(np.float32), M, (15, 15)).astype(np.uint8)
            k = (k > 0).astype(np.uint8)
            if k.sum() == 0:
                continue
        response = np.maximum(response, cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, k))
 
    inside = response[mask]
    if inside.size == 0:
        return 0.0
    # Vessels are a clear minority of pixels but a strong response; compare
    # against a high percentile so exposure differences do not dominate.
    thresh = max(6.0, float(np.percentile(inside, 97)) * 0.45)
    return float((inside > thresh).mean())
 
 
def verify_is_fundus(img_bgr: np.ndarray) -> dict:
    """
    Out-of-distribution gate: does this look like a retinal photograph at all?
 
    Without this, the most dangerous input is not a difficult retina -- it is
    something that is not a retina. A classifier asked to grade a photo of a
    wall, a chest X-ray, or a scanned form will return a confident number, and
    every offline metric you have will be silent about it.
 
    Three independent cues, each weak alone:
      1. Red dominance   -- retinal tissue is red/orange; R > G > B by a margin
      2. Circular FOV    -- the illuminated disc sits inside dark corners
      3. Optic disc      -- a bright, round, solid structure inside the retina
 
    KNOWN LIMITATION: a smooth, warm-toned, circular image (a sepia scan, some
    lens-flare photographs) can pass. All three cues genuinely fire on such an
    input, and the obvious fourth cue -- vessel structure -- was implemented and
    measured and did not separate the classes reliably. Treat this gate as
    catching gross mistakes (a document, a wall, a greyscale X-ray), not as a
    guarantee. A learned OOD detector -- distance from the training feature
    distribution -- is the right long-term answer, and it needs a trained model
    to exist first.
    """
    work = _to_work_size(img_bgr, 512)
    mask = _fov_mask(work)
    checks, reasons = {}, []
 
    if mask.sum() < 0.05 * mask.size:
        return {"is_fundus": False, "score": 0.0,
                "reasons": ["image is almost entirely dark"], "checks": {}}
 
    b, g, r = [c.astype(np.float32)[mask].mean() for c in cv2.split(work)]
    total = r + g + b + 1e-6
    redness = (r - g) / total
    checks["red_dominance"] = float(redness)
    # Fundus tissue is strongly red-dominant. Greyscale (X-ray, document scan)
    # gives ~0; a blue or green scene gives a negative value.
    s_red = max(0.0, min(1.0, (redness - 0.02) / 0.10))
    if s_red < 0.4:
        reasons.append("colour distribution is not retinal tissue "
                       f"(red dominance {redness:.3f})")
 
    # Corners dark, centre illuminated -- the signature of a fundus camera.
    h, w = mask.shape
    k = max(8, min(h, w) // 8)
    corners = np.mean([mask[:k, :k].mean(), mask[:k, -k:].mean(),
                       mask[-k:, :k].mean(), mask[-k:, -k:].mean()])
    centre = mask[h//2 - k:h//2 + k, w//2 - k:w//2 + k].mean()
    checks["corner_darkness"] = float(1.0 - corners)
    checks["centre_illumination"] = float(centre)
    s_fov = max(0.0, min(1.0, (centre - corners - 0.3) / 0.5))
    if s_fov < 0.4:
        reasons.append("no circular retinal field of view "
                       f"(centre {centre:.2f} vs corners {corners:.2f})")
 
    disc = detect_optic_disc(img_bgr)
    checks["optic_disc_score"] = disc["score"]
    if not disc["found"]:
        reasons.append("no plausible optic disc located")
 
    # Vessel energy is REPORTED but deliberately NOT scored. A morphological
    # vessel detector was tried and measured, and it did not separate real
    # retinas from smooth warm-toned circular images: blurred edges produce a
    # comparable black-hat response, and orientation-selectivity was worse
    # still (noise scored highest of anything tested). Rather than tune the
    # threshold until the test fixtures passed -- which would fit the heuristic
    # to synthetic negatives instead of to reality -- it is left out of the
    # decision and kept only as a diagnostic value.
    checks["vessel_energy"] = vessel_energy(img_bgr)
 
    score = float(0.35 * s_red + 0.35 * s_fov + 0.30 * disc["score"])
 
    # Red dominance is NECESSARY, not merely weighted. Retinal tissue is red;
    # nothing that is not red is a fundus photograph. Weighted averaging alone
    # lets a greyscale chest X-ray through on structure -- a bright oval inside
    # dark corners scores well on field-of-view and optic disc, and in testing
    # it passed at 0.51. Greyscale medical imagery is exactly the confusable
    # class here, so colour gets a veto.
    if s_red < 0.30:
        return {
            "is_fundus": False,
            "score": min(score, 0.49),
            "reasons": reasons or ["colour distribution is not retinal tissue"],
            "checks": checks,
            "optic_disc": disc,
        }
 
    return {
        "is_fundus": score >= 0.5,
        "score": score,
        "reasons": reasons,
        "checks": checks,
        "optic_disc": disc,
    }
 
 
# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
# A gradable-image check. In a real screening programme roughly 5-20% of
# captures are ungradable, and an ungradable image is a *referral*, not a
# negative. A model that silently classifies a blurred image as grade 0 is
# the exact failure mode that harms patients.
 
QUALITY_THRESHOLDS = {
    "min_fov_fraction": 0.25,   # retina should fill a reasonable share of frame
    "min_blur_var": 40.0,       # variance of Laplacian, at QUALITY_WORK_SIZE
    "min_mean_intensity": 20.0,
    "max_mean_intensity": 235.0,
}
 
# Quality metrics are computed at a FIXED working resolution, never at native
# resolution. Variance of the Laplacian scales with image size: the same eye
# stored at 512px vs 2048px yields blur variance of ~110 vs ~27, which flips the
# verdict against a fixed threshold. Left unnormalised, the gate silently
# becomes a camera-model detector -- and a quality gate whose strictness depends
# on the device is a fairness bug, not just a nuisance.
#
# Downsampling first is also ~10x faster, which is what makes the gate usable on
# a live preview stream.
QUALITY_WORK_SIZE = 512
 
 
def _to_work_size(img_bgr: np.ndarray, size: int = QUALITY_WORK_SIZE) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= size:
        return img_bgr
    s = size / longest
    return cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)
 
 
def assess_quality(img_bgr: np.ndarray, thresholds: dict | None = None) -> dict:
    """
    Gradability check. Run on the RAW image, before normalisation.
 
    In a real screening programme 5-20% of captures are ungradable, and an
    ungradable image is a REFERRAL, not a negative. A model that quietly grades
    a blurred photo as 'no retinopathy' is the failure mode that harms patients.
 
    Returns reasons, not just a flag, so the operator can be told what to fix.
    """
    t = thresholds or QUALITY_THRESHOLDS
    work = _to_work_size(img_bgr)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    mask = _fov_mask(work)
 
    fov_frac = float(mask.mean())
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    retina_px = gray[mask]
    mean_int = float(retina_px.mean()) if retina_px.size else 0.0
 
    reasons = []
    if fov_frac < t["min_fov_fraction"]:
        reasons.append(f"field of view too small ({fov_frac:.2f})")
    if blur_var < t["min_blur_var"]:
        reasons.append(f"image too blurred (laplacian var {blur_var:.1f})")
    if mean_int < t["min_mean_intensity"]:
        reasons.append(f"underexposed (mean {mean_int:.1f})")
    if mean_int > t["max_mean_intensity"]:
        reasons.append(f"overexposed (mean {mean_int:.1f})")
 
    return {
        "gradable": len(reasons) == 0,
        "reasons": reasons,
        "fov_fraction": fov_frac,
        "blur_var": blur_var,
        "mean_intensity": mean_int,
    }
 
 
# ---------------------------------------------------------------------------
# Offline cache builder
# ---------------------------------------------------------------------------
 
def _process_one(args):
    src, dst, cfg = args
    try:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return src.name, None, "unreadable"
        q = assess_quality(img)
        out = preprocess_image(img, cfg)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return src.name, q, None
    except Exception as e:                              # noqa: BLE001
        return src.name, None, f"{type(e).__name__}: {e}"
 
 
def build_cache(csv_path, images_dir, cache_dir, filename_col,
                cfg: PreprocessConfig, workers: int = 8) -> pd.DataFrame:
    """Preprocess every image once. Writes a quality report alongside the cache."""
    csv_path = Path(csv_path)
    images_dir, cache_dir = Path(images_dir), Path(cache_dir)
 
    # Validate up front with actionable messages. A raw pandas/OpenCV traceback
    # tells you what broke but not what to do, and these three mistakes account
    # for most first-run failures.
    if not csv_path.exists():
        raise SystemExit(
            f"\nLabels CSV not found: {csv_path.resolve()}\n"
            f"\nIf you have no dataset yet, generate a synthetic one and run the"
            f"\nwhole pipeline on that first:"
            f"\n    python make_demo_data.py --out demo_data --patients 80"
            f"\n    python preprocessing.py --csv demo_data/labels.csv "
            f"--images demo_data/images --cache demo_data/cache_512 "
            f"--filename-col id_code"
            f"\n\nIf you do have data, check the path. Run  python check_setup.py"
        )
    if not images_dir.exists():
        raise SystemExit(
            f"\nImages directory not found: {images_dir.resolve()}\n"
            f"Pass the correct folder with --images"
        )
 
    df = pd.read_csv(csv_path)
    if filename_col not in df.columns:
        raise SystemExit(
            f"\nColumn '{filename_col}' is not in {csv_path.name}."
            f"\nAvailable columns: {list(df.columns)}"
            f"\n\nPass the right one with --filename-col, e.g.:"
            f"\n    --filename-col {list(df.columns)[0]}"
        )
 
    cache_dir.mkdir(parents=True, exist_ok=True)
 
    jobs = []
    for name in df[filename_col].astype(str):
        src = images_dir / name
        dst = cache_dir / (Path(name).stem + ".png")
        if dst.exists():
            continue
        jobs.append((src, dst, cfg))
 
    # Catch the "images are there but named differently" case before spending
    # minutes preprocessing nothing. APTOS is the classic example: its CSV
    # stores id_code without a file extension.
    if jobs:
        missing = [j for j in jobs[:20] if not j[0].exists()]
        if len(missing) == min(20, len(jobs)):
            on_disk = [p.name for p in list(images_dir.iterdir())[:3]]
            raise SystemExit(
                f"\nNone of the first {min(20, len(jobs))} files named in the CSV exist "
                f"in {images_dir}.\n"
                f"  CSV expects : {missing[0][0].name}\n"
                f"  Folder has  : {on_disk}\n"
                f"\nUsually a missing/extra file extension. If the CSV lists names "
                f"without\nan extension (APTOS does this), add it to the CSV, or "
                f"rename the files."
            )
 
    print(f"Preprocessing {len(jobs)} images -> {cache_dir} "
          f"(fingerprint {cfg.fingerprint()})")
    if not jobs:
        print("  Nothing to do -- cache is already complete.")
 
    records = []
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                name, q, err = fut.result()
                rec = {"filename": name, "error": err}
                if q:
                    rec.update({k: v for k, v in q.items() if k != "reasons"})
                    rec["reasons"] = "; ".join(q["reasons"])
                records.append(rec)
                if i % 250 == 0:
                    print(f"  {i}/{len(jobs)}")
 
    qdf = pd.DataFrame(records)
    if not qdf.empty:
        report = cache_dir / "quality_report.csv"
        qdf.to_csv(report, index=False)
        n_bad = int((~qdf["gradable"].fillna(False)).sum()) if "gradable" in qdf else 0
        print(f"\nQuality report -> {report}")
        print(f"  ungradable / failed: {n_bad} / {len(qdf)} ({100*n_bad/max(len(qdf),1):.1f}%)")
        print("  These are NOT negatives. Exclude them from training and route "
              "them to manual review at serving time.")
 
    # Record the fingerprint so the dataset loader can verify the cache matches.
    (cache_dir / ".fingerprint").write_text(cfg.fingerprint())
    return qdf
 
 
if __name__ == "__main__":
    from config import CONFIG
 
    p = argparse.ArgumentParser(description="Build the preprocessed fundus cache.")
    p.add_argument("--csv", default=CONFIG.data.csv_path)
    p.add_argument("--images", default=CONFIG.data.images_dir)
    p.add_argument("--cache", default=CONFIG.data.cache_dir)
    p.add_argument("--filename-col", default=CONFIG.data.filename_col)
    p.add_argument("--size", type=int, default=CONFIG.preprocess.image_size)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    a = p.parse_args()
 
    cfg = PreprocessConfig(image_size=a.size)
    build_cache(a.csv, a.images, a.cache, a.filename_col, cfg, a.workers)