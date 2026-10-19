"""
Prepare DeepDRiD (ISBI 2020) regular fundus images for this pipeline.

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/deepdrdoc/DeepDRiD.git deepdrid/repo
    (cd deepdrid/repo && git sparse-checkout set regular_fundus_images)
    python prepare_deepdrid.py

Licence: CC BY-SA 4.0. Cite Liu et al., "DeepDRiD: Diabetic Retinopathy --
Grading and Image Quality Estimation Challenge", Patterns 3(6), 2022.

What it does:
  * Pools the training, validation and online-evaluation splits (2,000 images,
    500 patients). The pipeline makes its own patient-grouped split, so the
    challenge's split boundaries are not reused -- but no patient is shared
    between them, which the script checks.
  * Grades are given PER EYE (left_eye_DR_Level / right_eye_DR_Level); each
    image takes the grade of its own eye, read from the _l / _r in its id.
  * Grade 5 ("image quality too low to grade") is not an ICDR grade and is
    dropped from the labels, as are images with no grade. Both are kept in
    the quality-label file, where they count as human-rejected.
  * Images the graders marked "Overall quality 0" are KEPT for training when
    they carry a grade, as in the challenge itself; the human quality label is
    written out separately so validate_quality_gate.py can measure the gate.

Writes:
  deepdrid/labels.csv           filename, diagnosis, patient_id, eye, site, human_quality
  deepdrid/quality_labels.csv   filename, quality (1 good / 0 not diagnosable)
  dr.env                        points training here, IDRiD as the external set
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SPLITS = {
    "training": "regular-fundus-training",
    "validation": "regular-fundus-validation",
}
EVAL_DIR = "Online-Challenge1&2-Evaluation"


def _image_path(root: Path, folder: str, image_id: str) -> str:
    """Relative path (from root) of an image, as it exists on disk."""
    patient = image_id.split("_")[0]
    rel = Path(folder) / "Images" / patient / f"{image_id}.jpg"
    if not (root / rel).exists():
        raise SystemExit(f"  Missing image {root / rel}. Is the clone complete?")
    return rel.as_posix()


def _eye(image_id: str) -> str:
    m = re.search(r"_([lr])\d+$", image_id)
    if not m:
        raise ValueError(f"Cannot read the eye from image id {image_id!r}")
    return "left" if m.group(1) == "l" else "right"


def load(root: Path) -> pd.DataFrame:
    rows = []
    for split, folder in SPLITS.items():
        d = pd.read_csv(root / folder / f"{folder}.csv")
        src = pd.read_csv(root / folder / f"regular-fundus-source-{split}.csv")
        site = dict(zip(src["image_id"], src["Source"]))
        for r in d.itertuples(index=False):
            image_id = r.image_id
            eye = _eye(image_id)
            grade = r.left_eye_DR_Level if eye == "left" else r.right_eye_DR_Level
            rows.append({"image_id": image_id, "patient_id": f"DDR{int(r.patient_id):04d}",
                         "split": split, "eye": eye, "raw_grade": grade,
                         "human_quality": int(getattr(r, "_3")),   # "Overall quality"
                         "site": site.get(image_id, "unknown"),
                         "filename": _image_path(root, folder, image_id)})

    ev_dr = pd.read_excel(root / EVAL_DIR / "Challenge1_labels.xlsx")
    ev_q = pd.read_excel(root / EVAL_DIR / "Challenge2_labels.xlsx")
    q = dict(zip(ev_q["image_id"], ev_q["Overall quality"]))
    for r in ev_dr.itertuples(index=False):
        image_id = r.image_id
        rows.append({"image_id": image_id, "patient_id": f"DDR{int(image_id.split('_')[0]):04d}",
                     "split": "evaluation", "eye": _eye(image_id), "raw_grade": r.DR_Levels,
                     "human_quality": int(q[image_id]), "site": "unknown",
                     "filename": _image_path(root, EVAL_DIR, image_id)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="deepdrid/repo/regular_fundus_images")
    ap.add_argument("--out-dir", default="deepdrid")
    ap.add_argument("--external-csv", default="idrid/labels.csv",
                    help="external set written to dr.env if it exists")
    ap.add_argument("--no-env", action="store_true", help="do not rewrite dr.env")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out_dir)
    df = load(root)
    print(f"  {len(df)} images, {df.patient_id.nunique()} patients")

    # The challenge splits must not share patients, or pooling would hide a leak.
    by_split = df.groupby("split")["patient_id"].apply(set)
    for a in by_split.index:
        for b in by_split.index:
            if a < b and by_split[a] & by_split[b]:
                raise SystemExit(f"  Patients shared between {a} and {b}: "
                                 f"{sorted(by_split[a] & by_split[b])[:5]}")
    if df["image_id"].duplicated().any():
        raise SystemExit("  Duplicate image ids across splits.")

    graded = df["raw_grade"].isin([0, 1, 2, 3, 4])
    n_ungradable = int((df["raw_grade"] == 5).sum())
    n_missing = int(df["raw_grade"].isna().sum())
    print(f"  Dropped from labels: {n_ungradable} graded 'too poor to grade' (5), "
          f"{n_missing} with no grade")

    out.mkdir(parents=True, exist_ok=True)
    labels = df[graded].copy()
    labels["diagnosis"] = labels["raw_grade"].astype(int)
    labels = labels[["filename", "diagnosis", "patient_id", "eye", "site", "human_quality"]]
    labels.to_csv(out / "labels.csv", index=False)

    # Human-rejected = marked not diagnosable, or graded 5 (too poor to grade).
    ql = df[["filename"]].copy()
    ql["quality"] = ((df["human_quality"] == 1) & (df["raw_grade"] != 5)).astype(int)
    ql.to_csv(out / "quality_labels.csv", index=False)

    print(f"\n  Wrote {out / 'labels.csv'} ({len(labels)} graded images, "
          f"{labels.patient_id.nunique()} patients)")
    for g in range(5):
        n = int((labels["diagnosis"] == g).sum())
        print(f"    grade {g}: {n:>5}  ({100 * n / len(labels):5.1f}%)")
    ref = int((labels["diagnosis"] >= 2).sum())
    print(f"  Referable (>=2): {ref}/{len(labels)} ({100 * ref / len(labels):.1f}%)")
    print(f"  Human quality 'not diagnosable': {int((labels.human_quality == 0).sum())} "
          f"of the graded images (kept; see docstring)")
    print(f"  Wrote {out / 'quality_labels.csv'} for validate_quality_gate.py")

    if not args.no_env:
        ext = Path(args.external_csv)
        lines = [
            "# Written by prepare_deepdrid.py. Train on DeepDRiD; evaluate externally",
            "# on IDRiD (different country, cameras and clinic).",
            "# A real shell environment variable overrides anything here.",
            f"DR_CSV={(out / 'labels.csv').as_posix()}",
            f"DR_IMAGES={root.as_posix()}",
            f"DR_CACHE={(out / 'cache_512').as_posix()}",
            "DR_FILENAME_COL=filename",
            "DR_GRADE_COL=diagnosis",
            "DR_PATIENT_COL=patient_id",
            "DR_PATIENT_SOURCE=column",
            "DR_NUM_WORKERS=0",
            "DR_EPOCHS=30",
            "DR_BATCH_SIZE=8",
        ]
        if ext.exists():
            lines += [f"DR_EXTERNAL_CSV={ext.as_posix()}",
                      f"DR_EXTERNAL_IMAGES={(ext.parent / 'B. Disease Grading').as_posix()}",
                      f"DR_EXTERNAL_CACHE={(ext.parent / 'cache_512').as_posix()}"]
        Path("dr.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n  Wrote dr.env" + (" (with IDRiD as the external set)" if ext.exists() else ""))
    print("\n  Next:")
    print(f"    python preprocessing.py --csv {out / 'labels.csv'} --images {root} "
          f"--cache {out / 'cache_512'} --filename-col filename --size 512")


if __name__ == "__main__":
    main()
