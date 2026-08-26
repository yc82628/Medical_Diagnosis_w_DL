"""
Assemble a curated test batch with KNOWN ground truth.

    python pick_test_images.py                 # list candidates
    python pick_test_images.py --copy-to test_batch

Picks representative images across the severity range plus, if the quality
report exists, one ungradable image. Prints what each SHOULD produce so you can
compare the system's output against the answer instead of guessing.

Reads paths from dr.env, so it works for the synthetic demo set and for cleaned
APTOS without arguments.

A warning worth taking seriously: on synthetic data a correct-looking result
demonstrates that the pipeline is wired up, and nothing more. It is not evidence
that the model detects disease. Only a model trained on real fundus images and
evaluated on a held-out real dataset can support that claim.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from config import CONFIG, REFERABLE_THRESHOLD, ICDR_GRADES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-grade", type=int, default=2,
                    help="how many examples to pick from each ICDR grade")
    ap.add_argument("--copy-to", default=None,
                    help="folder to copy the batch into, for drag-and-drop testing")
    args = ap.parse_args()

    dc = CONFIG.data
    csv, images = Path(dc.csv_path), Path(dc.images_dir)
    if not csv.exists():
        raise SystemExit(
            f"\nLabels CSV not found: {csv.resolve()}\n"
            f"Run  python make_demo_data.py --out demo_data --patients 80\n"
            f"or   python check_setup.py  to diagnose."
        )

    df = pd.read_csv(csv)
    fcol = dc.filename_col if dc.filename_col in df.columns else "filename"
    gcol = dc.grade_col if dc.grade_col in df.columns else "diagnosis"
    if fcol not in df.columns or gcol not in df.columns:
        raise SystemExit(f"Expected columns '{fcol}' and '{gcol}'. Found: {list(df.columns)}")

    # Exclude images the quality gate rejected -- they get their own section.
    ungradable = set()
    qpath = Path(dc.cache_dir) / "quality_report.csv"
    if qpath.exists():
        q = pd.read_csv(qpath)
        if "gradable" in q.columns:
            ungradable = set(q.loc[~q["gradable"].fillna(False), "filename"])

    clean = df[~df[fcol].isin(ungradable)]

    print(f"\nSource: {csv}  ({len(df)} rows, {len(ungradable)} ungradable)\n")
    print(f"{'file':<26} {'grade':>5}  {'expected':<9} {'what it tests'}")
    print("-" * 92)

    picked = []
    for g in sorted(clean[gcol].unique()):
        rows = clean[clean[gcol] == g].head(args.per_grade)
        for _, r in rows.iterrows():
            expected = "REFER" if g >= REFERABLE_THRESHOLD else "ROUTINE"
            note = ICDR_GRADES.get(int(g), "")
            print(f"{str(r[fcol]):<26} {int(g):>5}  {expected:<9} {note}")
            picked.append(r[fcol])

    if ungradable:
        print("-" * 92)
        for name in sorted(ungradable)[:2]:
            print(f"{name:<26} {'--':>5}  {'MANUAL':<9} "
                  f"quality gate must catch this BEFORE the model sees it")
            picked.append(name)

    print("\nWhat to look for:")
    print(f"  grade 0-1  -> ROUTINE  (below the referable threshold of {REFERABLE_THRESHOLD})")
    print("  grade 2-4  -> REFER    (referable DR)")
    print( "  ungradable -> MANUAL REVIEW, never ROUTINE. An ungradable image scored")
    print( "                as healthy is the failure mode that actually harms patients.")
    print( "  borderline -> MANUAL REVIEW is correct, not a failure. The abstention")
    print( "                band exists so uncertain cases go to a human.")

    if args.copy_to:
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in picked:
            src = images / str(name)
            if src.exists():
                shutil.copy(src, dest / src.name)
                n += 1
        print(f"\nCopied {n} images -> {dest.resolve()}")
        print("Drag them into http://127.0.0.1:8000 after starting  python service.py")

    print("\n" + "=" * 92)
    print("If this is synthetic data, a correct result proves the PIPELINE works.")
    print("It is not evidence that the model detects disease. Only a model trained")
    print("on real fundus images and tested on a held-out real dataset shows that.")
    print("=" * 92)


if __name__ == "__main__":
    main()
