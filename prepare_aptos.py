from __future__ import annotations
 
import argparse
import hashlib
from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
 
 
# ---------------------------------------------------------------------------
 
def resolve_extensions(df: pd.DataFrame, images_dir: Path, col: str) -> pd.DataFrame:
    """Repair filenames against what actually exists on disk."""
    on_disk = {p.stem: p.name for p in images_dir.iterdir() if p.is_file()}
    if not on_disk:
        raise SystemExit(f"No files found in {images_dir}")
 
    resolved, missing = [], []
    for raw in df[col].astype(str):
        if (images_dir / raw).exists():
            resolved.append(raw)
            continue
        stem = Path(raw).stem
        if stem in on_disk:
            resolved.append(on_disk[stem])
        else:
            resolved.append(None)
            missing.append(raw)
 
    df = df.copy()
    df["filename"] = resolved
    if missing:
        print(f"  ! {len(missing)} rows have no matching file (e.g. {missing[:3]}); dropped.")
        df = df[df["filename"].notna()].reset_index(drop=True)
 
    changed = int((df["filename"] != df[col].astype(str)).sum())
    if changed:
        ext = Path(df["filename"].iloc[0]).suffix
        print(f"  Repaired {changed} filenames (added '{ext}').")
    return df
 
 
# ---------------------------------------------------------------------------
 
def _dhash(img_gray: np.ndarray, size: int = 16) -> int:
    """
    Difference hash. A 16x16 grid gives 240 bits.
 
    The usual 8x8 (64-bit) dHash is far too coarse here. Every fundus photograph
    is globally alike -- a round orange disc on a black surround -- so at 64 bits
    unrelated eyes collide constantly. Measured on 60 distinct synthetic eyes, an
    8x8 hash matched every one of them to every other.
    """
    small = cv2.resize(img_gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    out = 0
    for b in bits.flatten():
        out = (out << 1) | int(b)
    return out
 
 
def hash_images(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    """Exact hash on decoded pixels + perceptual hash, in a single read pass."""
    exact, perceptual, bad = [], [], 0
    for i, name in enumerate(df["filename"], 1):
        img = cv2.imread(str(images_dir / name), cv2.IMREAD_COLOR)
        if img is None:
            exact.append(None)
            perceptual.append(None)
            bad += 1
            continue
        # Hash the DECODED pixels, not the file bytes, so a re-encoded copy of
        # the same image still matches.
        exact.append(hashlib.md5(img.tobytes()).hexdigest())
        gray = cv2.cvtColor(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        perceptual.append(_dhash(gray))
        if i % 500 == 0:
            print(f"    hashed {i}/{len(df)}")
 
    df = df.copy()
    df["pixel_md5"], df["dhash"] = exact, perceptual
    if bad:
        print(f"  ! {bad} images could not be decoded; dropped.")
        df = df[df["pixel_md5"].notna()].reset_index(drop=True)
    return df
 
 
def find_near_duplicates(df: pd.DataFrame, max_distance: int = 12,
                         max_group_size: int = 8):
    """
    Group near-identical images. Returns (groups, suspicious_groups).
 
    Deliberately NOT a transitive closure. Union-find merges A with C whenever
    A~B and B~C, even when A and C are nothing alike. On globally similar images
    that chains the whole dataset into one component -- in testing it collapsed
    64 images into a single group of 64, and deduplication then deleted 63 of
    them. Here each image joins a group only if it is within `max_distance` of
    that group's REPRESENTATIVE, so similarity is never inherited second-hand.
 
    Groups larger than `max_group_size` are returned as SUSPICIOUS rather than
    deduplicated. A duplicate group of 40 images is not 40 duplicates; it is a
    threshold that is too loose, and silently deleting them would be worse than
    the problem being solved.
    """
    hashes = df["dhash"].to_dict()
    reps: list[int] = []
    members: dict[int, list[int]] = {}
 
    for idx in df.index:
        h = hashes[idx]
        for rep in reps:
            if bin(h ^ hashes[rep]).count("1") <= max_distance:
                members[rep].append(idx)
                break
        else:
            reps.append(idx)
            members[idx] = [idx]
 
    groups, suspicious = [], []
    for g in members.values():
        if len(g) < 2:
            continue
        (suspicious if len(g) > max_group_size else groups).append(g)
    return groups, suspicious
 
 
# ---------------------------------------------------------------------------
 
def main():
    ap = argparse.ArgumentParser(description="Clean APTOS 2019 for this pipeline.")
    ap.add_argument("--raw-csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="labels_clean.csv")
    ap.add_argument("--filename-col", default="id_code")
    ap.add_argument("--grade-col", default="diagnosis")
    ap.add_argument("--near-duplicate-distance", type=int, default=0,
                    help="dHash Hamming radius out of 240 bits. DEFAULT 0 = exact "
                         "matching only, which is the documented APTOS problem and "
                         "needs no tuning. Non-zero values require calibration per "
                         "dataset; start at 6 and inspect the flagged groups before "
                         "trusting them.")
    ap.add_argument("--max-group-size", type=int, default=8,
                    help="groups larger than this are flagged, not deduplicated")
    args = ap.parse_args()
 
    images_dir = Path(args.images)
    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
 
    print(f"\n[1/5] Reading {args.raw_csv}")
    df = pd.read_csv(args.raw_csv)
    for c in (args.filename_col, args.grade_col):
        if c not in df.columns:
            raise SystemExit(f"Column '{c}' not in CSV. Available: {list(df.columns)}")
    df = df.rename(columns={args.grade_col: "grade"})
    print(f"  {len(df)} rows")
 
    print("\n[2/5] Resolving filenames against disk")
    df = resolve_extensions(df, images_dir, args.filename_col)
 
    print("\n[3/5] Hashing images (exact + perceptual)")
    df = hash_images(df, images_dir)
 
    print("\n[4/5] Removing duplicates")
    n_start = len(df)
 
    exact_groups = [g.tolist() for _, g in df.groupby("pixel_md5").groups.items()
                    if len(g) > 1]
    exact_dupes = sum(len(g) - 1 for g in exact_groups)
    print(f"  exact duplicates : {exact_dupes} extra copies in {len(exact_groups)} groups")
 
    near_groups, suspicious = [], []
    if args.near_duplicate_distance > 0:
        near_groups, suspicious = find_near_duplicates(
            df, args.near_duplicate_distance, args.max_group_size)
        extra = sum(len(g) - 1 for g in near_groups)
        print(f"  near duplicates  : {extra} extra copies in {len(near_groups)} groups "
              f"(dHash distance <= {args.near_duplicate_distance})")
 
    if suspicious:
        biggest = max(len(g) for g in suspicious)
        print(f"\n  ! {len(suspicious)} group(s) exceed --max-group-size "
              f"(largest has {biggest} images). NOT deduplicated.")
        print("    A 'duplicate' group that large almost always means the distance")
        print("    threshold is too loose for this dataset, not that you have that")
        print("    many copies. Inspect them, then lower --near-duplicate-distance.")
        for g in suspicious[:2]:
            print(f"      e.g. {df.loc[g[:4], 'filename'].tolist()} ... (+{len(g)-4})")
 
    # Combine, preferring near-duplicate groups (they subsume exact ones).
    groups = near_groups if near_groups else exact_groups
    if near_groups and exact_groups:
        covered = {i for g in near_groups for i in g}
        groups = near_groups + [g for g in exact_groups
                                if not set(g) & covered]
 
    # Runaway guard. Deleting a large share of the dataset is never the right
    # outcome of a dedup pass; abort rather than quietly destroy the data.
    would_drop = sum(len(g) - (0 if df.loc[g, "grade"].nunique() > 1 else 1)
                     for g in groups)
    if would_drop > 0.25 * n_start:
        raise SystemExit(
            f"\n  ABORTING: deduplication would remove {would_drop}/{n_start} rows "
            f"({100*would_drop/n_start:.0f}%).\n"
            f"  That is not a duplicate problem, it is a threshold problem.\n"
            f"  Re-run with a tighter radius, e.g. --near-duplicate-distance 6,\n"
            f"  or --near-duplicate-distance 0 to use exact matching only."
        )
 
    # Label conflicts among duplicates are direct evidence of annotation noise,
    # and they cap the accuracy any model can reach. Report, never hide.
    conflicts = [g for g in groups if df.loc[g, "grade"].nunique() > 1]
    if conflicts:
        print(f"\n  ! {len(conflicts)} duplicate group(s) carry DIFFERENT grades.")
        print("    Same image, two labels -- direct evidence of label noise.")
        for g in conflicts[:5]:
            names = df.loc[g[:4], "filename"].tolist()
            grades = df.loc[g[:4], "grade"].tolist()
            more = f" ... (+{len(g)-4})" if len(g) > 4 else ""
            print(f"      {names}{more} -> grades {grades}")
        print("    These groups are dropped ENTIRELY. An image whose true grade is")
        print("    unknown is worse than no image: keeping one copy arbitrarily")
        print("    injects a label you already know to be unreliable.")
 
    drop = set()
    for g in groups:
        if df.loc[g, "grade"].nunique() > 1:
            drop.update(g)                      # conflicting grades: drop all
        else:
            drop.update(g[1:])                  # agreeing: keep the first
    df = df.drop(index=list(drop)).reset_index(drop=True)
 
    print(f"\n  {n_start} -> {len(df)} rows ({n_start - len(df)} removed)")
 
    print("\n[5/5] Writing cleaned labels")
    # Safe ONLY because deduplication has run: APTOS is one image per patient,
    # but duplicates would have made that claim false.
    df["patient_id"] = [f"APTOS{i:05d}" for i in range(len(df))]
    out_cols = ["filename", "grade", "patient_id", "pixel_md5"]
    out = df[out_cols].rename(columns={"grade": "diagnosis"})
    out.to_csv(args.out, index=False)
 
    print(f"  Wrote {args.out}")
    print("\n  Grade distribution after cleaning:")
    for g in sorted(out["diagnosis"].unique()):
        n = int((out["diagnosis"] == g).sum())
        print(f"    {g}: {n:>5}  ({100*n/len(out):5.1f}%)")
    ref = int((out["diagnosis"] >= 2).sum())
    print(f"  Referable (>=2): {ref}/{len(out)} ({100*ref/len(out):.1f}%)")
    print("\n  NOTE: APTOS referable prevalence is far higher than a real screening")
    print("  population (typically 8-25%). PPV measured here will be optimistic")
    print("  relative to deployment, even though sensitivity and specificity are")
    print("  prevalence-independent.")
 
    cache = Path(args.out).parent / "cache_512"
    env = Path("dr.env")
    env.write_text(
        "# Written by prepare_aptos.py. Points the pipeline at the CLEANED APTOS\n"
        "# labels (extensions repaired, duplicates removed, patient_id added).\n"
        "# A real shell environment variable overrides anything here.\n"
        f"DR_CSV={args.out}\n"
        f"DR_IMAGES={images_dir}\n"
        f"DR_CACHE={cache}\n"
        "DR_FILENAME_COL=filename\n"
        "DR_GRADE_COL=diagnosis\n"
        "DR_PATIENT_COL=patient_id\n"
        "DR_PATIENT_SOURCE=column\n"
        "DR_NUM_WORKERS=0\n"
        "DR_EPOCHS=30\n"
        "DR_BATCH_SIZE=8\n",
        encoding="utf-8")
    print(f"\n  Wrote {env} -- config persists across terminal windows.")
    print("\n  Next:")
    print(f"    python preprocessing.py --csv {args.out} --images {images_dir} "
          f"--cache {cache} --filename-col filename --size 512")
 
 
if __name__ == "__main__":
    main()