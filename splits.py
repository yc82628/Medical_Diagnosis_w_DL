from __future__ import annotations
 
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from config import CONFIG, DataConfig, REFERABLE_THRESHOLD, NUM_GRADES, ICDR_GRADES
 
 
# ---------------------------------------------------------------------------
# Patient identity
# ---------------------------------------------------------------------------
 
def resolve_patient_ids(df: pd.DataFrame, dc: DataConfig) -> pd.Series:
    src = dc.patient_id_source
 
    if src == "column":
        if not dc.patient_col or dc.patient_col not in df.columns:
            raise ValueError(
                f"\npatient_id_source='column' but column '{dc.patient_col}' is not in the CSV."
                f"\nAvailable columns: {list(df.columns)}"
                f"\n\nResolve this deliberately. Options:"
                f"\n  - add a patient_id column (best)"
                f"\n  - patient_id_source='filename_prefix' if filenames encode it"
                f"\n  - patient_id_source='unique_per_row' ONLY if this dataset is"
                f"\n    genuinely one image per patient (e.g. APTOS 2019)"
                f"\nDo not guess. A wrong choice silently inflates every metric you report."
            )
        return df[dc.patient_col].astype(str)
 
    if src == "filename_prefix":
        ids = df["filename"].astype(str).map(lambda f: re.split(r"[_.]", Path(f).name)[0])
        if ids.nunique() == len(df):
            print("  ! filename_prefix produced a unique id per row -- verify this is "
                  "correct and not a filename-convention mismatch.")
        return ids
 
    if src == "unique_per_row":
        print("  ! patient_id_source='unique_per_row': assuming one image per patient.")
        print("    If untrue, test metrics will be inflated.")
        return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)
 
    raise ValueError(f"Unknown patient_id_source: {src!r}")
 
 
# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
 
def load_labels(dc: DataConfig = CONFIG.data, verbose: bool = True) -> pd.DataFrame:
    if not Path(dc.csv_path).exists():
        import os
        overrides = [k for k in os.environ if k.startswith("DR_")]
        hint = (
            f"\n  DR_* environment overrides currently set: {sorted(overrides)}"
            if overrides else
            "\n  No DR_* environment overrides are set, so this is the default from"
            "\n  config.py. On Windows, `set` only applies to the Command Prompt window"
            "\n  where you ran it -- open a new window and it is gone. Run demo_env.bat"
            "\n  (or aptos_env.bat) in THIS window first."
        )
        raise FileNotFoundError(
            f"\nLabels CSV not found: {Path(dc.csv_path).resolve()}"
            f"{hint}"
            f"\n\n  To run on synthetic data:"
            f"\n      demo_env.bat"
            f"\n      python make_demo_data.py --out demo_data --patients 80"
            f"\n      python preprocessing.py --csv demo_data\\labels.csv "
            f"--images demo_data\\images --cache demo_data\\cache_512 "
            f"--filename-col id_code --size 512"
            f"\n      python train.py"
            f"\n\n  To diagnose:  python check_setup.py"
        )
    df = pd.read_csv(dc.csv_path)
    return prepare_labels(df, dc, verbose=verbose)
 
 
def prepare_labels(df: pd.DataFrame, dc: DataConfig = CONFIG.data,
                   verbose: bool = True) -> pd.DataFrame:
    """Validate, normalise column names, attach patient ids and the referable target."""
    for col in (dc.filename_col, dc.grade_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in CSV. Available: {list(df.columns)}")
 
    df = df.rename(columns={dc.filename_col: "filename", dc.grade_col: "grade"}).copy()
 
    before = len(df)
    df = df.dropna(subset=["filename", "grade"])
    if len(df) < before and verbose:
        print(f"  Dropped {before - len(df)} rows with missing filename/grade.")
 
    df["grade"] = df["grade"].astype(int)
    bad = ~df["grade"].between(0, NUM_GRADES - 1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows have grades outside 0-{NUM_GRADES-1}: "
            f"{sorted(df.loc[bad, 'grade'].unique())}"
        )
 
    df["patient_id"] = resolve_patient_ids(df, dc)
    df["referable"] = (df["grade"] >= REFERABLE_THRESHOLD).astype(int)
 
    # Exclude ungradable images if a quality report exists.
    qpath = Path(dc.cache_dir) / "quality_report.csv"
    if qpath.exists():
        q = pd.read_csv(qpath)
        if "gradable" in q.columns:
            ungradable = set(q.loc[~q["gradable"].fillna(False), "filename"])
            n = int(df["filename"].isin(ungradable).sum())
            if n and verbose:
                print(f"  Excluding {n} ungradable images from modelling "
                      f"(at serving time these are referrals, not negatives).")
            df = df[~df["filename"].isin(ungradable)]
 
    if verbose:
        describe(df)
    return df.reset_index(drop=True)
 
 
def describe(df: pd.DataFrame) -> None:
    n, npat = len(df), df["patient_id"].nunique()
    print(f"\n  {n} images from {npat} patients ({n/max(npat,1):.2f} images/patient)")
    print("  ICDR grade distribution:")
    for g in range(NUM_GRADES):
        c = int((df["grade"] == g).sum())
        print(f"    {g}  {ICDR_GRADES[g]:<32} {c:>6}  ({100*c/n:5.1f}%)")
    pos = int(df["referable"].sum())
    print(f"  Referable (grade>={REFERABLE_THRESHOLD}): {pos}/{n} ({100*pos/n:.1f}%)")
    print(f"  -> predicting 'not referable' always scores {100*(1-pos/n):.1f}% "
          f"accuracy. Accuracy is not a usable metric here.")
 
 
# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
 
def split_by_patient(df: pd.DataFrame, dc: DataConfig = CONFIG.data,
                     verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    Split at PATIENT level, stratified on each patient's worst-eye grade.
 
    Stratifying on the worst grade keeps referable-patient prevalence
    comparable across splits, which is the quantity screening metrics are
    defined over.
    """
    from sklearn.model_selection import train_test_split
 
    patients = (df.groupby("patient_id")["grade"].max()
                  .rename("worst_grade").reset_index())
 
    # A stratum must survive BOTH split stages. After stage 1 a stratum retains
    # ~(1-train_ratio) of its members, and stage 2 needs >=2 of those. So the
    # requirement on the full set is 2 / (1 - train_ratio), not a flat 3.
    min_count = max(3, int(np.ceil(2.0 / max(1e-9, 1.0 - dc.train_ratio))))
    strat, merged = collapse_rare_strata(patients["worst_grade"], min_count)
    if merged and verbose:
        print(f"  ! rare strata merged for stratification only (min {min_count} "
              f"patients/stratum): {merged}")
        print("    Grades are unchanged in the data; this affects split balancing only.")
 
    if strat.nunique() < 2:
        if verbose:
            print("  ! too few patients per grade to stratify; falling back to "
                  "an UNSTRATIFIED patient split. Check split prevalences below.")
        strat = None
 
    seed = dc.random_seed
    carrier = strat if strat is not None else pd.Series(0, index=patients.index)
    train_p, temp_p, _, temp_s = train_test_split(
        patients["patient_id"], carrier,
        test_size=1.0 - dc.train_ratio,
        stratify=(carrier if strat is not None else None), random_state=seed,
    )
    rel_test = dc.test_ratio / (dc.val_ratio + dc.test_ratio)
    val_p, test_p = train_test_split(
        temp_p, test_size=rel_test,
        stratify=(temp_s if strat is not None else None), random_state=seed,
    )
 
    sets = {"train": set(train_p), "val": set(val_p), "test": set(test_p)}
    splits = {k: df[df["patient_id"].isin(v)].reset_index(drop=True)
              for k, v in sets.items()}
 
    verify_no_leakage(splits)
    if verbose:
        print("\n  Patient-grouped split:")
        for k, s in splits.items():
            print(f"    {k:<5} {len(s):>6} img  {s['patient_id'].nunique():>6} pts  "
                  f"referable {100*s['referable'].mean():5.1f}%")
    return splits
 
 
def collapse_rare_strata(values: pd.Series, min_count: int) -> tuple[pd.Series, list]:
    """
    Merge sparse severity strata into ADJACENT ones, top-down.
 
    Grades are ordinal, so grade 4 belongs with grade 3, not in an undifferentiated
    "rare" bucket with grade 0. Severe grades are always the sparse ones, so we
    walk downward accumulating until each bucket clears min_count.
 
    Returns (stratification labels, description of merges performed).
    """
    counts = values.value_counts()
    buckets, current = [], []
    for g in sorted(counts.index, reverse=True):
        current.append(g)
        if sum(counts[x] for x in current) >= min_count:
            buckets.append(sorted(current))
            current = []
    if current:                      # leftover low grades fold into the neighbour
        if buckets:
            buckets[-1] = sorted(buckets[-1] + current)
        else:
            buckets.append(sorted(current))
 
    mapping, merges = {}, []
    for b in buckets:
        label = min(b)
        for g in b:
            mapping[g] = label
        if len(b) > 1:
            merges.append(f"grades {b} -> stratum {label}")
    return values.map(mapping), merges
 
 
def verify_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Assert rather than trust. Cheap check, fatal failure."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]["patient_id"]) & set(splits[b]["patient_id"])
            if overlap:
                raise RuntimeError(
                    f"PATIENT LEAKAGE: {len(overlap)} patients appear in both "
                    f"'{a}' and '{b}'. Examples: {sorted(overlap)[:5]}"
                )
 
 
# ---------------------------------------------------------------------------
# Ordinal encoding
# ---------------------------------------------------------------------------
 
def ordinal_targets(grade: int, num_grades: int = NUM_GRADES) -> np.ndarray:
    """
    Cumulative-link encoding: t[k] = 1 iff grade > k, for k = 0..K-2.
 
        grade 0 -> [0,0,0,0]
        grade 2 -> [1,1,0,0]
        grade 4 -> [1,1,1,1]
 
    Makes the loss ordinal-aware -- confusing 4 with 0 costs four wrong bits,
    confusing 4 with 3 costs one. Plain cross-entropy prices those identically,
    which is clinically wrong.
 
    Bonus: output index 1 is exactly P(grade > 1) = P(referable), so the
    screening decision is a single calibratable scalar.
    """
    return (np.arange(num_grades - 1) < grade).astype(np.float32)
 
 
def grades_from_cumulative(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """P(y>k) per threshold -> integer grade, after enforcing monotonicity."""
    probs = enforce_monotonic(probs)
    return (probs > threshold).sum(axis=1).astype(int)
 
 
def enforce_monotonic(probs: np.ndarray) -> np.ndarray:
    """
    P(y>0) >= P(y>1) >= ... must hold. Nothing in the loss guarantees it, so we
    impose it with a running minimum before decoding.
    """
    return np.minimum.accumulate(probs, axis=1)
 