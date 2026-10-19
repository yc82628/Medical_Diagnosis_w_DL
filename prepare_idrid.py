"""
Prepare IDRiD (Indian Diabetic Retinopathy Image Dataset) disease grading for
use as the EXTERNAL test set.

    Download "B. Disease Grading.zip" (212 MB) from
        https://zenodo.org/records/17219542   (or IEEE DataPort, the original)
    and extract it into idrid/, then:
    python prepare_idrid.py

Licence: CC BY 4.0. Cite Porwal et al., "Indian Diabetic Retinopathy Image
Dataset (IDRiD)", IEEE DataPort, 2018, doi:10.21227/H25W98.

IDRiD comes from a different country, camera and clinic than DeepDRiD, which
is the point: a random split of one dataset cannot show how a model survives
a new site. Its training and test partitions are pooled -- the model never
trains on any of it.

IDRiD has no patient identifiers, so each image is treated as its own patient.
If some patients contributed both eyes, the bootstrap confidence intervals on
this set are somewhat too narrow; the model card should say so.

IDRiD also grades risk of diabetic macular edema (0-2). It is written out
(`dme_risk`) for future maculopathy work; the model does not use it.

Writes idrid/labels.csv: filename, diagnosis, patient_id, dme_risk
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PARTS = {"a. Training Set": "a. IDRiD_Disease Grading_Training Labels.csv",
         "b. Testing Set": "b. IDRiD_Disease Grading_Testing Labels.csv"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="idrid/B. Disease Grading")
    ap.add_argument("--out", default="idrid/labels.csv")
    args = ap.parse_args(argv)
    root = Path(args.root)

    frames = []
    for folder, csv in PARTS.items():
        d = pd.read_csv(root / "2. Groundtruths" / csv)
        d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
        d.columns = [c.strip() for c in d.columns]
        d = d.dropna(subset=["Image name"])
        d["filename"] = [f"1. Original Images/{folder}/{n}.jpg" for n in d["Image name"]]
        # Both partitions number their images from IDRiD_001: the names collide
        # across folders although the images differ, so prefix the partition.
        d["image_key"] = [f"{folder[0]}_{n}" for n in d["Image name"]]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    missing = [f for f in df["filename"] if not (root / f).exists()]
    if missing:
        raise SystemExit(f"  {len(missing)} images missing, e.g. {missing[:3]}")
    if df["image_key"].duplicated().any():
        raise SystemExit("  Duplicate image names within an IDRiD partition.")

    out = pd.DataFrame({
        "filename": df["filename"],
        "diagnosis": df["Retinopathy grade"].astype(int),
        "patient_id": "IDRID_" + df["image_key"],   # no patient ids in IDRiD
        "dme_risk": df["Risk of macular edema"].astype(int),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"  Wrote {args.out} ({len(out)} images)")
    for g in range(5):
        n = int((out["diagnosis"] == g).sum())
        print(f"    grade {g}: {n:>5}  ({100 * n / len(out):5.1f}%)")
    ref = int((out["diagnosis"] >= 2).sum())
    print(f"  Referable (>=2): {ref}/{len(out)} ({100 * ref / len(out):.1f}%) -- far above "
          f"a screening population, so PPV here is optimistic")
    print("\n  Next:")
    print(f"    python preprocessing.py --csv {args.out} --images \"{root}\" "
          f"--cache {Path(args.out).parent / 'cache_512'} --filename-col filename --size 512")


if __name__ == "__main__":
    main()
