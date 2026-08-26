from __future__ import annotations
 
import argparse
from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
 
 
def _vessels(img, rng, cx, cy, r):
    """Crude branching vessel tree from the optic disc."""
    for _ in range(9):
        ang = rng.uniform(0, 2 * np.pi)
        x, y = cx, cy
        thick = rng.integers(4, 8)
        for _ in range(rng.integers(14, 22)):
            ang += rng.normal(0, 0.28)
            nx, ny = x + np.cos(ang) * r * 0.06, y + np.sin(ang) * r * 0.06
            cv2.line(img, (int(x), int(y)), (int(nx), int(ny)),
                     (28, 28, 138), max(1, int(thick)))
            x, y = nx, ny
            thick *= 0.92
            if (x - cx) ** 2 + (y - cy) ** 2 > (r * 0.95) ** 2:
                break
 
 
def make_fundus(grade: int, rng, size=1200) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    cx = cy = size // 2
    r = int(size * 0.46)
 
    # retina disc with a mild illumination gradient (the pipeline should remove it)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    base = np.zeros((size, size, 3), np.float32)
    base[..., 0], base[..., 1], base[..., 2] = 42, 88, 186          # BGR
    grad = 1.0 - 0.45 * ((xx - cx) / r) + 0.15 * ((yy - cy) / r)
    base *= np.clip(grad, 0.25, 1.35)[..., None]
    base += rng.normal(0, 6, base.shape)
    img[disc] = np.clip(base, 0, 255).astype(np.uint8)[disc]
 
    # optic disc, offset like a real eye
    ox = cx + int(r * rng.uniform(0.30, 0.42)) * rng.choice([-1, 1])
    oy = cy + int(r * rng.uniform(-0.08, 0.08))
    cv2.circle(img, (ox, oy), int(r * 0.13), (150, 215, 245), -1)
    _vessels(img, rng, ox, oy, r)
 
    # lesions scale with grade
    n_ma = [0, 6, 22, 45, 60][grade]
    n_ex = [0, 0, 5, 18, 30][grade]
    n_hem = [0, 0, 2, 10, 20][grade]
 
    def rand_pt(margin=0.85):
        while True:
            a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, r * margin)
            px, py = int(cx + np.cos(a) * d), int(cy + np.sin(a) * d)
            if (px - ox) ** 2 + (py - oy) ** 2 > (r * 0.16) ** 2:
                return px, py
 
    for _ in range(n_ma):                                   # microaneurysms
        cv2.circle(img, rand_pt(), rng.integers(3, 6), (18, 18, 105), -1)
    for _ in range(n_ex):                                   # hard exudates
        p = rand_pt()
        cv2.ellipse(img, p, (rng.integers(6, 14), rng.integers(5, 11)),
                    rng.integers(0, 180), 0, 360, (170, 235, 250), -1)
    for _ in range(n_hem):                                  # blot haemorrhages
        p = rand_pt()
        cv2.ellipse(img, p, (rng.integers(10, 22), rng.integers(8, 18)),
                    rng.integers(0, 180), 0, 360, (25, 25, 90), -1)
    if grade == 4:                                          # neovascularisation
        for _ in range(4):
            _vessels(img, rng, rand_pt(0.6)[0], rand_pt(0.6)[1], int(r * 0.25))
 
    return img
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="demo_data")
    p.add_argument("--patients", type=int, default=80)
    p.add_argument("--size", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ungradable-frac", type=float, default=0.06,
                   help="fraction blurred/dark, to exercise the quality gate")
    a = p.parse_args()
 
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
 
    rows = []
    for pid in range(a.patients):
        # realistic-ish DR distribution, correlated between a patient's two eyes
        worst = int(rng.choice([0, 1, 2, 3, 4], p=[0.55, 0.18, 0.14, 0.08, 0.05]))
        for eye in ("left", "right"):
            g = max(0, worst - int(rng.integers(0, 2)))
            img = make_fundus(g, rng, a.size)
 
            bad = rng.random() < a.ungradable_frac
            if bad:                                    # exercise the quality gate
                if rng.random() < 0.5:
                    img = cv2.GaussianBlur(img, (0, 0), 14)
                else:
                    img = (img * 0.06).astype(np.uint8)
 
            name = f"P{pid:04d}_{eye}.png"
            cv2.imwrite(str(img_dir / name), img)
            rows.append({
                "id_code": name,
                "diagnosis": g,
                "patient_id": f"P{pid:04d}",
                "eye": eye,
                "device": rng.choice(["CamA", "CamB", "CamC"], p=[0.5, 0.35, 0.15]),
                "sex": rng.choice(["M", "F"]),
                "age_group": rng.choice(["<50", "50-65", ">65"], p=[0.3, 0.45, 0.25]),
                "_synthetic_ungradable": bool(bad),
            })
 
    df = pd.DataFrame(rows)
    csv = out / "labels.csv"
    df.to_csv(csv, index=False)
 
    print(f"\nWrote {len(df)} images from {a.patients} patients -> {img_dir}")
    print(f"Labels -> {csv}")
    print("\nGrade distribution:")
    for g in range(5):
        n = int((df["diagnosis"] == g).sum())
        print(f"  {g}: {n:>4} ({100*n/len(df):5.1f}%)")
    print(f"Referable (>=2): {int((df['diagnosis']>=2).sum())} "
          f"({100*(df['diagnosis']>=2).mean():.1f}%)")
    print(f"Deliberately ungradable: {int(df['_synthetic_ungradable'].sum())}")
 
    # Write the config alongside the data, so the two cannot drift apart and so
    # settings survive closing the terminal. Shell variables still override it.
    env = Path("dr.env")
    env.write_text(
        "# Written by make_demo_data.py. Points the pipeline at the synthetic\n"
        "# demo dataset. Delete or edit to switch datasets; a real shell\n"
        "# environment variable overrides anything here.\n"
        f"DR_CSV={csv}\n"
        f"DR_IMAGES={img_dir}\n"
        f"DR_CACHE={out / 'cache_512'}\n"
        "DR_FILENAME_COL=id_code\n"
        "DR_GRADE_COL=diagnosis\n"
        "DR_PATIENT_COL=patient_id\n"
        "DR_PATIENT_SOURCE=column\n"
        "\n# Windows DataLoader workers use spawn, not fork: slow and crash-prone.\n"
        "DR_NUM_WORKERS=0\n"
        "\n# Small and short so a CPU-only machine finishes. For checking the\n"
        "# pipeline runs, not for producing a usable model.\n"
        "DR_EPOCHS=3\n"
        "DR_BATCH_SIZE=4\n",
        encoding="utf-8")
    print(f"\nWrote {env} -- config now persists across terminal windows.")
    print("Next:")
    print(f"  python preprocessing.py --csv {csv} --images {img_dir} "
          f"--cache {out / 'cache_512'} --filename-col id_code --size 512")
    print("\n*** SYNTHETIC DATA. Any metric obtained from it is meaningless "
          "clinically. This is a plumbing test only. ***")
 
 
if __name__ == "__main__":
    main()