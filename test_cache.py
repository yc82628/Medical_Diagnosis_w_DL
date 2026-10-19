"""
Cache naming and the quality report, for datasets whose images live in
subfolders (DeepDRiD, IDRiD). Torch-free.
"""
import cv2
import numpy as np
import pandas as pd
import pytest

from config import DataConfig, PreprocessConfig
from preprocessing import build_cache, cache_name
from splits import load_labels
from test_predictor import _fake_fundus


def test_flat_names_keep_their_old_cache_file():
    # Caches built before cache_name() existed must stay valid.
    assert cache_name("abc123.png") == "abc123.png"
    assert cache_name("10_left.jpeg") == "10_left.png"


def test_same_name_in_different_folders_gets_different_cache_files():
    a = cache_name("1. Original Images/a. Training Set/IDRiD_001.jpg")
    b = cache_name("1. Original Images/b. Testing Set/IDRiD_001.jpg")
    assert a != b
    assert cache_name(r"x\y\z.jpg") == cache_name("x/y/z.jpg")


def _dataset(tmp_path):
    """Two different images that share a basename, in two folders."""
    imgs = tmp_path / "imgs"
    good = _fake_fundus(seed=1)
    dark = (good * 0.03).astype(np.uint8)          # fails the quality gate
    for sub, img in (("a", good), ("b", dark)):
        (imgs / sub).mkdir(parents=True)
        cv2.imwrite(str(imgs / sub / "IDRiD_001.png"), img)
    csv = tmp_path / "labels.csv"
    pd.DataFrame({"filename": ["a/IDRiD_001.png", "b/IDRiD_001.png"],
                  "diagnosis": [0, 3], "patient_id": ["p1", "p2"]}).to_csv(csv, index=False)
    return csv, imgs, tmp_path / "cache"


def test_colliding_basenames_are_cached_separately(tmp_path):
    csv, imgs, cache = _dataset(tmp_path)
    build_cache(csv, imgs, cache, "filename", PreprocessConfig(image_size=128), workers=1)
    files = sorted(p.name for p in cache.glob("*.png"))
    assert files == ["a__IDRiD_001.png", "b__IDRiD_001.png"]
    # Different source images must give different cached pixels.
    a, b = (cv2.imread(str(cache / f)) for f in files)
    assert not np.array_equal(a, b)


def test_quality_report_excludes_ungradable_nested_images(tmp_path):
    csv, imgs, cache = _dataset(tmp_path)
    build_cache(csv, imgs, cache, "filename", PreprocessConfig(image_size=128), workers=1)
    report = pd.read_csv(cache / "quality_report.csv")
    assert set(report["filename"]) == {"a/IDRiD_001.png", "b/IDRiD_001.png"}
    dc = DataConfig(csv_path=str(csv), images_dir=str(imgs), cache_dir=str(cache),
                    filename_col="filename", grade_col="diagnosis",
                    patient_col="patient_id", patient_id_source="column")
    df = load_labels(dc, verbose=False)
    assert df["filename"].tolist() == ["a/IDRiD_001.png"], "dark image must be excluded"


def test_incremental_build_keeps_earlier_report_rows(tmp_path):
    csv, imgs, cache = _dataset(tmp_path)
    full = pd.read_csv(csv)
    full.iloc[:1].to_csv(csv, index=False)                       # first run: one image
    build_cache(csv, imgs, cache, "filename", PreprocessConfig(image_size=128), workers=1)
    full.to_csv(csv, index=False)                                # second run: both
    build_cache(csv, imgs, cache, "filename", PreprocessConfig(image_size=128), workers=1)
    report = pd.read_csv(cache / "quality_report.csv")
    assert len(report) == 2 and report["filename"].is_unique


def test_csv_listing_one_file_twice_is_refused(tmp_path):
    csv, imgs, cache = _dataset(tmp_path)
    pd.DataFrame({"filename": ["a/IDRiD_001.png", "a/IDRiD_001.png"],
                  "diagnosis": [0, 0]}).to_csv(csv, index=False)
    with pytest.raises(SystemExit, match="shared by different CSV rows"):
        build_cache(csv, imgs, cache, "filename", PreprocessConfig(image_size=128), workers=1)
