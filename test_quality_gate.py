"""validate_quality_gate.py, on synthetic labelled images."""
import cv2
import numpy as np

import validate_quality_gate as vq
from test_predictor import _fake_fundus


def test_validator_counts_catches_and_false_rejects(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    good = _fake_fundus(seed=1)
    blurred = cv2.GaussianBlur(good, (0, 0), 25)          # human would reject
    dark = (good * 0.03).astype(np.uint8)                  # human would reject
    cv2.imwrite(str(imgs / "g1.png"), good)
    cv2.imwrite(str(imgs / "g2.png"), _fake_fundus(seed=2))
    cv2.imwrite(str(imgs / "b.png"), blurred)
    cv2.imwrite(str(imgs / "d.png"), dark)
    csv = tmp_path / "labels.csv"
    csv.write_text("image,quality\ng1.png,0\ng2.png,1\nb.png,2\nd.png,2\nmissing.png,0\n")

    rep = vq.main(["--csv", str(csv), "--images", str(imgs),
                   "--out", str(tmp_path / "out.json")])
    assert rep["n_images"] == 4 and rep["unreadable"] == 1
    assert rep["n_human_reject"] == 2 and rep["n_human_accept"] == 2
    assert rep["catch_rate"] == 1.0
    assert rep["false_reject_rate"] == 0.0
    assert (tmp_path / "out.json").exists()
