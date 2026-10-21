"""model_card.py: the gate verdict, dataset licences and caveats reach the card."""
import json

import model_card
from config import PreprocessConfig


def _evaluation(**over):
    rep = {"n": 300, "prevalence": 0.4, "tp": 110, "fn": 12, "auroc": 0.95,
           "sensitivity": 0.9, "specificity": 0.9, "ppv": 0.87, "npv": 0.93,
           "ece": 0.03, "qwk": 0.73, "auroc_ci": {"point": 0.95, "lo": 0.91, "hi": 0.97}}
    ev = {"operating_point": 0.4, "calibration": {"method": "isotonic"},
          "internal_test": rep, "external_test": None, "abstention": [],
          "subgroups": [], "preprocess_fingerprint": PreprocessConfig().fingerprint()}
    ev.update(over)
    return ev


def test_card_reports_a_failed_gate_and_why(tmp_path):
    ev = tmp_path / "evaluation.json"
    ext = dict(_evaluation()["internal_test"], sensitivity=0.75, ece=0.15)
    ev.write_text(json.dumps(_evaluation(external_test=ext)))
    text = model_card.build(str(ev), str(tmp_path / "card.md"))
    assert "## Acceptance gate" in text and "**NOT PASSED.**" in text
    assert "external test: sensitivity 0.750" in text
    assert "| Expected calibration error | 0.0300 | 0.1500 |" in text


def test_known_datasets_carry_licence_and_citation():
    d = model_card._dataset_for("deepdrid/labels.csv")
    assert "CC BY-SA 4.0" in d["licence"] and "Patterns" in d["cite"]
    assert "CC BY 4.0" in model_card._dataset_for(r"idrid\labels.csv")["licence"]
    assert "NON-COMMERCIAL" in model_card._dataset_for("aptos/labels_clean.csv")["licence"]
    assert "unknown" in model_card._dataset_for("other/labels.csv")["licence"]


def test_quality_section_compares_learned_and_classical(tmp_path):
    ci = {"point": 0.6, "lo": 0.5, "hi": 0.7, "n_boot": 10}
    me = tmp_path / "quality_model_eval.json"
    me.write_text(json.dumps({
        "backbone": "resnet18", "image_size": 448, "labels": "C:/x/deepdrid/quality_labels.csv",
        "threshold": 0.7, "max_false_reject_target": 0.05, "checkpoint_sha256": "abc",
        "split": {"train": {"patients": 300}, "val": {"patients": 100},
                  "test": {"patients": 100}},
        "test": {"n": 400, "n_human_reject": 214, "catch_rate": 0.6,
                 "auroc_ci": {"point": 0.85, "lo": 0.8, "hi": 0.9},
                 "catch_rate_ci": ci, "false_reject_rate_ci": ci},
        "classical_test": {"auroc_blur_var": 0.51, "catch_rate": 0.05,
                           "false_reject_rate": 0.025}}))
    ck = tmp_path / "quality_model.pt"
    ck.write_bytes(b"other weights")
    served = tmp_path / "served.json"
    served.write_text(json.dumps({"quality_source": "learned", "n_images": 400,
                                  "catch_rate": 0.62, "false_reject_rate": 0.03,
                                  "false_rejects": 6, "n_human_accept": 186}))
    text = "\n".join(model_card._quality_section(str(me), str(tmp_path / "none.json"),
                                                 checkpoint=str(ck), served_eval=str(served)))
    assert "(`deepdrid/quality_labels.csv`)" in text      # never a local absolute path
    assert "catches **62.0%** of human-rejected images and rejects **3.0%**" in text
    assert "| AUROC (human reject = positive) | 0.850 (95% CI 0.800–0.900) | 0.510" in text
    assert "| Human-rejected images caught | 60.0% (50.0–70.0) | 5.0% |" in text
    assert "40.0% of human-rejected test images still reach" in text
    assert "**Stale:**" in text
    assert model_card._quality_section(str(tmp_path / "x.json"),
                                       str(tmp_path / "y.json")) == []


def test_card_explains_a_site_calibrated_profile(tmp_path):
    ev = tmp_path / "evaluation.json"
    ext = dict(_evaluation()["internal_test"], n=258)
    site = {"split_seed": 20261019, "n_calibration": 258, "n_evaluation": 258,
            "evaluation_half_under_source_profile": {"sensitivity": 0.75,
                                                     "specificity": 0.85, "ece": 0.14},
            "internal_test_under_site_profile": {"sensitivity": 0.97,
                                                 "specificity": 0.6, "ece": 0.12}}
    ev.write_text(json.dumps(_evaluation(external_test=ext, site_calibration=site,
                                         tta="d8")))
    text = model_card.build(str(ev), str(tmp_path / "card.md"))
    assert "### Site calibration" in text
    assert "calibration half of the external site" in text
    assert "| Source profile (no site step) | 0.750 | 0.850 | 0.1400 |" in text
    assert "not plug-and-play" in text
    assert "Test-time augmentation: **d8**" in text
    assert "nothing was refitted" not in text
