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
