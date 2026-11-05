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


def test_dme_section_reports_cv_and_experimental_status(tmp_path):
    ci = {"point": 0.9, "lo": 0.85, "hi": 0.94, "n_boot": 10}
    ev = tmp_path / "dme_eval.json"
    ev.write_text(json.dumps({
        "status": "experimental", "n_train": 514, "checkpoint_sha256": "abc",
        "cross_validation": {"folds": 5, "tta": "d8", "oof": {
            "n": 514, "positives": 243, "prevalence": 0.47, "fn": 20,
            "auroc_ci": ci, "sensitivity_ci": ci, "specificity_ci": ci}}}))
    ck = tmp_path / "dme_model.pt"
    ck.write_bytes(b"other")
    text = "\n".join(model_card._dme_section(str(ev), str(ck)))
    assert "## Maculopathy (experimental)" in text
    assert "| AUROC | 0.900 (95% CI 0.850–0.940) |" in text
    assert "advice only" in text and "never" in text
    assert "**20** of 243" in text and "**Stale:**" in text
    assert model_card._dme_section(str(tmp_path / "none.json")) == []


def test_quality_section_reports_recapture_reasons():
    ci = {"point": 0.9, "lo": 0.85, "hi": 0.95, "n_boot": 10}
    m = {"subscores": {"clarity": {
            "problem": "out of focus or hazy", "fix": "refocus and retake",
            "problem_definition": "clarity 1-6 of 10",
            "test": {"n": 400, "n_problem": 130, "auroc": 0.9, "auroc_ci": ci,
                     "sensitivity": 0.8, "specificity": 0.75, "weighted_kappa": 0.65}}},
         "recapture_reasons_test": {"n_gate_rejected": 70, "named_a_cause": 0.9,
                                    "top_cause_confirmed": 0.85,
                                    "any_named_cause_confirmed": 0.9,
                                    "human_saw_a_subscore_problem": 0.8}}
    text = "\n".join(model_card._recapture_lines(m))
    assert "### Recapture reasons" in text
    assert ("| Out of focus or hazy: refocus and retake | clarity 1-6 of 10 | 130 of 400 "
            "| 0.900 (95% CI 0.850–0.950) | 80.0% / 75.0% | 0.650 |") in text
    assert "For **85.0%** of those the first cause named" in text
    assert "never decide" in text
    assert model_card._recapture_lines({}) == []


def test_recapture_lines_compare_with_the_single_task_model():
    m = {"subscores": {"clarity": {"problem": "out of focus", "fix": "refocus",
                                   "test": {"n": 10, "n_problem": 3}}},
         "validation": {"auroc": 0.83, "catch_rate": 0.31, "false_reject_rate": 0.048,
                        "n_human_accept": 186},
         "test": {"auroc": 0.82, "catch_rate": 0.29, "false_reject_rate": 0.032},
         "single_task_before": {"validation": {"auroc": 0.82, "catch_rate": 0.32,
                                               "false_reject_rate": 0.048},
                                "test": {"auroc": 0.81, "catch_rate": 0.34,
                                         "false_reject_rate": 0.027},
                                "served": {"catch_rate": 0.35, "false_reject_rate": 0.032,
                                           "false_rejects": 6}}}
    text = "\n".join(model_card._recapture_lines(
        m, {"catch_rate": 0.3, "false_reject_rate": 0.043, "false_rejects": 8}))
    assert "| Test AUROC / caught / false rejects | 0.810 / 34.0% / 2.7% | 0.820 / 29.0% / 3.2% |" in text
    assert "| As served: caught / false rejects | 35.0% / 3.2% (6) | 30.0% / 4.3% (8) |" in text
    assert "5% of 186" in text
