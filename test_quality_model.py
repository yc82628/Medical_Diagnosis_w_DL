"""
Learned quality gate: wiring into run_gates, the classical fallback, and the
threshold rule. Torch-free: a stub stands in for the trained model.
"""
import re
from dataclasses import asdict

import cv2
import numpy as np
import pytest

import quality_model as qm
import validate_quality_gate as vq
from predictor import DECISION_MANUAL, DECISION_REFER, run_gates
from test_predictor import _fake_fundus, _Stub


class _Gate:
    """Stands in for LearnedQualityGate: a fixed P(ungradable)."""

    def __init__(self, p, threshold=0.5, seen_files=()):
        self.p, self.threshold, self.calls = p, threshold, 0
        self.seen_files = set(seen_files)

    def assess(self, img):
        self.calls += 1
        ok = self.p < self.threshold
        return {"gradable": ok, "p_ungradable": self.p, "threshold": self.threshold,
                "model": "stub", "reasons": [] if ok else ["learned: too poor"]}


def _use_default(monkeypatch, gate):
    monkeypatch.setitem(qm._default, "loaded", True)
    monkeypatch.setitem(qm._default, "gate", gate)


def test_learned_gate_rejects_an_image_the_classical_checks_pass():
    q = run_gates(_fake_fundus(), quality_gate=_Gate(0.9))
    assert q["quality_source"] == "learned"
    assert q["gradable"] is False and "learned: too poor" in q["reasons"]
    assert q["p_ungradable"] == 0.9 and q["quality_threshold"] == 0.5


def test_classical_backstop_still_rejects_when_learned_gate_passes():
    dark = (_fake_fundus() * 0.03).astype(np.uint8)
    q = run_gates(dark, quality_gate=_Gate(0.01))
    assert q["quality_source"] == "learned"
    assert q["gradable"] is False
    assert any("underexposed" in r for r in q["reasons"])


def test_learned_gate_passes_a_good_image():
    q = run_gates(_fake_fundus(), quality_gate=_Gate(0.1))
    assert q["gradable"] is True and q["reasons"] == []


def test_no_checkpoint_falls_back_to_classical():
    # conftest points DR_QUALITY_CHECKPOINT at a file that does not exist.
    assert qm.default_quality_gate() is None
    assert "classical" in qm.default_quality_note()
    q = run_gates(_fake_fundus())
    assert q["quality_source"] == "classical" and q["gradable"] is True
    assert "p_ungradable" not in q


def test_service_path_uses_learned_gate_when_present(monkeypatch):
    gate = _Gate(0.95)
    _use_default(monkeypatch, gate)
    p = _Stub(0.9, operating_point=0.5)
    out = p.predict(_fake_fundus())
    assert gate.calls == 1
    assert out.decision == DECISION_MANUAL and not p.called
    assert out.quality["quality_source"] == "learned"

    _use_default(monkeypatch, _Gate(0.05))
    assert _Stub(0.9, operating_point=0.5).predict(_fake_fundus()).decision == DECISION_REFER


def test_threshold_caps_false_rejects_on_accepted_images():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200, bool), np.ones(200, bool)]
    p = np.r_[rng.uniform(0, 0.7, 200), rng.uniform(0.3, 1, 200)]
    t = qm.choose_threshold(y, p, max_false_reject=0.05)
    m = qm.gate_metrics(y, p, t)
    assert m["false_rejects"] <= 10 and m["false_reject_rate"] <= 0.05
    # Lowest such threshold: one step lower would exceed the cap.
    below = p[~y][p[~y] < t].max()
    assert (p[~y] >= below).sum() > 10
    assert 0 < m["catch_rate"] < 1 and m["auroc"] > 0.5


def test_split_is_by_patient():
    files = [f"Images/{pid}/{pid}_{e}{k}.jpg" for pid in range(60)
             for e in "lr" for k in (1, 2)]
    y = np.arange(len(files)) % 3 == 0
    masks = qm.split_patients(files, y, seed=1)
    pats = {k: {qm.patient_of(f) for f, b in zip(files, m) if b} for k, m in masks.items()}
    assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"]
                or pats["val"] & pats["test"])
    assert sum(m.sum() for m in masks.values()) == len(files)
    assert qm.patient_of("regular-fundus-training/Images/347/347_l1.jpg") == "347"


def test_validator_excludes_images_the_learned_gate_trained_on(tmp_path, monkeypatch):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for n in ("a.png", "b.png", "c.png"):
        cv2.imwrite(str(imgs / n), _fake_fundus(seed=len(n)))
    csv = tmp_path / "labels.csv"
    csv.write_text("image,quality\na.png,0\nb.png,1\nc.png,1\n")
    _use_default(monkeypatch, _Gate(0.9, seen_files={"a.png"}))

    rep = vq.main(["--csv", str(csv), "--images", str(imgs), "--reject-values", "0",
                   "--out", str(tmp_path / "out.json")])
    assert rep["quality_source"] == "learned"
    assert rep["excluded_seen_by_model"] == 1 and rep["n_images"] == 2
    assert rep["false_reject_rate"] == 1.0          # the stub rejects everything

    rep = vq.main(["--csv", str(csv), "--images", str(imgs), "--reject-values", "0",
                   "--gate", "classical", "--out", str(tmp_path / "out2.json")])
    assert rep["quality_source"] == "classical" and rep["n_images"] == 3


# ---------------------------------------------------------------------------
# Recapture reasons: sub-score heads (clarity, field definition, artefact)
# ---------------------------------------------------------------------------

SPECS = {s.key: s for s in qm.SUBSCORES}


def _logit(p):
    return float(np.log(p / (1 - p)))


def _sub_logits(p_better):
    """{key: P(level > k) for each cumulative logit} -> flat logit vector."""
    return np.array([_logit(v) for s in qm.SUBSCORES for v in p_better[s.key]])


def _subs(**p_problem):
    """Sub-score dict as LearnedQualityGate reports it, threshold 0.5 each."""
    return {s.key: {"p_problem": p_problem.get(s.key, 0.1), "threshold": 0.5,
                    "flagged": p_problem.get(s.key, 0.1) >= 0.5, "level": s.levels[-1],
                    "problem": s.problem, "fix": s.fix} for s in qm.SUBSCORES}


class _SubGate(_Gate):
    """A stub with sub-score heads, through the same assessment() as the real gate."""

    def __init__(self, p, subscores, threshold=0.5):
        super().__init__(p, threshold)
        self.subscores = qm.SUBSCORES
        self.subs = subscores

    def assess(self, img):
        self.calls += 1
        return qm.assessment(self.p, self.threshold, "stub", self.subs)


def test_rejection_names_the_cause_and_what_to_do():
    q = run_gates(_fake_fundus(), quality_gate=_SubGate(0.9, _subs(clarity=0.8, artifact=0.6)))
    assert q["gradable"] is False
    # Most likely cause first, each with its fix.
    assert q["recapture"][0].startswith("out of focus or hazy: refocus")
    assert q["recapture"][1].startswith("artefact such as an eyelash")
    assert len(q["recapture"]) == 2                       # off-centre was not flagged
    assert all(r in q["reasons"] for r in q["recapture"])
    assert q["quality_subscores"]["clarity"]["flagged"] is True
    assert q["quality_subscores"]["field_definition"]["flagged"] is False


def test_reasons_have_no_nested_brackets():
    # The web page strips "(...)" for its plain view; nesting would leave debris.
    q = run_gates(_fake_fundus(), quality_gate=_SubGate(
        0.9, _subs(clarity=0.9, field_definition=0.9, artifact=0.9)))
    plain = re.compile(r"\s*\([^)]*\)")
    for r in q["reasons"]:
        depth = 0
        for ch in r:
            depth += (ch == "(") - (ch == ")")
            assert 0 <= depth <= 1, r
        assert "(" not in plain.sub("", r) and ")" not in plain.sub("", r)
    for s in qm.SUBSCORES:
        assert "(" not in s.problem + s.fix


def test_no_flagged_cause_says_so_instead_of_guessing():
    q = run_gates(_fake_fundus(), quality_gate=_SubGate(0.9, _subs()))
    assert q["gradable"] is False and q["recapture"] == [qm.NO_SINGLE_CAUSE]


def test_subscores_never_decide_gradability():
    # Every problem flagged, but P(ungradable) is under the threshold: passes.
    q = run_gates(_fake_fundus(), quality_gate=_SubGate(
        0.2, _subs(clarity=0.99, field_definition=0.99, artifact=0.99)))
    assert q["gradable"] is True and q["reasons"] == [] and q["recapture"] == []
    assert q["quality_subscores"]["clarity"]["flagged"] is True   # still reported


def test_gate_without_subscore_heads_still_works():
    q = run_gates(_fake_fundus(), quality_gate=_Gate(0.9))
    assert q["quality_subscores"] == {} and q["recapture"] == []


def test_label_values_map_to_ordinal_levels_worst_first():
    c, a = SPECS["clarity"], SPECS["artifact"]
    assert list(qm.subscore_targets([1, 6, 10, None, float("nan"), 7], c)) == [0, 2, 4, -1, -1, -1]
    # Artefact: a higher raw value is WORSE, so 10 is level 0.
    assert list(qm.subscore_targets([10, 6, 4, 0], a)) == [0, 2, 3, 5]
    assert a.levels[:a.problem_levels] == (10, 8, 6)          # over a quarter covered
    assert c.levels[:c.problem_levels] == (1, 4, 6)
    assert SPECS["field_definition"].levels[:3] == (1, 4, 6)  # not within 2 PD of centre


def test_cumulative_logits_give_problem_probability_and_level():
    # clarity: P(level > k) for k = 0..3, so P(level < 3) = 1 - P(level > 2) = 0.7
    lg = _sub_logits({"clarity": [0.9, 0.8, 0.3, 0.1],
                      "field_definition": [0.99, 0.99, 0.9, 0.6],
                      "artifact": [0.99, 0.9, 0.2, 0.1, 0.1]})
    parts = qm.split_sub_logits(lg, qm.SUBSCORES)
    assert np.isclose(qm.problem_probability(parts["clarity"], SPECS["clarity"])[0], 0.7)
    assert qm.predicted_level(parts["clarity"], SPECS["clarity"])[0] == 2       # raw 6
    d = qm.describe_subscores(lg, qm.SUBSCORES,
                              {"clarity": 0.5, "field_definition": 0.5, "artifact": 0.5})
    assert d["clarity"]["flagged"] and d["clarity"]["level"] == 6
    assert not d["field_definition"]["flagged"] and d["field_definition"]["level"] == 10
    assert d["artifact"]["flagged"] and d["artifact"]["level"] == 6
    with pytest.raises(ValueError):
        qm.split_sub_logits(lg[:-1], qm.SUBSCORES)


def test_subscore_metrics_on_perfect_and_missing_labels():
    s = SPECS["clarity"]
    t = np.array([0, 1, 2, 3, 4, -1, 4, 2])
    lg = np.array([[_logit(0.99 if lv > k else 0.01) for k in range(s.n_logits)]
                   for lv in np.maximum(t, 0)])
    m = qm.subscore_metrics(t, lg, s, threshold=0.5)
    assert m["n"] == 7 and m["n_problem"] == 4             # the -1 row is left out
    assert m["auroc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert m["weighted_kappa"] == 1.0 and m["exact_agreement"] == 1.0


def test_problem_threshold_chosen_by_youden():
    y = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    p = np.r_[np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)]
    t = qm.choose_problem_threshold(y, p)
    assert 0.4 < t <= 0.6
    assert qm.choose_problem_threshold(np.ones(5, bool), np.ones(5)) == 0.5


def test_checkpoint_subscores_round_trip_and_old_checkpoints():
    assert qm.subscores_from_checkpoint({}) == ((), {})
    stored = [{**asdict(s), "threshold": 0.3 + i / 10, "fix": "old wording"}
              for i, s in enumerate(qm.SUBSCORES)]
    specs, thr = qm.subscores_from_checkpoint({"subscores": stored})
    assert [s.key for s in specs] == [s.key for s in qm.SUBSCORES]
    assert specs[0].levels == qm.SUBSCORES[0].levels
    assert specs[0].fix == qm.SUBSCORES[0].fix                # wording from the code
    assert thr == pytest.approx({"clarity": 0.3, "field_definition": 0.4, "artifact": 0.5})


def test_reason_agreement_counts_confirmed_causes():
    blurred = _sub_logits({"clarity": [0.9, 0.5, 0.1, 0.05],
                           "field_definition": [0.99] * 4, "artifact": [0.99] * 5})
    fine = _sub_logits({"clarity": [0.99] * 4, "field_definition": [0.99] * 4,
                        "artifact": [0.99] * 5})
    logits = np.stack([blurred, blurred, fine, blurred])
    # Human level indices, worst first: clarity, field definition, artefact.
    human = np.array([[1, 4, 5], [4, 4, 5], [4, 4, 5], [0, -1, 5]])
    thr = {s.key: 0.5 for s in qm.SUBSCORES}
    r = qm.reason_agreement([True, True, True, False], logits, human, qm.SUBSCORES, thr)
    assert r["n_gate_rejected"] == 3
    assert np.isclose(r["named_a_cause"], 2 / 3)           # the third image: no cause
    assert r["top_cause_confirmed"] == 0.5                 # first image yes, second no
    assert r["per_cause_named"] == {"clarity": 2, "field_definition": 0, "artifact": 0}


def test_prepare_deepdrid_adds_subscores_and_keeps_columns(tmp_path):
    import pandas as pd

    import prepare_deepdrid as pdd

    root = tmp_path / "repo"
    img = _fake_fundus(64)

    def put(folder, image_id):
        d = root / folder / "Images" / image_id.split("_")[0]
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{image_id}.jpg"), img)

    for split, folder, pid in (("training", "regular-fundus-training", 1),
                               ("validation", "regular-fundus-validation", 2)):
        rows = []
        for k, (q, c, f, a) in enumerate([(1, 10, 8, 0), (0, 4, 6, 8)]):
            iid = f"{pid}_l{k + 1}"
            put(folder, iid)
            rows.append({"patient_id": pid, "image_id": iid, "image_path": "x",
                         "Overall quality": q, "left_eye_DR_Level": 1,
                         "right_eye_DR_Level": None, "patient_DR_Level": 1,
                         "Clarity": c, "Field definition": f, "Artifact": a})
        pd.DataFrame(rows).to_csv(root / folder / f"{folder}.csv", index=False)
        pd.DataFrame({"image_id": [r["image_id"] for r in rows], "Source": ["A", "B"]}).to_csv(
            root / folder / f"regular-fundus-source-{split}.csv", index=False)
    ev = root / pdd.EVAL_DIR
    put(pdd.EVAL_DIR, "3_r1")
    pd.DataFrame({"image_id": ["3_r1"], "DR_Levels": [5]}).to_excel(
        ev / "Challenge1_labels.xlsx", index=False)
    pd.DataFrame({"image_id": ["3_r1"], "Overall quality": [1], "Artifact": [1],
                  "Clarity": [6], "Field definition": [10]}).to_excel(
        ev / "Challenge2_labels.xlsx", index=False)

    pdd.main(["--root", str(root), "--out-dir", str(tmp_path / "out"), "--no-env"])
    ql = pd.read_csv(tmp_path / "out" / "quality_labels.csv")
    assert list(ql.columns) == ["filename", "quality", "clarity", "field_definition", "artifact"]
    assert ql["quality"].tolist() == [1, 0, 1, 0, 0]       # grade 5 counts as rejected
    assert ql["clarity"].tolist() == [10, 4, 10, 4, 6]
    assert ql["artifact"].tolist() == [0, 8, 0, 8, 1]
    assert ql["field_definition"].tolist() == [8, 6, 8, 6, 10]
    labels = pd.read_csv(tmp_path / "out" / "labels.csv")
    assert list(labels.columns) == ["filename", "diagnosis", "patient_id", "eye", "site",
                                    "human_quality"]


# ---------------------------------------------------------------------------
# Two models: the gate decides, the explainer only names the cause
# ---------------------------------------------------------------------------

class _PGate(_Gate):
    """A gate stub with the attributes ExplainedQualityGate reads."""

    def __init__(self, p, threshold=0.5, seen_files=(), version="gate@1"):
        super().__init__(p, threshold, seen_files)
        self.version, self.subscores = version, ()

    def p_ungradable(self, img):
        self.calls += 1
        return self.p


class _Explainer(_SubGate):
    def __init__(self, subscores, p=0.0, seen_files=()):
        super().__init__(p, subscores)
        self.version, self.seen_files = "explainer@2", set(seen_files)


def test_explained_gate_decides_with_the_gate_and_explains_with_the_other():
    # The explainer's own P(ungradable) (0.0 here) is never used: the gate rejects.
    ex = _Explainer(_subs(clarity=0.8))
    g = qm.ExplainedQualityGate(_PGate(0.9, seen_files={"a"}), ex)
    q = run_gates(_fake_fundus(), quality_gate=g)
    assert q["gradable"] is False and q["p_ungradable"] == 0.9
    assert q["quality_model"] == "gate@1"
    assert q["recapture"][0].startswith("out of focus or hazy: refocus")
    assert ex.calls == 1


def test_explainer_does_not_run_on_an_accepted_image():
    # Accepted images cost one forward pass, as before: older CPUs.
    ex = _Explainer(_subs(clarity=0.99, artifact=0.99), p=0.99)
    q = run_gates(_fake_fundus(), quality_gate=qm.ExplainedQualityGate(_PGate(0.1), ex))
    assert q["gradable"] is True and q["recapture"] == [] and q["quality_subscores"] == {}
    assert ex.calls == 0


def test_explained_gate_leaves_out_images_either_model_saw():
    g = qm.ExplainedQualityGate(_PGate(0.9, seen_files={"a"}),
                                _Explainer(_subs(), seen_files={"b"}))
    assert g.seen_files == {"a", "b"} and g.threshold == 0.5


def test_default_gate_pairs_with_an_explainer_only_when_it_has_heads(tmp_path, monkeypatch):
    explainer_path = tmp_path / "explainer.pt"
    explainer_path.write_bytes(b"x")
    monkeypatch.setenv("DR_QUALITY_EXPLAINER", str(explainer_path))
    gate = _PGate(0.9)

    monkeypatch.setattr(qm, "LearnedQualityGate", lambda path: _Explainer(_subs()))
    paired, note = qm._with_explainer(gate)
    assert isinstance(paired, qm.ExplainedQualityGate) and paired.gate is gate
    assert "never decides" in note

    headless = _Explainer(_subs())
    headless.subscores = ()
    monkeypatch.setattr(qm, "LearnedQualityGate", lambda path: headless)
    assert qm._with_explainer(gate)[0] is gate

    explainer_path.unlink()
    assert qm._with_explainer(gate) == (gate, "")
