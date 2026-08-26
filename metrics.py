from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    cohen_kappa_score, confusion_matrix,
)


# Operating point
def threshold_at_sensitivity(y_true, y_score, target_sensitivity=0.90) -> float:
    """
    Lowest-specificity-cost threshold achieving AT LEAST target sensitivity.

    Chosen on the VALIDATION set and then frozen. Choosing it on test is a
    subtle but complete invalidation of the reported operating point.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target_sensitivity)[0]
    if len(ok) == 0:
        return float(thr[np.argmax(tpr)])
    return float(thr[ok[np.argmin(fpr[ok])]])


def screening_metrics(y_true, y_score, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    out = {
        "n": len(y_true),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if 0 < y_true.sum() < len(y_true):
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


# Patient-level bootstrap
def bootstrap_ci(y_true, y_score, patient_ids, metric_fn, n_boot=1000,
                 alpha=0.05, seed=0) -> dict:
    """
    Percentile CI, resampling PATIENTS with replacement.

    metric_fn(y_true, y_score) -> float
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    patient_ids = np.asarray(patient_ids)

    uniq = np.unique(patient_ids)
    idx_by_patient = {p: np.where(patient_ids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)

    stats = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_patient[p] for p in chosen])
        try:
            v = metric_fn(y_true[idx], y_score[idx])
            if np.isfinite(v):
                stats.append(v)
        except ValueError:            # e.g. a resample with one class only
            continue

    if not stats:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    stats = np.array(stats)
    return {
        "point": float(metric_fn(y_true, y_score)),
        "lo": float(np.percentile(stats, 100 * alpha / 2)),
        "hi": float(np.percentile(stats, 100 * (1 - alpha / 2))),
        "n_boot": len(stats),
    }


def spec_at_sens_fn(target_sensitivity=0.90):
    """Metric function for bootstrapping: re-derives the threshold within each resample."""
    def _f(y_true, y_score):
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            raise ValueError("single class")
        t = threshold_at_sensitivity(y_true, y_score, target_sensitivity)
        m = screening_metrics(y_true, y_score, t)
        return m["specificity"]
    return _f


# Calibration
def expected_calibration_error(y_true, y_prob, n_bins=15) -> float:
    """
    Weighted mean gap between confidence and accuracy.

    A fine-tuned CNN is typically badly overconfident. If the interface says
    "82% confident" to a clinician, that number must mean something, or you are
    manufacturing false reassurance.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def reliability_table(y_true, y_prob, n_bins=10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        rows.append({
            "bin": f"({lo:.1f}, {hi:.1f}]",
            "n": int(m.sum()),
            "mean_predicted": float(y_prob[m].mean()) if m.sum() else np.nan,
            "observed_rate": float(y_true[m].mean()) if m.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def fit_temperature(logits, y_true, lo=0.05, hi=10.0, n=400) -> float:
    """
    Temperature scaling by 1-D grid + golden refinement on NLL.

    Single parameter fitted on VALIDATION. It cannot change the ranking of
    predictions, so AUROC is untouched; it only fixes the probability scale.
    """
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y_true, dtype=float)

    def nll(T):
        z = np.clip(logits / T, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    grid = np.linspace(lo, hi, n)
    best = grid[int(np.argmin([nll(t) for t in grid]))]
    step = (hi - lo) / n
    fine = np.linspace(max(lo, best - step), min(hi, best + step), 50)
    return float(fine[int(np.argmin([nll(t) for t in fine]))])


def apply_temperature(logits, T: float) -> np.ndarray:
    z = np.clip(np.asarray(logits, dtype=float) / T, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class Calibrator:
    def __init__(self):
        self.method = None
        self.temperature = None
        self._iso = None

    def fit(self, logits, y_true, seed=0, holdout_frac=0.3):
        from sklearn.isotonic import IsotonicRegression

        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y_true, dtype=float)

        rng = np.random.default_rng(seed)
        fit_mask = rng.random(len(y)) >= holdout_frac
        ho = ~fit_mask
        # Degenerate holdout -> fall back to fitting on everything with temperature.
        if ho.sum() < 20 or y[ho].sum() == 0 or y[fit_mask].sum() == 0:
            self.method = "temperature"
            self.temperature = fit_temperature(logits, y)
            return self

        T = fit_temperature(logits[fit_mask], y[fit_mask])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(logits[fit_mask], y[fit_mask])

        candidates = {
            "identity":    1 / (1 + np.exp(-np.clip(logits[ho], -30, 30))),
            "temperature": apply_temperature(logits[ho], T),
            "isotonic":    iso.predict(logits[ho]),
        }
        eces = {k: expected_calibration_error(y[ho], v) for k, v in candidates.items()}
        self.method = min(eces, key=eces.get)
        self.temperature, self._iso, self.holdout_ece = T, iso, eces

        print("  Calibration: held-out ECE " +
              ", ".join(f"{k}={v:.4f}" for k, v in eces.items()) +
              f"  -> selected '{self.method}'")
        if self.method == "identity":
            print("    (model was already well calibrated; leaving it alone)")
        return self

    def predict(self, logits) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        if self.method == "isotonic":
            return self._iso.predict(logits)
        if self.method == "temperature":
            return apply_temperature(logits, self.temperature)
        return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

    def to_dict(self) -> dict:
        d = {"method": self.method, "temperature": self.temperature}
        if self._iso is not None:
            d["isotonic_x"] = self._iso.X_thresholds_.tolist()
            d["isotonic_y"] = self._iso.y_thresholds_.tolist()
        return d


# Grade-level metrics
def quadratic_weighted_kappa(y_true, y_pred, num_grades=5) -> float:
    """
    The standard DR grading metric. Penalises distant disagreements
    quadratically, matching the clinical reality that 4-vs-0 is a catastrophe
    and 3-vs-4 is a quibble.
    """
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                   labels=list(range(num_grades))))


# Abstention
def _logit(p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def abstention_bands(y_prob, threshold, coverage: float) -> tuple[float, float]:
    """
    Two thresholds (rule-out, refer) that leave exactly `coverage` of cases
    auto-decided, abstaining on those nearest the decision boundary.

    Distance is measured in LOGIT space, not probability space. A symmetric
    probability band around a threshold of, say, 0.12 is deeply asymmetric in
    practice -- it can swallow every negative while touching no positives.
    Logit space is the scale the decision actually lives on.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    d = np.abs(_logit(y_prob) - _logit(threshold))
    if coverage >= 1.0:
        return threshold, threshold
    cut = np.quantile(d, 1.0 - coverage)             # abstain below this distance
    lt = _logit(threshold)
    return (
        float(1 / (1 + np.exp(-(lt - cut)))),        # below -> confident rule-out
        float(1 / (1 + np.exp(-(lt + cut)))),        # above -> confident refer
    )


def abstention_curve(y_true, y_prob, threshold, coverages=None) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    coverages = coverages or [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]

    rows = []
    for cov in coverages:
        lo, hi = abstention_bands(y_prob, threshold, cov)
        keep = (y_prob <= lo) | (y_prob >= hi)
        row = {"target_coverage": cov, "actual_coverage": float(keep.mean()),
               "n_retained": int(keep.sum()), "n_referred_to_human": int((~keep).sum())}
        if keep.sum() and 0 < y_true[keep].sum() < keep.sum():
            m = screening_metrics(y_true[keep], y_prob[keep], threshold)
            row.update({"sensitivity": m["sensitivity"], "specificity": m["specificity"],
                        "missed": m["fn"]})
        else:
            row.update({"sensitivity": np.nan, "specificity": np.nan, "missed": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


# Subgroups
def subgroup_report(df: pd.DataFrame, y_col, score_col, threshold,
                    subgroup_cols, min_n=30) -> pd.DataFrame:
    rows = []
    for col in subgroup_cols:
        if col not in df.columns:
            continue
        for val, g in df.groupby(col):
            if len(g) == 0:
                continue
            m = screening_metrics(g[y_col], g[score_col], threshold)
            rows.append({
                "attribute": col, "group": str(val), "n": len(g),
                "n_positive": int(np.asarray(g[y_col]).sum()),
                "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                "auroc": m["auroc"],
                "underpowered": len(g) < min_n or m["tp"] + m["fn"] < 10,
            })
    return pd.DataFrame(rows)


# Full report
def full_report(y_true, y_score, patient_ids, threshold,
                grades_true=None, grades_pred=None, n_boot=1000) -> dict:
    rep = screening_metrics(y_true, y_score, threshold)
    rep["ece"] = expected_calibration_error(y_true, y_score)

    rep["auroc_ci"] = bootstrap_ci(
        y_true, y_score, patient_ids,
        lambda a, b: roc_auc_score(a, b), n_boot=n_boot,
    )
    rep["spec_at_sens_ci"] = bootstrap_ci(
        y_true, y_score, patient_ids, spec_at_sens_fn(0.90), n_boot=n_boot,
    )
    if grades_true is not None and grades_pred is not None:
        rep["qwk"] = quadratic_weighted_kappa(grades_true, grades_pred)
    return rep


def print_report(rep: dict, title="Screening performance") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  n = {rep['n']}   prevalence = {100*rep['prevalence']:.1f}%   "
          f"threshold = {rep['threshold']:.4f}")
    a = rep.get("auroc_ci")
    if a:
        print(f"  AUROC              {a['point']:.3f}  (95% CI {a['lo']:.3f}-{a['hi']:.3f})")
    else:
        print(f"  AUROC              {rep['auroc']:.3f}")
    print(f"  Sensitivity        {rep['sensitivity']:.3f}")
    print(f"  Specificity        {rep['specificity']:.3f}")
    s = rep.get("spec_at_sens_ci")
    if s:
        print(f"    spec@sens90 CI   {s['lo']:.3f}-{s['hi']:.3f}")
    print(f"  PPV / NPV          {rep['ppv']:.3f} / {rep['npv']:.3f}")
    print(f"  ECE                {rep['ece']:.4f}")
    if "qwk" in rep:
        print(f"  Quadratic kappa    {rep['qwk']:.3f}")
    print(f"  Confusion          TP={rep['tp']} FP={rep['fp']} "
          f"TN={rep['tn']} FN={rep['fn']}")
    print(f"  -> {rep['fn']} referable cases missed at this operating point.")
