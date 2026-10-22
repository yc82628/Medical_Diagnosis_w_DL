"""
How many images does a hospital study need?

    python research/sample_size.py
    python research/sample_size.py --sensitivity 0.90 --floor 0.85 --prevalence 0.15

Two questions, answered with exact binomial (Clopper-Pearson) intervals rather
than the normal approximation, which is too optimistic near 90%:

  1. PRECISION: how many REFERABLE cases give a 95% CI no wider than
     +-half_width around the expected sensitivity?
  2. SUCCESS CRITERION: how many referable cases give `power` probability that
     the lower 95% confidence bound clears `floor`, if the true sensitivity is
     `sensitivity`? (The study's pre-specified pass/fail criterion.)

Specificity is computed the same way on the NON-referable cases. The total
number of images then follows from the expected prevalence of referable DR in
the hospital's screening population (typically 8-25%), and from the expected
share of ungradable images, which do not count towards either.
"""
from __future__ import annotations

import argparse
import math

from scipy.stats import beta, binom


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def n_for_precision(p: float, half_width: float, alpha: float = 0.05, n_max: int = 20000) -> int:
    """Smallest n whose exact CI at the expected count is within +-half_width."""
    for n in range(10, n_max):
        k = round(p * n)
        lo, hi = clopper_pearson(k, n, alpha)
        if hi - lo <= 2 * half_width:
            return n
    raise ValueError("half_width too small")


def power_lower_bound(p: float, floor: float, n: int, alpha: float = 0.05) -> float:
    """P(lower exact 95% bound > floor) when the true proportion is p."""
    # The lower bound is monotone in k: find the smallest k that clears the floor.
    k_min = next((k for k in range(n + 1) if clopper_pearson(k, n, alpha)[0] > floor), None)
    if k_min is None:
        return 0.0
    return float(binom.sf(k_min - 1, n, p))


def n_for_power(p: float, floor: float, power: float = 0.8, alpha: float = 0.05,
                n_max: int = 20000) -> int:
    for n in range(10, n_max):
        if power_lower_bound(p, floor, n, alpha) >= power:
            return n
    raise ValueError("not reachable: p too close to the floor")


def plan(sensitivity=0.90, sens_floor=0.80, specificity=0.85, spec_floor=0.75,
         prevalence=0.15, ungradable=0.10, power=0.80, half_width=0.05) -> dict:
    pos = max(n_for_power(sensitivity, sens_floor, power),
              n_for_precision(sensitivity, half_width))
    neg = max(n_for_power(specificity, spec_floor, power),
              n_for_precision(specificity, half_width))
    gradable = max(math.ceil(pos / prevalence), math.ceil(neg / (1 - prevalence)))
    total = math.ceil(gradable / (1 - ungradable))
    return {"referable_needed": pos, "non_referable_needed": neg,
            "gradable_images": gradable, "total_images": total,
            "inputs": dict(sensitivity=sensitivity, sens_floor=sens_floor,
                           specificity=specificity, spec_floor=spec_floor,
                           prevalence=prevalence, ungradable=ungradable,
                           power=power, half_width=half_width)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sensitivity", type=float, default=0.90, help="expected")
    ap.add_argument("--floor", type=float, default=0.80, help="lower CI bound must clear this")
    ap.add_argument("--specificity", type=float, default=0.85)
    ap.add_argument("--spec-floor", type=float, default=0.75)
    ap.add_argument("--prevalence", type=float, default=0.15)
    ap.add_argument("--ungradable", type=float, default=0.10)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--half-width", type=float, default=0.05)
    a = ap.parse_args(argv)
    r = plan(a.sensitivity, a.floor, a.specificity, a.spec_floor, a.prevalence,
             a.ungradable, a.power, a.half_width)
    print(f"  Referable cases needed:      {r['referable_needed']}")
    print(f"  Non-referable cases needed:  {r['non_referable_needed']}")
    print(f"  Gradable images at {a.prevalence:.0%} prevalence: {r['gradable_images']}")
    print(f"  Total images incl. {a.ungradable:.0%} ungradable: {r['total_images']}")
    print("\n  Prevalence is the big lever:")
    for prev in (0.08, 0.15, 0.25):
        t = plan(a.sensitivity, a.floor, a.specificity, a.spec_floor, prev, a.ungradable,
                 a.power, a.half_width)["total_images"]
        print(f"    {prev:.0%} referable -> {t} images")
    return r


if __name__ == "__main__":
    main()
