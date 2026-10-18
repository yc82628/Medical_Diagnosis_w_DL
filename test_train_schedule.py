"""The OneCycleLR warm-up never divides by zero, whatever the run length."""
import pytest

torch = pytest.importorskip("torch")

from train import one_cycle_pct_start  # noqa: E402


@pytest.mark.parametrize("steps_per_epoch, epochs, warmup", [
    (14, 2, 2),        # warm-up = whole run: crashed before the fix
    (5, 2, 2),
    (10, 1, 0.1),      # warm-up = exactly 1 step: crashed before the fix
    (3, 30, 2),        # APTOS-like: many epochs, few steps each
    (229, 30, 2),      # APTOS at batch 8, accumulation 2
    (2, 2, 0),
    (4, 1, 1),
])
def test_scheduler_runs_to_completion(steps_per_epoch, epochs, warmup):
    total = steps_per_epoch * epochs
    pct = one_cycle_pct_start(total, warmup, epochs)
    assert pct is not None and 0 < pct < 1
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=total,
                                                pct_start=pct)
    for _ in range(total):
        opt.step()
        sched.step()


def test_too_few_steps_means_no_schedule():
    assert one_cycle_pct_start(3, 2, 3) is None
