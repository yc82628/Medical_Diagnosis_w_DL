from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import CONFIG, Config, NUM_GRADES
from dataset import get_dataloaders, pos_weights
from model import (DRModel, OrdinalLoss, save_checkpoint, param_groups,
                   get_device, count_parameters, tta_logits)
from metrics import (threshold_at_sensitivity, screening_metrics,
                     quadratic_weighted_kappa)
from splits import enforce_monotonic, grades_from_cumulative


def one_cycle_pct_start(total_steps: int, warmup_epochs: float,
                        num_epochs: int) -> float | None:
    """
    pct_start for OneCycleLR that cannot divide by zero, or None for "too few
    steps to schedule".

    OneCycleLR's warm-up phase ends at step pct_start*total_steps - 1 and its
    annealing phase at total_steps - 1; it divides by each phase's length. So
    it raises ZeroDivisionError mid-training when the warm-up covers all of
    training (warmup_epochs >= num_epochs, e.g. a 2-epoch smoke run) or comes
    to exactly one step. Round the warm-up to whole steps, at least 2, and
    leave at least 2 steps to anneal.
    """
    if total_steps < 4:
        return None
    frac = warmup_epochs / max(num_epochs, 1)
    warm = int(round(total_steps * frac))
    warm = min(max(warm, 2), total_steps - 2)
    return warm / total_steps


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and torch. Best effort: cuDNN kernels and
    albumentations' own generators are not pinned, so runs still vary."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Epoch loops
def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    accum_steps=1, scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["ordinal"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * accum_steps * images.size(0)
        n += images.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, tta: str = "none") -> dict:
    """
    Returns raw referable logits plus cumulative probabilities and true labels.
    Runs in fp32: some logits reach -700, which overflows fp16 autocast.
    """
    model.eval()
    logits_all, y_all, g_all = [], [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = tta_logits(model, images, tta).cpu().numpy()
        logits_all.append(logits)
        y_all.append(batch["referable"].numpy())
        g_all.append(batch["grade"].numpy())

    logits = np.concatenate(logits_all)
    cum = enforce_monotonic(1 / (1 + np.exp(-np.clip(logits, -30, 30))))
    return {
        "logits": logits,                       # B x (K-1) raw
        "referable_logit": logits[:, 1],        # the screening decision
        "cumulative": cum,
        "grades_pred": grades_from_cumulative(cum),
        "y_referable": np.concatenate(y_all),
        "grades_true": np.concatenate(g_all),
    }

# Main
def main(cfg: Config = CONFIG):
    tc = cfg.train
    print("=" * 68)
    print("  Diabetic Retinopathy Screening -- training")
    print("  RESEARCH USE ONLY. Not a medical device. See INTENDED_USE.md")
    print("=" * 68)

    device = get_device()
    if tc.seed is not None:
        seed_everything(tc.seed)
    print(f"  Augmentation: {tc.augment}   seed: {tc.seed}")
    loaders, splits, df = get_dataloaders(cfg)

    print("\n  Building model...")
    model = DRModel(tc.backbone, NUM_GRADES, tc.dropout).to(device)
    count_parameters(model)

    pw = pos_weights(splits["train"]).to(device)
    print(f"  Ordinal pos_weights per threshold: "
          f"{[f'{v:.1f}' for v in pw.tolist()]}")
    # Upweight k=1: that threshold IS the clinical decision.
    thr_w = torch.tensor([1.0, 2.0, 1.0, 1.0], device=device)[:NUM_GRADES - 1]
    criterion = OrdinalLoss(pos_weight=pw, threshold_weight=thr_w).to(device)

    optimizer = torch.optim.AdamW(
        param_groups(model, tc.learning_rate, tc.backbone_lr_mult, tc.weight_decay)
    )
    steps_per_epoch = max(1, len(loaders["train"]) // tc.grad_accum_steps)
    total_steps = steps_per_epoch * tc.num_epochs
    pct = one_cycle_pct_start(total_steps, tc.warmup_epochs, tc.num_epochs)
    scheduler = None
    if pct is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            total_steps=total_steps, pct_start=pct,
        )
    else:
        print(f"  Only {total_steps} optimiser steps in total: too few for a "
              f"warm-up schedule, training at a constant learning rate.")
    use_amp = tc.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best, best_epoch, history = -np.inf, -1, []
    for epoch in range(1, tc.num_epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loaders["train"], criterion, optimizer,
                               scaler, device, tc.grad_accum_steps, scheduler)

        val = predict(model, loaders["val"], device)
        thr = threshold_at_sensitivity(val["y_referable"], val["referable_logit"],
                                       tc.target_sensitivity)
        m = screening_metrics(val["y_referable"], val["referable_logit"], thr)
        qwk = quadratic_weighted_kappa(val["grades_true"], val["grades_pred"])

        score = m["specificity"] if np.isfinite(m["specificity"]) else -np.inf
        history.append({"epoch": epoch, "train_loss": loss, "val_auroc": m["auroc"],
                        "val_spec_at_sens": score, "val_qwk": qwk,
                        "val_missed": m["fn"]})

        flag = ""
        if score > best:
            best, best_epoch = score, epoch
            # thr was chosen on RAW LOGITS. The predictor compares against a
            # probability, so store both, each named for its scale. (Older
            # checkpoints stored the logit as `val_threshold`; the predictor
            # still reads that key and converts it.)
            save_checkpoint(tc.save_path, model, cfg.preprocess,
                            extra={"epoch": epoch, "val_metrics": m,
                                   "augment": tc.augment, "seed": tc.seed,
                                   "val_threshold_logit": float(thr),
                                   "val_threshold_prob": float(1 / (1 + np.exp(-np.clip(thr, -30, 30)))),
                                   "history": history})
            flag = "  <- saved"

        print(f"  epoch {epoch:>3}/{tc.num_epochs}  loss {loss:.4f}  "
              f"AUROC {m['auroc']:.3f}  spec@sens{int(100*tc.target_sensitivity)} "
              f"{score:.3f}  QWK {qwk:.3f}  missed {m['fn']}  "
              f"({time.time()-t0:.0f}s){flag}")

        if epoch - best_epoch >= tc.patience:
            print(f"\n  Early stop: no improvement in {tc.patience} epochs.")
            break

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best epoch {best_epoch}: val spec@sens"
          f"{int(100*tc.target_sensitivity)} = {best:.3f}")
    print(f"  Checkpoint: {tc.save_path}")
    print("\n  NEXT: run  python calibrate.py  to fit calibration, freeze the")
    print("  operating point on val, and evaluate on test. Do NOT read the test")
    print("  numbers until you have stopped changing the model.")


if __name__ == "__main__":
    main()
