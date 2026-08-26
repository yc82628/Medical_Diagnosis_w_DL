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
                   get_device, count_parameters)
from metrics import (threshold_at_sensitivity, screening_metrics,
                     quadratic_weighted_kappa)
from splits import enforce_monotonic, grades_from_cumulative


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
def predict(model, loader, device) -> dict:
    """Returns raw referable logits plus cumulative probabilities and true labels."""
    model.eval()
    logits_all, y_all, g_all = [], [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images).float().cpu().numpy()
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        total_steps=steps_per_epoch * tc.num_epochs,
        pct_start=tc.warmup_epochs / max(tc.num_epochs, 1),
    )
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
            save_checkpoint(tc.save_path, model, cfg.preprocess,
                            extra={"epoch": epoch, "val_metrics": m,
                                   "val_threshold": thr, "history": history})
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
