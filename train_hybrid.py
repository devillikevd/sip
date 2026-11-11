"""Phase 6: Complete Training Pipeline for Hybrid MIRNet and Ablation Configs A-D.

Supports:
  - Paired LOL dataset loading & synchronized random crop/flip augmentation
  - Weighted multi-term loss (L1 + SSIM + Color Consistency + Illumination Smoothness + Optional Perceptual)
  - AdamW optimizer + CosineAnnealingLR scheduler + gradient clipping
  - Optional CUDA Automatic Mixed Precision (--amp) for fast Google Colab GPU training
  - Checkpoint saving (best_model.pth & latest_checkpoint.pth) and full --resume
  - Fixed random seeds, CSV/JSON/TensorBoard logs, and automatic curve plotting
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import LoLDataset, PairedRandomPatchDataset
from losses import HybridEnhancementLoss
from models.hybrid_mirnet import CONFIG_DESCRIPTIONS, build_ablation_model
from utils import batch_psnr, batch_ssim


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def train_one_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion: HybridEnhancementLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    running = {
        "total": 0.0,
        "l1": 0.0,
        "ssim_loss": 0.0,
        "color": 0.0,
        "smooth": 0.0,
        "perceptual": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    count = 0

    for idx, (low, high) in enumerate(tqdm(loader, desc="Train", leave=False)):
        if max_steps is not None and idx >= max_steps:
            break
        low = low.to(device)
        high = high.to(device)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(low, return_dict=True)
            loss, terms = criterion(outputs, high, low)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        pred = outputs["enhanced"].detach()
        for k, v in terms.items():
            running[k] += v
        running["psnr"] += batch_psnr(high, pred)
        running["ssim"] += batch_ssim(high, pred)
        count += 1

    if count == 0:
        raise RuntimeError("Training DataLoader produced 0 batches.")
    return {k: v / count for k, v in running.items()}


@torch.no_grad()
def validate_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion: HybridEnhancementLoss,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    running = {
        "total": 0.0,
        "l1": 0.0,
        "ssim_loss": 0.0,
        "color": 0.0,
        "smooth": 0.0,
        "perceptual": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    count = 0

    for idx, (low, high) in enumerate(tqdm(loader, desc="Valid", leave=False)):
        if max_steps is not None and idx >= max_steps:
            break
        low = low.to(device)
        high = high.to(device)
        outputs = model(low, return_dict=True)
        _, terms = criterion(outputs, high, low)

        pred = outputs["enhanced"]
        for k, v in terms.items():
            running[k] += v
        running["psnr"] += batch_psnr(high, pred)
        running["ssim"] += batch_ssim(high, pred)
        count += 1

    if count == 0:
        raise RuntimeError("Validation DataLoader produced 0 batches.")
    return {k: v / count for k, v in running.items()}


def save_training_curves(history_df: pd.DataFrame, out_path: str):
    if history_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history_df["epoch"], history_df["train_total"], label="Train Loss")
    axes[0].plot(history_df["epoch"], history_df["val_total"], label="Val Loss")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["train_psnr"], label="Train PSNR")
    axes[1].plot(history_df["epoch"], history_df["val_psnr"], label="Val PSNR")
    axes[1].set_title("PSNR (dB)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history_df["epoch"], history_df["train_ssim"], label="Train SSIM")
    axes[2].plot(history_df["epoch"], history_df["val_ssim"], label="Val SSIM")
    axes[2].set_title("SSIM")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hybrid MIRNet + Illumination Curve Fusion (Configs A/B/C/D)"
    )
    parser.add_argument("--config", type=str, default="D", choices=["A", "B", "C", "D"])
    parser.add_argument("--train_low", type=str, default="data/LOL/our485/low")
    parser.add_argument("--train_high", type=str, default="data/LOL/our485/high")
    parser.add_argument("--val_low", type=str, default="data/LOL/eval15/low")
    parser.add_argument("--val_high", type=str, default="data/LOL/eval15/high")
    parser.add_argument("--output_dir", type=str, default="experiments/config_D")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--val_mode", type=str, default="full", choices=["crop", "resize", "full"])
    parser.add_argument("--val_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.25)
    parser.add_argument("--color_weight", type=float, default=0.10)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--perceptual_weight", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_ds = PairedRandomPatchDataset(
        args.train_low, args.train_high, patch_size=args.patch_size, augment=True
    )
    val_ds = LoLDataset(
        args.val_low, args.val_high, img_size=args.val_size, mode=args.val_mode
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_ablation_model(args.config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Config {args.config}: {CONFIG_DESCRIPTIONS[args.config]}")
    print(f"Device: {device} | Parameters: {total_params:,}")

    criterion = HybridEnhancementLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        color_weight=args.color_weight,
        smooth_weight=args.smooth_weight,
        perceptual_weight=args.perceptual_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    start_epoch = 0
    best_psnr = -1.0
    history = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_psnr = float(ckpt.get("best_psnr", -1.0))
        history = list(ckpt.get("history", []))
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_psnr={best_psnr:.4f})")

    writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        tr = train_one_epoch(
            train_loader, model, criterion, optimizer, device, scaler=scaler, max_steps=args.max_steps
        )
        va = validate_epoch(val_loader, model, criterion, device, max_steps=args.max_steps)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch + 1,
            "lr": cur_lr,
            "epoch_time_s": elapsed,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        history.append(row)

        writer.add_scalar("Loss/train", tr["total"], epoch + 1)
        writer.add_scalar("Loss/val", va["total"], epoch + 1)
        writer.add_scalar("Metrics/train_psnr", tr["psnr"], epoch + 1)
        writer.add_scalar("Metrics/val_psnr", va["psnr"], epoch + 1)
        writer.add_scalar("Metrics/val_ssim", va["ssim"], epoch + 1)
        writer.add_scalar("LR", cur_lr, epoch + 1)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train Loss: {tr['total']:.4f}, PSNR: {tr['psnr']:.2f} dB | "
            f"Val Loss: {va['total']:.4f}, PSNR: {va['psnr']:.2f} dB, SSIM: {va['ssim']:.4f} "
            f"({elapsed:.1f}s)"
        )

        state = {
            "config": args.config,
            "epoch": epoch,
            "best_psnr": max(best_psnr, va["psnr"]),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "history": history,
            "args": vars(args),
        }
        torch.save(state, os.path.join(ckpt_dir, "latest_checkpoint.pth"))

        if va["psnr"] > best_psnr:
            best_psnr = va["psnr"]
            state["best_psnr"] = best_psnr
            torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))

        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)

    writer.close()
    history_df = pd.DataFrame(history)
    save_training_curves(history_df, os.path.join(args.output_dir, "loss_psnr_curves.png"))

    summary = {
        "config": args.config,
        "description": CONFIG_DESCRIPTIONS[args.config],
        "total_params": int(total_params),
        "device": str(device),
        "best_val_psnr": float(best_psnr),
        "final_epoch": int(history[-1]["epoch"]) if history else 0,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

