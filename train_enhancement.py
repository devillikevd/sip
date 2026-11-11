import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import LoLDataset
from model import MIRNet
from utils import MixUp, batch_psnr, batch_ssim


class Loss_fn(nn.Module):
    """Charbonnier reconstruction loss used in baseline MIRNet training."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, ip, target):
        diff = ip - target
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg="auto"):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_preview_tensor(path, device):
    if path and os.path.isfile(path):
        img = Image.open(path).convert("RGB")
        return T.Compose([T.Resize((128, 128)), T.ToTensor()])(img).unsqueeze(0).to(device)
    return None


def train(
    loader,
    model,
    criterion,
    opt,
    writer,
    step,
    epoch,
    mixup=True,
    device=None,
    fixed_ip_train=None,
    max_steps=None,
):
    if device is None:
        device = next(model.parameters()).device

    mixup_aug = mixup if isinstance(mixup, MixUp) else (MixUp() if mixup else None)

    train_loss, train_psnr = [], []
    epoch_psnr, epoch_loss = 0.0, 0.0
    loop = tqdm(loader, position=0, leave=True)
    model.train()

    for idx, (lol, target) in enumerate(loop):
        if max_steps is not None and idx >= max_steps:
            break

        lol = lol.to(device)
        target = target.to(device)

        if epoch > 5 and mixup_aug is not None and lol.size(0) > 1:
            target, lol = mixup_aug.augment(target, lol)

        opt.zero_grad()
        gen = model(lol)
        loss = criterion(gen, target)
        loss.backward()
        opt.step()

        train_loss.append(float(loss.item()))
        train_psnr.append(batch_psnr(target, gen.detach(), 1.0))

        epoch_psnr = sum(train_psnr) / len(train_psnr)
        epoch_loss = sum(train_loss) / len(train_loss)

        if writer is not None:
            writer.add_scalar("Training/Loss", epoch_loss, global_step=step)
            writer.add_scalar("Training/PSNR", epoch_psnr, global_step=step)

            if idx % 25 == 0 and fixed_ip_train is not None:
                model.eval()
                with torch.no_grad():
                    gen_img = torch.clamp(model(fixed_ip_train), 0.0, 1.0)
                writer.add_image("Observations/Train_img", gen_img.squeeze(0).cpu(), global_step=step)
                model.train()

        if device.type == "cuda":
            torch.cuda.empty_cache()
        step += 1

    print(f"Epoch: {epoch}, Loss: {epoch_loss:.6f} and PSNR: {epoch_psnr:.4f} dB.\n")
    return step, epoch_loss, epoch_psnr


def validate(
    loader,
    model,
    criterion,
    writer,
    step,
    epoch,
    device=None,
    fixed_ip_valid=None,
    max_steps=None,
):
    if device is None:
        device = next(model.parameters()).device

    val_loss, val_psnr, val_ssim = [], [], []
    loop = tqdm(loader, position=0, leave=True)
    model.eval()
    print("Validating...")

    with torch.no_grad():
        for idx, (lol, target) in enumerate(loop):
            if max_steps is not None and idx >= max_steps:
                break

            lol = lol.to(device)
            target = target.to(device)

            gen = model(lol)
            loss = criterion(gen, target)
            val_loss.append(float(loss.item()))
            val_psnr.append(batch_psnr(target, gen, 1.0))
            val_ssim.append(batch_ssim(target, gen))
            step += 1

        mean_loss = sum(val_loss) / len(val_loss)
        mean_psnr = sum(val_psnr) / len(val_psnr)
        mean_ssim = sum(val_ssim) / len(val_ssim)

        if writer is not None:
            if fixed_ip_valid is not None:
                gen_img = torch.clamp(model(fixed_ip_valid), 0.0, 1.0)
                writer.add_image("Observation/Valid_img", gen_img.squeeze(0).cpu(), global_step=step)
            writer.add_scalar("Validation/Loss", mean_loss, global_step=step)
            writer.add_scalar("Validation/PSNR", mean_psnr, global_step=step)
            writer.add_scalar("Validation/SSIM", mean_ssim, global_step=step)

    print(
        f"Validation Epoch: {epoch}, Loss: {mean_loss:.6f}, "
        f"PSNR: {mean_psnr:.4f} dB, SSIM: {mean_ssim:.4f}.\n"
    )
    return step, mean_loss, mean_psnr, mean_ssim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline MIRNet training script"
    )
    parser.add_argument("--train_low", type=str, default="data/LOL/our485/low")
    parser.add_argument("--train_high", type=str, default="data/LOL/our485/high")
    parser.add_argument("--val_low", type=str, default="data/LOL/eval15/low")
    parser.add_argument("--val_high", type=str, default="data/LOL/eval15/high")
    parser.add_argument("--save_dir", type=str, default="baseline_results/checkpoints")
    parser.add_argument("--log_dir", type=str, default="baseline_results/logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--transform_mode", type=str, default="crop", choices=["crop", "resize", "full"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--fixed_train_img", type=str, default=None)
    parser.add_argument("--fixed_valid_img", type=str, default=None)
    parser.add_argument("--fixed_target_img", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_data = LoLDataset(
        low_light_root=args.train_low,
        target_root=args.train_high,
        img_size=args.img_size,
        mode=args.transform_mode,
    )
    val_data = LoLDataset(
        low_light_root=args.val_low,
        target_root=args.val_high,
        img_size=args.img_size,
        mode=args.transform_mode,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        val_data, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = MIRNet().to(device)
    criterion = Loss_fn()
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, args.epochs), 5e-5)
    writer = SummaryWriter(args.log_dir)

    start_epoch = 0
    best_psnr = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_psnr = ckpt.get("best_psnr", -1.0)
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed baseline MIRNet checkpoint from {args.resume} (start_epoch={start_epoch})")

    fixed_train_path = args.fixed_train_img or train_data.lol_fnames[0]
    fixed_valid_path = args.fixed_valid_img or val_data.lol_fnames[0]
    fixed_target_path = args.fixed_target_img or val_data.target_fnames[0]

    fixed_ip_train = _load_preview_tensor(fixed_train_path, device)
    fixed_ip_valid = _load_preview_tensor(fixed_valid_path, device)
    target_img = _load_preview_tensor(fixed_target_path, device)

    mixup_obj = None if args.no_mixup else MixUp()
    step, val_step = 0, 0

    for epoch in range(start_epoch, args.epochs):
        if fixed_ip_valid is not None:
            model.eval()
            with torch.no_grad():
                gen_img1 = torch.clamp(model(fixed_ip_valid), 0.0, 1.0)
            vutils.save_image(
                gen_img1.cpu(), os.path.join(args.save_dir, f"img_epoch_{epoch}.png")
            )
            writer.add_image("Images/Input", fixed_ip_valid.squeeze(0).cpu(), global_step=step)
            writer.add_image("Images/Generated", gen_img1.squeeze(0).cpu(), global_step=step)
            if target_img is not None:
                writer.add_image("Images/Target", target_img.squeeze(0).cpu(), global_step=step)

        step, _, _ = train(
            train_loader,
            model,
            criterion,
            opt,
            writer,
            step,
            epoch,
            mixup=mixup_obj,
            device=device,
            fixed_ip_train=fixed_ip_train,
            max_steps=args.max_steps,
        )
        scheduler.step()

        ckpt_path = os.path.join(args.save_dir, "Mirnet_enhance.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_psnr": best_psnr,
            },
            ckpt_path,
        )

        val_step, _, val_psnr, _ = validate(
            valid_loader,
            model,
            criterion,
            writer,
            val_step,
            epoch,
            device=device,
            fixed_ip_valid=fixed_ip_valid,
            max_steps=args.max_steps,
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, "Mirnet_enhance_best.pth")
            )

    writer.close()


if __name__ == "__main__":
    main()
