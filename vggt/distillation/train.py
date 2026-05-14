# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for VGGT distillation.

Usage:
    # Single GPU
    python -m vggt.distillation.train \
        --teacher-path "facebook/VGGT-1B" \
        --student-depth 4 \
        --train-dir /path/to/training/images \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-4

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=8 -m vggt.distillation.train \
        --teacher-path /local/model/path \
        --student-depth 4 \
        --train-dir /path/to/images \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-4 \
        --save-dir ./checkpoints
"""

import argparse
import glob
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from vggt.distillation.distiller import VGGTDistiller, create_student_vggt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Distillation Training")
    # Model
    parser.add_argument("--teacher-path", type=str, required=True,
                        help="HuggingFace repo ID or local path to the teacher VGGT")
    parser.add_argument("--student-depth", type=int, default=4,
                        help="Number of aggregator blocks in the student. Default: 4")
    parser.add_argument("--embed-dim", type=int, default=1024,
                        help="Student embed_dim (must match teacher). Default: 1024")
    # Data
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Directory containing validation images (optional)")
    parser.add_argument("--image-size", type=int, default=518,
                        help="Resize images to this size before feeding to model. Default: 518")
    parser.add_argument("--target-size", type=int, default=378,
                        help="Target image size for feature pooling. Default: 378")
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm. Default: 1.0")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    # Logging & checkpoints
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs. Default: 5")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Print training metrics every N steps. Default: 10")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from")
    # DDP / hardware
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


class ImageFolderDataset(Dataset):
    """
    Simple dataset that loads all images from a directory.
    Accepts .jpg, .jpeg, .png, .bmp, .webp
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str, image_size: int = 518):
        self.root_dir = root_dir
        self.image_size = image_size
        self.paths = sorted([
            p for p in glob.glob(os.path.join(root_dir, "**/*"), recursive=True)
            if Path(p).suffix.lower() in self.EXTENSIONS
        ])
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}")
        print(f"Dataset: found {len(self.paths)} images in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        images, crop_box = load_and_preprocess_images_square(
            [self.paths[idx]], target_size=self.image_size
        )
        return images.squeeze(0)


@torch.no_grad()
def compute_batch_features(model: VGGT, images: torch.Tensor, target_size: int | None):
    """Extract patch features from the last block of the aggregator."""
    features = model.get_last_block_features(images, exclude_special_tokens=True, target_size=target_size)
    return features


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(
    distiller: VGGTDistiller,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    args,
    rank: int = 0,
    world_size: int = 1,
    global_step: int = 0,
):
    distiller.student.train()

    total_loss = 0.0
    total_cos_sim = 0.0
    num_batches = 0

    for batch_idx, images in enumerate(train_loader):
        images = images.cuda()
        if rank == 0 and batch_idx == 0:
            torch.cuda.reset_peak_memory_stats()

        loss, metrics = distiller(images, target_size=args.target_size)

        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(distiller.student.parameters(), args.grad_clip)

        optimizer.step()

        cos_sim = metrics["cos_sim"].mean().item()
        batch_loss = loss.item()

        total_loss += batch_loss
        total_cos_sim += cos_sim
        num_batches += 1
        global_step += 1

        if rank == 0 and batch_idx % args.log_every == 0:
            print(
                f"[Epoch {epoch}][Step {batch_idx}/{len(train_loader)}] "
                f"loss={batch_loss:.4f}  cos_sim={cos_sim:.4f}  "
                f"t_patches={metrics['teacher_patches']}  s_patches={metrics['student_patches']}"
            )

    avg_loss = total_loss / num_batches
    avg_cos_sim = total_cos_sim / num_batches
    return avg_loss, avg_cos_sim, global_step


@torch.no_grad()
def validate(distiller: VGGTDistiller, val_loader: DataLoader, args, rank: int = 0):
    distiller.student.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    num_batches = 0

    for images in val_loader:
        images = images.cuda()
        loss, metrics = distiller(images, target_size=args.target_size)
        total_loss += loss.item()
        total_cos_sim += metrics["cos_sim"].mean().item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_cos_sim = total_cos_sim / num_batches
    return avg_loss, avg_cos_sim


def build_loaders(args, world_size: int = 1, rank: int = 0):
    train_dataset = ImageFolderDataset(args.train_dir, image_size=args.image_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = ImageFolderDataset(args.val_dir, image_size=args.image_size)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) \
            if world_size > 1 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader


def main(args):
    if args.ddp:
        setup(args.local_rank, int(os.environ["WORLD_SIZE"]))
        rank = args.local_rank
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1

    torch.cuda.set_device(rank)

    os.makedirs(args.save_dir, exist_ok=True)

    if rank == 0:
        print("=" * 60)
        print("VGGT Distillation Training")
        print(f"  Teacher:    {args.teacher_path}")
        print(f"  Student depth: {args.student_depth}")
        print(f"  Train dir:  {args.train_dir}")
        print(f"  Image size: {args.image_size}")
        print(f"  Target size: {args.target_size}")
        print(f"  Epochs:     {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  LR:         {args.lr}")
        print("=" * 60)

    teacher = VGGT.from_pretrained(args.teacher_path)
    teacher = teacher.cuda()
    teacher.eval()

    student = create_student_vggt(embed_dim=args.embed_dim, depth=args.student_depth)
    student = student.cuda()

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda")
        student.load_state_dict(ckpt["student_state_dict"])
        if rank == 0:
            print(f"Resumed from {args.resume}")

    distiller = VGGTDistiller(teacher=teacher, student=student, dino_freeze=True)

    if args.ddp:
        student = DDP(student, device_ids=[rank])

    train_loader, val_loader = build_loaders(args, world_size=world_size, rank=rank)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            t0 = time.time()

        train_loss, train_cos_sim, global_step = train_one_epoch(
            distiller, train_loader, optimizer, epoch, args, rank, world_size, global_step
        )
        scheduler.step()

        if rank == 0:
            elapsed = time.time() - t0
            print(f"\n[Epoch {epoch}] train_loss={train_loss:.4f}  cos_sim={train_cos_sim:.4f}  time={elapsed:.1f}s")

        if args.val_dir and rank == 0:
            val_loss, val_cos_sim = validate(distiller, val_loader, args, rank)
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f}  val_cos_sim={val_cos_sim:.4f}")
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = (epoch == 0)

        if rank == 0 and (epoch % args.save_every == 0 or is_best):
            ckpt_name = f"epoch={epoch:04d}_loss={train_loss:.4f}.pth"
            if is_best:
                ckpt_name = "best.pth"
            ckpt_path = os.path.join(args.save_dir, ckpt_name)

            student_state = student.module.state_dict() if args.ddp else student.state_dict()
            torch.save({
                "epoch": epoch,
                "student_state_dict": student_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    if args.ddp:
        cleanup()

    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    if args.ddp:
        torchrun_entry = "torchrun"
        import sys
        sys.exit(0)
    main(args)
