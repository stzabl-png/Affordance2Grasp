#!/usr/bin/env python3
"""
PointNet++ Affordance Training v4 — Object-level Split

改进 (相比 v3):
  1. ⭐ 物体级别 train/val 分割: 验证集使用训练中从未见过的物体
     → 真正测试泛化能力, 不再是同物体不同帧的"假验证"
  2. 使用全部 OakInk 类别 (17类 65物体) 而不仅是 hold
     → 更多训练数据, 更多物体形状多样性
  3. 200 epochs + Cosine Annealing with Warm Restarts
  4. 使用 contact_radius=0.015 (已验证最优)

过夜训练预计 ~4-6 小时

用法:
    python -m model.train_v4 --epochs 200
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse model architecture from train.py
from model.train import (
    PointNet2Seg, CombinedLoss, AffordanceDataset,
    compute_metrics, train_epoch, eval_epoch, threshold_search,
    save_visualization
)


# ============================================================
# Object-level Split Dataset
# ============================================================

class ObjectSplitDataset(Dataset):
    """HDF5 + 按物体 ID 选择子集."""

    def __init__(self, h5_path, obj_ids_to_use, augment=True):
        self.augment = augment

        with h5py.File(h5_path, 'r') as f:
            all_points = f['data/points'][:]
            all_normals = f['data/normals'][:]
            all_labels = f['data/labels'][:]
            all_obj_ids = f['data/obj_ids'][:]

        # 筛选属于指定物体的样本
        decoded_ids = [s.decode() if isinstance(s, bytes) else s for s in all_obj_ids]
        mask = np.array([oid in obj_ids_to_use for oid in decoded_ids])

        self.points = all_points[mask]
        self.normals = all_normals[mask]
        self.labels = all_labels[mask]

        self.num_samples = len(self.points)
        print(f"    Loaded {self.num_samples} samples ({len(obj_ids_to_use)} objects)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pts = self.points[idx].copy()
        nrm = self.normals[idx].copy()
        lbl = self.labels[idx].copy()

        if self.augment:
            # SO(3) 随机旋转
            z = np.random.randn(3, 3).astype(np.float32)
            q, r = np.linalg.qr(z)
            d = np.diagonal(r)
            ph = d / np.abs(d)
            R = (q @ np.diag(ph)).astype(np.float32)
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1
            pts = pts @ R.T
            nrm = nrm @ R.T

            # 随机缩放
            scale = np.random.uniform(0.8, 1.2)
            pts *= scale

            # 随机平移
            pts += np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)

            # 随机抖动
            pts += np.random.normal(0, 0.002, size=pts.shape).astype(np.float32)

            # 随机丢点 (30%)
            if np.random.rand() < 0.3:
                n = len(pts)
                keep = np.random.choice(n, int(n * 0.9), replace=False)
                drop = np.setdiff1d(np.arange(n), keep)
                fill = np.random.choice(keep, len(drop), replace=True)
                pts[drop] = pts[fill]
                nrm[drop] = nrm[fill]
                lbl[drop] = lbl[fill]

        features = np.concatenate([pts, nrm], axis=-1)
        return (
            torch.from_numpy(pts),
            torch.from_numpy(features),
            torch.from_numpy(lbl).long()
        )


def get_object_split(h5_train_path, h5_val_path, val_ratio=0.2, seed=42):
    """
    从完整数据集中按物体 ID 划分 train/val.
    
    Returns:
        train_obj_ids: set of object IDs for training
        val_obj_ids: set of object IDs for validation
    """
    # 合并两个文件的物体列表
    all_obj_ids = set()
    for path in [h5_train_path, h5_val_path]:
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                ids = f['data/obj_ids'][:]
                for s in ids:
                    all_obj_ids.add(s.decode() if isinstance(s, bytes) else s)

    all_obj_ids = sorted(all_obj_ids)
    np.random.seed(seed)
    np.random.shuffle(all_obj_ids)

    n_val = max(1, int(len(all_obj_ids) * val_ratio))
    val_obj_ids = set(all_obj_ids[:n_val])
    train_obj_ids = set(all_obj_ids[n_val:])

    return train_obj_ids, val_obj_ids


def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ Affordance v4 (Object-level Split)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集物体比例 (default: 20%%)")
    parser.add_argument("--use_hold_only", action="store_true",
                        help="仅使用 hold intent 数据")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    # 确定数据目录
    if args.use_hold_only:
        dataset_dir = "/home/lyh/Project/ContactPoint/dataset_hold"
        tag = "hold"
    else:
        dataset_dir = "/home/lyh/Project/ContactPoint/dataset"
        tag = "all"

    if args.save_dir is None:
        args.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", f"checkpoints_v4_{tag}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PointNet++ Affordance Training v4 — Object-level Split")
    print("=" * 60)
    print(f"  ⭐ KEY: 物体级别 train/val 分割 (验证=未见物体)")
    print(f"  Device:      {device}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Dataset:     {dataset_dir} ({tag})")
    print(f"  Val ratio:   {args.val_ratio:.0%} of objects")
    print(f"  Checkpoints: {args.save_dir}")
    sys.stdout.flush()

    # ============================================================
    # 物体级别划分
    # ============================================================
    print(f"\n--- Object-level Split ---")
    train_h5 = os.path.join(dataset_dir, "affordance_train.h5")
    val_h5 = os.path.join(dataset_dir, "affordance_val.h5")

    train_obj_ids, val_obj_ids = get_object_split(
        train_h5, val_h5, val_ratio=args.val_ratio
    )
    print(f"  Train objects ({len(train_obj_ids)}): {sorted(train_obj_ids)[:8]}...")
    print(f"  Val objects   ({len(val_obj_ids)}):   {sorted(val_obj_ids)}")
    overlap = train_obj_ids & val_obj_ids
    assert len(overlap) == 0, f"Overlap detected: {overlap}"
    print(f"  ✅ Zero overlap between train/val objects")
    sys.stdout.flush()

    # 加载数据 — 从两个 H5 文件中按物体筛选
    all_obj_ids_combined = train_obj_ids | val_obj_ids
    print(f"\n  Loading train data...")
    train_dataset = ObjectSplitDataset(train_h5, train_obj_ids, augment=True)
    # 验证集: 先从 val.h5 加载指定物体, 如果不够就补 train.h5
    print(f"  Loading val data...")
    val_dataset_from_val = ObjectSplitDataset(val_h5, val_obj_ids, augment=False)
    val_dataset_from_train = ObjectSplitDataset(train_h5, val_obj_ids, augment=False)

    # 合并 val 数据
    val_points = np.concatenate([val_dataset_from_val.points, val_dataset_from_train.points])
    val_normals = np.concatenate([val_dataset_from_val.normals, val_dataset_from_train.normals])
    val_labels = np.concatenate([val_dataset_from_val.labels, val_dataset_from_train.labels])

    # 同时, train 也需要补充 val.h5 中属于 train 物体的数据
    train_from_val = ObjectSplitDataset(val_h5, train_obj_ids, augment=True)
    train_dataset.points = np.concatenate([train_dataset.points, train_from_val.points])
    train_dataset.normals = np.concatenate([train_dataset.normals, train_from_val.normals])
    train_dataset.labels = np.concatenate([train_dataset.labels, train_from_val.labels])
    train_dataset.num_samples = len(train_dataset.points)

    # 更新 val dataset
    val_dataset_from_val.points = val_points
    val_dataset_from_val.normals = val_normals
    val_dataset_from_val.labels = val_labels
    val_dataset_from_val.num_samples = len(val_points)
    val_dataset = val_dataset_from_val

    contact_ratio_train = train_dataset.labels.sum() / train_dataset.labels.size * 100
    contact_ratio_val = val_dataset.labels.sum() / val_dataset.labels.size * 100

    print(f"\n  Summary:")
    print(f"    Train: {len(train_dataset)} samples, {len(train_obj_ids)} objects, contact={contact_ratio_train:.1f}%")
    print(f"    Val:   {len(val_dataset)} samples, {len(val_obj_ids)} objects, contact={contact_ratio_val:.1f}%")
    sys.stdout.flush()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ============================================================
    # Model
    # ============================================================
    model = PointNet2Seg(num_classes=2, in_channel=6).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params:  {n_params:,}")

    criterion = CombinedLoss(focal_weight=0.6, tversky_weight=0.4)
    print(f"  Loss: Focal(α=0.75,γ=2) + Tversky(α=0.3,β=0.7)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine with warm restarts: 每 50 epoch 重启一次
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-5
    )

    print(f"  Optimizer: AdamW (改为 weight decay decoupled)")
    print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0=50)")

    print(f"\n{'='*70}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Val Acc':>7} | {'Val P':>7} | {'Val R':>7} | {'Val F1':>7} | {'LR':>8}")
    print(f"{'-'*70}")
    sys.stdout.flush()

    best_val_f1 = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{val_metrics['accuracy']:>6.1%} | {val_metrics['precision']:>6.1%} | "
              f"{val_metrics['recall']:>6.1%} | {val_metrics['f1']:>6.1%} | "
              f"{lr:>8.6f}  ({elapsed:.0f}s)")
        sys.stdout.flush()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
            "lr": round(lr, 7),
            "time_s": round(elapsed, 1)
        })

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_iou': val_metrics['iou'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'train_objects': sorted(train_obj_ids),
                'val_objects': sorted(val_obj_ids),
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"        ★ New best! F1={best_val_f1:.1%} (P={val_metrics['precision']:.1%} R={val_metrics['recall']:.1%})")
            sys.stdout.flush()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth"))
            save_visualization(
                model, val_dataset, device,
                os.path.join(args.save_dir, f"vis_epoch{epoch}.png"),
                epoch, history
            )

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
    }, os.path.join(args.save_dir, "final_model.pth"))

    with open(os.path.join(args.save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    # Save split info
    split_info = {
        "train_objects": sorted(train_obj_ids),
        "val_objects": sorted(val_obj_ids),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "dataset_dir": dataset_dir,
    }
    with open(os.path.join(args.save_dir, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    # Threshold Search
    print(f"\n{'='*60}")
    print(f"Threshold Search...")
    best_thresh, best_thresh_f1 = threshold_search(model, val_loader, device)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE (v4 Object-level Split)")
    print(f"  Train objects: {len(train_obj_ids)}")
    print(f"  Val objects:   {len(val_obj_ids)} (UNSEEN during training)")
    print(f"  Best Val F1 (thresh=0.5): {best_val_f1:.1%}")
    print(f"  Best Threshold:           {best_thresh:.2f} → F1={best_thresh_f1:.1%}")
    print(f"  Best model:   {args.save_dir}/best_model.pth")
    print(f"{'='*60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
