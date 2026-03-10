#!/usr/bin/env python3
"""
PointNet++ Affordance Training v5 — Multi-Task (Segmentation + Force Center)

改进 (相比 v4):
  1. ⭐ 多任务学习: 接触点分割 + 受力中心回归
  2. 新 PointNet2Seg(predict_force_center=True)
  3. Multi-task Loss: L_seg + λ * L_fc (λ=10.0)
  4. 数据目录改为 Affordance2Grasp/output/dataset

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python -m model.train_v5 --epochs 200
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
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pointnet2 import PointNet2Seg
from model.train import CombinedLoss, compute_metrics, threshold_search, save_visualization


# ============================================================
# Multi-Task Dataset
# ============================================================

class MultiTaskDataset(Dataset):
    """HDF5 dataset with contact labels + force centers."""

    def __init__(self, h5_path, obj_ids_to_use=None, augment=True):
        self.augment = augment

        with h5py.File(h5_path, 'r') as f:
            all_points = f['data/points'][:]
            all_normals = f['data/normals'][:]
            all_labels = f['data/labels'][:]
            all_obj_ids = f['data/obj_ids'][:]
            # Force centers (可能不存在于旧数据)
            if 'data/force_centers' in f:
                all_fc = f['data/force_centers'][:]
            else:
                all_fc = np.zeros((len(all_points), 3), dtype=np.float32)

        # 按物体筛选
        if obj_ids_to_use is not None:
            decoded_ids = [s.decode() if isinstance(s, bytes) else s for s in all_obj_ids]
            mask = np.array([oid in obj_ids_to_use for oid in decoded_ids])
            self.points = all_points[mask]
            self.normals = all_normals[mask]
            self.labels = all_labels[mask]
            self.force_centers = all_fc[mask]
        else:
            self.points = all_points
            self.normals = all_normals
            self.labels = all_labels
            self.force_centers = all_fc

        self.num_samples = len(self.points)
        n_objs = len(obj_ids_to_use) if obj_ids_to_use else "all"
        n_fc_valid = (np.linalg.norm(self.force_centers, axis=1) > 0.001).sum()
        print(f"    Loaded {self.num_samples} samples ({n_objs} objects), "
              f"force_centers: {n_fc_valid}/{self.num_samples} valid")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pts = self.points[idx].copy()
        nrm = self.normals[idx].copy()
        lbl = self.labels[idx].copy()
        fc = self.force_centers[idx].copy()

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
            fc = fc @ R.T  # 受力中心同步旋转

            # 随机缩放
            scale = np.random.uniform(0.8, 1.2)
            pts *= scale
            fc *= scale  # 受力中心同步缩放

            # 随机平移
            shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
            pts += shift
            fc += shift.flatten()  # 受力中心同步平移

            # 随机抖动 (仅点云, 不影响受力中心)
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
            torch.from_numpy(lbl).long(),
            torch.from_numpy(fc).float(),
        )


# ============================================================
# Object-level Split
# ============================================================

def get_object_split(h5_train_path, h5_val_path, val_ratio=0.2, seed=42):
    """从数据集中按物体 ID 划分 train/val."""
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


# ============================================================
# Multi-Task Training / Eval
# ============================================================

def train_epoch_mt(model, loader, optimizer, seg_criterion, fc_lambda, device):
    """多任务训练: 分割 + 受力中心回归."""
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_fc_loss = 0
    all_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
    n_batches = 0

    for xyz, features, labels, fc_gt in loader:
        xyz = xyz.to(device)
        features = features.to(device)
        labels = labels.to(device)
        fc_gt = fc_gt.to(device)

        optimizer.zero_grad()
        seg_pred, fc_pred = model(xyz, features)

        # 分割 loss
        seg_loss = seg_criterion(seg_pred.reshape(-1, 2), labels.reshape(-1))

        # 受力中心回归 loss (仅对有效 fc_gt 计算)
        fc_valid = (fc_gt.norm(dim=1) > 0.001)  # 排除 zero fc
        if fc_valid.any():
            fc_loss = F.mse_loss(fc_pred[fc_valid], fc_gt[fc_valid])
        else:
            fc_loss = torch.tensor(0.0, device=device)

        loss = seg_loss + fc_lambda * fc_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_fc_loss += fc_loss.item()
        metrics = compute_metrics(seg_pred.detach(), labels)
        for k in all_metrics:
            all_metrics[k] += metrics[k]
        n_batches += 1

    avg = lambda x: x / n_batches
    return (avg(total_loss), avg(total_seg_loss), avg(total_fc_loss),
            {k: v / n_batches for k, v in all_metrics.items()})


@torch.no_grad()
def eval_epoch_mt(model, loader, seg_criterion, fc_lambda, device):
    """多任务验证."""
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_fc_loss = 0
    total_fc_dist = 0
    n_fc_valid = 0
    all_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
    n_batches = 0

    for xyz, features, labels, fc_gt in loader:
        xyz = xyz.to(device)
        features = features.to(device)
        labels = labels.to(device)
        fc_gt = fc_gt.to(device)

        seg_pred, fc_pred = model(xyz, features)

        seg_loss = seg_criterion(seg_pred.reshape(-1, 2), labels.reshape(-1))

        fc_valid = (fc_gt.norm(dim=1) > 0.001)
        if fc_valid.any():
            fc_loss = F.mse_loss(fc_pred[fc_valid], fc_gt[fc_valid])
            # 欧氏距离 (mm)
            fc_dist = (fc_pred[fc_valid] - fc_gt[fc_valid]).norm(dim=1).mean()
            total_fc_dist += fc_dist.item()
            n_fc_valid += 1
        else:
            fc_loss = torch.tensor(0.0, device=device)

        loss = seg_loss + fc_lambda * fc_loss

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_fc_loss += fc_loss.item()
        metrics = compute_metrics(seg_pred, labels)
        for k in all_metrics:
            all_metrics[k] += metrics[k]
        n_batches += 1

    avg_fc_dist_mm = (total_fc_dist / max(n_fc_valid, 1)) * 1000  # m → mm
    avg = lambda x: x / n_batches
    return (avg(total_loss), avg(total_seg_loss), avg(total_fc_loss), avg_fc_dist_mm,
            {k: v / n_batches for k, v in all_metrics.items()})


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ Affordance v5 (Multi-Task)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--fc_lambda", type=float, default=10.0,
                        help="受力中心回归 loss 权重 (default: 10.0)")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    # 数据目录
    dataset_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "dataset"
    )

    if args.save_dir is None:
        args.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "checkpoints_v5"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PointNet++ Affordance Training v5 — Multi-Task")
    print("=" * 70)
    print(f"  ⭐ Multi-Task: 接触分割 + 受力中心回归")
    print(f"  Device:      {device}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  FC λ:        {args.fc_lambda}")
    print(f"  Dataset:     {dataset_dir}")
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

    # 加载数据
    print(f"\n  Loading train data...")
    train_dataset = MultiTaskDataset(train_h5, train_obj_ids, augment=True)
    print(f"  Loading val data (from val.h5)...")
    val_from_val = MultiTaskDataset(val_h5, val_obj_ids, augment=False)
    print(f"  Loading val data (from train.h5)...")
    val_from_train = MultiTaskDataset(train_h5, val_obj_ids, augment=False)

    # 合并 val 数据
    val_from_val.points = np.concatenate([val_from_val.points, val_from_train.points])
    val_from_val.normals = np.concatenate([val_from_val.normals, val_from_train.normals])
    val_from_val.labels = np.concatenate([val_from_val.labels, val_from_train.labels])
    val_from_val.force_centers = np.concatenate([val_from_val.force_centers, val_from_train.force_centers])
    val_from_val.num_samples = len(val_from_val.points)
    val_dataset = val_from_val

    # 补充 train: val.h5 中属于 train 物体的数据
    print(f"  Loading extra train data (from val.h5)...")
    train_from_val = MultiTaskDataset(val_h5, train_obj_ids, augment=True)
    train_dataset.points = np.concatenate([train_dataset.points, train_from_val.points])
    train_dataset.normals = np.concatenate([train_dataset.normals, train_from_val.normals])
    train_dataset.labels = np.concatenate([train_dataset.labels, train_from_val.labels])
    train_dataset.force_centers = np.concatenate([train_dataset.force_centers, train_from_val.force_centers])
    train_dataset.num_samples = len(train_dataset.points)

    contact_ratio_train = train_dataset.labels.sum() / train_dataset.labels.size * 100
    contact_ratio_val = val_dataset.labels.sum() / val_dataset.labels.size * 100
    fc_valid_train = (np.linalg.norm(train_dataset.force_centers, axis=1) > 0.001).sum()
    fc_valid_val = (np.linalg.norm(val_dataset.force_centers, axis=1) > 0.001).sum()

    print(f"\n  Summary:")
    print(f"    Train: {len(train_dataset)} samples, {len(train_obj_ids)} objects, "
          f"contact={contact_ratio_train:.1f}%, fc={fc_valid_train}")
    print(f"    Val:   {len(val_dataset)} samples, {len(val_obj_ids)} objects, "
          f"contact={contact_ratio_val:.1f}%, fc={fc_valid_val}")
    sys.stdout.flush()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ============================================================
    # Model (Multi-Task)
    # ============================================================
    model = PointNet2Seg(num_classes=2, in_channel=6, predict_force_center=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params:  {n_params:,}")

    seg_criterion = CombinedLoss(focal_weight=0.6, tversky_weight=0.4)
    print(f"  Seg Loss: Focal(α=0.75,γ=2) + Tversky(α=0.3,β=0.7)")
    print(f"  FC Loss:  MSE × λ={args.fc_lambda}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-5
    )

    print(f"\n{'='*80}")
    print(f"{'Ep':>4} | {'Loss':>8} | {'Seg':>8} | {'FC':>8} | "
          f"{'F1':>6} | {'P':>6} | {'R':>6} | {'FC mm':>6} | {'LR':>8}")
    print(f"{'-'*80}")
    sys.stdout.flush()

    best_val_f1 = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        t_loss, t_seg, t_fc, t_metrics = train_epoch_mt(
            model, train_loader, optimizer, seg_criterion, args.fc_lambda, device)
        v_loss, v_seg, v_fc, v_fc_mm, v_metrics = eval_epoch_mt(
            model, val_loader, seg_criterion, args.fc_lambda, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"{epoch:>4} | {t_loss:>8.4f} | {t_seg:>8.4f} | {t_fc:>8.5f} | "
              f"{v_metrics['f1']:>5.1%} | {v_metrics['precision']:>5.1%} | "
              f"{v_metrics['recall']:>5.1%} | {v_fc_mm:>5.1f} | "
              f"{lr:>8.6f}  ({elapsed:.0f}s)")
        sys.stdout.flush()

        history.append({
            "epoch": epoch,
            "train_loss": round(t_loss, 5), "train_seg": round(t_seg, 5), "train_fc": round(t_fc, 7),
            "val_loss": round(v_loss, 5), "val_seg": round(v_seg, 5), "val_fc": round(v_fc, 7),
            "val_fc_mm": round(v_fc_mm, 2),
            **{f"val_{k}": round(v, 4) for k, v in v_metrics.items()},
            "lr": round(lr, 7),
            "time_s": round(elapsed, 1)
        })

        if v_metrics['f1'] > best_val_f1:
            best_val_f1 = v_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_fc_mm': v_fc_mm,
                'val_iou': v_metrics['iou'],
                'fc_lambda': args.fc_lambda,
                'predict_force_center': True,
                'train_objects': sorted(train_obj_ids),
                'val_objects': sorted(val_obj_ids),
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"        ★ New best! F1={best_val_f1:.1%} FC={v_fc_mm:.1f}mm")
            sys.stdout.flush()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth"))

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'predict_force_center': True,
    }, os.path.join(args.save_dir, "final_model.pth"))

    with open(os.path.join(args.save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    split_info = {
        "train_objects": sorted(train_obj_ids),
        "val_objects": sorted(val_obj_ids),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "fc_lambda": args.fc_lambda,
        "version": "v5_multitask",
    }
    with open(os.path.join(args.save_dir, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE (v5 Multi-Task)")
    print(f"  Train objects: {len(train_obj_ids)}")
    print(f"  Val objects:   {len(val_obj_ids)} (UNSEEN during training)")
    print(f"  Best Val F1:   {best_val_f1:.1%}")
    print(f"  Best model:    {args.save_dir}/best_model.pth")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
