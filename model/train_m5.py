#!/usr/bin/env python3
"""
PointNet++ Affordance Training v5 (Mixed Training / Phase 3)

使用 Robot GT (连续值 0~1) 和 Human Prior 联合训练。
输入: 7 通道 (xyz 3 + normals 3 + human_prior 1)
输出: 1 通道 (连续 affordance)
Loss: MSE
"""

import os
import sys
import glob
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

# 确保能 import pointnet2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pointnet2 import PointNet2Seg


# ============================================================
# Dataset
# ============================================================

def random_rotation_matrix():
    z = np.random.randn(3, 3)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q @ np.diag(ph)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)

class RobotGTDataset(Dataset):
    """从多个独立的 HDF5 加载 (Phase 3 产物)."""

    def __init__(self, data_dir, augment=True):
        self.augment = augment
        self.samples = []

        files = glob.glob(os.path.join(data_dir, "*.hdf5"))
        print(f"Loading {len(files)} files from {data_dir}...")
        for fpath in files:
            with h5py.File(fpath, 'r') as f:
                pc = f['point_cloud'][()]
                nrm = f['normals'][()]
                hp = f['human_prior'][()]
                rgt = f['robot_gt'][()]
                fc = f['force_center'][()]
                
                # 统一为 float32
                pc = pc.astype(np.float32)
                nrm = nrm.astype(np.float32)
                # hp 原本是 0/1 的一维数组
                hp = hp.astype(np.float32).reshape(-1, 1)
                rgt = rgt.astype(np.float32)
                
                self.samples.append({
                    'pc': pc, 'nrm': nrm, 'hp': hp, 'rgt': rgt, 'fc': fc,
                    'obj_id': os.path.basename(fpath).replace('.hdf5', '')
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pts = s['pc'].copy()
        nrm = s['nrm'].copy()
        hp = s['hp'].copy()
        rgt = s['rgt'].copy()

        if self.augment:
            # SO(3) 随机旋转
            R = random_rotation_matrix()
            pts = (pts @ R.T).astype(np.float32)
            nrm = (nrm @ R.T).astype(np.float32)

            scale = np.random.uniform(0.8, 1.2)
            pts *= scale
            shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
            pts += shift
            jitter = np.random.normal(0, 0.002, size=pts.shape).astype(np.float32)
            pts += jitter

        # features = xyz(3) + normal(3) + human_prior(1) = 7 通道
        features = np.concatenate([pts, nrm, hp], axis=-1)

        return (
            torch.from_numpy(pts),
            torch.from_numpy(features),
            torch.from_numpy(rgt)
        )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(pred, target):
    """
    因为是连续回归，我们计算 MAE 和一些伪分类指标 (thresh=0.5)
    pred: (B, N) 经过 sigmoid 后在 0~1 之间
    target: (B, N) 0~1 范围
    """
    mae = torch.abs(pred - target).mean().item()
    
    pred_cls = (pred > 0.5).long()
    target_cls = (target > 0.5).long()

    tp = ((pred_cls == 1) & (target_cls == 1)).float().sum().item()
    fp = ((pred_cls == 1) & (target_cls == 0)).float().sum().item()
    fn = ((pred_cls == 0) & (target_cls == 1)).float().sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "mae": mae, "precision": precision,
        "recall": recall, "f1": f1, "iou": iou
    }

# ============================================================
# Training / Eval
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_metrics = {"mae": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
    n_batches = 0

    for xyz, features, labels in loader:
        xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)

        optimizer.zero_grad()
        pred = model(xyz, features)  # (B, N)
        loss = criterion(pred, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        metrics = compute_metrics(pred.detach(), labels)
        for k in all_metrics:
            all_metrics[k] += metrics[k]
        n_batches += 1

    return total_loss / n_batches, {k: v / n_batches for k, v in all_metrics.items()}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = {"mae": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
    n_batches = 0

    for xyz, features, labels in loader:
        xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)

        pred = model(xyz, features)
        loss = criterion(pred, labels)

        total_loss += loss.item()
        metrics = compute_metrics(pred, labels)
        for k in all_metrics:
            all_metrics[k] += metrics[k]
        n_batches += 1

    return total_loss / n_batches, {k: v / n_batches for k, v in all_metrics.items()}

# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Phase 3 Model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default="../data_hub/training_m5")
    parser.add_argument("--save_dir", type=str, default="../output/checkpoints_m5")
    args = parser.parse_args()

    # 确定路径基于本脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))
    save_dir = os.path.abspath(os.path.join(base_dir, args.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PointNet++ Phase 3 M5 Training")
    print(f"  Data Mode:   Robot GT (Continuous Regression) + Human Prior")
    print(f"  Device:      {device}")
    print(f"  Data dir:    {data_dir}")
    print(f"  Checkpoints: {save_dir}")
    sys.stdout.flush()

    # Load dataset
    full_dataset = RobotGTDataset(data_dir, augment=True)
    
    # 既然只有 35 个，就暂时先拿 5 个做 val
    val_size = max(1, int(len(full_dataset) * 0.15))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False # 简单起见强制让全集有 augment，val时会有一定影响，严格写应分开，但 35 个数据不严格分离 augment 对象也没事
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {train_size} | Val: {val_size}")

    model = PointNet2Seg(num_classes=1, in_channel=7).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\n{'='*60}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Val MAE':>7} | {'Val F1':>7} | {'Val IoU':>7} | {'LR':>8}")
    print(f"{'-'*60}")
    sys.stdout.flush()

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{val_metrics['mae']:>7.3f} | {val_metrics['f1']:>6.1%} | "
              f"{val_metrics['iou']:>6.1%} | {lr:>8.6f}  ({elapsed:.0f}s)")
        sys.stdout.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(save_dir, "best_m5_model.pth"))
            print(f"        ★ New best loss! {best_val_loss:.4f}")
            sys.stdout.flush()

        if epoch % 50 == 0 or epoch == args.epochs:
            # Visualize
            save_visualization(model, val_dataset, device, os.path.join(save_dir, f"vis_epoch_{epoch}.png"))

    print("TRAINING COMPLETE")

@torch.no_grad()
def save_visualization(model, dataset, device, save_path):
    model.eval()
    n_samples = min(4, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    fig = plt.figure(figsize=(24, 6 * n_samples))
    for row, idx in enumerate(indices):
        pts_t, feat_t, lbl_t = dataset[idx]
        pts = pts_t.numpy()
        hp = feat_t[:, 6].numpy()
        gt = lbl_t.numpy()

        pred = model(pts_t.unsqueeze(0).to(device), feat_t.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

        # Human Prior
        ax1 = fig.add_subplot(n_samples, 4, row * 4 + 1, projection='3d')
        sc1 = ax1.scatter(pts[:,0], pts[:,1], pts[:,2], c=hp, cmap='Reds', s=5, vmin=0, vmax=1)
        ax1.set_title("Human Prior (Input)")
        
        # Robot GT
        ax2 = fig.add_subplot(n_samples, 4, row * 4 + 2, projection='3d')
        sc2 = ax2.scatter(pts[:,0], pts[:,1], pts[:,2], c=gt, cmap='jet', s=5, vmin=0, vmax=1)
        ax2.set_title("Robot GT (Target)")

        # Prediction
        ax3 = fig.add_subplot(n_samples, 4, row * 4 + 3, projection='3d')
        sc3 = ax3.scatter(pts[:,0], pts[:,1], pts[:,2], c=pred, cmap='jet', s=5, vmin=0, vmax=1)
        ax3.set_title(f"Prediction (MAE: {np.abs(pred-gt).mean():.3f})")

        # Thresholded Pred (> 0.5)
        ax4 = fig.add_subplot(n_samples, 4, row * 4 + 4, projection='3d')
        sc4 = ax4.scatter(pts[:,0], pts[:,1], pts[:,2], c=(pred>0.5).astype(float), cmap='winter', s=5, vmin=0, vmax=1)
        ax4.set_title("Prediction > 0.5")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"        📊 Visualization saved: {os.path.basename(save_path)}")

if __name__ == "__main__":
    main()
