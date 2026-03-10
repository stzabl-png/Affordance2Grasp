#!/usr/bin/env python3
"""
PointNet++ Affordance Training v3

改进:
  1. SA 半径放大: 0.02/0.04/0.08 → 0.05/0.1/0.2
  2. Tversky + Focal Loss (针对 Precision 低)
  3. SO(3) 全轴随机旋转 (代替仅 Y 轴)
  4. Dropout 0.5 → 0.3
  5. 训练后 Threshold Search (找最优 F1 的阈值)

用法:
    python train_v3.py --epochs 100
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
matplotlib.use('Agg')  # 无头模式, 不需要 display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# Dataset (SO(3) random rotation)
# ============================================================

def random_rotation_matrix():
    """生成均匀分布的 SO(3) 随机旋转矩阵."""
    # 使用 QR 分解方法
    z = np.random.randn(3, 3)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q @ np.diag(ph)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


class AffordanceDataset(Dataset):
    """从 HDF5 文件加载点云 + 接触标签."""

    def __init__(self, h5_path, augment=True):
        self.h5_path = h5_path
        self.augment = augment

        with h5py.File(h5_path, 'r') as f:
            self.points = f['data/points'][:]
            self.normals = f['data/normals'][:]
            self.labels = f['data/labels'][:]

        self.num_samples = len(self.points)
        print(f"  Loaded {self.num_samples} samples from {os.path.basename(h5_path)}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pts = self.points[idx].copy()
        nrm = self.normals[idx].copy()
        lbl = self.labels[idx].copy()

        if self.augment:
            # ⭐ SO(3) 全轴随机旋转 (替代仅 Y 轴)
            R = random_rotation_matrix()
            pts = pts @ R.T
            nrm = nrm @ R.T

            # 随机缩放
            scale = np.random.uniform(0.8, 1.2)
            pts *= scale

            # 随机平移
            shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
            pts += shift

            # 随机抖动
            jitter = np.random.normal(0, 0.002, size=pts.shape).astype(np.float32)
            pts += jitter

            # 随机丢点 (30% 概率丢 10% 的点)
            if np.random.rand() < 0.3:
                n = len(pts)
                keep = np.random.choice(n, int(n * 0.9), replace=False)
                drop = np.setdiff1d(np.arange(n), keep)
                # 用随机已有点填充被丢的点
                fill = np.random.choice(keep, len(drop), replace=True)
                pts[drop] = pts[fill]
                nrm[drop] = nrm[fill]
                # 标签也对应填充
                lbl[drop] = lbl[fill]

        features = np.concatenate([pts, nrm], axis=-1)

        return (
            torch.from_numpy(pts),
            torch.from_numpy(features),
            torch.from_numpy(lbl).long()
        )


# ============================================================
# PointNet++ Modules
# ============================================================

def square_distance(src, dst):
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)

        grouped_xyz = index_points(xyz, idx)
        # ⭐ 相对位置归一化 (PointNeXt)
        grouped_xyz_norm = (grouped_xyz - new_xyz.unsqueeze(2)) / self.radius

        if points is not None:
            grouped_points = index_points(points, idx)
            grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz_norm

        grouped_points = grouped_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1), dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)


class PointNet2Seg(nn.Module):
    """PointNet++ Seg v3: 大半径 + 低 Dropout"""

    def __init__(self, num_classes=2, in_channel=6):
        super().__init__()

        # ⭐ Encoder: 半径从 0.02/0.04/0.08 → 0.05/0.1/0.2
        self.sa1 = PointNetSetAbstraction(256, 0.05, 32, in_channel + 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.10, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(64, 0.20, 128, 256 + 3, [256, 256, 512])

        # Decoder
        self.fp3 = PointNetFeaturePropagation(256 + 512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(128 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel + 128, [128, 128, 128])

        # Head: ⭐ Dropout 0.5 → 0.3
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, features):
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)

        x = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


# ============================================================
# Loss: Focal + Tversky
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred: (B*N, 2), target: (B*N,)
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        # alpha weighting
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss — 可以独立调 FP 和 FN 的权重."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # FP 权重 (低 = 容忍 FP)
        self.beta = beta    # FN 权重 (高 = 惩罚 FN)
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B*N, 2), target: (B*N,)
        prob = F.softmax(pred, dim=-1)[:, 1]  # contact prob
        target_f = target.float()

        tp = (prob * target_f).sum()
        fp = (prob * (1 - target_f)).sum()
        fn = ((1 - prob) * target_f).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Focal + Tversky 组合 Loss."""
    def __init__(self, focal_weight=0.6, tversky_weight=0.4):
        super().__init__()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.fw = focal_weight
        self.tw = tversky_weight

    def forward(self, pred, target):
        return self.fw * self.focal(pred, target) + self.tw * self.tversky(pred, target)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(pred, target, threshold=0.5):
    """计算 metrics. 支持自定义 threshold."""
    prob = F.softmax(pred, dim=-1)[:, :, 1] if pred.dim() == 3 else F.softmax(pred, dim=-1)[:, 1]
    pred_cls = (prob > threshold).long()
    target_flat = target.reshape(-1)
    pred_flat = pred_cls.reshape(-1)

    correct = (pred_flat == target_flat).float().mean().item()
    tp = ((pred_flat == 1) & (target_flat == 1)).float().sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).float().sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).float().sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "accuracy": correct, "precision": precision,
        "recall": recall, "f1": f1, "iou": iou
    }


# ============================================================
# Training / Eval
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
    n_batches = 0

    for xyz, features, labels in loader:
        xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)

        optimizer.zero_grad()
        pred = model(xyz, features)
        loss = criterion(pred.reshape(-1, 2), labels.reshape(-1))
        loss.backward()

        # 梯度裁剪
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
    all_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
    n_batches = 0

    for xyz, features, labels in loader:
        xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)

        pred = model(xyz, features)
        loss = criterion(pred.reshape(-1, 2), labels.reshape(-1))

        total_loss += loss.item()
        metrics = compute_metrics(pred, labels)
        for k in all_metrics:
            all_metrics[k] += metrics[k]
        n_batches += 1

    return total_loss / n_batches, {k: v / n_batches for k, v in all_metrics.items()}


@torch.no_grad()
def threshold_search(model, loader, device):
    """训练后扫描 threshold 找最优 F1."""
    model.eval()
    all_probs = []
    all_labels = []

    for xyz, features, labels in loader:
        xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)
        pred = model(xyz, features)
        probs = F.softmax(pred, dim=-1)[:, :, 1]
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).reshape(-1)
    all_labels = torch.cat(all_labels, dim=0).reshape(-1)

    best_f1 = 0
    best_thresh = 0.5
    results = []

    for thresh in np.arange(0.3, 0.85, 0.05):
        pred_cls = (all_probs > thresh).long()
        tp = ((pred_cls == 1) & (all_labels == 1)).float().sum().item()
        fp = ((pred_cls == 1) & (all_labels == 0)).float().sum().item()
        fn = ((pred_cls == 0) & (all_labels == 1)).float().sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        results.append((thresh, prec, rec, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print(f"{'-'*48}")
    for thresh, prec, rec, f1 in results:
        marker = " ★" if abs(thresh - best_thresh) < 0.01 else ""
        print(f"{thresh:>10.2f} | {prec:>9.1%} | {rec:>9.1%} | {f1:>9.1%}{marker}")

    return best_thresh, best_f1


# ============================================================
# Visualization (自动保存 PNG)
# ============================================================

@torch.no_grad()
def save_visualization(model, val_dataset, device, save_path, epoch, history=None):
    """保存可视化 PNG: 4 个样本 × 3 列 (GT / Pred / Heatmap) + 训练曲线."""
    model.eval()
    n_samples = 4
    indices = np.random.choice(len(val_dataset), n_samples, replace=False)

    fig = plt.figure(figsize=(18, 5 * n_samples + (4 if history else 0)))
    n_rows = n_samples + (1 if history else 0)

    for row, idx in enumerate(indices):
        pts_t, feat_t, lbl_t = val_dataset[idx]
        pts = pts_t.numpy()
        lbl = lbl_t.numpy()

        # 推理
        pred = model(pts_t.unsqueeze(0).to(device), feat_t.unsqueeze(0).to(device))
        prob = F.softmax(pred, dim=-1)[0, :, 1].cpu().numpy()
        pred_mask = prob > 0.5
        gt_mask = lbl > 0

        # 统计
        tp = (gt_mask & pred_mask).sum()
        fp = (~gt_mask & pred_mask).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(gt_mask.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        for col, (title, colors) in enumerate([
            (f'GT (contact={gt_mask.sum()})', _gt_colors(gt_mask)),
            (f'Pred (TP={tp} FP={fp} F1={f1:.0%})', _pred_colors(gt_mask, pred_mask)),
            (f'Heatmap (max={prob.max():.2f})', _heat_colors(prob)),
        ]):
            ax = fig.add_subplot(n_rows, 3, row * 3 + col + 1, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=3, alpha=0.8)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # 训练曲线
    if history and len(history) > 1:
        ax_loss = fig.add_subplot(n_rows, 3, n_samples * 3 + 1)
        epochs = [h['epoch'] for h in history]
        ax_loss.plot(epochs, [h['train_loss'] for h in history], label='Train', color='blue')
        ax_loss.plot(epochs, [h['val_loss'] for h in history], label='Val', color='red')
        ax_loss.set_title('Loss'); ax_loss.legend(); ax_loss.grid(True, alpha=0.3)

        ax_f1 = fig.add_subplot(n_rows, 3, n_samples * 3 + 2)
        ax_f1.plot(epochs, [h['val_f1'] for h in history], color='green', linewidth=2)
        ax_f1.set_title(f'Val F1 (best={max(h["val_f1"] for h in history):.1%})')
        ax_f1.grid(True, alpha=0.3)

        ax_pr = fig.add_subplot(n_rows, 3, n_samples * 3 + 3)
        ax_pr.plot(epochs, [h['val_precision'] for h in history], label='Precision', color='orange')
        ax_pr.plot(epochs, [h['val_recall'] for h in history], label='Recall', color='purple')
        ax_pr.set_title('Precision / Recall'); ax_pr.legend(); ax_pr.grid(True, alpha=0.3)

    fig.suptitle(f'Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"        📊 Visualization saved: {os.path.basename(save_path)}")


def _gt_colors(gt_mask):
    colors = np.full((len(gt_mask), 3), 0.75)  # gray
    colors[gt_mask] = [1, 0.2, 0.2]  # red
    return colors

def _pred_colors(gt_mask, pred_mask):
    colors = np.full((len(gt_mask), 3), 0.75)  # gray = TN
    colors[gt_mask & pred_mask] = [0.2, 0.9, 0.2]   # green = TP
    colors[~gt_mask & pred_mask] = [1.0, 0.5, 0.0]  # orange = FP
    colors[gt_mask & ~pred_mask] = [0.5, 0.0, 0.5]  # purple = FN
    return colors

def _heat_colors(prob):
    cmap = plt.cm.jet
    return cmap(prob)[:, :3]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ Affordance v3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_v3"))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PointNet++ Affordance Training v3")
    print("=" * 60)
    print(f"  改进: SA半径↑, Tversky+Focal Loss, SO(3)旋转, Dropout↓")
    print(f"  Device:      {device}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Dataset:     {args.dataset_dir}")
    print(f"  Checkpoints: {args.save_dir}")
    sys.stdout.flush()

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = AffordanceDataset(
        os.path.join(args.dataset_dir, "affordance_train.h5"), augment=True)
    val_dataset = AffordanceDataset(
        os.path.join(args.dataset_dir, "affordance_val.h5"), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    sys.stdout.flush()

    # Model
    model = PointNet2Seg(num_classes=2, in_channel=6).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params: {n_params:,}")

    # ⭐ Focal + Tversky Loss
    criterion = CombinedLoss(focal_weight=0.6, tversky_weight=0.4)
    print(f"  Loss: Focal(α=0.75,γ=2) + Tversky(α=0.3,β=0.7)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\n{'='*60}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Val Acc':>7} | {'Val F1':>7} | {'Val IoU':>7} | {'LR':>8}")
    print(f"{'-'*60}")
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
              f"{val_metrics['accuracy']:>6.1%} | {val_metrics['f1']:>6.1%} | "
              f"{val_metrics['iou']:>6.1%} | {lr:>8.6f}  ({elapsed:.0f}s)")
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
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"        ★ New best! F1={best_val_f1:.1%}")
            sys.stdout.flush()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth"))
            # ⭐ 每 10 epoch 保存可视化
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

    # ⭐ Threshold Search
    print(f"\n{'='*60}")
    print(f"Threshold Search (finding optimal F1)...")
    best_thresh, best_thresh_f1 = threshold_search(model, val_loader, device)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best Val F1 (thresh=0.5): {best_val_f1:.1%}")
    print(f"  Best Threshold:           {best_thresh:.2f} → F1={best_thresh_f1:.1%}")
    print(f"  Best model:   {args.save_dir}/best_model.pth")
    print(f"  History:      {args.save_dir}/training_history.json")
    print(f"{'='*60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
