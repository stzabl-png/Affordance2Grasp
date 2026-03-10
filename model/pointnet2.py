#!/usr/bin/env python3
"""
PointNet++ Affordance 模型定义

纯 PyTorch 实现的 PointNet++ Part Segmentation。
输入: 物体点云 (N, 3) + 法线 (N, 3)
输出: 每个点的接触概率 (N,)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PointNet++ 基础操作
# ============================================================

def square_distance(src, dst):
    """计算两组点之间的平方距离. src: (B,N,3), dst: (B,M,3) → (B,N,M)"""
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)


def farthest_point_sample(xyz, npoint):
    """最远点采样. xyz: (B,N,3) → indices: (B,npoint)"""
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
    """根据索引取点. points: (B,N,C), idx: (B,S) → (B,S,C)"""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def query_ball_point(radius, nsample, xyz, new_xyz):
    """球查询. xyz: (B,N,3), new_xyz: (B,S,3) → group_idx: (B,S,nsample)"""
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
        grouped_xyz_norm = (grouped_xyz - new_xyz.unsqueeze(2)) / self.radius  # relative pos normalization

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


# ============================================================
# PointNet++ Segmentation Model
# ============================================================

class PointNet2Seg(nn.Module):
    """PointNet++ Part Segmentation for Affordance Prediction
    
    多任务版本:
      - 分割头: 每个点的接触概率 (N, 2)
      - 回归头: 全局受力中心 (3,)  [可选]
    """

    def __init__(self, num_classes=2, in_channel=6, predict_force_center=False):
        super().__init__()
        self.predict_force_center = predict_force_center
        
        # v3: SA radii 0.05/0.1/0.2 (was 0.02/0.04/0.08)
        self.sa1 = PointNetSetAbstraction(256, 0.05, 32, in_channel + 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.10, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(64, 0.20, 128, 256 + 3, [256, 256, 512])

        self.fp3 = PointNetFeaturePropagation(256 + 512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(128 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel + 128, [128, 128, 128])

        # 分割头
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)  # v3: was 0.5
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        # 回归头 (受力中心): SA3 全局特征 512-d → 3D
        if predict_force_center:
            self.fc_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 3),
            )

    def forward(self, xyz, features):
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # 分割路径
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)

        x = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        seg_logits = x.permute(0, 2, 1)  # (B, N, 2)

        if self.predict_force_center:
            # 全局特征: SA3 输出 max pool over 64 points → (B, 512)
            global_feat = l3_points.permute(0, 2, 1)  # (B, 512, 64)
            global_feat = torch.max(global_feat, dim=2)[0]  # (B, 512)
            fc_pred = self.fc_head(global_feat)  # (B, 3)
            return seg_logits, fc_pred
        
        return seg_logits

