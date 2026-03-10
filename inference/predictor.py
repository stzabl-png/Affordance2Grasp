#!/usr/bin/env python3
"""
Affordance Predictor API
========================
加载训练好的 PointNet++ 模型, 对任意物体 mesh 预测接触概率。

用法:
    from inference.predictor import AffordancePredictor

    predictor = AffordancePredictor()
    points, normals, probs = predictor.predict("/path/to/object.obj")
    contact_mask = probs > 0.5
"""

import os
import numpy as np
import trimesh
import torch

from model.pointnet2 import PointNet2Seg
import config

DEFAULT_CHECKPOINT = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")


class AffordancePredictor:
    """Affordance 预测器 — 封装模型加载 + 推理.
    
    v5: 自动检测 multi-task 模型, 同时输出接触概率 + 受力中心。
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cuda", num_points=1024):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_points = num_points

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        
        # 自动检测是否为 multi-task 模型
        self.predict_fc = ckpt.get('predict_force_center', False)
        if not self.predict_fc:
            # 检查 state_dict 中是否有 fc_head
            self.predict_fc = any('fc_head' in k for k in ckpt['model_state_dict'])
        
        self.model = PointNet2Seg(
            num_classes=2, in_channel=6, predict_force_center=self.predict_fc
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        mode = "multi-task (seg + fc)" if self.predict_fc else "seg-only"
        print(f"  AffordancePredictor loaded: {mode}")

    def predict(self, mesh_path, num_points=None):
        """
        对物体 mesh 进行 affordance 预测。

        Args:
            mesh_path: str, 物体 mesh 文件路径 (.obj / .ply)
            num_points: int, 采样点数 (默认 1024)

        Returns:
            points:       (N, 3) np.ndarray  采样的表面点坐标
            normals:      (N, 3) np.ndarray  对应法线
            contact_prob: (N,)   np.ndarray  每点接触概率 [0, 1]
            force_center: (3,)   np.ndarray  预测的受力中心 (仅 multi-task, 否则 None)
        """
        n = num_points or self.num_points

        mesh = trimesh.load(mesh_path, force='mesh')
        points, face_idx = mesh.sample(n, return_index=True)
        normals = mesh.face_normals[face_idx]
        points = points.astype(np.float32)
        normals = normals.astype(np.float32)

        contact_prob, force_center = self.predict_from_points(points, normals)
        return points, normals, contact_prob, force_center

    def predict_from_points(self, points, normals):
        """
        对已有点云进行 affordance 预测 (不需要 mesh)。

        Args:
            points:  (N, 3) np.ndarray
            normals: (N, 3) np.ndarray

        Returns:
            contact_prob: (N,) np.ndarray  每点接触概率 [0, 1]
            force_center: (3,) np.ndarray  预测的受力中心 (仅 multi-task, 否则 None)
        """
        points = points.astype(np.float32)
        normals = normals.astype(np.float32)

        pts_t = torch.from_numpy(points).unsqueeze(0).to(self.device)
        feat_t = torch.from_numpy(
            np.concatenate([points, normals], axis=-1)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(pts_t, feat_t)

        if self.predict_fc:
            seg_pred, fc_pred = output
            contact_prob = torch.softmax(seg_pred, dim=-1)[0, :, 1].cpu().numpy()
            force_center = fc_pred[0].cpu().numpy()
            return contact_prob, force_center
        else:
            contact_prob = torch.softmax(output, dim=-1)[0, :, 1].cpu().numpy()
            return contact_prob, None

