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
M5_CHECKPOINT = os.path.join("output", "checkpoints_m5", "best_m5_model.pth")


class AffordancePredictor:
    """Affordance 预测器 — 封装模型加载 + 推理.
    
    v6: 自动检测 M5 回归模型 vs 旧版分类模型.
      - M5: 7通道输入 (xyz+normal+human_prior), 1通道 sigmoid 回归输出
      - Legacy: 6通道输入 (xyz+normal), 2通道 softmax 分类输出
    """

    def __init__(self, checkpoint=None, device="cuda", num_points=1024):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_points = num_points

        # 自动选择 checkpoint: 优先 M5, fallback 旧模型
        if checkpoint is None:
            if os.path.exists(M5_CHECKPOINT):
                checkpoint = M5_CHECKPOINT
            else:
                checkpoint = DEFAULT_CHECKPOINT

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        
        # 自动检测模型类型
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        # 判断 M5: sa1 的第一层权重的输入通道 = 7+3=10 (M5) vs 6+3=9 (legacy)
        sa1_key = 'sa1.mlp_convs.0.weight'
        if sa1_key in state_dict:
            in_features = state_dict[sa1_key].shape[1]
            self.is_m5 = (in_features == 10)  # 7 + 3 = 10
        else:
            self.is_m5 = False

        if self.is_m5:
            # M5: 7 通道输入, 1 通道连续回归
            self.model = PointNet2Seg(num_classes=1, in_channel=7).to(self.device)
            self.model.load_state_dict(state_dict)
            print(f"  AffordancePredictor loaded: M5 regression (7ch→sigmoid)")
        else:
            # Legacy: 6 通道输入, 2 类分类
            self.predict_fc = any('fc_head' in k for k in state_dict)
            self.model = PointNet2Seg(
                num_classes=2, in_channel=6, predict_force_center=self.predict_fc
            ).to(self.device)
            self.model.load_state_dict(state_dict)
            mode = "multi-task (seg + fc)" if self.predict_fc else "seg-only"
            print(f"  AffordancePredictor loaded: legacy {mode}")
        
        self.model.eval()

    def predict(self, mesh_path, num_points=None):
        """
        对物体 mesh 进行 affordance 预测。

        Returns:
            points:       (N, 3) np.ndarray
            normals:      (N, 3) np.ndarray
            contact_prob: (N,)   np.ndarray  每点接触概率 [0, 1]
            force_center: (3,)   np.ndarray  预测的受力中心 (可能 None)
        """
        n = num_points or self.num_points

        mesh = trimesh.load(mesh_path, force='mesh')
        points, face_idx = mesh.sample(n, return_index=True)
        normals = mesh.face_normals[face_idx]
        points = points.astype(np.float32)
        normals = normals.astype(np.float32)

        if self.is_m5:
            contact_prob, force_center = self._predict_m5(points, normals, mesh_path)
        else:
            contact_prob, force_center = self.predict_from_points(points, normals)
        return points, normals, contact_prob, force_center

    def _predict_m5(self, points, normals, mesh_path):
        """M5 推理: 需要额外加载 human_prior 作为第 7 通道."""
        import h5py
        
        # 尝试从 human_prior 里找对应的数据
        obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
        hp_candidates = [
            os.path.join("data_hub", "human_prior", f"{obj_name}.hdf5"),
            os.path.join("data_hub", "human_prior", f"grab_{obj_name}.hdf5"),
        ]
        
        human_prior = np.zeros(len(points), dtype=np.float32)
        for hp_path in hp_candidates:
            if os.path.exists(hp_path):
                with h5py.File(hp_path, 'r') as f:
                    hp_pc = f['point_cloud'][()]
                    hp_val = f['human_prior'][()]
                # KNN: 为当前采样点找最近的 human_prior 点
                from scipy.spatial import cKDTree
                tree = cKDTree(hp_pc)
                _, idx = tree.query(points)
                human_prior = hp_val[idx].astype(np.float32)
                break
        
        # 7 通道: xyz(3) + normal(3) + human_prior(1)
        hp_feat = human_prior.reshape(-1, 1)
        features = np.concatenate([points, normals, hp_feat], axis=-1)
        
        pts_t = torch.from_numpy(points).unsqueeze(0).to(self.device)
        feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.model(pts_t, feat_t)  # (B, N) sigmoid output
        
        contact_prob = pred.squeeze(0).cpu().numpy()
        
        # force_center: 用 affordance 加权平均
        if contact_prob.max() > 0.1:
            weights = contact_prob / (contact_prob.sum() + 1e-8)
            force_center = (points * weights[:, None]).sum(axis=0)
        else:
            force_center = points.mean(axis=0)
        
        return contact_prob, force_center

    def predict_from_points(self, points, normals):
        """Legacy: 对已有点云进行 affordance 预测 (6通道)."""
        points = points.astype(np.float32)
        normals = normals.astype(np.float32)

        pts_t = torch.from_numpy(points).unsqueeze(0).to(self.device)
        feat_t = torch.from_numpy(
            np.concatenate([points, normals], axis=-1)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(pts_t, feat_t)

        if hasattr(self, 'predict_fc') and self.predict_fc:
            seg_pred, fc_pred = output
            contact_prob = torch.softmax(seg_pred, dim=-1)[0, :, 1].cpu().numpy()
            force_center = fc_pred[0].cpu().numpy()
            return contact_prob, force_center
        else:
            contact_prob = torch.softmax(output, dim=-1)[0, :, 1].cpu().numpy()
            return contact_prob, None


