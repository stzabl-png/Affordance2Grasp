#!/usr/bin/env python3
"""Threshold search on best_model.pth."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from model.pointnet2 import PointNet2Seg
from model.train_v5 import MultiTaskDataset, get_object_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load best model
ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'output', 'checkpoints_v5', 'best_model.pth')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = PointNet2Seg(num_classes=2, in_channel=6, predict_force_center=True).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded: epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.1%}")
sys.stdout.flush()

# Load val data
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'output', 'dataset')
train_h5 = os.path.join(dataset_dir, 'affordance_train.h5')
val_h5 = os.path.join(dataset_dir, 'affordance_val.h5')
train_obj_ids, val_obj_ids = get_object_split(train_h5, val_h5)

val_v = MultiTaskDataset(val_h5, val_obj_ids, augment=False)
val_t = MultiTaskDataset(train_h5, val_obj_ids, augment=False)
val_v.points = np.concatenate([val_v.points, val_t.points])
val_v.normals = np.concatenate([val_v.normals, val_t.normals])
val_v.labels = np.concatenate([val_v.labels, val_t.labels])
val_v.force_centers = np.concatenate([val_v.force_centers, val_t.force_centers])
val_v.num_samples = len(val_v.points)
print(f"Val samples: {val_v.num_samples}")
sys.stdout.flush()

val_loader = DataLoader(val_v, batch_size=32, shuffle=False, num_workers=0)

# Collect predictions
all_probs = []
all_labels = []
with torch.no_grad():
    for xyz, features, labels, _ in val_loader:
        seg_pred, _ = model(xyz.to(device), features.to(device))
        probs = F.softmax(seg_pred, dim=-1)[:, :, 1]
        all_probs.append(probs.cpu())
        all_labels.append(labels)

all_probs = torch.cat(all_probs).reshape(-1)
all_labels = torch.cat(all_labels).reshape(-1)
print(f"Total predictions: {len(all_probs):,}")
sys.stdout.flush()

# Threshold search
print(f"\n{'Thresh':>7} | {'P':>7} | {'R':>7} | {'F1':>7} | {'IoU':>7}")
print('-' * 50)
best_f1 = 0; best_t = 0.5
for t in np.arange(0.10, 0.90, 0.05):
    pred = (all_probs > t).long()
    tp = ((pred == 1) & (all_labels == 1)).sum().item()
    fp = ((pred == 1) & (all_labels == 0)).sum().item()
    fn = ((pred == 0) & (all_labels == 1)).sum().item()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    marker = ' ★' if f1 > best_f1 else ''
    print(f"{t:>7.2f} | {p:>6.1%} | {r:>6.1%} | {f1:>6.1%} | {iou:>6.1%}{marker}")
    sys.stdout.flush()
    if f1 > best_f1:
        best_f1 = f1; best_t = t

print(f"\n★ Best threshold: {best_t:.2f} → F1={best_f1:.1%}")
