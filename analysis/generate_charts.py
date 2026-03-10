#!/usr/bin/env python3
"""
Affordance2Grasp — 完整分析图表生成脚本

生成:
  1. 数据集统计（类别分布、接触比例）
  2. V4 训练曲线（Loss, F1, P/R）
  3. Per-category 分析
"""

import os
import sys
import json
import re
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_training_log(log_path):
    """从训练日志解析 epoch 数据."""
    epochs, train_loss, val_loss = [], [], []
    val_acc, val_p, val_r, val_f1, lrs = [], [], [], [], []
    
    with open(log_path) as f:
        for line in f:
            m = re.match(
                r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)',
                line
            )
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                val_acc.append(float(m.group(4)))
                val_p.append(float(m.group(5)))
                val_r.append(float(m.group(6)))
                val_f1.append(float(m.group(7)))
                lrs.append(float(m.group(8)))
    
    return {
        'epochs': epochs, 'train_loss': train_loss, 'val_loss': val_loss,
        'val_acc': val_acc, 'val_p': val_p, 'val_r': val_r,
        'val_f1': val_f1, 'lrs': lrs
    }


# ============================================================
# Chart 1: 数据集类别分布
# ============================================================
def plot_dataset_stats():
    print("📊 [1/3] Dataset Statistics...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('OakInk Contact Dataset — Statistics', fontsize=14, fontweight='bold')
    
    for idx, (name, data_dir) in enumerate([
        ("All Intents (106K)", "/home/lyh/Project/ContactPoint/dataset"),
        ("Hold Only (30K)", "/home/lyh/Project/ContactPoint/dataset_hold"),
    ]):
        info_path = os.path.join(data_dir, "dataset_info.json")
        if not os.path.exists(info_path):
            continue
        with open(info_path) as f:
            info = json.load(f)
        
        # 从 H5 提取 per-category 统计
        categories = []
        contact_by_cat = {}
        for split in ['train', 'val']:
            h5_path = os.path.join(data_dir, f"affordance_{split}.h5")
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    if 'data/categories' in f:
                        cats = [c.decode() if isinstance(c, bytes) else c for c in f['data/categories'][:]]
                        labels = f['data/labels'][:]
                        categories.extend(cats)
                        for i, c in enumerate(cats):
                            if c not in contact_by_cat:
                                contact_by_cat[c] = []
                            contact_by_cat[c].append(labels[i].mean() * 100)
        
        if categories:
            cat_counts = Counter(categories)
            sorted_cats = sorted(cat_counts.keys(), key=lambda x: cat_counts[x], reverse=True)
            counts = [cat_counts[c] for c in sorted_cats]
            
            ax = axes[idx]
            bars = ax.barh(range(len(sorted_cats)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(sorted_cats))))
            ax.set_yticks(range(len(sorted_cats)))
            ax.set_yticklabels(sorted_cats, fontsize=9)
            ax.set_xlabel('Number of Samples')
            ax.set_title(f'{name}\n({info["total_samples"]} samples, {info["num_objects"]} objects)')
            ax.invert_yaxis()
            
            for bar, count in zip(bars, counts):
                ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                       str(count), va='center', fontsize=8)
        else:
            # 没有 categories 字段, 用 info 里的信息
            ax = axes[idx]
            cats = info.get('categories', [])
            ax.barh(range(len(cats)), [1]*len(cats))
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(cats, fontsize=9)
            ax.set_title(f'{name}\n({info["total_samples"]} samples, {info["num_objects"]} objects)')
            ax.invert_yaxis()
            ax.set_xlabel('(count not available per category)')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "01_dataset_stats.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


# ============================================================
# Chart 2: 训练曲线
# ============================================================
def plot_training_curves():
    print("📊 [2/3] Training Curves...")
    
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "output", "train_v4.log")
    if not os.path.exists(log_path):
        print("   ⚠️  train_v4.log not found, skipping")
        return
    
    data = parse_training_log(log_path)
    epochs = data['epochs']
    
    if len(epochs) == 0:
        print("   ⚠️  No epoch data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Affordance Training v4 — Object-level Split (Val = 13 unseen objects)',
                 fontsize=13, fontweight='bold')
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, data['train_loss'], label='Train Loss', color='#2196F3', linewidth=1.5)
    ax.plot(epochs, data['val_loss'], label='Val Loss', color='#F44336', linewidth=1.5, alpha=0.7)
    ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    
    # F1
    ax = axes[0, 1]
    ax.plot(epochs, data['val_f1'], color='#4CAF50', linewidth=2)
    best_idx = np.argmax(data['val_f1'])
    ax.scatter([epochs[best_idx]], [data['val_f1'][best_idx]], color='red', s=100, zorder=5)
    ax.annotate(f'Best: {data["val_f1"][best_idx]:.1f}% (ep{epochs[best_idx]})',
                xy=(epochs[best_idx], data['val_f1'][best_idx]), fontsize=11, fontweight='bold',
                xytext=(epochs[best_idx]+10, data['val_f1'][best_idx]+1.5),
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_title('Val F1 Score'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 (%)')
    
    # Precision & Recall
    ax = axes[1, 0]
    ax.plot(epochs, data['val_p'], label='Precision', color='#FF9800', linewidth=1.5)
    ax.plot(epochs, data['val_r'], label='Recall', color='#9C27B0', linewidth=1.5)
    ax.axhline(y=np.mean(data['val_p']), color='#FF9800', linestyle='--', alpha=0.3)
    ax.axhline(y=np.mean(data['val_r']), color='#9C27B0', linestyle='--', alpha=0.3)
    ax.set_title(f'Val Precision (avg {np.mean(data["val_p"]):.1f}%) & Recall (avg {np.mean(data["val_r"]):.1f}%)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('%')
    
    # LR
    ax = axes[1, 1]
    ax.plot(epochs, data['lrs'], color='#607D8B', linewidth=1.5)
    ax.set_title('Learning Rate (CosineWarmRestarts T₀=50)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "02_training_curves_v4.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


# ============================================================
# Chart 3: Object-level Split 信息
# ============================================================
def plot_split_info():
    print("📊 [3/3] Object Split Info...")
    
    split_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "output", "checkpoints_v4_all", "split_info.json")
    if not os.path.exists(split_path):
        print("   ⚠️  split_info.json not found, skipping")
        return
    
    with open(split_path) as f:
        split = json.load(f)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Object-level Train/Val Split', fontsize=14, fontweight='bold')
    
    train_objs = split['train_objects']
    val_objs = split['val_objects']
    
    # 简单柱状图
    labels = ['Train', 'Val']
    obj_counts = [len(train_objs), len(val_objs)]
    sample_counts = [split['train_samples'], split['val_samples']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, obj_counts, width, label='Objects', color='#2196F3')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, sample_counts, width, label='Samples', color='#FF9800', alpha=0.7)
    
    ax.set_ylabel('Number of Objects', color='#2196F3')
    ax2.set_ylabel('Number of Samples', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    ax.bar_label(bars1, padding=3)
    ax2.bar_label(bars2, padding=3)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 添加物体列表文字
    val_text = f"Val objects (unseen): {', '.join(val_objs)}"
    fig.text(0.5, -0.02, val_text, ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "03_object_split.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("Generating Analysis Charts")
    print("=" * 50)
    
    plot_dataset_stats()
    plot_training_curves()
    plot_split_info()
    
    print(f"\n✅ All charts saved to: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  {f}")
