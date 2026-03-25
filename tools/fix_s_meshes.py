#!/usr/bin/env python3
"""修复 S 系列 mesh 缩放: 归一化最大维度到 ~10cm."""
import trimesh, os, glob

mesh_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_hub', 'meshes', 'v1')
s_meshes = sorted(glob.glob(os.path.join(mesh_dir, 'S*.obj')))

fixed, skipped = 0, 0
for mp in s_meshes:
    name = os.path.basename(mp)
    mesh = trimesh.load(mp, force='mesh')
    max_dim = mesh.bounding_box.extents.max()
    
    if max_dim > 1.0:  # >1m = 需要修复
        scale = 0.101 / max_dim  # 归一化到 ~10cm
        mesh.apply_scale(scale)
        new_ext = mesh.bounding_box.extents * 100
        mesh.export(mp)
        print(f"✅ {name}: {max_dim:.1f}m → {new_ext[0]:.1f}×{new_ext[1]:.1f}×{new_ext[2]:.1f} cm")
        fixed += 1
    else:
        ext_cm = mesh.bounding_box.extents * 100
        print(f"⏭️ {name}: {ext_cm[0]:.1f}×{ext_cm[1]:.1f}×{ext_cm[2]:.1f} cm (OK)")
        skipped += 1

print(f"\n修复: {fixed}, 跳过: {skipped}")
