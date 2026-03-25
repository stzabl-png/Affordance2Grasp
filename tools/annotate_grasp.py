#!/usr/bin/env python3
"""
交互式抓取标注工具 — Web 3D Viewer + 自动穿透 + HDF5 导出.

Shift+单击表面 → 自动沿法线穿透物体找到对面 → 两个点一步完成
箭头显示机器人接近方向

Usage:
    python3 tools/annotate_grasp.py --obj A16013
"""

import os, sys, json, argparse, threading, webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import numpy as np
import trimesh
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PORT = 8765
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# Grasp 计算
# ============================================================
def compute_grasp_from_two_points(p1, p2, mesh, approach_hint=None):
    TCP_OFFSET = 0.105
    MAX_GRIPPER_OPEN = 0.08
    p1, p2 = np.array(p1, dtype=np.float64), np.array(p2, dtype=np.float64)
    grasp_center = (p1 + p2) / 2.0
    finger_dir = p2 - p1
    width = np.linalg.norm(finger_dir)
    finger_dir /= (width + 1e-8)
    if width > MAX_GRIPPER_OPEN:
        return None, f"宽度 {width*100:.1f}cm > 8cm"

    if approach_hint is not None:
        approach = np.array(approach_hint, dtype=np.float64)
        approach -= np.dot(approach, finger_dir) * finger_dir
        approach /= (np.linalg.norm(approach) + 1e-8)
    else:
        candidates = [np.array([0,1,0]), np.array([0,-1,0]),
                      np.array([1,0,0]), np.array([-1,0,0]), np.array([0,0,-1])]
        best, best_orth = candidates[0], 0
        for c in candidates:
            orth = 1 - abs(np.dot(c, finger_dir))
            if orth > best_orth: best_orth, best = orth, c.copy()
        approach = best - np.dot(best, finger_dir) * finger_dir
        approach /= (np.linalg.norm(approach) + 1e-8)

    y_body = np.cross(approach, finger_dir)
    y_body /= (np.linalg.norm(y_body) + 1e-8)
    finger_dir = np.cross(y_body, approach)
    finger_dir /= (np.linalg.norm(finger_dir) + 1e-8)
    rot = np.column_stack([finger_dir, y_body, approach])
    if np.linalg.det(rot) < 0:
        finger_dir = -finger_dir
        rot = np.column_stack([finger_dir, y_body, approach])

    panda_hand_pos = grasp_center - approach * TCP_OFFSET
    gripper_width = float(np.clip(width + 0.005, 0.01, MAX_GRIPPER_OPEN))

    return {
        "position": grasp_center.astype(np.float32),       # ⭐ 指尖夹持中心 (sim 内部处理 TCP 偏移)
        "rotation": rot.astype(np.float32),
        "gripper_width": gripper_width,
        "grasp_point": grasp_center.astype(np.float32),
        "width": float(width),
        "approach": approach.tolist(),
        "finger_dir": finger_dir.tolist(),
    }, None


def raycast_through_mesh(mesh, origin, direction):
    """从 origin 沿 direction 射入 mesh, 返回所有交点."""
    origin = np.array(origin, dtype=np.float64)
    direction = np.array(direction, dtype=np.float64)
    direction /= (np.linalg.norm(direction) + 1e-8)

    try:
        locations, _, face_ids = mesh.ray.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
            multiple_hits=True
        )
    except Exception:
        return []

    if len(locations) == 0:
        return []

    dists = np.linalg.norm(locations - origin, axis=1)
    order = np.argsort(dists)
    return [locations[i].tolist() for i in order]


def save_grasps_hdf5(grasps, obj_id, mesh_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{obj_id}_grasp.hdf5")
    with h5py.File(out_path, 'w') as f:
        f.attrs['object_id'] = obj_id
        f.attrs['mesh_path'] = mesh_path
        f.attrs['num_candidates'] = len(grasps)
        f.attrs['source'] = 'manual_annotation'
        for i, g in enumerate(grasps):
            grp = f.create_group(f'candidate_{i}')
            grp.create_dataset('position', data=g['position'])
            grp.create_dataset('rotation', data=g['rotation'])
            grp.create_dataset('grasp_point', data=g['grasp_point'])
            grp.attrs['gripper_width'] = g['gripper_width']
            grp.attrs['name'] = g.get('name', f'manual_{i}')
            grp.attrs['approach_type'] = g.get('approach_type', 'horizontal')
            grp.attrs['score'] = g.get('score', 80.0)
    print(f"✅ 保存 {len(grasps)} 个抓取 → {out_path}")
    return out_path


# ============================================================
# HTML
# ============================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Grasp Annotator — {OBJ_ID}</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; overflow: hidden; }
#canvas-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
#panel {
    position: absolute; top: 10px; right: 10px; width: 340px;
    background: rgba(20,20,40,0.95); border-radius: 12px; padding: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6); z-index: 10;
    max-height: 92vh; overflow-y: auto;
}
h2 { color: #64ffda; font-size: 16px; margin-bottom: 6px; }
h3 { color: #ffd54f; font-size: 13px; margin: 10px 0 4px; }
.info { font-size: 12px; color: #aaa; margin-bottom: 4px; line-height: 1.5; }
.highlight { color: #64ffda; font-weight: bold; }
.point-display { font-family: monospace; font-size: 11px; padding: 4px 8px; background: #2a2a4a; border-radius: 4px; margin: 2px 0; }
.btn { display: inline-block; padding: 8px 14px; border: none; border-radius: 6px;
       cursor: pointer; font-size: 13px; margin: 3px 2px; transition: all 0.2s; }
.btn-primary { background: #64ffda; color: #1a1a2e; font-weight: bold; }
.btn-primary:hover { background: #4dd0b8; }
.btn-danger { background: #ff5252; color: white; }
.btn-secondary { background: #444; color: #ccc; }
.grasp-item { background: #2a2a4a; border-radius: 8px; padding: 8px; margin: 5px 0; border-left: 3px solid #64ffda; font-size: 12px; }
.grasp-item.error { border-left-color: #ff5252; }
#status { position: absolute; bottom: 10px; left: 10px; font-size: 13px; color: #64ffda; z-index: 10;
          background: rgba(20,20,40,0.8); padding: 6px 12px; border-radius: 8px; }
select { background: #2a2a4a; color: #e0e0e0; border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }
#axis-indicator {
    position: absolute; top: 10px; left: 10px; z-index: 10;
    font-size: 20px; font-weight: bold; padding: 8px 16px;
    border-radius: 8px; display: none;
}
#axis-indicator.active { display: block; }
#axis-indicator.axis-x { background: rgba(255,68,68,0.85); color: white; }
#axis-indicator.axis-y { background: rgba(68,255,68,0.85); color: #1a1a2e; }
#axis-indicator.axis-z { background: rgba(68,68,255,0.85); color: white; }
.kbd { display: inline-block; background: #444; color: #64ffda; padding: 1px 6px; border-radius: 3px;
       font-family: monospace; font-size: 12px; border: 1px solid #666; }
</style>
</head>
<body>
<div id="canvas-container"></div>

<div id="panel">
    <h2>🤖 Grasp Annotator — {OBJ_ID} {GT_STATUS}</h2>
    <div class="info">左键旋转 | 右键平移 | 滚轮缩放</div>
    <div class="info"><span class="highlight">Shift+单击</span> 表面 → 自动穿透找对面</div>
    <div class="info">
        <span class="kbd">X</span> <span class="kbd">Y</span> <span class="kbd">Z</span> 锁定轴向 |
        <span class="kbd">Q</span> <span class="kbd">E</span> 旋转手指方向±10° |
        <span class="kbd">Esc</span> 解锁
    </div>

    <h3>📍 当前抓取点</h3>
    <div id="point1" class="point-display">🔴 手指1: —</div>
    <div id="point2" class="point-display">🔵 手指2: —</div>
    <div id="width-info" class="point-display">📏 宽度: —</div>

    <h3>🧭 接近方向 (机器人从哪侧来)</h3>
    <select id="approach-select">
        <option value="auto">自动选择 (偏好水平)</option>
        <option value="0,1,0">+Y 正面 (朝你)</option>
        <option value="0,-1,0">-Y 背面</option>
        <option value="1,0,0">+X 右侧</option>
        <option value="-1,0,0">-X 左侧</option>
        <option value="0,0,-1">-Z Top-down (从上)</option>
    </select>

    <div style="margin-top: 8px;">
        <button class="btn btn-primary" onclick="addGrasp()">➕ 确认并添加</button>
        <button class="btn btn-danger" onclick="clearPoints()">🗑️ 清除</button>
    </div>

    <h3>📋 已标注 (<span id="grasp-count">0</span>)</h3>
    <div id="grasp-list"></div>

    <div style="margin-top: 10px; display: flex; gap: 6px;">
        <button class="btn btn-primary" onclick="saveAll()" style="flex:1;">💾 保存 HDF5</button>
        <button class="btn btn-secondary" onclick="clearAll()">全部清空</button>
    </div>
    <div id="save-result" style="margin-top: 6px; font-size: 12px;"></div>

    <div style="margin-top: 12px; border-top: 1px solid #444; padding-top: 10px;">
        <div style="display:flex; gap:6px;">
            <button class="btn btn-secondary" onclick="prevObject()" style="flex:1; font-size: 14px;">⏮️ 上一个</button>
            <button class="btn btn-primary" onclick="nextObject()" style="flex:1; font-size: 14px;">⏭️ 下一个</button>
        </div>
        <div id="progress-info" class="info" style="color:#64ffda; margin-top:4px;"></div>
        <div id="existing-info" class="info" style="color:#ffd54f;"></div>
    </div>
</div>

<div id="axis-indicator"></div>
<div id="status">加载中...</div>

<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

let scene, camera, renderer, controls, meshObj, raycaster, mouse;
let currentVisuals = [];
let point1 = null, point2 = null;
let grasps = [], graspVisuals = [];
let lockedAxis = null;  // null | 'x' | 'y' | 'z'
let objCenter = new THREE.Vector3();
let objSize = new THREE.Vector3();
let currentApproach = null;  // THREE.Vector3, 当前接近方向

function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.001, 10);
    camera.position.set(0.15, 0.15, 0.25);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('canvas-container').appendChild(renderer.domElement);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.1;

    scene.add(new THREE.AmbientLight(0x404040, 2));
    const dl1 = new THREE.DirectionalLight(0xffffff, 1.5); dl1.position.set(1,1,1); scene.add(dl1);
    const dl2 = new THREE.DirectionalLight(0x8888ff, 0.8); dl2.position.set(-1,-1,0.5); scene.add(dl2);

    // 坐标轴 (红X 绿Y 蓝Z)
    const axisHelper = new THREE.AxesHelper(0.06);
    scene.add(axisHelper);

    // 坐标轴标签
    addAxisLabel('X', new THREE.Vector3(0.065, 0, 0), '#ff4444');
    addAxisLabel('Y', new THREE.Vector3(0, 0.065, 0), '#44ff44');
    addAxisLabel('Z', new THREE.Vector3(0, 0, 0.065), '#4444ff');

    const grid = new THREE.GridHelper(0.3, 30, 0x444444, 0x333333);
    scene.add(grid);
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    const loader = new OBJLoader();
    loader.load('/mesh.obj', (obj) => {
        obj.traverse(child => {
            if (child.isMesh) {
                child.material = new THREE.MeshPhongMaterial({
                    color: 0x6688aa, transparent: true, opacity: 0.8, side: THREE.DoubleSide,
                });
            }
        });
        meshObj = obj;
        scene.add(obj);
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        objCenter.copy(center);
        objSize.copy(size);
        controls.target.copy(center);
        camera.position.set(center.x + size.x*2, center.y + size.y, center.z + size.z*2);
        controls.update();
        document.getElementById('status').textContent =
            `✅ 已加载 | ${(size.x*100).toFixed(1)}×${(size.y*100).toFixed(1)}×${(size.z*100).toFixed(1)} cm | Shift+点击穿透`;
    });

    renderer.domElement.addEventListener('click', onClick);
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth/window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    window.addEventListener('keydown', onKeyDown);
    animate();
}

function addAxisLabel(text, pos, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 64; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'bold 48px Arial';
    ctx.fillStyle = color;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, 32, 32);
    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
    sprite.position.copy(pos);
    sprite.scale.set(0.02, 0.02, 1);
    scene.add(sprite);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// ---- 轴锁定 + 相机对齐 ----
function onKeyDown(e) {
    const key = e.key.toLowerCase();
    if (key === 'escape') {
        lockedAxis = null;
        updateAxisIndicator();
        return;
    }
    if (key === 'q') { rotateGrasp(-10); return; }
    if (key === 'e') { rotateGrasp(10); return; }
    if (!['x','y','z'].includes(key)) return;
    if (lockedAxis === key) {
        lockedAxis = null;
    } else {
        lockedAxis = key;
    }
    updateAxisIndicator();
    if (lockedAxis) snapCameraToAxis(lockedAxis);
}

function snapCameraToAxis(axis) {
    const dist = Math.max(objSize.x, objSize.y, objSize.z) * 3;
    const target = objCenter.clone();
    controls.target.copy(target);
    switch(axis) {
        case 'x':  // 沿 X 轴看 → 看到 YZ 平面
            camera.position.set(target.x + dist, target.y, target.z);
            camera.up.set(0, 0, 1);
            break;
        case 'y':  // 沿 Y 轴看 → 看到 XZ 平面
            camera.position.set(target.x, target.y + dist, target.z);
            camera.up.set(0, 0, 1);
            break;
        case 'z':  // 沿 Z 轴看 → 看到 XY 平面 (俯视)
            camera.position.set(target.x, target.y, target.z + dist);
            camera.up.set(0, 1, 0);
            break;
    }
    camera.lookAt(target);
    controls.update();
}

function updateAxisIndicator() {
    const el = document.getElementById('axis-indicator');
    if (!lockedAxis) {
        el.className = '';
        el.style.display = 'none';
        document.getElementById('status').textContent = '自由模式 | Shift+点击穿透';
        return;
    }
    const labels = { x: '🔴 X轴锁定 — 沿X穿透', y: '🟢 Y轴锁定 — 沿Y穿透', z: '🔵 Z轴锁定 — 沿Z穿透' };
    el.textContent = labels[lockedAxis];
    el.className = `active axis-${lockedAxis}`;
    el.style.display = 'block';
    document.getElementById('status').textContent = `${lockedAxis.toUpperCase()}轴锁定 | Shift+点击沿${lockedAxis.toUpperCase()}轴穿透 | 再按${lockedAxis.toUpperCase()}或Esc解锁`;
}

async function onClick(e) {
    if (!e.shiftKey || !meshObj) return;

    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(meshObj, true);
    if (hits.length === 0) return;

    const hit = hits[0];
    const clickPt = hit.point;

    // 确定穿透方向
    let rayDir;
    if (lockedAxis) {
        // 轴锁定模式: 沿锁定轴的正方向射穿, 另一个方向也试
        const axisVecs = { x: [1,0,0], y: [0,1,0], z: [0,0,1] };
        const av = axisVecs[lockedAxis];
        // 选择让射线从点击点出发,朝向物体中心方向的那个方向
        const toCenter = objCenter.clone().sub(clickPt);
        const axisV = new THREE.Vector3(av[0], av[1], av[2]);
        rayDir = toCenter.dot(axisV) > 0 ? axisV.clone() : axisV.clone().negate();
    } else {
        // 自动模式: 用面法线
        const faceNormal = hit.face.normal.clone();
        const normalMatrix = new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld);
        faceNormal.applyMatrix3(normalMatrix).normalize();
        rayDir = faceNormal.clone().negate();
    }

    clearCurrentVisuals();

    const response = await fetch('/raycast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            origin: [clickPt.x + rayDir.x*0.001,
                     clickPt.y + rayDir.y*0.001,
                     clickPt.z + rayDir.z*0.001],
            direction: [rayDir.x, rayDir.y, rayDir.z],
            click_point: [clickPt.x, clickPt.y, clickPt.z],
            locked_axis: lockedAxis,
        })
    });
    const data = await response.json();

    if (!data.success) {
        document.getElementById('status').textContent = `⚠️ 穿透失败: ${data.message}`;
        return;
    }

    // 设置两个点
    point1 = clickPt.clone();
    point2 = new THREE.Vector3(data.exit_point[0], data.exit_point[1], data.exit_point[2]);

    const width = point1.distanceTo(point2);

    // 可视化: 两个手指球 + 连线
    const m1 = createSphere(point1, 0xff4444, 0.003);  // 红 = 点击点
    const m2 = createSphere(point2, 0x4488ff, 0.003);  // 蓝 = 穿透点
    scene.add(m1); scene.add(m2);
    currentVisuals.push(m1, m2);

    // 连线 (黄色 = 手指方向)
    const lineGeo = new THREE.BufferGeometry().setFromPoints([point1, point2]);
    const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0xffff00 }));
    scene.add(line);
    currentVisuals.push(line);

    // 抓取中心
    const center = point1.clone().add(point2).multiplyScalar(0.5);
    const centerM = createSphere(center, 0x00ff88, 0.002);
    scene.add(centerM);
    currentVisuals.push(centerM);

    // 计算并展示接近方向箭头
    const approachSel = document.getElementById('approach-select').value;
    let approachDir;
    if (approachSel !== 'auto') {
        const parts = approachSel.split(',').map(Number);
        approachDir = new THREE.Vector3(parts[0], parts[1], parts[2]);
    } else {
        // 自动: 取法线方向作为接近方向参考
        approachDir = faceNormal.clone();
        // 确保与 finger_dir 垂直
    }

    // finger_dir
    const fd = point2.clone().sub(point1).normalize();

    // 正交化 approach
    approachDir.sub(fd.clone().multiplyScalar(approachDir.dot(fd))).normalize();
    if (approachDir.length() < 0.1) {
        approachDir.set(0, 0, -1);
        approachDir.sub(fd.clone().multiplyScalar(approachDir.dot(fd))).normalize();
    }

    // 画接近方向箭头 (绿色)
    const arrowLen = 0.08;
    const arrowDir = approachDir.clone();
    const arrow = new THREE.ArrowHelper(arrowDir, center, arrowLen, 0x00ff88, 0.012, 0.008);
    scene.add(arrow);
    currentVisuals.push(arrow);
    currentApproach = approachDir.clone();  // 保存当前接近方向用于 Q/E 旋转

    // 画 panda_hand 位置 (紫色球)
    const handPos = center.clone().sub(approachDir.clone().multiplyScalar(0.105));
    const handM = createSphere(handPos, 0xaa44ff, 0.005);
    scene.add(handM);
    currentVisuals.push(handM);

    // 画从 hand 到 center 的虚线 (手臂)
    const armGeo = new THREE.BufferGeometry().setFromPoints([handPos, center]);
    const armLine = new THREE.Line(armGeo, new THREE.LineDashedMaterial({
        color: 0xaa44ff, dashSize: 0.005, gapSize: 0.003
    }));
    armLine.computeLineDistances();
    scene.add(armLine);
    currentVisuals.push(armLine);

    // 更新 UI
    document.getElementById('point1').textContent =
        `🔴 手指1: [${point1.x.toFixed(4)}, ${point1.y.toFixed(4)}, ${point1.z.toFixed(4)}]`;
    document.getElementById('point2').textContent =
        `🔵 手指2: [${point2.x.toFixed(4)}, ${point2.y.toFixed(4)}, ${point2.z.toFixed(4)}]`;
    document.getElementById('width-info').textContent =
        `📏 宽度: ${(width*100).toFixed(2)} cm ${width > 0.08 ? '⚠️ 超过8cm!' : '✅ 可抓'}`;
    document.getElementById('status').textContent =
        `宽度 ${(width*100).toFixed(2)}cm | 🟢绿箭头=机器人来向 🟣紫球=手腕位置 | 确认后点"添加"`;
}

function createSphere(pos, color, radius) {
    const m = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 16, 16),
        new THREE.MeshBasicMaterial({ color })
    );
    m.position.copy(pos);
    return m;
}

function clearCurrentVisuals() {
    currentVisuals.forEach(o => { scene.remove(o); });
    currentVisuals = [];
    currentApproach = null;
}

function redrawCurrentGrasp() {
    // 清除旧的可视化并重新绘制
    currentVisuals.forEach(o => { scene.remove(o); });
    currentVisuals = [];
    if (!point1 || !point2) return;

    const width = point1.distanceTo(point2);
    const m1 = createSphere(point1, 0xff4444, 0.003);
    const m2 = createSphere(point2, 0x4488ff, 0.003);
    scene.add(m1); scene.add(m2);
    currentVisuals.push(m1, m2);

    const lineGeo = new THREE.BufferGeometry().setFromPoints([point1, point2]);
    const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0xffff00 }));
    scene.add(line); currentVisuals.push(line);

    const center = point1.clone().add(point2).multiplyScalar(0.5);
    const centerM = createSphere(center, 0x00ff88, 0.002);
    scene.add(centerM); currentVisuals.push(centerM);

    if (currentApproach) {
        const arrow = new THREE.ArrowHelper(currentApproach.clone(), center, 0.08, 0x00ff88, 0.012, 0.008);
        scene.add(arrow); currentVisuals.push(arrow);

        const handPos = center.clone().sub(currentApproach.clone().multiplyScalar(0.105));
        const handM = createSphere(handPos, 0xaa44ff, 0.005);
        scene.add(handM); currentVisuals.push(handM);

        const armGeo = new THREE.BufferGeometry().setFromPoints([handPos, center]);
        const armLine = new THREE.Line(armGeo, new THREE.LineDashedMaterial({
            color: 0xaa44ff, dashSize: 0.005, gapSize: 0.003
        }));
        armLine.computeLineDistances();
        scene.add(armLine); currentVisuals.push(armLine);
    }

    document.getElementById('point1').textContent =
        `🔴 手指1: [${point1.x.toFixed(4)}, ${point1.y.toFixed(4)}, ${point1.z.toFixed(4)}]`;
    document.getElementById('point2').textContent =
        `🔵 手指2: [${point2.x.toFixed(4)}, ${point2.y.toFixed(4)}, ${point2.z.toFixed(4)}]`;
    document.getElementById('width-info').textContent =
        `📏 宽度: ${(width*100).toFixed(2)} cm ${width > 0.08 ? '⚠️ 超过8cm!' : '✅ 可抓'}`;
}

function rotateGrasp(degrees) {
    if (!point1 || !point2 || !currentApproach) {
        document.getElementById('status').textContent = '⚠️ 先 Shift+点击选择抓取点后再旋转';
        return;
    }
    const center = point1.clone().add(point2).multiplyScalar(0.5);
    const angle = degrees * Math.PI / 180;
    const axis = currentApproach.clone().normalize();
    const quat = new THREE.Quaternion().setFromAxisAngle(axis, angle);

    point1.sub(center).applyQuaternion(quat).add(center);
    point2.sub(center).applyQuaternion(quat).add(center);

    redrawCurrentGrasp();
    document.getElementById('status').textContent = `旋转 ${degrees > 0 ? '+' : ''}${degrees}° | Q/E 继续旋转`;
}

window.clearPoints = function() {
    clearCurrentVisuals();
    point1 = null; point2 = null;
    document.getElementById('point1').textContent = '🔴 手指1: —';
    document.getElementById('point2').textContent = '🔵 手指2: —';
    document.getElementById('width-info').textContent = '📏 宽度: —';
    document.getElementById('status').textContent = 'Shift+点击表面穿透';
};

window.addGrasp = function() {
    if (!point1 || !point2) { alert('Shift+点击表面先生成抓取'); return; }
    const approachSel = document.getElementById('approach-select').value;
    let approach = null;
    if (approachSel !== 'auto') approach = approachSel.split(',').map(Number);

    grasps.push({
        p1: [point1.x, point1.y, point1.z],
        p2: [point2.x, point2.y, point2.z],
        approach: approach,
    });

    // 保留可视化, 移到 graspVisuals
    graspVisuals.push([...currentVisuals]);
    currentVisuals = [];
    point1 = null; point2 = null;
    updateGraspList();
    document.getElementById('point1').textContent = '🔴 手指1: —';
    document.getElementById('point2').textContent = '🔵 手指2: —';
    document.getElementById('width-info').textContent = '📏 宽度: —';
    document.getElementById('status').textContent = `✅ 已添加 #${grasps.length} | 继续 Shift+点击`;
};

function updateGraspList() {
    const list = document.getElementById('grasp-list');
    document.getElementById('grasp-count').textContent = grasps.length;
    list.innerHTML = '';
    grasps.forEach((g, i) => {
        const d = Math.sqrt((g.p1[0]-g.p2[0])**2 + (g.p1[1]-g.p2[1])**2 + (g.p1[2]-g.p2[2])**2);
        const cls = d > 0.08 ? 'grasp-item error' : 'grasp-item';
        const ap = g.approach ? g.approach.join(',') : 'auto';
        list.innerHTML += `<div class="${cls}">
            <b>#${i+1}</b> 宽${(d*100).toFixed(1)}cm 接近:${ap}
            <button class="btn btn-danger" style="float:right;padding:1px 6px;font-size:11px;"
                onclick="removeGrasp(${i})">✕</button>
        </div>`;
    });
}

window.removeGrasp = function(idx) {
    graspVisuals[idx].forEach(v => scene.remove(v));
    grasps.splice(idx, 1);
    graspVisuals.splice(idx, 1);
    updateGraspList();
};

window.clearAll = function() {
    graspVisuals.forEach(vs => vs.forEach(v => scene.remove(v)));
    grasps = []; graspVisuals = [];
    window.clearPoints();
    updateGraspList();
};

window.saveAll = function() {
    if (!grasps.length) { alert('先添加抓取'); return; }
    fetch('/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grasps }),
    }).then(r => r.json()).then(data => {
        document.getElementById('save-result').innerHTML = data.success
            ? `<span style="color:#64ffda;">✅ ${data.message}</span>`
            : `<span style="color:#ff5252;">❌ ${data.message}</span>`;
    });
};

window.nextObject = function() {
    fetch('/next', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.success) { window.location.reload(); }
            else { alert(data.message); }
        });
};

window.prevObject = function() {
    fetch('/prev', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.success) { window.location.reload(); }
            else { alert(data.message); }
        });
};

// 加载后获取进度 + 已有标注
fetch('/progress').then(r=>r.json()).then(data => {
    if (data.total > 1) {
        document.getElementById('progress-info').textContent =
            `进度: ${data.current+1}/${data.total} | 已标: ${data.annotated}`;
    }
    if (data.existing_grasps > 0) {
        document.getElementById('existing-info').textContent =
            `ℹ️ 该物体已有 ${data.existing_grasps} 个已保存的抓取 (橙色显示)`;
    }

});

// 加载并显示已保存的抓取
function loadExistingGrasps() {
    fetch('/load_existing').then(r=>r.json()).then(data => {
        if (!data.success || !data.grasps || !data.grasps.length) return;
        // 等 mesh 加载完成后再渲染
        const waitForMesh = setInterval(() => {
            if (!meshObj) return;
            clearInterval(waitForMesh);
            data.grasps.forEach((g, i) => {
                const p1 = new THREE.Vector3(g.p1[0], g.p1[1], g.p1[2]);
                const p2 = new THREE.Vector3(g.p2[0], g.p2[1], g.p2[2]);
                const center = p1.clone().add(p2).multiplyScalar(0.5);
                // 橙色显示已保存的抓取
                const m1 = createSphere(p1, 0xff8800, 0.0025);
                const m2 = createSphere(p2, 0xff8800, 0.0025);
                scene.add(m1); scene.add(m2);
                const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({
                    color: 0xff8800, transparent: true, opacity: 0.6
                }));
                scene.add(line);
                // 接近方向箭头
                if (g.approach) {
                    const ap = new THREE.Vector3(g.approach[0], g.approach[1], g.approach[2]);
                    const arr = new THREE.ArrowHelper(ap, center, 0.05, 0xff8800, 0.008, 0.005);
                    scene.add(arr);
                }
            });
        }, 200);
    });
}
loadExistingGrasps();

init();
</script>
</body>
</html>"""


# ============================================================
# HTTP Server
# ============================================================
class Handler(SimpleHTTPRequestHandler):
    mesh_path = None
    obj_id = None
    output_dir = None
    mesh_trimesh = None

    obj_list = []       # batch mode: list of (obj_id, mesh_path)
    obj_index = 0       # current index in obj_list

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            # GT 状态: 直接检查文件是否存在 + 简单读取
            gt_label = '<span style="font-size:16px; padding:2px 8px; border-radius:4px; background:rgba(255,170,0,0.2); color:#ffaa00;">🆕 未测试</span>'
            gt_dir = os.path.join(os.path.dirname(self.output_dir), 'robot_gt')
            gt_auto_dir = os.path.join(os.path.dirname(self.output_dir), 'robot_gt_auto')
            for gd in [gt_dir, gt_auto_dir]:
                gp = os.path.join(gd, f'{self.obj_id}_robot_gt.hdf5')
                if os.path.exists(gp):
                    try:
                        with h5py.File(gp, 'r') as gf:
                            if gf.attrs.get('success', False):
                                gt_label = '<span style="font-size:16px; padding:2px 8px; border-radius:4px; background:rgba(0,255,0,0.2); color:#00ff00;">✅ GT成功</span>'
                            else:
                                gt_label = '<span style="font-size:16px; padding:2px 8px; border-radius:4px; background:rgba(255,0,0,0.2); color:#ff4444;">❌ GT失败</span>'
                            break
                    except:
                        gt_label = '<span style="font-size:16px; padding:2px 8px; border-radius:4px; background:rgba(255,170,0,0.2); color:#ffaa00;">⚠️ 读取错误</span>'
            html = HTML_PAGE.replace('{OBJ_ID}', self.obj_id).replace('{GT_STATUS}', gt_label)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        elif path == '/progress':
            annotated = sum(1 for oid, _ in self.obj_list
                           if os.path.exists(os.path.join(self.output_dir, f'{oid}_grasp.hdf5')))
            # 检查当前物体是否已有保存的抓取
            existing = 0
            cur_hdf5 = os.path.join(self.output_dir, f'{self.obj_id}_grasp.hdf5')
            if os.path.exists(cur_hdf5):
                try:
                    with h5py.File(cur_hdf5, 'r') as f:
                        existing = f.attrs.get('num_candidates', 0)
                except: pass
            # 检查 Robot GT 状态
            gt_status = 'new'  # new / success / fail
            gt_dir = os.path.join(os.path.dirname(self.output_dir), 'robot_gt')
            gt_auto_dir = os.path.join(os.path.dirname(self.output_dir), 'robot_gt_auto')
            for gd in [gt_dir, gt_auto_dir]:
                gp = os.path.join(gd, f'{self.obj_id}_robot_gt.hdf5')
                if os.path.exists(gp):
                    try:
                        with h5py.File(gp, 'r') as gf:
                            gt_status = 'success' if gf.attrs.get('success', False) else 'fail'
                            break
                    except: pass
            result = {'total': len(self.obj_list), 'current': self.obj_index,
                      'annotated': annotated, 'existing_grasps': existing,
                      'gt_status': gt_status}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        elif path == '/mesh.obj':
            with open(self.mesh_path, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f.read())
        elif path == '/load_existing':
            result = self.handle_load_existing()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length) if length > 0 else b''
        body = json.loads(raw) if raw else {}
        path = urlparse(self.path).path

        if path == '/raycast':
            result = self.handle_raycast(body)
        elif path == '/save':
            result = self.handle_save(body)
        elif path == '/next':
            result = self.handle_next()
        elif path == '/prev':
            result = self.handle_prev()
        else:
            result = {'success': False, 'message': 'Unknown endpoint'}

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))

    def handle_raycast(self, body):
        """沿法线射穿物体, 返回出口点."""
        origin = body['origin']
        direction = body['direction']
        click_pt = np.array(body['click_point'])

        hits = raycast_through_mesh(self.mesh_trimesh, origin, direction)
        if not hits:
            return {'success': False, 'message': '射线未穿透物体'}

        # 找最远的命中点 (出口)
        best_dist, best_pt = 0, hits[0]
        for h in hits:
            d = np.linalg.norm(np.array(h) - click_pt)
            if d > best_dist and d < 0.08:  # 只接受 <8cm 的
                best_dist, best_pt = d, h

        if best_dist < 0.003:
            # 如果最远点太近, 可能是自交, 试试所有交点
            for h in hits:
                d = np.linalg.norm(np.array(h) - click_pt)
                if d > best_dist:
                    best_dist, best_pt = d, h

        return {
            'success': True,
            'exit_point': best_pt,
            'width': best_dist,
            'all_hits': hits[:5],  # 最多返回 5 个交点供调试
        }

    def handle_save(self, body):
        grasps_data = body.get('grasps', [])
        if not grasps_data:
            return {'success': False, 'message': '没有抓取'}

        valid = []
        for i, g in enumerate(grasps_data):
            result, err = compute_grasp_from_two_points(
                g['p1'], g['p2'], self.mesh_trimesh, g.get('approach'))
            if err:
                print(f"  ⚠️ #{i+1} 无效: {err}")
                continue
            result['name'] = f'manual_{i}'
            result['approach_type'] = 'top_down' if abs(result['approach'][2]) > 0.7 else 'horizontal'
            result['score'] = 80.0
            valid.append(result)

        if not valid:
            return {'success': False, 'message': '所有抓取无效'}

        path = save_grasps_hdf5(valid, self.obj_id, self.mesh_path, self.output_dir)
        return {'success': True, 'message': f'{len(valid)} 个抓取 → {os.path.basename(path)}'}

    def handle_next(self):
        if not self.obj_list:
            return {'success': False, 'message': '非批量模式'}
        Handler.obj_index += 1
        if Handler.obj_index >= len(self.obj_list):
            Handler.obj_index = len(self.obj_list) - 1
            return {'success': False, 'message': '🎉 已是最后一个!'}
        obj_id, mesh_path = self.obj_list[Handler.obj_index]
        self._load_object(obj_id, mesh_path)
        return {'success': True, 'message': f'切换到 {obj_id}', 'obj_id': obj_id}

    def handle_prev(self):
        if not self.obj_list:
            return {'success': False, 'message': '非批量模式'}
        if Handler.obj_index <= 0:
            return {'success': False, 'message': '已是第一个!'}
        Handler.obj_index -= 1
        obj_id, mesh_path = self.obj_list[Handler.obj_index]
        self._load_object(obj_id, mesh_path)
        return {'success': True, 'message': f'切换到 {obj_id}', 'obj_id': obj_id}

    def handle_load_existing(self):
        """读取已保存的 HDF5，返回抓取点供前端显示."""
        hdf5_path = os.path.join(self.output_dir, f'{self.obj_id}_grasp.hdf5')
        if not os.path.exists(hdf5_path):
            return {'success': False, 'grasps': []}
        try:
            grasps = []
            with h5py.File(hdf5_path, 'r') as f:
                n = f.attrs.get('num_candidates', 0)
                for i in range(n):
                    key = f'candidate_{i}'
                    if key not in f: continue
                    ci = f[key]
                    gp = ci['grasp_point'][:]
                    rot = ci['rotation'][:]
                    w = ci.attrs.get('gripper_width', 0.04)
                    finger_dir = rot[:, 0]
                    approach = rot[:, 2]
                    p1 = gp + finger_dir * w / 2
                    p2 = gp - finger_dir * w / 2
                    grasps.append({
                        'p1': p1.tolist(),
                        'p2': p2.tolist(),
                        'approach': approach.tolist(),
                        'name': ci.attrs.get('name', f'saved_{i}'),
                        'width': float(w),
                    })
            return {'success': True, 'grasps': grasps}
        except Exception as e:
            return {'success': False, 'grasps': [], 'error': str(e)}

    @classmethod
    def _load_object(cls, obj_id, mesh_path):
        mesh = trimesh.load(mesh_path, force='mesh')
        ext = mesh.bounding_box.extents * 100
        print(f"\n📦 [{cls.obj_index+1}/{len(cls.obj_list)}] {obj_id}: {ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} cm")
        cls.mesh_path = os.path.abspath(mesh_path)
        cls.obj_id = obj_id
        cls.mesh_trimesh = mesh

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description='交互式抓取标注')
    parser.add_argument('--obj', help='单个 Object ID')
    parser.add_argument('--mesh', help='单个 Mesh file')
    parser.add_argument('--id', help='Object ID (配合 --mesh)')
    parser.add_argument('--all', action='store_true', help='批量标注所有物体')
    parser.add_argument('--start', help='批量模式: 从哪个 ID 开始 (跳过之前的)')
    parser.add_argument('--skip-done', action='store_true', default=True,
                        help='跳过已有 grasp HDF5 的物体 (默认开启)')
    parser.add_argument('--no-skip', dest='skip_done', action='store_false',
                        help='不跳过已标注的物体')
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--output-dir', default=os.path.join(PROJ, 'output', 'grasps'))
    args = parser.parse_args()

    Handler.output_dir = args.output_dir

    if args.all:
        # 批量模式: 加载所有 mesh
        import glob
        mesh_dir = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
        all_meshes = sorted(glob.glob(os.path.join(mesh_dir, '*.obj')))
        obj_list = []
        for mp in all_meshes:
            oid = os.path.splitext(os.path.basename(mp))[0]
            obj_list.append((oid, mp))

        if args.start:
            idx = next((i for i, (oid, _) in enumerate(obj_list) if oid == args.start), 0)
            obj_list = obj_list[idx:]
        elif not args.skip_done:
            # --no-skip: 找到最后一个已标注的，从那里开始
            last_done = -1
            for i, (oid, _) in enumerate(obj_list):
                if os.path.exists(os.path.join(args.output_dir, f'{oid}_grasp.hdf5')):
                    last_done = i
            if last_done >= 0:
                start_idx = max(0, last_done - 1)  # 回退1个方便检查
                Handler.obj_index = start_idx  # 不截断列表，只移动 index

        if not obj_list:
            print("✅ 所有物体都已标注!"); return

        # 检查 robot_gt 状态
        gt_dir = os.path.join(PROJ, 'output', 'robot_gt')
        gt_auto_dir = os.path.join(PROJ, 'output', 'robot_gt_auto')
        n_ok, n_fail, n_new = 0, 0, 0
        print(f"📋 批量标注模式: {len(obj_list)} 个物体待标注")
        for i, (oid, _) in enumerate(obj_list[:20]):
            gt_path = os.path.join(gt_dir, f'{oid}_robot_gt.hdf5')
            gt_auto_path = os.path.join(gt_auto_dir, f'{oid}_robot_gt.hdf5')
            status = "🆕"
            for gp in [gt_path, gt_auto_path]:
                if os.path.exists(gp):
                    try:
                        import h5py
                        with h5py.File(gp, 'r') as hf:
                            if hf.attrs.get('success', False):
                                status = "✅ GT成功"; n_ok += 1; break
                            else:
                                status = "❌ GT失败"; n_fail += 1; break
                    except: pass
            if status == "🆕": n_new += 1
            print(f"   [{i+1}] {oid}  {status}")
        if len(obj_list) > 20:
            print(f"   ... 还有 {len(obj_list)-20} 个")
        print(f"   📊 已显示: ✅{n_ok} 成功  ❌{n_fail} 失败  🆕{n_new} 未测试")

        Handler.obj_list = obj_list
        Handler.obj_index = 0
        first_id, first_path = obj_list[0]
        Handler._load_object(first_id, first_path)

    elif args.obj:
        obj_id = args.obj
        mesh_path = os.path.join(PROJ, 'data_hub', 'meshes', 'v1', f'{obj_id}.obj')
        if not os.path.exists(mesh_path):
            print(f"❌ 不存在: {mesh_path}"); sys.exit(1)
        Handler.obj_list = [(obj_id, mesh_path)]
        Handler.obj_index = 0
        Handler._load_object(obj_id, mesh_path)

    elif args.mesh:
        mesh_path = args.mesh
        obj_id = args.id or os.path.splitext(os.path.basename(mesh_path))[0]
        if not os.path.exists(mesh_path):
            print(f"❌ 不存在: {mesh_path}"); sys.exit(1)
        Handler.obj_list = [(obj_id, mesh_path)]
        Handler.obj_index = 0
        Handler._load_object(obj_id, mesh_path)

    else:
        parser.error('请指定 --obj 或 --all')

    server = HTTPServer(('0.0.0.0', args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n🌐 {url}")
    print(f"   Shift+点击穿透 | X/Y/Z 轴锁定 | 保存后点 '下一个'")
    print(f"   Ctrl+C 退出\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n👋 退出 (已标注到第 {Handler.obj_index+1}/{len(Handler.obj_list)} 个)")
        server.server_close()


if __name__ == '__main__':
    main()
