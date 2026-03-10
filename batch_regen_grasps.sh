#!/bin/bash
# Batch regenerate all HDF5 grasp files with new world-axis-aligned approach
# Usage: bash batch_regen_grasps.sh

cd /home/lyh/Project/Affordance2Grasp

MESH_DIR="/home/lyh/Project/OakInk/image/obj"
OUT_DIR="/home/lyh/Project/Affordance2Grasp/output/grasps"
TOTAL=0
SUCCESS=0
FAIL=0
SKIP=0

echo "============================================================"
echo " Batch Regenerate HDF5 Grasps (World-Axis Aligned)"
echo "============================================================"

for obj_file in "$MESH_DIR"/*.obj; do
    obj_id=$(basename "$obj_file" .obj)
    TOTAL=$((TOTAL + 1))
    echo "[$TOTAL] $obj_id ..."

    # 用 pipefail 确保 $? 反映 python3 的退出码, 而非 tail
    set -o pipefail
    python3 -m inference.grasp_pose --mesh "$obj_file" 2>&1 | tail -3
    exit_code=$?
    set +o pipefail

    if [ $exit_code -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
    else
        # 检查是否因为物体太大而跳过
        if python3 -m inference.grasp_pose --mesh "$obj_file" 2>&1 | grep -q "无法抓取"; then
            SKIP=$((SKIP + 1))
            echo "  ⏭️  SKIP (too large): $obj_id"
        else
            FAIL=$((FAIL + 1))
            echo "  ❌ FAILED: $obj_id"
        fi
    fi
done

echo ""
echo "============================================================"
echo " Done! Total=$TOTAL  Success=$SUCCESS  Skipped=$SKIP  Failed=$FAIL"
echo "============================================================"
