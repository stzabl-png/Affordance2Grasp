#!/bin/bash
# Batch regenerate all HDF5 grasp files with fixed-axis approach
# Covers: OakInk V1 (157 obj) + GRAB (50 stl)
# Usage: bash batch_regen_grasps.sh

cd /home/lyh/Project/Affordance2Grasp

TOTAL=0
SUCCESS=0
FAIL=0
SKIP=0

echo "============================================================"
echo " Batch Regenerate HDF5 Grasps (Fixed-Axis: Front/Left/Right/TopDown)"
echo "============================================================"

# --- OakInk V1 ---
echo ""
echo "--- Dataset: OakInk V1 ---"
for obj_file in /home/lyh/Project/Affordance2Grasp/data_hub/meshes/v1/*.obj; do
    [ -f "$obj_file" ] || continue
    obj_id=$(basename "$obj_file" .obj)
    TOTAL=$((TOTAL + 1))
    echo -n "[$TOTAL] $obj_id ... "

    set -o pipefail
    output=$(python3 -m inference.grasp_pose --mesh "$obj_file" 2>&1)
    exit_code=$?
    set +o pipefail

    if [ $exit_code -eq 0 ]; then
        n_cands=$(echo "$output" | grep -c "✅")
        SUCCESS=$((SUCCESS + 1))
        echo "✅ ($n_cands candidates)"
    else
        if echo "$output" | grep -q "无法抓取"; then
            SKIP=$((SKIP + 1))
            echo "⏭️  SKIP (too large)"
        else
            FAIL=$((FAIL + 1))
            echo "❌ FAILED"
            echo "$output" | tail -3
        fi
    fi
done

# --- GRAB (stl files, trimesh can read directly) ---
echo ""
echo "--- Dataset: GRAB ---"
for stl_file in /home/lyh/Project/GRAB_data/_unzipped/*.stl; do
    [ -f "$stl_file" ] || continue
    obj_id=$(basename "$stl_file" .stl)
    TOTAL=$((TOTAL + 1))
    echo -n "[$TOTAL] $obj_id ... "

    set -o pipefail
    output=$(python3 -m inference.grasp_pose --mesh "$stl_file" 2>&1)
    exit_code=$?
    set +o pipefail

    if [ $exit_code -eq 0 ]; then
        n_cands=$(echo "$output" | grep -c "✅")
        SUCCESS=$((SUCCESS + 1))
        echo "✅ ($n_cands candidates)"
    else
        if echo "$output" | grep -q "无法抓取"; then
            SKIP=$((SKIP + 1))
            echo "⏭️  SKIP (too large)"
        else
            FAIL=$((FAIL + 1))
            echo "❌ FAILED"
            echo "$output" | tail -3
        fi
    fi
done

echo ""
echo "============================================================"
echo " Done! Total=$TOTAL  Success=$SUCCESS  Skipped=$SKIP  Failed=$FAIL"
echo "============================================================"
