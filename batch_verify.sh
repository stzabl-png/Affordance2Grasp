#!/bin/bash
# ============================================================
# 二次验证: 只跑初论成功的抓取, 提取接触点
# ============================================================
# 用法:
#     conda deactivate
#     cd /home/lyh/Project/Affordance2Grasp
#     bash batch_verify.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRASP_DIR="$SCRIPT_DIR/output/grasps_verified"
GT_DIR="$SCRIPT_DIR/output/robot_gt_verified"
LOG_DIR="$SCRIPT_DIR/output/sim_logs_verified"

mkdir -p "$GT_DIR" "$LOG_DIR"

HDF5_LIST=($(ls "$GRASP_DIR"/*_grasp.hdf5 2>/dev/null))
TOTAL=${#HDF5_LIST[@]}

echo "============================================================"
echo "  二次验证 + 接触点提取"
echo "============================================================"
echo "  Total: $TOTAL objects"
echo "  Input: $GRASP_DIR"
echo "  Output: $GT_DIR"
echo "============================================================"

SUCCESS=0; FAILED=0; SKIPPED=0

for i in $(seq 0 $((TOTAL-1))); do
    HDF5="${HDF5_LIST[$i]}"
    OBJ_ID=$(basename "$HDF5" _grasp.hdf5)
    N=$((i+1))

    # 跳过已验证的
    RESULT_FILE="$GT_DIR/${OBJ_ID}_robot_gt.hdf5"
    if [ -f "$RESULT_FILE" ]; then
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    echo "  [$N/$TOTAL] $OBJ_ID ..."

    timeout 600 /home/lyh/isaac-sim/python.sh "$SCRIPT_DIR/sim/run_grasp_sim.py" \
        --hdf5 "$HDF5" \
        --headless \
        --save-result \
        --result-dir "$GT_DIR" \
        2>&1 | tee "$LOG_DIR/${OBJ_ID}.log" | tail -3

    if [ -f "$RESULT_FILE" ]; then
        RESULT=$(python3 -c "
import h5py
with h5py.File('$RESULT_FILE','r') as f:
    ns = f.attrs.get('n_successful',0)
    has_cp = False
    if 'successful_grasps' in f:
        for k in f['successful_grasps']:
            if f[f'successful_grasps/{k}'].attrs.get('has_contact_points', False):
                has_cp = True; break
    print(f'{ns} {has_cp}')
" 2>/dev/null || echo "0 False")
        NS=$(echo $RESULT | cut -d' ' -f1)
        HAS_CP=$(echo $RESULT | cut -d' ' -f2)
        if [ "$NS" -gt 0 ]; then
            SUCCESS=$((SUCCESS+1))
            echo "  ✅ $OBJ_ID — $NS verified GTs (contact_pts=$HAS_CP)"
        else
            FAILED=$((FAILED+1))
        fi
    else
        FAILED=$((FAILED+1))
    fi
done

echo ""
echo "============================================================"
echo "  DONE  ✅ $SUCCESS  ❌ $FAILED  ⏭️ $SKIPPED  Total: $TOTAL"
echo "  Results: $GT_DIR"
echo "============================================================"
