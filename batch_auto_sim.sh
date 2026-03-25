#!/bin/bash
# ============================================================
# Step 2: 对自动生成的抓取跑 Sim 验证 (Isaac Sim 环境)
# ============================================================
# 用法:
#     conda deactivate
#     cd /home/lyh/Project/Affordance2Grasp
#     bash batch_auto_sim.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUTO_GRASP_DIR="$SCRIPT_DIR/output/grasps_auto"
AUTO_GT_DIR="$SCRIPT_DIR/output/robot_gt_auto"
LOG_DIR="$SCRIPT_DIR/output/sim_logs_auto"

mkdir -p "$AUTO_GT_DIR" "$LOG_DIR"

HDF5_LIST=($(ls "$AUTO_GRASP_DIR"/*_grasp.hdf5 2>/dev/null))
TOTAL=${#HDF5_LIST[@]}

echo "============================================================"
echo "  Auto Grasp Sim Verification"
echo "============================================================"
echo "  Total:      $TOTAL"
echo "  Grasp dir:  $AUTO_GRASP_DIR"
echo "  Result dir: $AUTO_GT_DIR"
echo "============================================================"

SUCCESS=0; FAILED=0; SKIPPED=0

for i in $(seq 0 $((TOTAL-1))); do
    HDF5="${HDF5_LIST[$i]}"
    OBJ_ID=$(basename "$HDF5" _grasp.hdf5)
    N=$((i+1))

    RESULT_FILE="$AUTO_GT_DIR/${OBJ_ID}_robot_gt.hdf5"

    if [ -f "$RESULT_FILE" ]; then
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    echo "  [$N/$TOTAL] $OBJ_ID ..."

    timeout 300 /home/lyh/isaac-sim/python.sh "$SCRIPT_DIR/sim/run_grasp_sim.py" \
        --hdf5 "$HDF5" \
        --headless \
        --save-result \
        --result-dir "$AUTO_GT_DIR" \
        2>&1 | tee "$LOG_DIR/${OBJ_ID}.log" | tail -3

    if [ -f "$RESULT_FILE" ]; then
        RESULT=$(python3 -c "
import h5py
with h5py.File('$RESULT_FILE','r') as f:
    print('SUCCESS' if f.attrs['success'] else 'FAILED')
" 2>/dev/null || echo "UNKNOWN")
        if [ "$RESULT" == "SUCCESS" ]; then
            SUCCESS=$((SUCCESS+1))
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
echo "  Results: $AUTO_GT_DIR"
echo "============================================================"
