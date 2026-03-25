#!/bin/bash
# ============================================================
# M2: Random Grasp Sim 验证 (Isaac Sim 环境)
# ============================================================
# 用法:
#     conda deactivate
#     cd /home/lyh/Project/Affordance2Grasp
#     bash batch_random_sim.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRASP_DIR="$SCRIPT_DIR/output/grasps_random"
GT_DIR="$SCRIPT_DIR/output/robot_gt_random"
GT_MANUAL="$SCRIPT_DIR/output/robot_gt"
LOG_DIR="$SCRIPT_DIR/output/sim_logs_random"

mkdir -p "$GT_DIR" "$LOG_DIR"

HDF5_LIST=($(ls "$GRASP_DIR"/*_grasp.hdf5 2>/dev/null))
TOTAL=${#HDF5_LIST[@]}

echo "============================================================"
echo "  M2: Random Grasp Sim Verification"
echo "============================================================"
echo "  Total grasp files:  $TOTAL"
echo "  Grasp dir:          $GRASP_DIR"
echo "  Result dir:         $GT_DIR"
echo "============================================================"

SUCCESS=0; FAILED=0; SKIPPED=0

for i in $(seq 0 $((TOTAL-1))); do
    HDF5="${HDF5_LIST[$i]}"
    OBJ_ID=$(basename "$HDF5" _grasp.hdf5)
    N=$((i+1))

    # 跳过已有成功 GT 的 (手动 or 随机)
    for gdir in "$GT_MANUAL" "$GT_DIR"; do
        RESULT_FILE="$gdir/${OBJ_ID}_robot_gt.hdf5"
        if [ -f "$RESULT_FILE" ]; then
            RESULT=$(python3 -c "
import h5py
with h5py.File('$RESULT_FILE','r') as f:
    print('SUCCESS' if f.attrs.get('success',False) else 'FAILED')
" 2>/dev/null || echo "UNKNOWN")
            if [ "$RESULT" == "SUCCESS" ]; then
                SKIPPED=$((SKIPPED+1))
                continue 2
            fi
        fi
    done

    echo "  [$N/$TOTAL] $OBJ_ID ..."

    timeout 600 /home/lyh/isaac-sim/python.sh "$SCRIPT_DIR/sim/run_grasp_sim.py" \
        --hdf5 "$HDF5" \
        --headless \
        --save-result \
        --result-dir "$GT_DIR" \
        2>&1 | tee "$LOG_DIR/${OBJ_ID}.log" | tail -5

    RESULT_FILE="$GT_DIR/${OBJ_ID}_robot_gt.hdf5"
    if [ -f "$RESULT_FILE" ]; then
        RESULT=$(python3 -c "
import h5py
with h5py.File('$RESULT_FILE','r') as f:
    s = f.attrs.get('success', False)
    ns = f.attrs.get('n_successful', 0)
    print(f'SUCCESS {ns}' if s else 'FAILED')
" 2>/dev/null || echo "UNKNOWN")
        if [[ "$RESULT" == SUCCESS* ]]; then
            SUCCESS=$((SUCCESS+1))
            echo "  ✅ $OBJ_ID — $RESULT GTs"
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
