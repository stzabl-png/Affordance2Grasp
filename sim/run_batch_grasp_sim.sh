#!/bin/bash
#
# M4: 批量 Sim 抓取执行 (Affordance2Grasp 内部)
# ==============================================
# 用法:
#     conda deactivate
#     cd /home/lyh/Project/Affordance2Grasp
#     bash sim/run_batch_grasp_sim.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
A2G_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GRASP_DIR="$A2G_ROOT/output/grasps"
RESULT_DIR="$A2G_ROOT/output/robot_gt"
LOG_DIR="$A2G_ROOT/output/sim_logs"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

# Discover grasp HDF5 files
HDF5_LIST=($(ls "$GRASP_DIR"/*_grasp.hdf5 2>/dev/null))
TOTAL=${#HDF5_LIST[@]}

echo "============================================================"
echo "  M4: Batch Grasp Simulation"
echo "============================================================"
echo "  Total objects: $TOTAL"
echo "  Grasp dir:     $GRASP_DIR"
echo "  Result dir:    $RESULT_DIR"
echo "============================================================"

SUCCESS=0
FAILED=0
SKIPPED=0

for i in $(seq 0 $((TOTAL-1))); do
    HDF5="${HDF5_LIST[$i]}"
    OBJ_ID=$(basename "$HDF5" _grasp.hdf5)
    N=$((i+1))

    RESULT_FILE="$RESULT_DIR/${OBJ_ID}_robot_gt.hdf5"

    # Skip if already done
    if [ -f "$RESULT_FILE" ]; then
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    echo "  [$N/$TOTAL] $OBJ_ID ..."

    # Run sim (timeout 5 min)
    timeout 300 /home/lyh/isaac-sim/python.sh "$SCRIPT_DIR/run_grasp_sim.py" \
        --hdf5 "$HDF5" \
        --headless \
        --save-result \
        --result-dir "$RESULT_DIR" \
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
echo "  Results: $RESULT_DIR"
echo "============================================================"
