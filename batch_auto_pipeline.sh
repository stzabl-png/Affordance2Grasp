#!/bin/bash
# ============================================================
# 自动模型抓取全流程: 生成抓取位姿 → Sim 验证
# 结果存到 output/robot_gt_auto/ (不影响手动标注的 robot_gt/)
# ============================================================
# 用法:
#     conda activate base
#     cd /home/lyh/Project/Affordance2Grasp
#     bash batch_auto_pipeline.sh
# ============================================================

set -e
cd /home/lyh/Project/Affordance2Grasp

AUTO_GRASP_DIR="output/grasps_auto"
AUTO_GT_DIR="output/robot_gt_auto"
LOG_DIR="output/sim_logs_auto"
mkdir -p "$AUTO_GRASP_DIR" "$AUTO_GT_DIR" "$LOG_DIR"

TOTAL=0; GEN_OK=0; GEN_SKIP=0; GEN_FAIL=0

echo "============================================================"
echo "  Step 1: 用 v5 模型生成所有物体的抓取位姿"
echo "  输出:   $AUTO_GRASP_DIR"
echo "============================================================"

# --- OakInk V1 (157 obj) ---
for obj_file in data_hub/meshes/v1/*.obj; do
    [ -f "$obj_file" ] || continue
    obj_id=$(basename "$obj_file" .obj)
    TOTAL=$((TOTAL + 1))

    # 跳过已生成的
    if [ -f "$AUTO_GRASP_DIR/${obj_id}_grasp.hdf5" ]; then
        GEN_SKIP=$((GEN_SKIP + 1))
        continue
    fi

    echo -n "  [$TOTAL] $obj_id ... "

    set -o pipefail
    output=$(python3 -m inference.grasp_pose \
        --mesh "$obj_file" \
        --output "$AUTO_GRASP_DIR/${obj_id}_grasp.hdf5" \
        2>&1) && {
        GEN_OK=$((GEN_OK + 1))
        echo "✅"
    } || {
        if echo "$output" | grep -q "无法抓取"; then
            GEN_SKIP=$((GEN_SKIP + 1))
            echo "⏭️ (too large)"
        else
            GEN_FAIL=$((GEN_FAIL + 1))
            echo "❌"
        fi
    }
    set +o pipefail
done

echo ""
echo "============================================================"
echo "  Step 1 完成: Gen=$GEN_OK  Skip=$GEN_SKIP  Fail=$GEN_FAIL"
echo "============================================================"
echo ""
echo "  Step 2: 请在 Isaac Sim 环境跑 sim 验证:"
echo ""
echo "    conda deactivate"
echo "    cd /home/lyh/Project/Affordance2Grasp"
echo "    bash batch_auto_sim.sh"
echo ""
echo "============================================================"
