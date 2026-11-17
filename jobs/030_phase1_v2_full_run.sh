# Phase 1 v2: full pipeline with the fixed rule sheet (873b2c6). Same seed, sizes, student,
# SFT and GRPO hyperparameters as results/rlm_final; only differences:
#   - rule sheet v2 (residence cutoff, CAPI loss, age bands, member count)
#   - speed: teacher on vLLM batch 64 (was HF batch 1, 4.6 h), eval batch 16 merged (was 2, 4.1 h)
# Resumable: if the runner timeout kills it, re-queue with RESUME=1 (pipeline --resume).
set -e
OUT="$(realpath "$OUT")"
RUN="$ARCA_WORK_DIR/arca-runs/rlm-v2"
mkdir -p "$RUN"
uv sync --extra train -q
start=$(date +%s)
uv run --extra train python -m rlm.pipeline --run-dir "$RUN" --quality-gates warn \
    --teacher-backend vllm --teacher-batch-size 64 --eval-batch-size 16 \
    ${RESUME:+--resume}
echo "pipeline wall: $(( $(date +%s) - start )) s"
cp "$RUN"/manifest.json "$OUT/"
cp "$RUN"/reports/*.json "$RUN"/reports/*.md "$RUN"/reports/*.png "$OUT/" 2>/dev/null || true
cp "$RUN"/weights/grpo_lora/log_history.json "$OUT/grpo_log_history.json" 2>/dev/null || true
# Adapters to the Hub so the API and later jobs can use them (needs a write token).
for a in sft grpo; do
  uvx --from huggingface_hub hf upload JES0406/imv-sft-lora "$RUN/weights/${a}_lora" "${a}_v2" \
      --include "adapter_*" --commit-message "Add ${a} adapter of the rule-sheet v2 run" \
      || echo "upload ${a}_v2 failed (token without write access?)"
done
