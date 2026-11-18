# Phase 1 v2: full pipeline with the fixed rule sheet (873b2c6). Same seed, sizes, student,
# SFT and GRPO hyperparameters as results/rlm_final; only differences:
#   - rule sheet v2 (residence cutoff, CAPI loss, age bands, member count)
#   - speed: teacher on vLLM batch 64 (was HF batch 1, 4.6 h), eval on vLLM (was HF batch 2, 4.1 h)
# Resumable: if the runner timeout kills it, re-queue with RESUME=1 (pipeline --resume).
set -e
OUT="$(realpath "$OUT")"
RUN="$ARCA_WORK_DIR/arca-runs/rlm-v2"
mkdir -p "$RUN"
uv sync --extra train -q
# vLLM may hang on this slice (job 011): use it only if a 15-min probe passes, else HF.
SPEED="--teacher-backend hf --teacher-batch-size 1 --eval-backend hf --eval-batch-size 16"
if UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-distill" timeout 900 uv sync --extra distill -q &&
   UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-distill" timeout 900 uv run --extra distill \
       python smoke/vllm_probe.py | tee /dev/stderr | grep -q VLLM_OK; then
  SPEED="--teacher-backend vllm --teacher-batch-size 64 --eval-backend vllm --eval-batch-size 16"
fi
echo "backends: $SPEED"
start=$(date +%s)
uv run --extra train python -m rlm.pipeline --run-dir "$RUN" --quality-gates warn $SPEED \
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
