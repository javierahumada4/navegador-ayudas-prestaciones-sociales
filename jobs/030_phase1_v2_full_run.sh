# est-minutes: 200
# (full run ~270 min; resumable per stage, so starting it even if the deadline cuts it is worth it)
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
# Probe output goes to a file, never a pipe: an orphaned vLLM engine kept a pipe open and hung
# job 029 for two sessions (rlm/proc.py now kills it; this is the second guard).
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-distill" timeout 900 uv sync --extra distill -q || true
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-distill" timeout -k 30 900 uv run --extra distill \
    python smoke/vllm_probe.py > "$OUT/vllm_probe.txt" 2>&1 || true
pkill -f "VLLM::EngineCore" || true  # never leave an engine holding the GPU before the pipeline
cat "$OUT/vllm_probe.txt" | grep -E "tok/s|VLLM_OK|engine up" || true
if grep -q VLLM_OK "$OUT/vllm_probe.txt"; then
  SPEED="--teacher-backend vllm --teacher-batch-size 64 --eval-backend vllm --eval-batch-size 16"
fi
echo "backends: $SPEED"
# Cut at a session deadline? Resume from the first stage whose outputs are missing.
RESUME_FLAGS=""
if [ -f "$RUN/manifest.json" ]; then
  stage=generate
  [ -f "$RUN/reports/dataset_validation.json" ] && stage=distill
  [ -f "$RUN/reports/distillation_validation.json" ] && stage=sft
  [ -f "$RUN/weights/sft_lora/adapter_config.json" ] && stage=grpo
  [ -f "$RUN/weights/grpo_lora/adapter_config.json" ] && stage=evaluate
  RESUME_FLAGS="--from-stage $stage"
  # --resume only where there is something to resume; pipeline aborts otherwise.
  case "$stage" in
    distill) [ -s "$RUN/data/sft_traces.jsonl" ] && RESUME_FLAGS="$RESUME_FLAGS --resume" ;;
    sft) ls -d "$RUN"/weights/sft_lora/checkpoint-* >/dev/null 2>&1 && RESUME_FLAGS="$RESUME_FLAGS --resume" ;;
    grpo) ls -d "$RUN"/weights/grpo_lora/checkpoint-* >/dev/null 2>&1 && RESUME_FLAGS="$RESUME_FLAGS --resume" ;;
    evaluate) RESUME_FLAGS="$RESUME_FLAGS --resume" ;;
  esac
fi
echo "resume: ${RESUME_FLAGS:-fresh run}${THROUGH:+, through $THROUGH}"
start=$(date +%s)
uv run --extra train python -m rlm.pipeline --run-dir "$RUN" --quality-gates warn $SPEED $RESUME_FLAGS \
    ${THROUGH:+--through-stage $THROUGH}
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
