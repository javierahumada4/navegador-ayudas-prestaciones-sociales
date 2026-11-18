# Reward ablation on top of job 030: same SFT v2 init, data, seed and hyperparameters as the
# v2 GRPO; only the third reward changes, domain (two-decimal format, redundant) ->
# eligibility (answers 0 exactly when the oracle does). Needs job 030's run directory.
set -e
OUT="$(realpath "$OUT")"
RUN="$ARCA_WORK_DIR/arca-runs/rlm-v2"
[ -f "$RUN/weights/sft_lora/adapter_config.json" ] || { echo "job 030 SFT adapter missing"; exit 1; }
uv sync --extra train -q
start=$(date +%s)
uv run --extra train python -m rlm.train_grpo --data "$RUN/data/train.jsonl" \
    --model Qwen/Qwen3-1.7B --init-adapter "$RUN/weights/sft_lora" --output "$RUN/weights/grpo_elig" \
    --steps 300 --num-generations 4 --temperature 0.6 --top-p 0.95 --top-k 20 \
    --micro-batch-size 1 --max-completion-length 1024 --zero-signal-patience 0 --seed 42 \
    --third-reward eligibility
echo "grpo wall: $(( $(date +%s) - start )) s"
# Same evaluation backend as job 030 chose (manifest), for a fair comparison.
BACKEND=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['configuration'].get('eval_backend', 'hf'))" "$RUN/manifest.json")
EVAL="--data $RUN/data/test.jsonl --model Qwen/Qwen3-1.7B --adapters grpo_elig=$RUN/weights/grpo_elig \
  --verifier imv --samples 1 --batch-size 16 --max-new-tokens 1024 --seed 42 --out $OUT/eval_test_elig.json"
echo "eval backend: $BACKEND"
if [ "$BACKEND" = vllm ]; then
  UV_PROJECT_ENVIRONMENT="$RUN/.venv-distill" timeout 5400 uv run --extra distill python -m rlm.evaluate $EVAL --backend vllm \
    || { echo "vLLM eval failed or hung: HF fallback"; uv run --extra train python -m rlm.evaluate $EVAL --backend hf; }
else
  uv run --extra train python -m rlm.evaluate $EVAL --backend hf
fi
uv run --extra train python -m rlm.curves --runs "domain=$RUN/weights/grpo_lora" \
    "eligibility=$RUN/weights/grpo_elig" --out "$OUT/grpo_curves_domain_vs_eligibility.png"
cp "$RUN/weights/grpo_elig/log_history.json" "$OUT/grpo_elig_log_history.json"
uvx --from huggingface_hub hf upload JES0406/imv-sft-lora "$RUN/weights/grpo_elig" grpo_elig_v2 \
    --include "adapter_*" --commit-message "Add eligibility-reward GRPO adapter (v2 ablation)" \
    || echo "upload grpo_elig_v2 failed"
