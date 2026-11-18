# est-minutes: 30
# Rerun of job 011 after the vLLM LoRA-swap hang (now max_loras=4). 011 got, on v1 with vLLM:
# base .045 and SFT .410 in ~3.5 min per model, then hung loading the GRPO adapter.
# v1 (final run) vs v2 (fixed residence/CAPI wording), base / SFT / SFT+GRPO.
# Generation on vLLM (one engine, LoRA per request) in a separate distill venv; if vLLM fails
# for a split, that split falls back to HF batch 16 with cuDNN SDPA disabled.
set -e
OUT="$(realpath "$OUT")"
DISTILL_ENV="$HOME/.venvs/arca-distill"
TRAIN_ENV="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/arca-train}"
UV_PROJECT_ENVIRONMENT="$DISTILL_ENV" uv sync --extra distill -q
UV_PROJECT_ENVIRONMENT="$DISTILL_ENV" uv run --extra distill python -c "import vllm; print('vllm', vllm.__version__)"
uv run --extra train python -m rlm.generate_problems --n 200 --split test --seed 42 --out "$OUT/test_v2.jsonl" >/dev/null
ADAPTERS="base=none sft=JES0406/imv-sft-lora/sft_final grpo=JES0406/imv-sft-lora/grpo_final"
COMMON="--model Qwen/Qwen3-1.7B --adapters $ADAPTERS --verifier imv --samples 1 --max-new-tokens 1024 --seed 42"
for v in v1 v2; do
  data=results/rlm_final/data/test.jsonl; [ $v = v2 ] && data="$OUT/test_v2.jsonl"
  start=$(date +%s)
  if UV_PROJECT_ENVIRONMENT="$DISTILL_ENV" timeout 2700 uv run --extra distill python -m rlm.evaluate \
       --data "$data" $COMMON --backend vllm --out "$OUT/eval_test_${v}_vllm.json"; then
    echo "eval $v vllm wall: $(( $(date +%s) - start )) s"
  else
    echo "eval $v vllm FAILED after $(( $(date +%s) - start )) s, falling back to HF"
    start=$(date +%s)
    UV_PROJECT_ENVIRONMENT="$TRAIN_ENV" uv run --extra train python -m rlm.evaluate \
        --data "$data" $COMMON --batch-size 16 --resume --out "$OUT/eval_test_${v}_hf.json"
    echo "eval $v hf wall: $(( $(date +%s) - start )) s"
  fi
done
rm -f "$OUT/test_v2.jsonl"  # regenerable from seed 42; keep the push small
