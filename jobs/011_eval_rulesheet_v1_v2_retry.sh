# Retry of job 010 (cuDNN SDPA crash at batch 16, now disabled in rlm/inference.py); --resume
# keeps models already evaluated. Phase 1 ablation + speed check, no training.
# Same 200 test problems (seed 42), rule sheet v1 (final run) vs v2 (fixed residence/CAPI
# wording), evaluated with base, SFT and SFT+GRPO adapters at batch 16 with merged LoRA.
# Original final run: batch 2, unmerged, 2.13 h for this test split.
set -e
uv sync --extra train -q
uv run --extra train python -m rlm.generate_problems --n 200 --split test --seed 42 --out "$OUT/test_v2.jsonl"
# Same problems? answers and params must match the v1 file row by row.
uv run python - "$OUT/test_v2.jsonl" <<'PY'
import json, sys
v1 = [json.loads(l) for l in open("results/rlm_final/data/test.jsonl")]
v2 = [json.loads(l) for l in open(sys.argv[1])]
same = sum(a["answer"] == b["answer"] and a.get("params") == b.get("params") for a, b in zip(v1, v2))
print(f"rows v1={len(v1)} v2={len(v2)} identical problems={same}")
assert same == len(v1) == len(v2), "seed 42 no longer reproduces the v1 problems"
PY
ADAPTERS="base=none sft=JES0406/imv-sft-lora/sft_final grpo=JES0406/imv-sft-lora/grpo_final"
for v in v1 v2; do
  data=results/rlm_final/data/test.jsonl; [ $v = v2 ] && data="$OUT/test_v2.jsonl"
  start=$(date +%s)
  uv run --extra train python -m rlm.evaluate --data "$data" --model Qwen/Qwen3-1.7B \
      --adapters $ADAPTERS --verifier imv --samples 1 --batch-size 16 \
      --max-new-tokens 1024 --seed 42 --resume --out "$OUT/eval_test_$v.json"
  echo "eval $v wall: $(( $(date +%s) - start )) s"
done
rm -f "$OUT/test_v2.jsonl"  # regenerable from seed 42; keep the push small
