# Phase 2 v5: the 40-case bench with feat/fase2-speed (merged LoRA, stop on repeated failed
# output, 1280-token cap, id rule + two-person example, schema sent once, lenient braces).
# Compare with v4 (same bench, same models): e2e base .500 / GRPO .425, 2193 s / 2287 s.
set -e
OUT="$(realpath "$OUT")"
W="$ARCA_WORK_DIR/arca-fase2"
git fetch -q origin feat/fase2-speed
rm -rf "$W"; git worktree prune
git worktree add -q --detach "$W" origin/feat/fase2-speed
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"  # reuse the runner venv instead of a second torch install
cd "$W"
git log --oneline -1
uv sync --extra train -q
for brain in base grpo; do
  args="--brain base"
  [ "$brain" = grpo ] && args="--brain rlm --adapter JES0406/imv-sft-lora/grpo_final"
  start=$(date +%s)
  uv run --extra train python -m tool_use.compare --model Qwen/Qwen3-1.7B $args \
      --bench tool_use/bench/cases.jsonl --max-turns 5 --out "reports/phase2_${brain}_v5.json"
  echo "phase2 $brain wall: $(( $(date +%s) - start )) s"
  cp "reports/phase2_${brain}_v5.json" "$OUT/"  # summary + rows; full traces stay on the DGX
done
