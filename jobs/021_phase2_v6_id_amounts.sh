# Phase 2 v6 = v5 + id example as «rol id» -> id (v5 copied the example id "titular") + rule to
# always state IMV/CAPI/total (v5 denials gave no amounts). v5 was feat/fase2-speed (merged LoRA, stop on repeated failed
# output, 1280-token cap, id rule + two-person example, schema sent once, lenient braces).
# Compare with v5 (e2e base .400 / GRPO .400) and v4 (.500 / .425).
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
      --bench tool_use/bench/cases.jsonl --max-turns 5 --out "reports/phase2_${brain}_v6.json"
  echo "phase2 $brain wall: $(( $(date +%s) - start )) s"
  cp "reports/phase2_${brain}_v6.json" "$OUT/"  # summary + rows; full traces stay on the DGX
done
