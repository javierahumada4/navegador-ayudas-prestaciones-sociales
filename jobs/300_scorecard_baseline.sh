# est-minutes: 180
# Scorecard of the BASELINE (branch eval/pipeline-baseline = feat/fase4-agent as deployed), run from
# this branch so both variants are measured with the same scorecard and benches. The API is
# started from a worktree of the baseline branch. Needs the GPU alone: stop any other API first.
set -u
OUT="$(realpath "$OUT")"
W="$HOME/arca-runs/scorecard/baseline-code"
rm -rf "$W"; git worktree prune
git fetch -q origin eval/pipeline-baseline
git worktree add -q --detach "$W" origin/eval/pipeline-baseline
PERSIST="$HOME/arca-runs/scorecard/baseline"; mkdir -p "$PERSIST"
bash pipeline_eval/run_api_scorecard.sh "$W" "$PERSIST" baseline
cp "$PERSIST"/scorecard.json "$PERSIST"/scorecard.md "$PERSIST"/health.json "$OUT/" 2>/dev/null || true
tail -200 "$PERSIST/api.log" > "$OUT/api_tail.log" 2>/dev/null || true
