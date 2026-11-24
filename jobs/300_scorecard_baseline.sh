# est-minutes: 180
# Whole-pipeline scorecard of THIS branch (eval/pipeline-baseline = feat/fase4-agent as deployed):
# the API is started here and every phase is measured through HTTP, like the grader does.
# Phase 1: 40 test problems; phase 2: 40-case bench; phase 3: 50 gold + 8 unanswerable (hybrid);
# phase 4: 23 tasks x base/rlm/thinking + 2 two-turn conversations. Needs the GPU alone.
set -u
OUT="$(realpath "$OUT")"
PERSIST="$HOME/arca-runs/scorecard/baseline"; mkdir -p "$PERSIST"
bash pipeline_eval/run_api_scorecard.sh . "$PERSIST" baseline
cp "$PERSIST"/scorecard.json "$PERSIST"/scorecard.md "$PERSIST"/health.json "$OUT/" 2>/dev/null || true
tail -200 "$PERSIST/api.log" > "$OUT/api_tail.log" 2>/dev/null || true
