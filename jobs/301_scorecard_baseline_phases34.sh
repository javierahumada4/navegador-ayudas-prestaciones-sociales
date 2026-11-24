# est-minutes: 120
# Retry of phases 3 and 4 of the baseline scorecard: in job 300 they ran while a demo API shared
# the 16 GB slice and every /agent call (and 15 of 58 /rag calls) died with CUDA OOM. The scorecard
# resumes from job 300's file and only re-runs rows without HTTP 200. Needs the GPU alone.
set -u
OUT="$(realpath "$OUT")"
W="$HOME/arca-runs/scorecard/baseline-code"
rm -rf "$W"; git worktree prune
git fetch -q origin eval/pipeline-baseline
git worktree add -q --detach "$W" origin/eval/pipeline-baseline
PERSIST="$HOME/arca-runs/scorecard/baseline"; mkdir -p "$PERSIST"
bash pipeline_eval/run_api_scorecard.sh "$W" "$PERSIST" baseline --phases 3 4
cp "$PERSIST"/scorecard.json "$PERSIST"/scorecard.md "$OUT/" 2>/dev/null || true
tail -200 "$PERSIST/api.log" > "$OUT/api_tail.log" 2>/dev/null || true
