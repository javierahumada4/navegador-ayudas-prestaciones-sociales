# est-minutes: 120
# Baseline phases 3 and 4, clean: job 301 never ran (port 8899 still held by job 310's API, two
# runners alive) and jobs 300/310 shared the GPU with a demo API (CUDA OOM). Resumes job 300's
# scorecard and re-runs only the rows without HTTP 200. The script now refuses to start without
# 13 GB free and the scorecard aborts after 3 OOM in a row. Needs the GPU alone.
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
