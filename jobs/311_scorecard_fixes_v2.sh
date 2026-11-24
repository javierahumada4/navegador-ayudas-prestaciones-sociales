# est-minutes: 180
# Fixes variant, clean and complete: job 310 shared the GPU with a demo API (phases 3-4 OOM) and
# ran before fixes 8-10 (shared RAG weights, start-up warm-up, seeded /reasoning). Fresh scorecard
# directory, RAG parameters from job 305. Needs the GPU alone.
set -u
OUT="$(realpath "$OUT")"
TUNED="$HOME/arca-runs/scorecard/rag-tuning/rag_tuned.env"
if [ -f "$TUNED" ]; then . "$TUNED"; cat "$TUNED"; else echo "no rag_tuned.env: default RAG parameters"; fi
PERSIST="$HOME/arca-runs/scorecard/fixes-v2"; mkdir -p "$PERSIST"
bash pipeline_eval/run_api_scorecard.sh . "$PERSIST" fixes
cp "$PERSIST"/scorecard.json "$PERSIST"/scorecard.md "$PERSIST"/health.json "$OUT/" 2>/dev/null || true
tail -200 "$PERSIST/api.log" > "$OUT/api_tail.log" 2>/dev/null || true
