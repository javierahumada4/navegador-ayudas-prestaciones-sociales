# est-minutes: 180
# Scorecard of THIS branch (eval/pipeline-fixes): same scorecard and benches as job 300, API from
# this checkout with the RAG parameters chosen by job 305 (falls back to the defaults if missing).
set -u
OUT="$(realpath "$OUT")"
TUNED="$HOME/arca-runs/scorecard/rag-tuning/rag_tuned.env"
if [ -f "$TUNED" ]; then . "$TUNED"; cat "$TUNED"; else echo "no rag_tuned.env: default RAG parameters"; fi
PERSIST="$HOME/arca-runs/scorecard/fixes"; mkdir -p "$PERSIST"
bash pipeline_eval/run_api_scorecard.sh . "$PERSIST" fixes
cp "$PERSIST"/scorecard.json "$PERSIST"/scorecard.md "$PERSIST"/health.json "$OUT/" 2>/dev/null || true
tail -200 "$PERSIST/api.log" > "$OUT/api_tail.log" 2>/dev/null || true
