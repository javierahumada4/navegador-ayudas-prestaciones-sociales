#!/usr/bin/env bash
# Start the API from CODE_DIR on a private port, run the scorecard against it from this checkout
# (same benches = same measuring stick for every variant), stop the API.
#   bash pipeline_eval/run_api_scorecard.sh <code_dir> <out_dir> <label> [scorecard args...]
# Needs the GPU alone: stop any other API first. Extra API settings come from the environment
# (e.g. ARCA_RAG_LAMBDA=0.3 bash pipeline_eval/run_api_scorecard.sh ...).
set -u
CODE_DIR="$(realpath "$1")"; OUT_DIR="$(mkdir -p "$2" && realpath "$2")"; LABEL="$3"; shift 3
HERE="$(pwd)"
PORT="${SCORECARD_PORT:-8899}"
export UV_PROJECT_ENVIRONMENT="${SCORECARD_VENV:-$HOME/.venvs/arca-api}"
export ARCA_API_PORT="$PORT"
export ARCA_RLM_ADAPTER="${ARCA_RLM_ADAPTER:-JES0406/imv-sft-lora/grpo_final}"
export ARCA_TOOLS_BRAIN=base ARCA_TOOLS_MODEL_ID=Qwen/Qwen3-1.7B ARCA_TOOLS_ENABLE_BOE=1
export ARCA_IMV_REPORTS_DIR="$HOME/arca-runs/scorecard/reports"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
if curl -s "localhost:$PORT/health" >/dev/null; then echo "port $PORT busy: stop the other API"; exit 1; fi
(cd "$CODE_DIR" && uv sync --frozen --extra train --extra rag -q)  # --frozen: never rewrite uv.lock
# Refuse to measure on a shared GPU: a demo API next to this one made every /agent call OOM.
FREE_MIB=$(cd "$CODE_DIR" && uv run --no-sync python -c "import torch; f, t = torch.cuda.mem_get_info(); print(int(f / 2**20))" 2>/dev/null)
echo "GPU free before start: ${FREE_MIB:-?} MiB"
if [ "${FREE_MIB:-0}" -lt "${SCORECARD_MIN_FREE_MIB:-13000}" ]; then
  echo "GPU busy (${FREE_MIB:-?} MiB free): stop the other API / job and re-queue"; exit 1
fi
start=$(date +%s)
(cd "$CODE_DIR" && exec setsid uv run --no-sync arca-api) > "$OUT_DIR/api.log" 2>&1 &
API_PID=$!
cleanup() {
  kill -- -"$API_PID" 2>/dev/null; kill "$API_PID" 2>/dev/null; sleep 3
  for p in $(lsof -t -i:"$PORT" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done  # never leave it holding the GPU
}
trap cleanup EXIT
echo "waiting for the API ($LABEL, $CODE_DIR) on port $PORT"
until curl -s "localhost:$PORT/health" >/dev/null; do
  kill -0 "$API_PID" 2>/dev/null || { echo "API died:"; tail -30 "$OUT_DIR/api.log"; exit 1; }
  [ $(( $(date +%s) - start )) -gt 2400 ] && { echo "API not up after 40 min"; exit 1; }
  sleep 10
done
echo "API up in $(( $(date +%s) - start )) s"
# Code with a start-up warm-up (eval/pipeline-fixes): wait for it, so the scorecard measures the
# warm API the grader would meet. The baseline has none and is measured cold, as it is deployed.
if grep -q "def warmup" "$CODE_DIR/api/app.py"; then
  t=$(date +%s)
  until grep -q "warm-up agent" "$OUT_DIR/api.log"; do
    [ $(( $(date +%s) - t )) -gt 900 ] && { echo "warm-up not finished after 15 min"; break; }
    sleep 10
  done
  grep "warm-up" "$OUT_DIR/api.log" | sed 's/^/  /'
  echo "warm-up wait: $(( $(date +%s) - t )) s"
fi
# The first /health loads every phase (embedder + corpus index): time it separately.
t=$(date +%s); curl -s "localhost:$PORT/health" > "$OUT_DIR/health.json"; echo "first /health: $(( $(date +%s) - t )) s"
cd "$HERE"
start=$(date +%s)
uv run --no-sync --project "$CODE_DIR" python -m pipeline_eval.scorecard --api "http://localhost:$PORT" \
    --label "$LABEL" --out "$OUT_DIR/scorecard.json" --resume "$@" || echo "scorecard FAILED"
echo "scorecard wall: $(( $(date +%s) - start )) s"
