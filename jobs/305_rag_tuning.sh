# est-minutes: 40
# Phase-3 parameters from data for the fixes variant: hybrid lambda sweep (with 5-fold CV) on two
# recursive chunkings, the abstention-gate calibration on the better one, then
# pipeline_eval/tune_rag.py writes rag_tuned.env (chunking, lambda, gate) for job 310.
set -u
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-api"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync --extra train --extra rag -q
R="$HOME/arca-runs/scorecard/rag-tuning"; mkdir -p "$R"
for c in recursive:800:100 recursive:1200:150; do
  uv run --no-sync python -m rag.evaluate --reports-dir "$R" lambda --chunking "$c" || echo "FAILED lambda $c"
done
BEST=$(uv run --no-sync python -c "from pathlib import Path; from pipeline_eval.tune_rag import best_lambda; print(best_lambda(Path('$R'))['chunking'])")
echo "best chunking: $BEST"
uv run --no-sync python -m rag.evaluate --reports-dir "$R" calibrate --chunking "$BEST" || echo "FAILED calibrate"
uv run --no-sync python -m pipeline_eval.tune_rag --reports "$R" --out-dir "$R"
cp "$R"/*.json "$R"/rag_tuned.env "$OUT/" 2>/dev/null || true
