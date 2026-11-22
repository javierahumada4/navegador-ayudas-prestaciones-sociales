# est-minutes: 45
# Phase 3, chunking comparison: recursive (4 sizes) vs semantic (threshold sweep, two size caps),
# each scored with BM25, dense and hybrid(lambda=0.5) on the 50 gold questions. One evaluation per
# spec, so that a failure or a cut loses at most one row. Local rows already measured (RTX 4070):
# recursive 400/50, 800/100, 1200/150 and semantic 0.5 and 0.65 (1000 chars).
set -u
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-rag"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync --extra rag --extra train -q
for spec in recursive:400:50 recursive:800:100 recursive:1200:150 recursive:1600:200 \
            semantic:0.5:1000 semantic:0.55:1000 semantic:0.6:1000 semantic:0.65:1000 \
            semantic:0.7:1000 semantic:0.75:1000 semantic:0.8:1000 \
            semantic:0.65:1500 semantic:0.7:1500 semantic:0.75:1500; do
  start=$(date +%s)
  uv run --no-sync python -m rag.evaluate --reports-dir "$OUT" chunking --specs "$spec" \
      | grep "^| \(rec\|sem\)" || echo "FAILED $spec"
  echo "  ($spec: $(( $(date +%s) - start )) s)"
done
