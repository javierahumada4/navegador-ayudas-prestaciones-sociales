# est-minutes: 50
# Phase 3, retrieval ablations on four chunkings: sweep of the hybrid weight lambda (with 5-fold
# cross-validation), dense retrieval vs query instruction, BM25 tokenisation, and the questions
# each retriever misses. Independent evaluations: a failure only loses its own table.
set -u
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-rag"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync --extra rag --extra train -q
ev() { uv run --no-sync python -m rag.evaluate --reports-dir "$OUT" "$@" || echo "FAILED $*"; }
for chunking in recursive:800:100 recursive:1200:150 semantic:0.65:1000 semantic:0.7:1000; do
  echo "=== lambda sweep: $chunking"; ev lambda --chunking "$chunking"
done
for chunking in recursive:800:100 semantic:0.65:1000; do
  echo "=== query instructions: $chunking"; ev instructions --chunking "$chunking"
  echo "=== failures: $chunking"; ev failures --chunking "$chunking" -k 5
done
echo "=== BM25 tokenisation"; ev tokenizer --chunking recursive:800:100
