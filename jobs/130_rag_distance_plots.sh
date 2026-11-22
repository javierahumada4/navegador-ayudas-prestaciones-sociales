# est-minutes: 10
# Phase 3: cosine distance between consecutive sentences, with the candidate thresholds drawn
# (the figure that justifies the semantic-chunking threshold). Three documents of different style.
set -u
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-rag"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync --extra rag --extra train -q
for doc in ley-19-2021 ss-imv rd-789-2022; do
  uv run --no-sync python -m rag.evaluate --reports-dir "$OUT" plot-distances --doc "$doc" \
      --thresholds 0.25 0.5 0.65 0.75 --out "$OUT/distances_$doc.png" || echo "FAILED $doc"
done
