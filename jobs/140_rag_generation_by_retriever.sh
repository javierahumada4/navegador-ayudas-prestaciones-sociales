# est-minutes: 60
# Phase 3, end to end: Qwen3-1.7B answers the 50 gold questions and the 8 unanswerable ones with
# the augmented prompt, once per retriever (hybrid, dense, bm25), the abstention gate OFF so that
# the model's own "no lo se" is what gets measured. Metrics: answered / false abstentions, a
# relevant chunk retrieved and cited, numbers in the answer supported by the cited chunks, correct
# abstentions, and how often each safety net fired. HF generation (58 questions, <= 400 tokens
# each) is minutes: no need for vLLM.
set -u
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-rag"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync --extra rag --extra train -q
for retriever in hybrid dense bm25; do
  echo "=== generation with $retriever"
  uv run --no-sync python -m rag.evaluate --reports-dir "$OUT" generation \
      --chunking recursive:800:100 --retriever "$retriever" --min-score 0.0 \
      | grep -v -i "warn" || echo "FAILED $retriever"
done
