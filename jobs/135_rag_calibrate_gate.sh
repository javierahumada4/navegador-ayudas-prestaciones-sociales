# est-minutes: 10
# Phase 3, "no lo se": top-1 dense cosine of the 50 answerable questions vs the 8 questions the
# corpus cannot answer, and what each possible gate would keep and reject. No generator needed.
set -u
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-rag"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync --extra rag --extra train -q
uv run --no-sync python -m rag.evaluate --reports-dir "$OUT" calibrate --chunking recursive:800:100
