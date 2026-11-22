# est-minutes: 20
# Phase 3 (RAG), step 0: does the environment work, and do the DGX numbers match the local ones?
# Private venv with the rag + train extras (train brings CUDA torch and matplotlib); the RAG
# tests; one quick evaluation. Local reference (RTX 4070, recursive 800/100, 50 gold questions):
# bm25 R@5 .510 MRR .326 | dense R@5 .700 MRR .444 | hybrid(0.5) R@5 .700 MRR .463.
set -e
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-rag"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync --extra rag --extra train -q
PY="uv run --no-sync python"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$PY -m pytest tests/test_rag.py tests/test_rag_corpus.py tests/test_rag_pipeline.py \
    tests/test_rag_retriever.py tests/test_rag_chunking.py tests/test_rag_generate.py -q -p no:warnings
$PY -m rag.evaluate --reports-dir "$OUT" chunking --specs recursive:800:100
