# est-minutes: 260
# Phase 4: the 23-task agent bench (20 chained >= 2 tools, 3 impossible) with our JSON ReAct agent
# and the three brains (base, rlm = phase-1 GRPO LoRA, thinking = Qwen3 native reasoning), then
# the smolagents CodeAgent on the same model/tools/timeouts with the base brain (JSON vs code).
# Resumable: results are written after every task to a persistent file (--resume).
# Needs the GPU alone (two 1.7B copies + embedder): stop the API while it runs.
set -u
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-agent"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export ARCA_RLM_ADAPTER="${ARCA_RLM_ADAPTER:-JES0406/imv-sft-lora/grpo_final}"
export ARCA_IMV_REPORTS_DIR="$HOME/arca-runs/phase4/reports"
uv sync --extra train --extra rag --extra agent -q
PERSIST="$HOME/arca-runs/phase4"; mkdir -p "$PERSIST"
start=$(date +%s)
uv run --no-sync python -m agent.evaluate --bench agent/bench/tasks.jsonl \
    --agents json --brains base rlm thinking --max-steps 8 --out "$PERSIST/phase4_eval.json" --resume \
    || echo "FAILED json agent"
echo "json agent wall: $(( $(date +%s) - start )) s"
start=$(date +%s)
uv run --no-sync python -m agent.evaluate --bench agent/bench/tasks.jsonl \
    --agents code --brains base --max-steps 8 --out "$PERSIST/phase4_code.json" --resume \
    || echo "FAILED code agent"
echo "code agent wall: $(( $(date +%s) - start )) s"
# Ablations with the base brain: a broken tool (BDNS always fails) on the 4 BDNS tasks, and
# the memory budget cut from 6000 to 1500 tokens (summaries kick in) on the whole bench.
uv run --no-sync python -m agent.evaluate --bench agent/bench/tasks.jsonl --agents json --brains base \
    --only T03 T09 T12 T18 --break-tool consultar_ayudas --tag @bdns_broken --max-steps 8 \
    --out "$PERSIST/phase4_broken.json" --resume || echo "FAILED broken-tool run"
ARCA_AGENT_TOKEN_BUDGET=1500 uv run --no-sync python -m agent.evaluate --bench agent/bench/tasks.jsonl \
    --agents json --brains base --tag @budget1500 --max-steps 8 \
    --out "$PERSIST/phase4_memory.json" --resume || echo "FAILED memory run"
cp "$PERSIST"/phase4_*.json "$PERSIST"/phase4_*.md "$OUT/" 2>/dev/null || true
