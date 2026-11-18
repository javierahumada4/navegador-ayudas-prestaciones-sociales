# Does vLLM work on this MIG slice at all? Job 011 went > 7 h silent, likely a vLLM hang.
# Hard 15-min cap: engine start, LoRA request, 16 prompts x 256 tokens; prints tokens/s.
# Jobs 030/040 run the same probe and fall back to HF when it fails.
set -e
OUT="$(realpath "$OUT")"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/arca-distill"
timeout 900 uv sync --extra distill -q
timeout 900 uv run --extra distill python smoke/vllm_probe.py
