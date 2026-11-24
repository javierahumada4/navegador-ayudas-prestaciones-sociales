# Runner smoke test: proves pull -> run -> push works and records the GPU and toolchain.
nvidia-smi
uv --version
uv sync --extra train -q
uv run python -c "import torch, transformers, trl; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, trl.__version__)"
uv run python -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm not installed"
