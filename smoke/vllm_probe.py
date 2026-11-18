"""Smoke test of vLLM on the DGX slice: engine start, one LoRA request, throughput.

    UV_PROJECT_ENVIRONMENT=~/.venvs/arca-distill timeout 900 uv run --extra distill python smoke/vllm_probe.py

Exit 0 only if every completion came back; jobs use it to choose vLLM or HF.
"""

import os
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from rlm.inference import _resolve_adapter

    t0 = time.time()
    llm = LLM(model="Qwen/Qwen3-1.7B", dtype="bfloat16", enable_lora=True, max_lora_rank=64,
              gpu_memory_utilization=0.85, max_model_len=8192, seed=0)
    print(f"engine up in {time.time() - t0:.0f} s", flush=True)
    prompts = [f"Suma {i} + {i * 3} y explica el resultado." for i in range(16)]
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256, seed=0)
    lora = LoRARequest("grpo", 1, _resolve_adapter("JES0406/imv-sft-lora/grpo_final"))
    for name, request in (("base", None), ("lora", lora)):
        t1 = time.time()
        outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=False)
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        assert len(outputs) == len(prompts) and tokens > 0
        print(f"{name}: {tokens} tokens in {time.time() - t1:.1f} s "
              f"= {tokens / (time.time() - t1):.0f} tok/s", flush=True)
    print("VLLM_OK")


if __name__ == "__main__":
    main()
