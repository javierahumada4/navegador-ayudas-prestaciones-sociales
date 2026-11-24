"""The three brains of the agent and a process-wide pool so no model is loaded twice.

The DGX slice has 16 GB and the API already holds the phase-1 model (``/reasoning``) and the
phase-2 base model (``/tools``). Loading two more copies of Qwen3-1.7B for ``/agent`` would not
fit next to the phase-3 embedder, so the pool first looks for a model another phase already
loaded with the same weights and only loads its own copy when there is none.

Brains (``brain`` of ``POST /agent``):

- ``base``: ``ARCA_AGENT_BASE_MODEL`` (default ``Qwen/Qwen3-1.7B``), thinking disabled.
- ``rlm``: the same base plus the phase-1 GRPO LoRA (``ARCA_RLM_ADAPTER``), thinking disabled:
  the adapter was trained to write its own ``<think>`` block.
- ``thinking``: ``ARCA_THINKING_MODEL`` (default ``Qwen/Qwen3-1.7B``) with Qwen3's native
  reasoning on (``enable_thinking=True``). Same weights as ``base`` by default, so the comparison
  isolates the effect of reasoning: base vs native reasoning vs our trained reasoning.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


@dataclass(frozen=True)
class BrainSpec:
    name: str
    model_id: str
    adapter: str | None
    enable_thinking: bool
    max_new_tokens: int

    @property
    def label(self) -> str:
        return f"{self.model_id}+{self.adapter}" if self.adapter else self.model_id


def brain_spec(name: str) -> BrainSpec:
    base = os.environ.get("ARCA_AGENT_BASE_MODEL", DEFAULT_MODEL)
    if name == "base":
        return BrainSpec("base", base, None, False, 1024)
    if name == "rlm":
        adapter = os.environ.get("ARCA_RLM_ADAPTER", "").strip()
        if not adapter:
            raise NotImplementedError(
                "brain='rlm' needs ARCA_RLM_ADAPTER (the phase-1 GRPO LoRA, local or on the Hub)"
            )
        return BrainSpec("rlm", base, adapter, False, 1024)
    if name == "thinking":
        model = os.environ.get("ARCA_THINKING_MODEL", DEFAULT_MODEL)
        # Native reasoning spends tokens before the action: give it room.
        return BrainSpec("thinking", model, None, True, 2048)
    raise ValueError(f"unknown brain {name!r}: use base, rlm or thinking")


class ModelPool:
    """(tokenizer, model) per (model_id, adapter), shared with the other phases when possible."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str | None], tuple[object, object]] = {}
        self._lock = threading.Lock()

    def get(self, spec: BrainSpec) -> tuple[object, object]:
        key = (spec.model_id, spec.adapter)
        with self._lock:
            if key not in self._models:
                self._models[key] = self._borrow(spec) or self._load(spec)
            return self._models[key]

    @staticmethod
    def _borrow(spec: BrainSpec) -> tuple[object, object] | None:
        """Load through the phase that owns the same weights, so the process holds one copy.

        Base weights: the phase-2 ``ToolLoop`` (when it runs the base brain on the same model).
        Base + phase-1 LoRA: the API's ``ReasoningModel`` (when the API is the process).
        Inside the API either phase may be called first; whichever comes first loads, the
        other reuses. Outside the API (evaluation scripts) the pool loads its own copy.
        """
        if spec.adapter is None:
            if os.environ.get("ARCA_TOOLS_BRAIN", "base").lower() != "base":
                return None
            if os.environ.get("ARCA_TOOLS_MODEL_ID", DEFAULT_MODEL) != spec.model_id:
                return None
            if "api.app" not in sys.modules and "tool_use.executor" not in sys.modules:
                return None
            from tool_use.executor import build_tool_loop

            loop = build_tool_loop()
            loop.load()
            return loop.tokenizer, loop.model
        app = sys.modules.get("api.app")
        backends = getattr(app, "backends", None)
        if backends is None:
            return None
        try:
            reasoning = backends.reasoning()
        except Exception:  # noqa: BLE001 - phase 1 not configured: load our own copy
            return None
        if getattr(reasoning, "base_model", None) != spec.model_id:
            return None
        from rlm.inference import _resolve_adapter

        if reasoning.adapter_path in {spec.adapter, _resolve_adapter(spec.adapter)}:
            return reasoning.tokenizer, reasoning.model
        return None

    @staticmethod
    def _load(spec: BrainSpec) -> tuple[object, object]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.backends.cuda.enable_cudnn_sdp(False)  # crashes on padded batches on the MIG slice
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, dtype=dtype, device_map="auto" if cuda else "cpu"
        )
        if spec.adapter:
            from peft import PeftModel

            from rlm.inference import _resolve_adapter

            # Merged LoRA: ~75 instead of ~40 tok/s measured on the MIG slice (phase 2).
            model = PeftModel.from_pretrained(model, _resolve_adapter(spec.adapter)).merge_and_unload()
        return tokenizer, model.eval()


POOL = ModelPool()


def generate(spec: BrainSpec, messages: list[dict], tools: list[dict] | None) -> tuple[str, int, int]:
    """One greedy completion: (text, prompt tokens, completion tokens)."""
    import torch

    tokenizer, model = POOL.get(spec)
    options = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        options["tools"] = tools
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=spec.enable_thinking, **options
        )
    except TypeError as exc:  # templates without the Qwen3 thinking switch
        if "enable_thinking" not in str(exc):
            raise
        prompt = tokenizer.apply_chat_template(messages, **options)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=spec.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    continuation = output[0, encoded["input_ids"].shape[-1] :]
    # Keep <think> (it is part of the trace) but drop the end-of-turn tokens.
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    return text, int(encoded["input_ids"].shape[-1]), int(continuation.shape[-1])


def count_tokens(spec: BrainSpec, text: str) -> int:
    tokenizer, _ = POOL.get(spec)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
