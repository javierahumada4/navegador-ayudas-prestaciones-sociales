"""Inference for the phase-1 IMV reasoning model and POST /reasoning."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from api.schemas import ReasoningResponse, VerifierVerdict
from rlm.data import build_prompt
from rlm.rewards import extract_answer, has_valid_format
from rlm.verifier import build_verifier

THINK_PATTERN = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL)


def split_thinking(raw: str) -> tuple[str, str]:
    match = THINK_PATTERN.search(raw)
    thinking = match.group("think").strip() if match else ""
    answer = extract_answer(raw)
    if answer is None:
        answer = raw.split("</think>")[-1].strip() if match else raw.strip()
    return thinking, answer


def _base_model_from_adapter(adapter: str) -> str:
    config = Path(adapter) / "adapter_config.json"
    if config.exists():
        base = json.loads(config.read_text()).get("base_model_name_or_path")
        if base:
            return base
    return "Qwen/Qwen3-0.6B"


def _resolve_adapter(adapter: str) -> str:
    """
    Return a local adapter folder, downloading it from the Hub when given a repo id.

    Args:
        adapter (str): Local folder, or Hugging Face repo id such as ``JES0406/imv-sft-lora``.

    Returns:
        str: Path to a local folder that holds ``adapter_config.json``.
    """
    if Path(adapter).exists():
        return adapter
    if adapter.count("/") == 1 and not adapter.startswith((".", "/")):
        from huggingface_hub import snapshot_download

        return snapshot_download(adapter, allow_patterns=["adapter_*", "*.json", "*.txt"])
    raise FileNotFoundError(f"ARCA_RLM_ADAPTER points to a missing folder: {adapter}")


@dataclass
class ReasoningModel:
    base_model: str
    adapter_path: str | None = None
    verifier_name: str = "imv"

    @classmethod
    def from_env(cls) -> "ReasoningModel":
        adapter = os.environ.get("ARCA_RLM_ADAPTER") or None
        if adapter is None:
            raise NotImplementedError(
                "Phase 1 is not wired yet: set ARCA_RLM_ADAPTER to the trained LoRA adapter."
            )
        adapter = _resolve_adapter(adapter)
        base_model = os.environ.get("ARCA_RLM_BASE_MODEL") or _base_model_from_adapter(adapter)
        return cls(
            base_model=base_model,
            adapter_path=adapter,
            verifier_name=os.environ.get("ARCA_RLM_VERIFIER", "imv"),
        )

    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model, dtype=dtype, device_map=device
        )
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        self.model = model.eval()
        self.verifier = build_verifier(self.verifier_name)

    def generate(
        self,
        question: str,
        max_new_tokens: int = 1024,
        *,
        do_sample: bool = True,
        temperature: float = 0.6,
    ) -> tuple[str, int]:
        import torch

        text = self.tokenizer.apply_chat_template(
            build_prompt(question), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": 0.95})
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), int(new_tokens.numel())

    def generate_batch(
        self,
        questions: list[str],
        max_new_tokens: int = 1024,
        *,
        do_sample: bool = True,
        temperature: float = 0.6,
    ) -> list[tuple[str, int]]:
        """Batched ``generate``: same prompt and decoding, one (raw, n_tokens) per question."""
        import torch

        texts = [
            self.tokenizer.apply_chat_template(
                build_prompt(q), tokenize=False, add_generation_prompt=True
            )
            for q in questions
        ]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        # Decoder-only models must be left-padded so every prompt ends where generation starts.
        padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        finally:
            self.tokenizer.padding_side = padding_side
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": pad_id}
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": 0.95})
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        return [
            (
                self.tokenizer.decode(row, skip_special_tokens=True),
                int((row != pad_id).sum()),
            )
            for row in new_tokens
        ]

    def answer(
        self, question: str, expected_answer: str | None = None, max_new_tokens: int = 1024
    ) -> ReasoningResponse:
        raw, n_tokens = self.generate(question, max_new_tokens)
        thinking, answer = split_thinking(raw)
        verdict = None
        if expected_answer is not None:
            result = self.verifier.verify(raw, expected_answer)
            verdict = VerifierVerdict(
                is_correct=result.is_correct,
                predicted=result.predicted,
                expected=result.expected,
                detail=result.detail,
            )
        return ReasoningResponse(
            thinking=thinking,
            answer=answer,
            raw=raw,
            has_valid_format=has_valid_format(raw),
            verifier=verdict,
            tokens_generated=n_tokens,
            model=f"{self.base_model}+{Path(self.adapter_path).name}"
            if self.adapter_path
            else self.base_model,
        )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Calcula el IMV/CAPI mensual del caso indicado."
    model = ReasoningModel.from_env()
    model.load()
    print(model.answer(question).model_dump_json(indent=2))
