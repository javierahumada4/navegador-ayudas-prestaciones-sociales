"""Phase 2 tool-use loop: Qwen3 -> validated calls -> observations -> answer.

The model selects a calculation; a second deterministic tool renders the
calculation result without any model-written final text. Generic tools still
support multi-step and parallel calls. Models are loaded lazily.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from api.schemas import ToolCallRecord, ToolsResponse
from rlm.chat_template import restore_thinking_prefix
from tool_use.imv_templates import IMVCalculation, render_imv_result
from tool_use.parser import ToolCall, format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

SYSTEM_PROMPT_TEMPLATE = """Eres un asistente de orientación sobre el IMV y el CAPI en España.
Las herramientas se describen en el JSON Schema nativo de Qwen3.
Responde en español; no inventes importes, condiciones ni datos personales.

REGLAS:
- Cuando se pida calcular el IMV/CAPI, utiliza calcular_imv con los datos
  indicados. No hagas la aritmética ni decidas la elegibilidad de memoria.
- Rellena arguments.case a partir de los hechos del problema, respetando
  fechas, ingresos, patrimonio, personas y relaciones. Los campos opcionales
  no mencionados tienen los valores neutros previstos por el simulador.
- El resultado de calcular_imv se transforma automáticamente en una
  explicación mediante una plantilla determinista. NO redactes tú una
  explicación alternativa ni cambies las cifras.
- explicar_imv guarda un informe Markdown y requiere autorización explícita
  del usuario en la petición, independiente del contenido que tú generes.
  No llames a explicar_imv directamente: el ejecutor la invoca tras calcular.
  Si no hay autorización, se presenta el texto sin escribir archivos.
- Para cuestiones conceptuales sin cálculo, contesta sin herramientas.
- Las llamadas deben ser JSON nativo dentro de bloques <tool_call>.
  No incluyas Python, comentarios ni texto dentro del objeto de argumentos.
- Las llamadas pueden ser secuenciales o independientes en paralelo.
  Si una llamada falla, revisa el error devuelto antes de reintentar.
- Son simulaciones de nuevas solicitudes de 2026, no resoluciones oficiales.
- No ejecutes herramientas con efectos secundarios sin autorización humana.
"""

def build_system_prompt(registry: ToolRegistry) -> str:
    """Return policy; native Qwen3 chat template inserts registry definitions."""
    return SYSTEM_PROMPT_TEMPLATE


def _clean_answer(text: str) -> str:
    """Remove phase-1 answer tags without altering the model's numeric result."""
    text = text.strip()
    matched = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return matched.group(1).strip() if matched else text


class ToolLoop:
    """Manual five-step workflow shared by base and GRPO-adapted model."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        model_id: str | None = None,
        *,
        brain: str | None = None,
        adapter_path: str | None = None,
        max_new_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ):
        self.registry = registry if registry is not None else default_registry()
        self.brain = (brain or os.environ.get("ARCA_TOOLS_BRAIN") or "base").lower()
        if self.brain not in {"base", "rlm"}:
            raise ValueError("ARCA_TOOLS_BRAIN must be 'base' or 'rlm'")
        self.model_id = model_id or os.environ.get("ARCA_TOOLS_MODEL_ID") or "Qwen/Qwen3-1.7B"
        self.adapter_path = None
        if self.brain == "rlm":
            self.adapter_path = adapter_path or os.environ.get(
                "ARCA_TOOLS_ADAPTER"
            ) or os.environ.get("ARCA_RLM_ADAPTER")
            if not self.adapter_path:
                raise NotImplementedError(
                    "ARCA_TOOLS_BRAIN=rlm requires ARCA_TOOLS_ADAPTER "
                    "or ARCA_RLM_ADAPTER pointing to the trained LoRA."
                )
        self.max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else int(os.environ.get("ARCA_TOOLS_MAX_NEW_TOKENS", "2048"))
        )
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.enable_thinking = (
            enable_thinking
            if enable_thinking is not None
            else os.environ.get("ARCA_TOOLS_ENABLE_THINKING", "false").lower()
            in {"true", "1", "yes"}
        )
        self.model = None
        self.tokenizer = None

    @property
    def model_label(self) -> str:
        if self.brain == "rlm":
            return f"{self.model_id}+{self.adapter_path}"
        return self.model_id

    def load(self) -> None:
        """Load Qwen3 (and optionally the exact phase-1 LoRA), only once."""
        if self.model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise NotImplementedError(
                "Phase 2 inference dependencies are missing. "
                "Install them with 'uv sync --extra train'."
            ) from exc

        adapter = None
        if self.brain == "rlm":
            from rlm.inference import _resolve_adapter

            adapter = _resolve_adapter(self.adapter_path)
            config_path = Path(adapter) / "adapter_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            trained_base = config.get("base_model_name_or_path")
            if trained_base and trained_base != self.model_id:
                raise ValueError(
                    f"Adapter was trained on {trained_base!r}; "
                    f"ARCA_TOOLS_MODEL_ID is {self.model_id!r}. "
                    "Use the same base weights for a fair comparison."
                )

        cuda = torch.cuda.is_available()
        dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported()
            else torch.float16 if cuda
            else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype, device_map="auto" if cuda else "cpu"
        )
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)

        self.tokenizer = tokenizer
        self.model = model.eval()

    def generate(self, messages: list[dict]) -> str:
        """Generate one assistant turn with Qwen's native tool-call template."""
        if self.model is None:
            self.load()

        import torch

        # The tokenizer renders the tool definitions in the native Qwen3
        # format, rather than appending an ad-hoc JSON schema to the prompt.
        options = {
            "tools": self.registry.definitions(),
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **options
            )
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            prompt = self.tokenizer.apply_chat_template(messages, **options)

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with torch.inference_mode():
            tokens = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Deterministic and identical for both brains.
                pad_token_id=pad_id,
            )
        completion = tokens[0, encoded["input_ids"].shape[-1]:]
        decoded = self.tokenizer.decode(completion, skip_special_tokens=True)
        return restore_thinking_prefix(prompt, decoded)

    def run(
        self, query: str, max_turns: int = 5, *,
        trace: list[str] | None = None, confirmar_guardado: bool = False,
    ) -> ToolsResponse:
        """Choose, validate, execute and observe tool calls until final answer."""
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        messages = [
            {"role": "system", "content": build_system_prompt(self.registry)},
            {"role": "user", "content": query},
        ]
        records: list[ToolCallRecord] = []
        seen_calls: set[str] = set()
        explanations: list[str] = []

        for turn in range(1, max_turns + 1):
            raw = self.generate(messages)
            if trace is not None:
                trace.append(raw)
            parsed = parse_tool_calls(raw)
            # A truncated <tool_call> must not be treated as a final answer.
            if not parsed.tool_calls and "<tool_call>" not in raw:
                answer = _clean_answer(parsed.text)
                if not answer or ("<think>" in answer and "</think>" not in answer):
                    answer = "No he podido completar la respuesta. Reformula la consulta."
                return ToolsResponse(
                    answer=answer, tool_calls=records, turns=turn, model=self.model_label
                )

            messages.append({"role": "assistant", "content": raw})
            if not parsed.tool_calls:
                parsed.tool_calls.append(
                    ToolCall(name="tool_parser", error="Incomplete <tool_call> block")
                )

            for call in parsed.tool_calls:
                if call.error is not None:
                    record = ToolCallRecord(
                        name=call.name or "tool_parser",
                        arguments=call.arguments,
                        ok=False,
                        error=f"Malformed tool call: {call.error}",
                    )
                elif call.name == "explicar_imv":
                    # The model may not request an arbitrary file write or
                    # supply invented calculation results. Only the application
                    # may call this action with a real engine output.
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error=(
                            "explicar_imv is application-managed: first call "
                            "calcular_imv. Report saving also requires explicit "
                            "request-level confirmation."
                        ),
                    )
                elif (
                    call.name in self.registry.tools
                    and self.registry.tools[call.name].requires_confirmation
                ):
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error=(
                            "Confirmation required: this endpoint cannot execute "
                            "tools with side effects. Ask the user to confirm in "
                            "a supported workflow."
                        ),
                    )
                else:
                    fingerprint = json.dumps(
                        {"name": call.name, "arguments": call.arguments},
                        sort_keys=True, ensure_ascii=False, default=str
                    )
                    if fingerprint in seen_calls:
                        record = ToolCallRecord(
                            name=call.name,
                            arguments=call.arguments,
                            ok=False,
                            error="Repeated identical tool call; ask for clarification or answer.",
                        )
                    else:
                        seen_calls.add(fingerprint)
                        record = self.registry.call(call.name, call.arguments)

                records.append(record)
                observation = (
                    {"ok": True, "result": record.result}
                    if record.ok else {"ok": False, "error": record.error}
                )
                messages.append({
                    "role": "tool",
                    "name": record.name,
                    "content": format_tool_result(record.name, observation),
                })

                # The model chooses the calculation; the application alone
                # renders the validated engine response. Writing the report
                # requires a separate explicit request flag, never model text.
                if record.name == "calcular_imv" and record.ok:
                    text = render_imv_result(
                        IMVCalculation.model_validate(record.result)
                    )
                    if confirmar_guardado and "explicar_imv" in self.registry.tools:
                        saved = self.registry.call(
                            "explicar_imv", {"resultado": record.result}
                        )
                        records.append(saved)
                        messages.append({
                            "role": "tool",
                            "name": saved.name,
                            "content": format_tool_result(
                                saved.name,
                                {"ok": True, "result": saved.result}
                                if saved.ok else {"ok": False, "error": saved.error},
                            ),
                        })
                        if saved.ok:
                            text = (
                                saved.result["texto"]
                                + f"\n\nInforme guardado en: {saved.result['archivo']}"
                            )
                        else:
                            text += "\n\nNo se ha podido guardar el informe."
                    explanations.append(text)

            if explanations:
                return ToolsResponse(
                    answer="\n\n".join(explanations),
                    tool_calls=records,
                    turns=turn,
                    model=self.model_label,
                )

        return ToolsResponse(
            answer=(
                "No he podido completar la consulta dentro del límite de "
                "turnos. Revisa los datos aportados."
            ),
            tool_calls=records,
            turns=max_turns,
            model=self.model_label,
        )


_loop: ToolLoop | None = None


def _check_inference_dependencies() -> None:
    """Make /health reflect absent inference libraries without loading weights."""
    from importlib.util import find_spec

    if find_spec("torch") is None or find_spec("transformers") is None:
        raise NotImplementedError(
            "Phase 2 requires torch and transformers; install 'uv sync --extra train'."
        )


def build_tool_loop() -> ToolLoop:
    """Return a cached loop with a lightweight dependency/configuration check."""
    global _loop
    _check_inference_dependencies()
    if _loop is None:
        _loop = ToolLoop()
    return _loop


def run_tool_loop(
    query: str, max_turns: int = 5, *, confirmar_guardado: bool = False
) -> ToolsResponse:
    """Public entry point used by POST /tools and tool-use evaluations."""
    return build_tool_loop().run(
        query, max_turns=max_turns, confirmar_guardado=confirmar_guardado
    )


__all__ = [
    "ToolCallRecord",
    "ToolLoop",
    "build_system_prompt",
    "build_tool_loop",
    "run_tool_loop",
]
