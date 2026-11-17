"""Hand-written five-step tool use: prompt -> generate -> parse -> execute -> observe.

No agent framework. Qwen3 base and phase-1 LoRA share generation settings.
"""
import json
import os
import re

from api.schemas import ToolCallRecord, ToolsResponse
from tool_use.parser import format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

SYSTEM_PROMPT = """Eres un asistente español de IMV/CAPI y navegación de ayudas públicas.
Las herramientas están descritas con sus esquemas JSON. Usa herramientas SOLO si la
petición las requiere, nunca para preguntas conceptuales contestables sin ellas.
- calcular_imv: extrae HECHOS, no conclusiones, y pide aclaraciones si faltan datos
  necesarios. Nunca inventes residencia, fecha, ingresos, patrimonio o parentesco.
  El motor de 2026 es un simulador parcial, NO una decisión administrativa.
- consultar_ayudas: consulta convocatorias en la BDNS, no asumas que están abiertas
  o que el ciudadano tiene derecho a ellas. El IMV/CAPI no se buscan en BDNS.
- guardar_informe: primero recibe el resultado real del motor. Redacta TÚ el texto
  del informe en español, en lenguaje natural, respetando literalmente importes,
  distinción IMV/CAPI, motivos y limitaciones; luego envía ese texto a la herramienta.
  Solo guarda si el usuario pide explícitamente guardar, generar o descargar un
  informe/archivo. No escribas un fichero para responder a una pregunta normal.
Para llamar a una herramienta emite exactamente:
<tool_call>{"name":"nombre","arguments":{"campo":"valor"}}</tool_call>
Se permiten varias llamadas INDEPENDIENTES en una generación. Las llamadas que
dependen de otras requieren otra generación tras observar el resultado.
Al recibir un error de validación, corrige los argumentos o pide los datos; jamás
inventes una cuantía como sustituto del motor. Explica los límites con claridad.
Cuando termines responde directamente en español SIN llamadas a herramientas."""


def build_system_prompt(registry: ToolRegistry) -> str:
    return SYSTEM_PROMPT + "\n\nDefiniciones disponibles:\n" + registry.render_for_prompt()


def _report_requested(query: str) -> bool:
    """Treat an explicit file request as authorization for the local report action."""
    return bool(re.search(
        r"(?i)\b(guarda|guardar|genera|generar|crea|crear|descarga|descargar|"
        r"exporta|exportar|escribe|escribir)\b", query
    ) and re.search(r"(?i)\b(informe|archivo|fichero|txt|documento)\b", query))


def _money_is_mentioned(content: str, value: float) -> bool:
    number = f"{value:.2f}"
    possible = {number, number.replace(".", ",")}
    if value >= 1000:
        possible |= {
            f"{value:,.2f}", f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        }
    return any(v in content for v in possible)


class ToolLoop:
    def __init__(
        self, registry: ToolRegistry | None = None, model_id: str | None = None,
        *, brain: str | None = None, adapter_path: str | None = None,
        use_tools: bool = True, max_new_tokens: int = 2048,
    ):
        self.registry = registry if registry is not None else default_registry()
        self.brain = (brain or os.getenv("ARCA_TOOLS_BRAIN", "base")).lower()
        if self.brain not in {"base", "rlm"}:
            raise ValueError("ARCA_TOOLS_BRAIN must be base or rlm")
        self.model_id = model_id or os.getenv("ARCA_TOOLS_MODEL_ID", "Qwen/Qwen3-1.7B")
        self.adapter_path = adapter_path or os.getenv("ARCA_TOOLS_ADAPTER") or os.getenv("ARCA_RLM_ADAPTER")
        self.use_tools = use_tools
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        if self.brain == "rlm" and not self.adapter_path:
            raise ValueError("ARCA_TOOLS_BRAIN=rlm needs ARCA_TOOLS_ADAPTER or ARCA_RLM_ADAPTER")

    @property
    def model_label(self):
        return self.model_id if self.brain == "base" else f"{self.model_id}+LoRA"

    def load(self):
        """Lazy model initialization avoids loading GPU weights during /health."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = None
        if self.brain == "rlm":
            from rlm.inference import _resolve_adapter
            adapter = _resolve_adapter(self.adapter_path)
            from pathlib import Path
            config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
            trained_base = config.get("base_model_name_or_path")
            if trained_base and trained_base != self.model_id:
                raise ValueError("LoRA base model differs from ARCA_TOOLS_MODEL_ID")
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (
            torch.float16 if cuda else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype, device_map="auto" if cuda else "cpu"
        )
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.tokenizer, self.model = tokenizer, model.eval()

    def generate(self, messages: list[dict]) -> str:
        """One model completion; injected in unit tests to avoid GPU requirements."""
        self.load()
        import torch

        options = {"tokenize": False, "add_generation_prompt": True}
        if self.use_tools:
            options["tools"] = self.registry.definitions()
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **options
            )
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            prompt = self.tokenizer.apply_chat_template(messages, **options)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, do_sample=False, max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        continuation = output[0, encoded["input_ids"].shape[-1]:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True)

    def run(self, query: str, max_turns: int = 5) -> ToolsResponse:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        messages = [
            {"role": "system", "content": build_system_prompt(self.registry)
             if self.use_tools else "Responde en español sin herramientas ni acceso externo."},
            {"role": "user", "content": query},
        ]
        records: list[ToolCallRecord] = []
        seen: set[str] = set()
        for turn in range(1, max_turns + 1):
            raw = self.generate(messages)
            parsed = parse_tool_calls(raw)
            if not self.use_tools:
                return ToolsResponse(answer=parsed.text or raw, tool_calls=[], turns=turn,
                                     model=self.model_label)
            # An incomplete tool block is a recoverable error, never a final answer.
            malformed_block = "<tool_call>" in raw and "</tool_call>" not in raw
            if not parsed.tool_calls and not malformed_block:
                answer = parsed.text.strip()
                if not answer:
                    answer = "No he podido completar la consulta con los datos disponibles."
                return ToolsResponse(answer=answer, tool_calls=records, turns=turn,
                                     model=self.model_label)
            messages.append({"role": "assistant", "content": raw})
            calls = parsed.tool_calls
            if malformed_block and not calls:
                records.append(ToolCallRecord(name="tool_parser", arguments={}, ok=False,
                                              error="Incomplete tool_call block"))
                messages.append({"role": "tool", "name": "tool_parser",
                                 "content": "Error: bloque <tool_call> incompleto. Reintenta."})
                continue
            for call in calls:
                if call.error:
                    record = ToolCallRecord(name=call.name or "tool_parser", arguments=call.arguments,
                                            ok=False, error=f"Malformed tool call: {call.error}")
                elif call.name == "guardar_informe" and not _report_requested(query):
                    record = ToolCallRecord(
                        name=call.name, arguments=call.arguments, ok=False,
                        error="Report writing denied: user did not explicitly request a file."
                    )
                elif call.name == "guardar_informe" and not any(
                    r.ok and r.name == "calcular_imv" for r in records
                ):
                    record = ToolCallRecord(
                        name=call.name, arguments=call.arguments, ok=False,
                        error="Run calcular_imv successfully before saving an IMV report."
                    )
                elif call.name == "guardar_informe" and not all(
                    _money_is_mentioned(str(call.arguments.get("contenido", "")), r.result[key])
                    for r in records if r.ok and r.name == "calcular_imv"
                    for key in ("imv_monthly_eur", "capi_monthly_eur", "total_monthly_eur")
                ):
                    record = ToolCallRecord(
                        name=call.name, arguments=call.arguments, ok=False,
                        error="Report must reproduce every IMV/CAPI/total amount from the engine."
                    )
                elif (
                    call.name in self.registry.tools
                    and self.registry.tools[call.name].requires_confirmation
                    and call.name != "guardar_informe"
                ):
                    record = ToolCallRecord(
                        name=call.name, arguments=call.arguments, ok=False,
                        error="Tool requires explicit confirmation."
                    )
                else:
                    signature = json.dumps(
                        [call.name, call.arguments], sort_keys=True, default=str, ensure_ascii=False
                    )
                    if signature in seen:
                        record = ToolCallRecord(name=call.name, arguments=call.arguments,
                                                ok=False, error="Repeated identical call.")
                    else:
                        seen.add(signature)
                        record = self.registry.call(call.name, call.arguments)
                records.append(record)
                observation = {"ok": record.ok}
                observation.update({"result": record.result} if record.ok else {"error": record.error})
                messages.append({
                    "role": "tool", "name": record.name,
                    "content": format_tool_result(record.name, observation),
                })
        return ToolsResponse(
            answer="No he podido completar la consulta dentro del límite de turnos.",
            tool_calls=records, turns=max_turns, model=self.model_label,
        )


_loop: ToolLoop | None = None


def build_tool_loop() -> ToolLoop:
    global _loop
    if _loop is None:
        from importlib.util import find_spec
        if find_spec("torch") is None or find_spec("transformers") is None:
            raise NotImplementedError("Phase 2 needs torch/transformers: uv sync --extra train")
        _loop = ToolLoop()
    return _loop


def run_tool_loop(query: str, max_turns: int = 5) -> ToolsResponse:
    return build_tool_loop().run(query, max_turns)


__all__ = ["ToolLoop", "ToolCallRecord", "build_tool_loop", "run_tool_loop", "build_system_prompt"]
