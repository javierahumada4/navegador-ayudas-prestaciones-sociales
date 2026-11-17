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

SYSTEM_PROMPT = """Eres un asistente de simulación IMV/CAPI y búsqueda de ayudas públicas.
Utiliza SOLO las herramientas disponibles en las definiciones JSON Schema.
Cuando un argumento obligatorio se desconoce, PREGUNTA; nunca inventes fechas,
residencia, convivencia, ingresos o patrimonio. El IMV/CAPI se calcula con el
motor determinista, no con estimaciones del modelo. Una subvención en BDNS
NO equivale a concesión, compatibilidad o plazo abierto.
Las llamadas deben escribirse exactamente como:
<tool_call>{"name":"nombre","arguments":{"campo":"valor"}}</tool_call>
Puedes emitir varias llamadas independientes en una generación. No hagas
llamadas dependientes hasta recibir las observaciones. Corrige errores de
validación usando el mensaje de la herramienta. Responde en español y
distingue simulaciones de resoluciones oficiales."""

CALCULATION_PROMPT = """Esta etapa SOLO dispone de calcular_imv.
El argumento es un ÚNICO objeto llamado case, directamente con los campos
application_date, applicant_id, persons, domicile, economic y, cuando
corresponda, relationships y applicant_history. No crees un nivel IMVCase,
ni pongas los campos de case en la raíz de arguments.
La fecha es YYYY-MM-DD; income es countable_income_annual_eur en euros/año,
y el patrimonio y activos deben introducirse separadamente.
Para persona única comprueba independencia parental.
Todos los indicadores obligatorios deben proceder de la petición.
Si falta alguno, NO invoques calcular_imv; pide los hechos necesarios.
Una validación Pydantic fallida es un error de argumentos, no una
denegación del IMV: corrige el JSON si hay suficientes datos o pregunta.
En una simulación válida, reproduce exactamente el resultado del motor."""

BDNS_PROMPT = """Esta etapa SOLO dispone de consultar_ayudas.
Identifica la temática/tema solicitado y el límite max_resultados (1 a 20).
Consulta la BDNS por HTTP real y usa solo los títulos y entidades
efectivamente devueltos. Si falla la red, informa del fallo.
No presentes convocatorias como prestaciones ya concedidas."""

REPORT_PROMPT = """Esta etapa SOLO dispone de guardar_informe.
Los mensajes anteriores con rol tool contienen las observaciones YA
EJECUTADAS del cálculo IMV/CAPI y de la búsqueda BDNS. NO vuelvas a
calcular ni a consultar. Redacta tú mismo un informe natural en español,
SIN PLANTILLAS; incluye importes IMV, CAPI y TOTAL exactamente como
figuran en calcular_imv, y explica su condición orientativa y los motivos.
Incluye convocatorias BDNS únicamente si aparecen en las observaciones;
si no hubo resultados o la API falló, indícalo. No inventes títulos,
requisitos, fechas de cierre ni compatibilidad.
Luego llama a guardar_informe con {"contenido": "TODO EL TEXTO REDACTADO"}.
Nunca afirmes que se guardó sin una observación de éxito."""

INTEGRATION_PROMPT = """Si el usuario pide cálculo y búsquedas BDNS, puedes llamar
a calcular_imv y consultar_ayudas en la MISMA generación cuando sus
argumentos ya estén disponibles. Si faltan datos del cálculo, pregunta
por ellos; la búsqueda BDNS independiente puede realizarse.
Si además pide archivo, espera las observaciones de cálculo y BDNS
antes de redactar el informe y llamar a guardar_informe.
El informe lo redacta el modelo sin plantillas, usando solamente
observaciones verificadas; nunca inventes cifras ni ayudas.
Confirma el guardado únicamente cuando la herramienta lo confirme."""


def build_system_prompt(registry: ToolRegistry) -> str:
    available = set(registry.tools)
    if available == {"calcular_imv"}:
        guidance = CALCULATION_PROMPT
    elif available == {"consultar_ayudas"}:
        guidance = BDNS_PROMPT
    elif available == {"guardar_informe"}:
        guidance = REPORT_PROMPT
    else:
        guidance = INTEGRATION_PROMPT
    return (
        SYSTEM_PROMPT + "\n\n" + guidance
        + "\n\nDefiniciones disponibles (JSON Schema Pydantic):\n"
        + registry.render_for_prompt()
    )


def _report_requested(query: str) -> bool:
    """Recognize an affirmative request to create a report file, not just discuss one."""
    if re.search(
        r"(?i)\b(?:no|sin)\s+(?:quiero\s+)?(?:un\s+)?"
        r"(?:informe|archivo|fichero|documento|txt)\b", query
    ):
        return False
    if re.search(
        r"(?i)\b(?:no|sin)\s+(?:quiero\s+)?(?:guardar|generar|crear)\b", query
    ):
        return False
    if re.search(
        r"(?i)^\s*¿?\s*(?:como|cómo|que|qué)\s+(?:se\s+)?"
        r"(?:guardar|generar|crear|hacer)\s+(?:un\s+)?"
        r"(?:informe|archivo|fichero|documento|txt)\b", query
    ):
        return False
    action = re.search(
        r"(?i)\b(?:guarda|guardar|guárdame|genera|generar|genérame|"
        r"crea|crear|descarga|descargar|exporta|exportar|escribe|escribir|"
        r"redacta|redactar|prepara|preparar|hazme|quiero|necesito)\b", query
    )
    document = re.search(r"(?i)\b(?:informe|archivo|fichero|txt|documento)\b", query)
    return bool(action and document)


def _api_requested(query: str) -> bool:
    """Recognize a request for grant discovery before report composition."""
    return bool(re.search(
        r"(?i)\b(?:bdns|convocatorias|subvenciones|"
        r"busca(?:r)?\s+ayudas|consulta(?:r)?\s+ayudas)\b", query
    ))


def _money_is_mentioned(content: str, value: float) -> bool:
    """Accept decimal commas/dots, one decimal place and thousands separators."""
    from decimal import Decimal, InvalidOperation

    expected = Decimal(str(value)).quantize(Decimal("0.01"))
    pattern = r"(?<![\w])\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?(?![\w])"
    for match in re.finditer(pattern, content):
        raw = match.group()
        try:
            if "," in raw and "." in raw:
                decimal_separator = "," if raw.rfind(",") > raw.rfind(".") else "."
                thousands = "." if decimal_separator == "," else ","
                raw = raw.replace(thousands, "").replace(decimal_separator, ".")
            elif raw.count(",") > 1 or raw.count(".") > 1:
                sep = "," if "," in raw else "."
                parts = raw.split(sep)
                raw = "".join(parts[:-1]) + ("." + parts[-1] if len(parts[-1]) <= 2 else parts[-1])
            elif "," in raw:
                raw = raw.replace(",", ".")
            elif "." in raw and len(raw.rsplit(".", 1)[1]) == 3:
                raw = raw.replace(".", "")
            number = Decimal(raw).quantize(Decimal("0.01"))
        except InvalidOperation:
            continue
        if number == expected:
            return True
    return False


def _report_matches_calculation(content: str, records: list[ToolCallRecord]) -> bool:
    """Minimum numeric guard, independent of the model's reporting style."""
    successful = [
        record.result for record in records
        if record.ok and record.name == "calcular_imv" and isinstance(record.result, dict)
    ]
    if not successful:
        return False
    latest = successful[-1]
    return all(
        _money_is_mentioned(content, latest[key])
        for key in ("imv_monthly_eur", "capi_monthly_eur", "total_monthly_eur")
    )


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

    def _save_report(self, content: str, records: list[ToolCallRecord], query: str) -> ToolCallRecord:
        """Invoke the file tool only with model-authored, numerically checked prose."""
        if not _report_requested(query):
            return ToolCallRecord(
                name="guardar_informe", arguments={"contenido": content}, ok=False,
                error="User has not explicitly requested writing an IMV report."
            )
        if _api_requested(query) and not any(
            r.name == "consultar_ayudas" for r in records
        ):
            return ToolCallRecord(
                name="guardar_informe", arguments={"contenido": content}, ok=False,
                error="Wait for the BDNS tool observation before writing a combined report."
            )
        if not _report_matches_calculation(content, records):
            return ToolCallRecord(
                name="guardar_informe", arguments={"contenido": content}, ok=False,
                error="Report must contain the IMV, CAPI and total amounts from the engine."
            )
        if any(r.ok and r.name == "guardar_informe" for r in records):
            return ToolCallRecord(
                name="guardar_informe", arguments={}, ok=False,
                error="A report has already been saved for this request."
            )
        return self.registry.call("guardar_informe", {"contenido": content})

    def _finish(self, answer: str, records: list[ToolCallRecord], turns: int,
                report_requested: bool, *, reason: str = "") -> ToolsResponse:
        saved = [
            record for record in records if record.ok and record.name == "guardar_informe"
        ]
        if report_requested:
            if saved:
                # The file path and success are taken ONLY from actual tool execution.
                path = saved[-1].result.get("archivo")
                answer = (
                    f"Informe guardado correctamente en: {path}. "
                    "Los importes proceden del motor determinista."
                )
            else:
                cause = reason or next(
                    (r.error for r in reversed(records)
                     if r.name in {"guardar_informe", "calcular_imv"} and not r.ok),
                    None,
                )
                if not cause:
                    cause = "el modelo no completó la escritura del archivo"
                original = answer.strip()
                answer = f"No se ha guardado ningún informe. Motivo: {cause}."
                # Preserve an informational request for missing facts but never
                # reproduce an unverified assertion that a file was saved.
                if not any(r.ok and r.name == "calcular_imv" for r in records):
                    if original and not re.search(
                        r"(?i)\b(?:he|hemos|se ha|está|ya está)\s+"
                        r"(?:guardado|creado|generado)\b", original
                    ):
                        answer += " " + original
        else:
            # If the user did not authorize saving, the model cannot assert it did.
            if not saved and re.search(
                r"(?i)\b(?:he|hemos|se ha|está)\s+"
                r"(?:guardado|creado|generado)\b.{0,35}\b(?:archivo|fichero|informe)\b",
                answer
            ):
                answer = (
                    "No se ha creado ningún archivo. "
                    "La consulta solo ha producido una respuesta de texto."
                )
        return ToolsResponse(
            answer=answer, tool_calls=records, turns=turns, model=self.model_label
        )

    def run(
        self, query: str, max_turns: int = 5, *,
        prior_observations: list[ToolCallRecord] | None = None,
    ) -> ToolsResponse:
        """Perform normal tool use, optionally continuing from trusted tool observations.

        prior_observations is for trusted orchestrators / offline stage tests only.
        Never expose it as a user-supplied API argument. It contains actual
        completed tool records that the model is meant to have already seen.
        Tool calls from this invocation alone are returned in tool_calls.
        """
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        messages = [
            {"role": "system", "content": build_system_prompt(self.registry)
             if self.use_tools else "Responde en español sin herramientas ni acceso externo."},
            {"role": "user", "content": query},
        ]
        if prior_observations and not self.use_tools:
            raise ValueError("Cannot seed observations without tool access")
        history = list(prior_observations or [])
        # Reconstruct the *same* assistant-call/tool-observation chat sequence
        # produced by run(): never insert golden results as user prose.
        for item in history:
            call = json.dumps(
                {"name": item.name, "arguments": item.arguments},
                ensure_ascii=False, default=str,
            )
            messages.append({"role": "assistant", "content": f"<tool_call>{call}</tool_call>"})
            observation = {"ok": item.ok}
            observation.update(
                {"result": item.result} if item.ok else {"error": item.error}
            )
            messages.append({
                "role": "tool", "name": item.name,
                "content": format_tool_result(item.name, observation),
            })
        records: list[ToolCallRecord] = []
        seen: set[str] = set()
        wants_report = self.use_tools and _report_requested(query)
        reminders = 0

        for turn in range(1, max_turns + 1):
            raw = self.generate(messages)
            parsed = parse_tool_calls(raw)
            if not self.use_tools:
                return self._finish(parsed.text or raw, [], turn, False)

            malformed_block = "<tool_call>" in raw and "</tool_call>" not in raw
            if not parsed.tool_calls and not malformed_block:
                if wants_report and not any(
                    r.ok and r.name == "guardar_informe" for r in records
                ):
                    if not any(r.ok and r.name == "calcular_imv" for r in history):
                        return self._finish(parsed.text, records, turn, True,
                                            reason="falta un cálculo IMV válido para redactarlo")

                    # First request a real tool call: keep generation and tool use separate.
                    if reminders == 0 and turn < max_turns:
                        reminders += 1
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({
                            "role": "system",
                            "content": (
                                "La petición NO está completa: el usuario solicitó guardar "
                                "un informe TXT y guardar_informe todavía NO se ha ejecutado. "
                                "Con los resultados de calcular_imv disponibles, escribe tú "
                                "el informe en prosa española con importes IMV, CAPI y total, "
                                "y haz AHORA una llamada a guardar_informe(contenido=texto). "
                                "NO afirmes que está guardado sin observación."
                            ),
                        })
                        continue

                    # Recovery: the model has composed its own text but omitted the
                    # tool-call markup. The application passes that exact text to
                    # the file tool after deterministic checks; no template is used.
                    if parsed.text:
                        record = self._save_report(parsed.text, history, query)
                        records.append(record)
                        history.append(record)
                        if record.ok:
                            return self._finish("", records, turn, True)
                    return self._finish("", records, turn, True,
                                        reason="el modelo no generó un informe válido para guardar")

                return self._finish(parsed.text or
                                    "No he podido completar la consulta con estos datos.",
                                    records, turn, wants_report)

            messages.append({"role": "assistant", "content": raw})
            if malformed_block and not parsed.tool_calls:
                record = ToolCallRecord(
                    name="tool_parser", arguments={}, ok=False,
                    error="Incomplete tool_call block",
                )
                records.append(record)
                messages.append({
                    "role": "tool", "name": "tool_parser",
                    "content": "Error: bloque <tool_call> incompleto. Reintenta.",
                })
                continue

            # Calls emitted in the same completion are independent. A report cannot
            # use results which the model has not observed in a previous turn.
            calculated_before_turn = any(
                r.ok and r.name == "calcular_imv" for r in history
            )
            for call in parsed.tool_calls:
                if call.error:
                    record = ToolCallRecord(
                        name=call.name or "tool_parser", arguments=call.arguments,
                        ok=False, error=f"Malformed tool call: {call.error}",
                    )
                elif call.name == "guardar_informe":
                    if not calculated_before_turn:
                        record = ToolCallRecord(
                            name=call.name, arguments=call.arguments, ok=False,
                            error="Calculate IMV and observe its result in an earlier turn first."
                        )
                    else:
                        record = self._save_report(
                            str(call.arguments.get("contenido", "")), history, query
                        )
                elif (
                    call.name in self.registry.tools
                    and self.registry.tools[call.name].requires_confirmation
                ):
                    record = ToolCallRecord(
                        name=call.name, arguments=call.arguments, ok=False,
                        error="Tool requires explicit confirmation."
                    )
                else:
                    signature = json.dumps(
                        [call.name, call.arguments], sort_keys=True,
                        default=str, ensure_ascii=False,
                    )
                    if signature in seen:
                        record = ToolCallRecord(
                            name=call.name, arguments=call.arguments, ok=False,
                            error="Repeated identical call.",
                        )
                    else:
                        seen.add(signature)
                        record = self.registry.call(call.name, call.arguments)
                records.append(record)
                history.append(record)
                observation = {"ok": record.ok}
                observation.update(
                    {"result": record.result} if record.ok else {"error": record.error}
                )
                messages.append({
                    "role": "tool", "name": record.name,
                    "content": format_tool_result(record.name, observation),
                })

            if wants_report and any(
                r.ok and r.name == "guardar_informe" for r in records
            ):
                # No extra model turn required to establish file creation.
                return self._finish("", records, turn, True)

        return self._finish(
            "", records, max_turns, wants_report,
            reason="se alcanzó el límite de turnos sin generar el fichero"
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
