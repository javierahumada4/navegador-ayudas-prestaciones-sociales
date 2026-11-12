# Fase 2 — Acción: el modelo habla con el mundo

**Endpoint que se evalúa:** `POST /tools` · **Clase:** The augmented LLM, bloque de tool use

## De qué va

Un modelo de lenguaje consume texto y produce texto. No puede consultar el tiempo, ni
ejecutar código, ni escribir un fichero. Tool use es la forma de dárselo: la aplicación
describe al modelo qué herramientas existen, el modelo genera una llamada en un formato
reconocible, la aplicación la detecta, la ejecuta y le devuelve el resultado, y el modelo
sigue. Cinco pasos, que en clase dibujamos con `get_weather` y que aquí vais a implementar a
mano, sin frameworks, con herramientas de vuestro dominio.

## Lo que tendréis que hacer

**1. Tres herramientas reales.** En `tool_use/tools.py`. Una que consulte algo fuera del
modelo (una API HTTP de verdad, no simulada), una que calcule o ejecute (un intérprete
acotado, una función numérica, una consulta a una base de datos), y una que actúe con efecto
observable (escribir un fichero, crear un evento de calendario, enviar algo a un webhook de
pruebas). `get_weather` y `calculator` os las doy como ejemplo de cómo se escribe una: la
docstring es la descripción que lee el modelo y los tipos son el esquema. Las que tienen
efectos se registran con `requires_confirmation=True`; en la fase 4 decidiréis qué hace el
agente con eso.

**2. Definiciones precisas y argumentos validados.** El registro (`registry.py`) convierte
la función en JSON Schema y valida con Pydantic lo que el modelo genera antes de ejecutar
nada. Si la validación falla, el error vuelve al modelo como observación para que se corrija.
Esto ya está hecho; lo que os pido es que lo uséis bien y que vuestras descripciones sean lo
bastante precisas para que el modelo acierte los argumentos. Es más difícil de lo que parece.

**3. El flujo de cinco pasos.** En `tool_use/executor.py`: el prompt de sistema con las
definiciones (la plantilla es la de las diapositivas), la generación, el parser de
`<tool_call>` (os lo doy, es infraestructura), la ejecución, la reinyección del resultado
como mensaje con rol `tool`, y la respuesta final. Tiene que soportar varias llamadas
encadenadas en distintos turnos (multi-step) y varias llamadas en una misma generación
(paralelas). Sin `smolagents` ni nada parecido en esta fase: quiero ver el bucle.

**4. El banco de pruebas.** Unos treinta casos en `tool_use/bench/cases.jsonl` (formato en
`example_cases.jsonl`): para cada petición, la herramienta esperada y sus argumentos.
Incluid casos que **no** necesitan herramienta: un modelo que llama a una herramienta para
todo no las está usando bien. `tool_use/evaluate.py` calcula la precisión de selección y la
de argumentos. Y comparad el mismo modelo con y sin herramientas, como en el notebook de
clase.

## Flujo de rulesets y BOE (herramientas de dominio)

La Fase 2 dispone de **cinco tools** en `default_registry()`:

| Tool | Tipo | Funcion |
|---|---|---|
| `fecha_actual()` | lectura | Fecha actual en Europe/Madrid, solo si no se proporciona fecha de solicitud |
| `comprobar_ruleset(fecha_solicitud)` | lectura local | Inspecciona `rlm/rulesets/imv_*.json` y sus intervalos `effective_from/effective_to` |
| `consultar_boe(boe_id, seccion)` | red HTTP | Consulta **de verdad** la API oficial de legislación consolidada del BOE |
| `crear_ruleset(ruleset, confirmacion_humana)` | escritura local | Publica fichero JSON completo **solo con revisión y aprobación humana** |
| `calcular_imv(case)` | calculo | Selecciona el ruleset según `case.application_date` y ejecuta el motor |

**Orden recomendado:** fecha de solicitud conocida (o `fecha_actual` si de verdad
se pregunta por hoy) → `comprobar_ruleset` → si hay un único fichero aplicable,
`calcular_imv` con datos completos; si no lo hay, `consultar_boe` y
**parar hasta disponer de una propuesta completa y revisada**. El simple
intervalo del JSON no demuestra que no haya reformas legales posteriores.

El endpoint del BOE es
`https://www.boe.es/datosabiertos/api/legislacion-consolidada/id/BOE-A-2021-21007/metadatos`
y ofrece también `analisis` y `texto`. Se usa `httpx` con timeout y
un `boe_id` validado; no se permite una URL arbitraria.
El texto consolidado tiene carácter **informativo** y puede contener varias
versiones según fecha. La API **no entrega automáticamente** la estructura
numérica completa (`amounts`, `patrimony`, `capi`, etc.) del motor.
Las cuantías anuales también requieren contrastarse con la información
oficial de Seguridad Social. No reutilizar valores del año anterior.

`crear_ruleset` comprueba estructura, fechas, secciones numéricas y fuentes
oficiales, impide solapamientos y nunca sobrescribe archivos. Está registrada
con `requires_confirmation=True`: el bucle de la Fase 2 **la bloquea**.
La publicación queda preparada para una confirmación supervisada en la Fase 4;
no es una generación desatendida de normas legales a partir de texto libre.
El único ruleset de producción actualmente presente sigue siendo
`rlm/rulesets/imv_2026.json`.

El motor `evaluate_imv_case` selecciona automáticamente un fichero que cubra
la **fecha de solicitud**, nunca usa 2026 como fallback de 2027, y rechaza
intervalos ambiguos o catálogos corruptos. `load_ruleset()` sin fecha
mantiene el comportamiento de la Fase 1 para generación de datasets.

**Prueba sin modelo ni red** (la consulta al BOE está simulada en tests):

```bash
uv run pytest -q tests/test_ruleset_tools.py tests/test_calcular_imv_tool.py
uv run python - <<'PY'
from tool_use.tools import default_registry
tools = default_registry()
print(tools.call("fecha_actual", {}).result)
print(tools.call("comprobar_ruleset", {"fecha_solicitud": "2026-11-12"}).result)
PY
```

**Consulta HTTP en vivo** (requiere conectividad desde la DGX):

```bash
uv run python - <<'PY'
from tool_use.tools import default_registry
result = default_registry().call("consultar_boe", {"boe_id": "BOE-A-2021-21007"})
print(result.result if result.ok else result.error)
PY
```

Para el benchmark de base frente a RLM, evaluar esta **nueva secuencia**
por separado: el smoke antiguo esperaba `calcular_imv` como primera llamada;
ahora el flujo de consulta puede empezar con `comprobar_ruleset`.

## Herramienta implementada: `calcular_imv`

La herramienta de dominio vive en [`imv_tools.py`](imv_tools.py) y está registrada
como `calcular_imv` en `default_registry()` de `tools.py`. Reutiliza sin duplicar
reglas `rlm.imv_engine.evaluate_imv_case` y el catálogo de rulesets por fecha.

**Contrato:** argumento `case` con un caso IMV normalizado. El esquema Pydantic
`IMVCase` exige identificación y fechas de todas las personas, situación de
residencia, parentescos y custodia, convivencia, historial y datos económicos.
Los datos que afectan a la cuantía (ingresos, patrimonio, prestaciones, etc.)
**no tienen valores por defecto inventados**. Una llamada sin algún dato
obligatorio devuelve un error de validación a través del registro, no un cálculo
sobre supuestos. El LLM debe solicitar esos datos antes de llamar a la herramienta.

La salida contiene `status`, `imv_eligible`, `imv_monthly_eur`,
`capi_eligible`, `capi_monthly_eur`, `total_monthly_eur`,
`failed_requirements` y más información generada por el motor. Su alcance son
**solicitudes cubiertas por un ruleset revisado**, no resoluciones oficiales;
las fechas sin reglas se rechazan. No se efectúa acceso a la red ni se modifican archivos.

Prueba local sin GPU ni modelo de lenguaje:

```bash
uv run pytest -q tests/test_calcular_imv_tool.py tests/test_tool_registry.py
```

**Implementado en `executor.py`:** bucle de herramientas con prompt, llamadas
nativas Qwen3 `<tool_call>`, Pydantic, observaciones `tool`, varias llamadas
por respuesta, llamadas encadenadas y control del límite `max_turns`.
La aplicación intercepta llamadas con `requires_confirmation=True` sin ejecutar
sus efectos secundarios (el contrato de fase 2 no contempla confirmaciones
interactivas). Las llamadas mal formadas y los errores vuelven al modelo.

El tokenizer de Qwen3 recibe `tools=registry.definitions()` directamente en
`apply_chat_template`; por tanto, **la plantilla nativa**, y no una concatenación
manual, inserta nombres, descripciones y JSON Schema. Se usa generación greedy
en ambos cerebros. Por defecto `enable_thinking=False`; puede activarse para
comparar el efecto en los experimentos.

**Smoke test inicial (4 casos, NO sustituye al benchmark de 30):**

El fichero `tool_use/bench/imv_smoke_cases.jsonl` contiene dos casos IMV
normalizados completos (incluido CAPI sin IMV), una consulta con datos
insuficientes y una pregunta conceptual sin herramienta. El benchmark completo
de la fase debe incluir también peticiones en lenguaje natural y herramientas
encadenadas.

```bash
uv run --extra train python -m tool_use.evaluate \
  --brain base --bench tool_use/bench/imv_smoke_cases.jsonl \
  --out reports/imv_smoke_base.json
uv run --extra train python -m tool_use.evaluate \
  --brain rlm --adapter /ruta/al/adaptador/grpo_lora \
  --bench tool_use/bench/imv_smoke_cases.jsonl \
  --out reports/imv_smoke_rlm.json
```

**Diagnóstico de llamadas mal formadas:** el evaluador guarda ahora
`raw_generations` (texto completo que produjo el modelo en cada turno),
`tool_execution_ok` y `answer_check_ok` en cada fila. El resumen añade
`tool_execution_accuracy` (sobre casos que necesitan herramienta) y
`answer_check_accuracy` (sólo sobre casos con `expected_answer_contains`).
Esto permite distinguir entre *selección de herramienta*, *JSON válido*,
*ejecución real* y *respuesta factual*. Los primeros resultados del smoke test
mostraron llamadas inválidas por JSON `Extra data`; no se puede determinar
si falta longitud de generación o si se ha emitido texto extra sin revisar
los nuevos traces. Los JSON de evaluación pueden contener datos personales,
por lo que no deben publicarse sin revisarlos.

Para probar si la longitud de salida influye, repetir los smoke tests con
`--max-new-tokens 4096` en **ambos** cerebros, manteniendo el resto idéntico.

**Comparación controlada base + tools frente a RLM + tools:** usar exactamente
el mismo modelo Qwen3-1.7B, prompt, herramientas y banco de pruebas. Sólo cambian
los pesos del adaptador. Primero instalar dependencias y seleccionar la rama.

```bash
uv sync --extra train
# Con un banco IMV/CAPI creado en tool_use/bench/cases.jsonl:
ARCA_TOOLS_BRAIN=base uv run --extra train python -m tool_use.evaluate \
  --brain base --bench tool_use/bench/cases.jsonl \
  --out reports/phase2_base.json
ARCA_TOOLS_BRAIN=rlm ARCA_TOOLS_ADAPTER=runs/phase1-rebuild/weights/grpo_lora \
  uv run --extra train python -m tool_use.evaluate \
  --brain rlm --bench tool_use/bench/cases.jsonl \
  --out reports/phase2_rlm.json
```

Puede indicarse el adaptador en Hugging Face Hub mediante
`ARCA_TOOLS_ADAPTER` o `ARCA_RLM_ADAPTER`. El modelo base predeterminado
es `Qwen/Qwen3-1.7B`; si el adaptador fue entrenado con otro, el ejecutor
rechaza la combinación. Ambas ejecuciones tienen que usar las mismas preguntas.
Los pesos se cargan de manera diferida al recibir la primera petición; `/health`
sólo verifica que el bucle está configurado.

```bash
uv run --extra train arca-api
# En otra terminal:
curl -X POST http://localhost:8000/tools \
  -H 'Content-Type: application/json' \
  -d '{"query":"¿Qué datos necesitas para calcularme el IMV?", "max_turns":3}'
```

Para verificar la lógica sin descargar modelos:

```bash
uv run pytest -q tests/test_tool_executor.py tests/test_calcular_imv_tool.py
```

**Pendiente en la fase 2:** benchmark de ~30 casos de dominio, evaluación GPU real
con las nuevas herramientas y flujo de aprobación de efectos secundarios.
La creación de rulesets está implementada, pero requiere revisión humana.
El bucle no gestiona historial entre peticiones HTTP: preguntar y recopilar
datos del usuario a lo largo de varias peticiones corresponde a la fase 4.
Detectar los datos ausentes depende todavía de la calidad del modelo; el
schema evita la omisión de campos, no que el modelo invente un dato plausible.

## Para ir más allá

- La "opción 2" de las diapositivas: **SFT para tool use**. Generad un dataset sintético de
  llamadas a vuestras herramientas, ajustad el modelo de la fase 1 con él y medid la mejora
  en el banco de pruebas.
- Una herramienta **con estado** (SQLite) y tareas donde las acciones tienen consecuencias
  sobre llamadas posteriores.
- Una **política de seguridad**: qué herramientas piden confirmación, cómo se pide, qué pasa
  si el modelo insiste.

## Cómo se corrige

Llamaré a `POST /tools` con peticiones parecidas a las de vuestro banco de pruebas y alguna
que requiera dos herramientas encadenadas, y miraré:

- **Implementación (30 %):** el modelo elige la herramienta correcta con los argumentos
  correctos, el resultado real aparece en `tool_calls` con su latencia, y la respuesta final
  lo usa. Una petición que no necesita herramienta se responde sin llamar a ninguna.
- **Completitud (30 %):** tres herramientas de los tres tipos, definiciones desde Pydantic,
  flujo de cinco pasos con multi-step y paralelo, manejo de errores de validación, banco de
  treinta casos con métricas, comparación con/sin herramientas.
- **Interpretación (30 %):** por qué esas herramientas y no otras; qué descripciones
  funcionaron y cuáles no; en qué casos falla el modelo y por qué; qué cambió al pasar de
  single-step a multi-step.
- **Limpieza (10 %):** herramientas con docstrings claras, sin claves en el código, tests
  para las herramientas que se puedan probar sin red.

## Ficheros

| Fichero | Estado | Qué es |
|---|---|---|
| `registry.py` | hecho | `@tool`, definiciones JSON Schema, validación y ejecución con captura de errores |
| `parser.py` | hecho | Detecta `<think>` y uno o varios `<tool_call>` en la salida del modelo |
| `tools.py` | ejemplo + tu turno | `get_weather` y `calculator`; las vuestras van aquí |
| `executor.py` | tu turno | El flujo de cinco pasos; detrás de `/tools` |
| `evaluate.py` | hecho | Métricas de selección y argumentos sobre el banco de pruebas |
| `bench/` | vuestro | Casos de prueba (`example_cases.jsonl` muestra el formato) |
