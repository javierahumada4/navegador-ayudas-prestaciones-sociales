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

## Herramienta implementada: `calcular_imv`

La herramienta de dominio vive en [`imv_tools.py`](imv_tools.py) y está registrada
como `calcular_imv` en `default_registry()` de `tools.py`. Reutiliza sin duplicar
reglas `rlm.imv_engine.evaluate_imv_case` y el fichero de reglas de 2026.

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
**nuevas solicitudes de 2026**, no resoluciones oficiales; los casos de otros
años se rechazan. No se efectúa acceso a la red ni se modifican archivos.

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

**Pendiente en la fase 2:** `consultar_boe`, `preparar_solicitud`, un
benchmark de ~30 casos de dominio y la evaluación GPU real de ambos modelos.
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
