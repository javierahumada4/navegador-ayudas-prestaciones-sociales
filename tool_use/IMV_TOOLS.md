# Fase 2 — Herramientas IMV/CAPI (sin GRPO de Tool Use)

Esta rama nace **directamente de `main`** y conserva la infraestructura
docente `registry.py`, `parser.py`, `evaluate.py`, el API `POST /tools`,
los ejemplos de clase y los artefactos de Fase 1. No añade ningún
entrenamiento ni adaptador GRPO nuevo.

## Herramientas disponibles

| Herramienta registrada | Tipo | Función |
|---|---|---|
| `calcular_imv(case)` | Cálculo | Convierte hechos sintéticos estructurados en resultados IMV+CAPI de 2026 mediante el motor determinista de `rlm/imv_engine.py` |
| `explicar_imv(resultado)` | Presentación determinista | Convierte la salida **completa** del cálculo en español natural mediante una plantilla fija; no usa un LLM |

Las dos definiciones JSON Schema se construyen desde Pydantic y sus
argumentos se validan antes de ejecutar. `calcular_imv` utiliza
`IMVCase` (campos imprescindibles obligatorios, condiciones opcionales
neutras para problemas sintéticos). `explicar_imv` usa `IMVCalculation`
(resultado de referencia con validación de coherencia: estado,
importes no negativos y total IMV+CAPI). El texto muestra cuantía total,
desglose, indicadores del motor y sus comprobaciones desfavorables,
acompañados de un aviso de carácter orientativo.

El registro **no** publica las herramientas ilustrativas
`get_weather` y `calculator` del profesor, pero permanecen importables.

## Bucle manual y explicación sin generación de texto

`tool_use/executor.py` conserva la capacidad de llamadas sucesivas,
paralelas, manejo de JSON erróneo y reinyección de observaciones.

Para un cálculo correcto, el flujo concreto es:

1. Qwen3 selecciona `calcular_imv` y produce `arguments.case`.
2. `ToolRegistry` valida y ejecuta el motor IMV.
3. **El ejecutor** pasa ese resultado real a `explicar_imv` y registra
   una segunda llamada, sin pedir al modelo que fabrique argumentos.
4. Devuelve **literalmente** la cadena generada por la plantilla; el
   LLM no vuelve a redactar el resultado. Las dos llamadas quedan en
   `tool_calls`, con latencias, argumentos y observaciones.

En preguntas sin cálculo, el modelo puede responder sin herramientas.
Una llamada inválida vuelve como error/observación para permitir corrección
en otro turno. No hay reparaciones silenciosas de JSON.

La explicación es una *segunda herramienta determinista*, pero la
orquestación automática de `calcular_imv` → `explicar_imv` no sustituye
a la demostración de decisiones **multi-step** del modelo que solicita
el enunciado original. El sistema sigue permitiendo ese flujo general.

## Probar en CPU (sin pesos)

```bash
uv run pytest -q \
  tests/test_imv_tool.py tests/test_imv_templates.py \
  tests/test_imv_tool_loop.py tests/test_tool_registry.py \
  tests/test_tool_parser.py
```

El test de plantillas verifica que el desglose 373,76 € de IMV +
57,50 € de CAPI produce **431,26 € mensuales**, que se explican
correctamente casos solo CAPI y no elegibles, que resultados inconsistentes
se rechazan y que no se hace una segunda generación del modelo.

## Evaluación con Qwen3 en DGX

```bash
git fetch origin
git switch feat/fase2-imv-plantillas
git pull origin feat/fase2-imv-plantillas
uv sync --extra train

# Opcional: reutilizar el LoRA de razonamiento de Fase 1 (no un GRPO de Tool Use).
export ARCA_TOOLS_BRAIN=rlm
export ARCA_TOOLS_ADAPTER="$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora"

uv run --extra train python -m tool_use.evaluate \
  --bench tool_use/bench/imv_smoke_cases.jsonl \
  --out reports/imv_tools_smoke.json --max-turns 3
```

Para probar Qwen3 base en lugar de la Fase 1, usar
`ARCA_TOOLS_BRAIN=base` sin adaptador. El banco heredado de cuatro
casos es solo una **prueba de humo**; no es el banco definitivo de
30 casos requerido por el profesor.

Para servir `POST /tools`, ejecutar `uv run arca-api` en la DGX
(con `--extra train` instalado) y probar desde `/docs`. El contrato
público permanece el de `api/schemas.py`.

## Pendiente para completar el enunciado docente

La Fase 2 pide **tres tipos distintos**: consulta a una API HTTP,
cálculo y una acción con efecto observable. Ahora hay **dos herramientas
de cálculo/presentación**, por lo que **no cubrimos todavía** la
consulta HTTP ni la acción observable, aunque haya dos nombres de tool.

La tercera aún no está decidida; tampoco está construido el banco final
de aproximadamente 30 casos incluyendo casos sin herramienta y
encadenamientos. No se registran herramientas ficticias ni se consulta
el BOE. Antes de cerrar la Fase 2 habrá que acordar con el profesor
si se admite un alcance adaptado o cubrir explícitamente las categorías
y la evaluación exigidas.

No hay `train_tool_grpo.py`, recompensas de GRPO de herramientas,
few-shot de entrenamiento ni checkpoints nuevos en esta rama.
