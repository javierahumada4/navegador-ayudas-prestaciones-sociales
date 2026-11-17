# Fase 2 — Herramientas IMV/CAPI (sin GRPO de Tool Use)

Esta rama nace **directamente de `main`** y conserva la infraestructura
docente `registry.py`, `parser.py`, `evaluate.py`, el API `POST /tools`,
los ejemplos de clase y los artefactos de Fase 1. No añade ningún
entrenamiento ni adaptador GRPO nuevo.

## Herramientas disponibles

| Herramienta registrada | Tipo | Función |
|---|---|---|
| `calcular_imv(case)` | Cálculo | Convierte hechos sintéticos estructurados en resultados IMV+CAPI de 2026 mediante el motor determinista de `rlm/imv_engine.py` |
| `explicar_imv(resultado)` | Acción con efecto observable | Aplica la misma plantilla fija de español y **guarda un informe Markdown real**. Requiere confirmación explícita; no usa un LLM |

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
3. **Sin autorización para guardar**, el ejecutor aplica
   `render_imv_result`, devuelve la misma plantilla literal y **no crea
   ningún fichero**. La traza contiene solo `calcular_imv`.
4. **Con `confirmar_guardado: true` en la petición HTTP**, el ejecutor
   entrega la salida real del motor a `explicar_imv` sin intervención del
   LLM. La segunda tool guarda un Markdown y devuelve `texto`,
   `archivo` y `sha256`. Ambas llamadas quedan registradas en
   `tool_calls`, con argumentos, observaciones y latencias.

`explicar_imv` está declarada `requires_confirmation=True`; un
`<tool_call>` producido por el LLM que la invoque directamente **se
rechaza incluso con confirmación**. La única ruta admitida en
`ToolLoop` es un cálculo ejecutado de verdad y la confirmación que
aporta el consumidor de `POST /tools`, no palabras o parámetros del
modelo. Sin confirmación explícita, no existe efecto externo.

Los informes se crean en `reports/imv/` (o en
`ARCA_IMV_REPORTS_DIR`, variable de entorno del servidor), con nombres
aleatorios, apertura exclusiva para no sobrescribir y permisos privados
cuando el sistema operativo lo permite. El cliente no puede elegir rutas
ni nombres a través de argumentos de la herramienta. Se excluye
`reports/imv/` de Git para no subir informes privados.

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
se rechazan, que se guarda el Markdown solo con confirmación y que
no se hace una segunda generación del modelo.

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

## Guardar un informe real por HTTP

En la API, el campo opcional `confirmar_guardado` es `false` por
defecto y **solo la petición de usuario**, no el modelo, puede
activarlo. Por ejemplo, con el servidor arrancado:

```bash
curl -s http://localhost:8000/tools \
  -H 'Content-Type: application/json' \
  -d '{"query":"Calcula el IMV de este caso [incluye aquí los hechos completos]","confirmar_guardado":true}'
```

La respuesta incluye la ruta del informe en `answer` y en
`tool_calls[1].result.archivo` cuando ambas herramientas se ejecutan
correctamente. Los ejemplos y los tests CPU verifican que se guarda el
fichero con el desglose y aviso de simulación, que es nuevo en cada
ejecución, que los datos inconsistentes no llegan al disco y que la
petición sin confirmación es de **solo lectura**.

La confirmación es una autorización de quien **llama al endpoint**;
si se expone públicamente `/tools`, habrá que añadir autenticación y
autorización por usuario antes de permitir guardados. No publicar los
ficheros de informes como estáticos ni confiar en frases de aprobación
generadas por el modelo.

## Pendiente para completar el enunciado docente

La Fase 2 pide **tres tipos distintos**: consulta a una API HTTP,
cálculo y una acción con efecto observable. **Ahora cubrimos dos de esas
categorías**: `calcular_imv` calcula y `explicar_imv` guarda un
documento persistente, previa autorización. **Falta una herramienta
que consulte una API HTTP externa real**, distinta de la API del BOE
si así lo acordamos.

La tercera aún no está decidida; tampoco está construido el banco final
de aproximadamente 30 casos incluyendo casos sin herramienta y
encadenamientos. No se registran herramientas ficticias ni se consulta
el BOE. Antes de cerrar la Fase 2 habrá que acordar con el profesor
si se admite un alcance adaptado o cubrir explícitamente las categorías
y la evaluación exigidas.

No hay `train_tool_grpo.py`, recompensas de GRPO de herramientas,
few-shot de entrenamiento ni checkpoints nuevos en esta rama.
