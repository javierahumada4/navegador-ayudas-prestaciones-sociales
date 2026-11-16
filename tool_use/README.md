# Experimento focalizado: GRPO para usar `calcular_imv`

## Objetivo

Partimos de **Qwen3-1.7B → SFT → GRPO de razonamiento del IMV**, cuyo
adaptador final figura en `results/rlm_final/manifest.json` y se guarda en la
DGX en:

```text
$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora
```

Sobre **ese único checkpoint** entrenamos un GRPO nuevo para aprender a usar
`calcular_imv`. No entrenamos otras ramas (base, SFT aislado, GRPO sin SFT)
ni mantenemos la matriz antigua de 12 variantes.

**La única herramienta de producción** sigue siendo `calcular_imv(case)`,
que delega en `rlm.imv_engine.evaluate_imv_case` para obtener la cuantía
IMV+CAPI del caso sintético de 2026. No se incorporan BOE, rulesets
dinámicos, efectos secundarios ni reparaciones de JSON. Tampoco hay un
flujo de entrevista sobre datos ausentes: queda fuera de este experimento.

## Código necesario

| Archivo | Función |
|---|---|
| `train_tool_grpo.py` | Continuar el LoRA de SFT+GRPO con `trl.GRPOTrainer` y rollouts de herramientas |
| `tool_grpo_rewards.py` | Recompensar llamadas válidas, argumentos, ejecución y cifra final |
| `imv_tools.py` | Tool y esquema Pydantic del caso |
| `registry.py`, `tools.py` | Publicar y validar la única herramienta |
| `executor.py`, `parser.py` | Inferencia Qwen3 con llamadas nativas, resultados y trazas |
| `evaluate_shared.py` | Evaluar el modelo antes/después con el mismo test de Fase 1 |

Se conservan los archivos `rlm/`, los tests existentes y el endpoint
`POST /tools`: todos se reutilizan. No se modifican pesos en Git.

## Datos y recompensas

El entrenamiento consume **únicamente**
`results/rlm_final/data/train.jsonl`. Los enunciados y sus
`rule_context` llegan al modelo; `answer` y `params` se reservan para
puntuar el rollout, no para construir el prompt. Se verifica que las
preguntas de train y test no coincidan exactamente.

TRL invoca la herramienta de entrenamiento `calcular_imv(case: dict)`.
Su firma ligera permite generar el JSON Schema en Transformers, pero **su
implementación llama al registro Pydantic real**, sin duplicar el cálculo.

Las cinco recompensas, con pesos por defecto, son:

| Recompensa | Peso |
|---|---:|
| Tool seleccionada y llamada bien formada | 0,25 |
| Argumentos normalizados iguales al caso de train | 1,00 |
| Ejecución con argumentos correctos y observación de herramienta | 1,00 |
| Respuesta final correcta tras ejecutar la herramienta | 2,00 |
| Respuesta con un importe de dos decimales | 0,25 |

El GRPO de razonamiento previo **no era** específico de herramientas. El
nuevo entrenamiento continúa sus pesos, no crea un modelo desde cero.
Se entrenan los mismos parámetros LoRA con `PeftModel(..., is_trainable=True)`.
El directorio de salida contendrá un único adaptador final acumulativo.

## Pasos en la DGX

Todos los comandos se ejecutan en la raíz del repositorio.

### 1. Actualizar rama y dependencias

```bash
git fetch origin
git switch feat/evaluacion-12-variantes-imv
git pull origin feat/evaluacion-12-variantes-imv
uv sync --extra train
```

### 2. Comprobar el adaptador original y los datos sin entrenar

```bash
ls "$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora/adapter_config.json"

uv run --extra train python -m tool_use.train_tool_grpo \
  --output "$HOME/clusters/dgx/arca-runs/tool-grpo-smoke/weights/grpo_lora" \
  --dry-run
```

### 3. Comprobar compatibilidad TRL sin cargar el modelo

```bash
uv run --extra train python -m tool_use.train_tool_grpo \
  --output "$HOME/clusters/dgx/arca-runs/tool-grpo-smoke/weights/grpo_lora" \
  --preflight
```

Este paso revisa la firma de las APIs `TRL`/`Transformers`, pero **no
equivale a ejecutar una actualización GPU**.

### 4. Entrenar dos pasos de prueba en GPU

```bash
uv run --extra train python -m tool_use.train_tool_grpo \
  --output "$HOME/clusters/dgx/arca-runs/tool-grpo-smoke/weights/grpo_lora" \
  --steps 2 \
  --n-examples 24 \
  --num-generations 4 \
  --micro-batch-size 1 \
  --max-completion-length 3072 \
  --seed 42
```

Revisar `log_history.json`, que existan llamadas reales y que las
recompensas no estén colapsadas a cero. Ante incompatibilidades, falta de
memoria o señal nula, corregir antes de entrenar cientos de pasos.

### 5. Entrenar el adaptador final

```bash
uv run --extra train python -m tool_use.train_tool_grpo \
  --output "$HOME/clusters/dgx/arca-runs/tool-grpo-sft-grpo/weights/grpo_lora" \
  --steps 300 \
  --num-generations 4 \
  --micro-batch-size 1 \
  --max-completion-length 3072 \
  --save-steps 50 \
  --seed 42
```

Los resultados se guardan en `--output`: adaptador PEFT final,
`tool_grpo_run.json` con hashes y configuración, `log_history.json`
y checkpoints intermedios. **El entrenamiento GPU no se ha ejecutado
desde este repositorio por el asistente.**

### 6. Evaluar antes y después con el mismo test congelado

```bash
uv run --extra train python -m tool_use.evaluate_shared \
  --data results/rlm_final/data/test.jsonl \
  --adapter "$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora" \
  --max-new-tokens 4096 --max-turns 3 \
  --out reports/imv_before_tool_grpo.json

uv run --extra train python -m tool_use.evaluate_shared \
  --data results/rlm_final/data/test.jsonl \
  --adapter "$HOME/clusters/dgx/arca-runs/tool-grpo-sft-grpo/weights/grpo_lora" \
  --max-new-tokens 4096 --max-turns 3 \
  --out reports/imv_after_tool_grpo.json
```

La métrica principal es `numeric_accuracy`; también se guardan
`tool_execution_rate`, `correct_with_tool_rate`, JSON sin reparar y
`raw_generations` por problema. Para una prueba breve, añadir
`--n-examples 8` y elegir otros nombres de salida.

## Pruebas CPU (sin modelos)

```bash
uv run pytest -q tests/test_tool_grpo.py tests/test_calcular_imv_tool.py \
  tests/test_tool_evaluate_shared.py tests/test_tool_executor.py \
  tests/test_tool_parser.py tests/test_tool_registry.py
```

**Nota sobre el alcance docente:** centrarnos en una única tool es una
adaptación del proyecto; el enunciado original de la Fase 2 planteaba tres
tipos de herramientas. El experimento aquí descrito **no cubre** esa
rúbrica completa.
