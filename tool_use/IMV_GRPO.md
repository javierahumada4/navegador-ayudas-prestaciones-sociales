# IMV/CAPI: comparación de cinco sistemas y GRPO para `calcular_imv`

Esta extensión de la Fase 2 vive **sobre la plantilla original de ARCA**.
No modifica las reglas ni el motor `rlm/imv_engine.py`, ni las etapas
SFT/GRPO de razonamiento. La única herramienta publicada por
`default_registry()` es `calcular_imv`; los ejemplos `get_weather` y
`calculator` de la plantilla siguen existiendo, pero **no se registran**.

## Experimento

| Nombre en CLI | Checkpoint | Tool |
|---|---|---|
| `base` | Qwen3-1.7B | No |
| `sft` | SFT de Fase 1 | No |
| `sft_grpo` | SFT + GRPO de razonamiento IMV | No |
| `sft_grpo_tool` | SFT + GRPO de razonamiento IMV | `calcular_imv` |
| `sft_grpo_tool_grpo` | SFT + GRPO de razonamiento + GRPO específico de Tool Use | `calcular_imv` |

Los cinco reciben **los mismos enunciados de test** de Fase 1. Se puntúa
el **importe total mensual IMV+CAPI** con el verificador decimal del
repositorio: coincidencia exacta tras `ROUND_HALF_UP` a céntimos y
**una sola cantidad final no ambigua**. Se informa por separado de las
llamadas correctas y fallidas a la herramienta, formatos, duración y
tokens. Las variantes con herramientas reciben adicionalmente el
JSON Schema y pueden usar varios turnos; su coste computacional no es
idéntico al de las variantes sin herramientas.

No se comprueba exhaustivamente información faltante de una solicitud
real: estos son problemas sintéticos normalizados, y las condiciones
opcionales ausentes usan valores neutros. Los campos imprescindibles
del motor siguen siendo obligatorios y Pydantic comprueba consistencia.

**Datos:** `results/rlm_final/data/train.jsonl` (solo entrenar) y
`results/rlm_final/data/test.jsonl` (solo evaluar). Se comprueba
solapamiento exacto de preguntas. Congelar esos archivos y no elegir
hiperparámetros mirando el test final: los smoke tests deberían usar un
conjunto de validación separado si se ajusta la política de entrenamiento.

## Preparación en la DGX

```bash
git fetch origin
git switch feat/imv-tool-grpo-comparativa-5
git pull origin feat/imv-tool-grpo-comparativa-5
uv sync --extra train

cp tool_use/bench/imv_five_models.example.json "$HOME/imv-five-models.json"
ls "$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/sft_lora/adapter_config.json"
ls "$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora/adapter_config.json"
```

Las rutas de SFT y SFT+GRPO provienen del manifiesto de Fase 1, pero
su existencia física debe comprobarse en la DGX. Si la ejecución previa
guardó los adaptadores en otro directorio, editar la copia local JSON.

### Comprobar disponibilidad sin GPU

```bash
uv run python -m tool_use.compare_imv --all --dry-run \
  --config "$HOME/imv-five-models.json"
```

El quinto checkpoint figura como `missing_checkpoint` hasta que
termine el nuevo entrenamiento. **Nunca se sustituye por SFT+GRPO
sin avisar ni se inventan resultados.**

### Comprobar la integración de entrenamiento (sin GPU)

```bash
uv run --extra train python -m tool_use.train_tool_grpo \
  --init-adapter "$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora" \
  --output "$HOME/clusters/dgx/arca-runs/imv-tool-grpo-smoke/weights/grpo_lora" \
  --preflight
```

`--preflight` comprueba que la versión instalada de TRL dispone de
`tools=` y del GRPO de rollouts con herramientas. Tanto en entrenamiento
como en inferencia Qwen3 recibe **exactamente el mismo JSON Schema de
`calcular_imv`**, derivado del registro Pydantic
(`default_registry().definitions()[0]`), incluidos los campos anidados
de `IMVCase`. En `tool_use/trl_tool_schema.py` se sustituye *solo* la
representación del esquema al renderizar el chat de TRL: la función
ejecutable `calcular_imv(case: dict)` se mantiene para que TRL pueda
invocarla, y delega en el mismo registro Pydantic y motor de producción.
No se parchea globalmente Transformers ni se alteran las reglas IMV.
Los tests CPU verifican la igualdad exacta de esquemas; la
compatibilidad de generación y aprendizaje real debe comprobarse en DGX.

### Diagnóstico de recompensa cero antes de entrenar

El adaptador SFT+GRPO de razonamiento puede preferir generar directamente
`<think>/<answer>` en vez de llamar a herramientas. Si
`tools/call_frequency=0` y todas las recompensas son cero durante varios
pasos, GRPO **no tiene gradiente útil** y se detiene deliberadamente.

Para favorecer la exploración sin añadir una nueva etapa de SFT,
las dos variantes con herramienta incluyen ahora **la misma demostración
sintética fija** en el contexto: una llamada nativa completa a
`calcular_imv`, una observación del motor y una respuesta final.
La demostración está en `tool_use/tool_prompt.py`, es distinta del test
y se valida mediante el motor IMV. No es una respuesta-oráculo del
problema actual. Se registra esta intervención al interpretar la
comparación: el efecto del acceso a herramientas incluye también esta
demostración, idéntica para los brazos 4 y 5.

Antes de entrenar de nuevo, ejecutar una **sonda de inferencia** con
cuatro preguntas de train (ningún dato del test ni pesos modificados):

```bash
uv run --extra train python -m tool_use.train_tool_grpo \
  --init-adapter "$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora" \
  --train-data "$HOME/clusters/dgx/arca-runs/rlm-final-v3/data/train.jsonl" \
  --test-data "$HOME/clusters/dgx/arca-runs/rlm-final-v3/data/test.jsonl" \
  --output "$HOME/clusters/dgx/arca-runs/imv-tool-grpo-probe" \
  --max-completion-length 2048 --probe 4
```

`--probe` imprime las generaciones crudas y los registros de tool por caso.
Si sigue sin haber ninguna llamada, **no iniciar 300 pasos**: hay que
inspeccionar los textos generados y posiblemente introducir una
inicialización supervisada de Tool Use previa a GRPO. No aumentar el
límite de pasos ni anular `zero-signal-patience` para ocultarlo.

### Smoke GPU de dos pasos

```bash
uv run --extra train python -m tool_use.train_tool_grpo \
  --init-adapter "$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora" \
  --output "$HOME/clusters/dgx/arca-runs/imv-tool-grpo-smoke/weights/grpo_lora" \
  --steps 2 --n-examples 24 --num-generations 4 \
  --micro-batch-size 1 --max-completion-length 3072 --seed 42
```

Inspeccionar el historial de recompensas, errores de llamadas, memoria
GPU y que los grupos tengan varianza no nula. **Los tests CPU no ejecutan
ninguna actualización real de pesos**.

### GRPO de Tool Use completo

```bash
uv run --extra train python -m tool_use.train_tool_grpo \
  --init-adapter "$HOME/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora" \
  --output "$HOME/clusters/dgx/arca-runs/imv-tool-grpo-final/weights/grpo_lora" \
  --steps 300 --num-generations 4 --micro-batch-size 1 \
  --max-completion-length 3072 --save-steps 50 --seed 42
```

El entrenamiento continúa el **adaptador final SFT + GRPO de
razonamiento**, con `PeftModel(..., is_trainable=True)`. Genera un
**adaptador PEFT acumulativo** en un directorio nuevo. No sobreescribe
los pesos anteriores. Las recompensas se calculan solo en train:
llamada válida (0.25), argumentos correctos (1), ejecución correcta
(1), importe final correcto tras uso real de la tool (2) y formato (0.25).
No hay reparaciones automáticas del JSON.

### Registrar el quinto checkpoint

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path.home() / "imv-five-models.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
adapter = (
    Path.home() /
    "clusters/dgx/arca-runs/imv-tool-grpo-final/weights/grpo_lora"
)
if not (adapter / "adapter_config.json").is_file():
    raise FileNotFoundError(adapter)
cfg["adapters"]["tool_grpo"] = str(adapter)
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
PY
```

### Comparativa completa de cinco modelos

```bash
uv run --extra train python -m tool_use.compare_imv \
  --all --config "$HOME/imv-five-models.json" \
  --data results/rlm_final/data/test.jsonl \
  --train-data results/rlm_final/data/train.jsonl \
  --max-new-tokens 2048 --max-turns 3 --seed 42 \
  --out-dir reports/imv_five_models_final
```

Antes del test final puede lanzarse una prueba rápida con
`--n-examples 8`, pero si esos mismos casos se utilizan para escoger
hiperparámetros se debe separar un conjunto de validación adecuado.

**Ficheros de resultados:**
`comparison.csv` (cinco filas);
`manifest.json` (rutas, hashes, estados);
`<variante>.json` (predicciones, referencia, llamadas, errores y trazas).
Para evaluar una sola variante, usar `--variant nombre`; no combinar
`--all` con `--variant`.

## Pruebas CPU

```bash
uv run pytest -q tests/test_imv_tool.py tests/test_imv_tool_grpo.py \
  tests/test_imv_tool_loop.py tests/test_imv_compare.py
```

**Límite de alcance:** una sola herramienta de cálculo es una
adaptación del proyecto; el enunciado original de Fase 2 pide más
herramientas. `tool_use/README.md` mantiene el enunciado sin cambios.
