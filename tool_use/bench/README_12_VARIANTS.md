# Evaluación IMV común: 12 variantes

Esta rama prepara la **infraestructura de evaluación**, no entrena nuevos
pesos ni ejecuta inferencia GPU en GitHub Actions. La herramienta disponible
continúa siendo exclusivamente `calcular_imv`.

## Qué se compara

Es un diseño **4 x 3**:

| Checkpoint de razonamiento | Sin tool | Con tool | Con GRPO de tool use + tool |
|---|---|---|---|
| Qwen3 base | `base` | `base_tool` | `base_tool_grpo` |
| Qwen3 + SFT | `sft` | `sft_tool` | `sft_tool_grpo` |
| Qwen3 + GRPO sin SFT | `grpo_zero` | `grpo_zero_tool` | `grpo_zero_tool_grpo` |
| Qwen3 + SFT + GRPO numérico | `sft_grpo` | `sft_grpo_tool` | `sft_grpo_tool_grpo` |

Los checkpoints con `*_tool_grpo` requieren un GRPO adicional, orientado al
uso de `calcular_imv`. **No equivale** al GRPO numérico de Fase 1.
Las cuatro ramas requieren finales entrenados individualmente y sus adaptadores
deben contener **todas las etapas previas necesarias para inferencia**:
`ToolLoop` y `ReasoningModel` cargan un único checkpoint final por variante.

## Datos y criterio de éxito

**Test congelado:** `results/rlm_final/data/test.jsonl` (200 problemas originales).
**Train:** `results/rlm_final/data/train.jsonl`. El programa aborta si detecta
una pregunta exactamente idéntica en ambos. El manifiesto registra hashes
SHA-256, configuración y semilla para reproducibilidad.

Cada variante ve **exactamente los mismos enunciados y hojas de reglas**
(`question` y `rule_context`), jamás `answer` ni `params`.
Las variantes sin tool utilizan el prompt de Fase 1 y generan con
`ReasoningModel`; las variantes con tool reciben el prompt del ejecutor,
el esquema de `calcular_imv` y el bucle nativo de llamadas a herramientas.
Es una diferencia experimental inevitable que debe documentarse.

Métrica principal: **`numeric_accuracy`**, coincidencia con la cuantía
total IMV+CAPI del oráculo, tras redondeo HALF_UP a céntimos, con **un único
resultado numérico final no ambiguo**. Para el modelo sin tool se extrae solo
la respuesta final etiquetada `<answer>...</answer>` (o un número sin tags ni
razonamiento), nunca cifras intermedias del pensamiento. Para Tool Use se
evalúa la respuesta final del bucle. Las llamadas fallidas no se reparan
automáticamente ni se convierten en aciertos.

Métricas secundarias: error absoluto medio en euros (solo calculado cuando
**todas** las respuestas son parseables; en otro caso `null`),
`parseable_rate`, formato, ejecución de la tool, errores de la tool,
duración y tokens generados. En Tool Use los tokens se **estiman** tokenizando
las respuestas del modelo; no son cifras facturables. La suma de tokens de
varios turnos puede superar el presupuesto individual de una variante sin
tool: no interpretar esto como igualdad estricta de coste computacional.

## Preparación en la DGX

```bash
git fetch origin
git switch feat/evaluacion-12-variantes-imv
git pull origin feat/evaluacion-12-variantes-imv
uv sync --extra train
```

El fichero `tool_use/bench/variants.example.json` ya declara los
**dos adaptadores existentes** según el manifiesto de Fase 1: SFT y
SFT+GRPO. Resuelve `${HOME}` en la DGX. No inventa rutas para
GRPO sin SFT ni los cuatro GRPO de tool use: esos cinco checkpoints
figuran como `null` hasta entrenarlos.

Copia esta configuración, por ejemplo, a
`tool_use/bench/variants.local.json` y rellena las rutas que existan.
No copies pesos de modelos a Git. Las rutas deben contener
`adapter_config.json` (adaptador PEFT final compatible con Qwen3-1.7B).

Verificación **sin descargar pesos ni usar GPU**:

```bash
uv run python -m tool_use.compare_variants --all --dry-run \
  --config tool_use/bench/variants.example.json
```

Para empezar con el checkpoint existente y probar el ejecutor:

```bash
uv run --extra train python -m tool_use.compare_variants \
  --variant base --variant base_tool \
  --n-examples 8 --max-new-tokens 2048 \
  --config tool_use/bench/variants.example.json \
  --out-dir reports/imv_12_variants_smoke
```

Ejecutar las doce configuraciones disponibles, **una por una en GPU**:

```bash
uv run --extra train python -m tool_use.compare_variants \
  --all --max-new-tokens 2048 --max-turns 3 \
  --config tool_use/bench/variants.local.json \
  --out-dir reports/imv_12_variants
```

`--all` mantiene en el CSV las variantes cuyos adaptadores faltan con
`status=missing_checkpoint` y métricas vacías; **nunca genera resultados
ficticios**. Si se selecciona explícitamente una variante no disponible
mediante `--variant`, el comando falla con una explicación.
`--n-examples N` está pensado para el smoke test: la comparación
definitiva debe incluir el test congelado completo.

**Resultados:**
- `reports/imv_12_variants/comparison.csv`: una fila por variante, incluidas pendientes.
- `reports/imv_12_variants/manifest.json`: cobertura de las 12 variantes,
  configuración, hashes y estados.
- `reports/imv_12_variants/<nombre>.json`: respuestas, trazas, llamadas,
  latencia, tokens y puntuación por problema de cada variante ejecutada.

Los informes se escriben de manera incremental. Si una ejecución se corta,
vuelve a lanzar la variante necesaria con `--variant`; para consolidar
varios informes de ejecuciones independientes, mantener el CSV/manifest del
comando `--all` y los JSON individuales. No asumir que `--all` reanuda
las variantes terminadas: actualmente **las vuelve a ejecutar**.

## Entrenamientos pendientes

**GRPO de razonamiento sin SFT**: el código existente sí permite
omitir `--init-adapter`:

```bash
uv run --extra train python -m rlm.train_grpo \
  --data results/rlm_final/data/train.jsonl \
  --model Qwen/Qwen3-1.7B \
  --output "$HOME/clusters/dgx/arca-runs/grpo-zero/weights/grpo_lora" \
  --steps 300 --num-generations 4 --micro-batch-size 1 \
  --max-completion-length 1024 --seed 42
```

Este es un **ejemplo de comando**, no un experimento ejecutado ni
un resultado prometido. Ajustar la configuración con los recursos DGX,
revisar señales de recompensa y guardar los logs.

**GRPO de tool use:** requiere una implementación nueva que entrene con
llamadas a `calcular_imv` y recompensa sobre ejecución real y respuesta
final; el GRPO de razonamiento existente **no sirve sin adaptarlo**.
Serán cuatro entrenamientos adicionales, con datos de **train** exclusivamente,
nunca con las preguntas del test.

Los checkpoints SFT/SFT+GRPO ya aparecen en el manifiesto de la ejecución
anterior, pero su existencia física se verifica en la DGX con el inventario.

## Pruebas CPU

```bash
uv run pytest -q tests/test_compare_variants.py tests/test_tool_evaluate_shared.py
```
