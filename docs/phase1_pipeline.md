# Fase 1 reproducible — IMV + CAPI

Esta guía describe cómo generar **desde cero** problemas, trazas verificadas y adaptadores,
sin incorporar trazas ni checkpoints históricos. No publica datos personales.

## Requisitos

- Python 3.11 o 3.12 y [uv](https://docs.astral.sh/uv/).
- GPU NVIDIA y entorno CUDA operativo para destilación y entrenamiento. En la
  DGX comprobar cuánta VRAM tiene la partición asignada; no está garantizado
  que los valores por defecto quepan en cualquier MIG de 16 GB.
- Acceso a Hugging Face (definir `HF_TOKEN` si algún modelo lo requiere).
- Ejecución desde la **raíz del repositorio**.

El prompt del **alumno IMV** está en español en `rlm/data.py`, al igual que
el del **profesor** Qwen3 en modo thinking de `rlm/distill.py` y los
enunciados. El formato de salida mantiene las etiquetas `<think>` y
`<answer>`, así como el punto decimal. Los prompts no incluyen un importe
de ejemplo fijo que el modelo pueda copiar. El profesor debe razonar en
español y de forma concisa, aunque **no se puede garantizar el idioma del
pensamiento nativo de Qwen3 solo mediante instrucciones**. **El idioma no es
un filtro de aceptación**: se conservan las trazas inglesas cuando el formato
y el importe final son correctos. El código ya no rechaza trazas por la
heurística `non_spanish_reasoning`. Para generar trazas de razonamiento
**debe utilizarse el thinking nativo** (`enable_thinking=True`).
El pipeline utiliza exclusivamente thinking nativo, sin una alternativa que lo desactive.
No se añaden etiquetas de cierre artificiales a las trazas truncadas.
El prompt genérico de GSM8K continúa en inglés, porque se evalúa por separado.

**No mezcles checkpoints ni trazas de runs anteriores con este cambio de
prompt.** Para comparar base/SFT/GRPO, genera un `--run-dir` nuevo y mantén
el mismo prompt durante toda la ejecución.

### Misma hoja de reglas en profesor y alumno

Por defecto, el **profesor** y el **alumno** reciben el mismo texto
`rule_context` generado por `build_rule_context(load_ruleset())`, delante
del enunciado del caso. El alumno lo usa en **SFT, GRPO, evaluación y
`/reasoning`**. Las trazas nuevas almacenan una copia exacta de esa hoja de
reglas, que SFT reutiliza; para trazas antiguas sin `rule_context`, se
reconstruye desde las reglas actuales del repositorio. La destilación
sigue añadiendo las reglas **una sola vez**, antes de generar la respuesta
del profesor. El dominio GSM8K no recibe las reglas IMV.

Se ha eliminado el marcador literal `IMPORTE` de ambos prompts, pues
los modelos podían reproducirlo como respuesta. Ahora se indica que el
número debe escribirse entre las etiquetas de apertura y cierre
`<answer>` y `</answer>`, sin ejemplo numérico.

Como los prompts IMV incluyen reglas legales largas, `--sft-max-length 8192`
limita el **prompt más la respuesta**; no debe ponerse a 1024. El
argumento único `--completion-tokens 1024` limita las **respuestas** del
SFT, el readiness gate, GRPO y la evaluación usando el tokenizador del
alumno. No se recortan trazas ni se fabrican cierres de etiquetas.
Este cambio invalida los LoRA anteriores: inicia un `--run-dir` nuevo.
Que los modelos vean las reglas no garantiza por sí solo una mejora en pass@1.

Los extras `distill` (vLLM y su versión de torch) y `train`
(TRL/Transformers/PEFT) tienen dependencias incompatibles. El lanzador prepara
dos entornos aislados bajo el run, `.venv-distill` y `.venv-train`, mediante
`UV_PROJECT_ENVIRONMENT`. No instalar ambos extras en un mismo entorno.

## Comandos

El pipeline usa `--quality-gates warn` **por defecto**. Esto permite
recorrer generación, destilación, SFT, GRPO y evaluación aunque los
mínimos de calidad no se alcancen, para observar los resultados y
comparar etapas. Si se desea que los controles por calidad detengan
el entrenamiento, añadir `--quality-gates strict`. Los fallos
técnicos o los datos imposibles de entrenar siguen deteniendo la
ejecución. No es necesario añadir `--quality-gates warn` explícitamente.


### Entrenamiento completo para estudiar el aprendizaje de SFT

Con una carpeta nueva, sin `--smoke`:

```bash
python -u -m rlm.pipeline \
  --run-dir runs/phase1-dgx-1024-full \
  --seed 42 \
  --teacher Qwen/Qwen3-4B --teacher-backend vllm \
  --teacher-samples 8 --teacher-batch-size 1 --teacher-gpu-memory 0.75 \
  --student Qwen/Qwen3-1.7B \
  --completion-tokens 1024 --sft-max-length 8192 \
  --sft-epochs 2 --sft-batch-size 1 --sft-grad-accum 8 \
  --min-sft-problem-coverage 0.70 \
  --grpo-micro-batch-size 1 --eval-batch-size 1
```

El profesor mantiene `--teacher-max-new-tokens 2048` para poder completar
una traza; **SFT solo acepta completions de hasta 1024 tokens**
según el tokenizador del alumno. No se mezclan datasets ni adaptadores
de ensayos anteriores. El entrenamiento puede superar una sesión de
24 h de DGX; conserva los checkpoints y los informes por etapas.


Inspeccionar el plan sin escribir ni usar GPU:

```bash
python -m rlm.pipeline --dry-run
make rlm-plan
```

Ejecutar un ensayo de extremo a extremo con pocos datos (usa GPU y descarga
los modelos, no es solo una simulación):

```bash
python -m rlm.pipeline --smoke --run-dir runs/phase1-smoke
```

Entrenamiento completo:

```bash
python -m rlm.pipeline \
  --run-dir runs/phase1-2026-01 \
  --seed 42 \
  --teacher Qwen/Qwen3-4B \
  --teacher-backend vllm \
  --student Qwen/Qwen3-0.6B
```

De forma predeterminada genera 800 casos de entrenamiento y 200 de
test, con cobertura casi uniforme de **10 familias** (incluida
`boundary`, que sitúa renta, patrimonio, activos o residencia a un
céntimo/día del umbral). Los casos se calculan con el mismo oracle
determinista; el control de splits verifica etiquetas y ausencia de
duplicados. No es una muestra representativa de la población española.

Además, genera **cuatro suites OOD independientes** de 100 casos cada
una: `ood` (CAPI sin IMV, familia ausente de train/test),
`ood_threshold` (interacciones de dos condiciones próximas al límite),
`ood_composition` (unidades con **4 adultos**, fuera del rango de
train/test) y `ood_wording` (formulaciones alternativas de los
enunciados; es un cambio léxico, no una población independiente).
Se comprueban los duplicados *entre todas las suites* y se evalúan
base/SFT/GRPO por separado. Un smoke genera 8 casos por suite.

El profesor produce cuatro muestras por caso (dos en smoke), filtradas
por importe contra el oráculo. **SFT utiliza como máximo 2 trazas
verificadas, diferentes y preferentemente breves, por problema**
(`--sft-max-traces-per-problem`). Las trazas antiguas se limpian de
las etiquetas `<imv>` y `<capi>` cuando aparezcan: ya no forman parte
del contrato del estudiante y nunca se añaden de nuevo.

El **único formato IMV admitido** es
`<think>razonamiento</think><answer>123.45</answer>`, con un único
`<answer>` final fuera de `<think>`, sin anidamientos y con dos
decimales exactos (sin símbolos ni separadores). El detector genérico
para GSM8K sigue permitiendo respuestas enteras. La evaluación,
`SFT readiness`, el validador de trazas, la inferencia y la recompensa
IMV usan el contrato estricto. Esto puede disminuir la tasa de formato
publicada respecto de los runs anteriores, que aceptaban erróneamente
`<answer>` dentro del razonamiento.

En **GRPO hay solo tres recompensas**: formato estricto (1.0),
importe total correcto contra el oráculo (2.0) y respuesta final
única con dos decimales (0.5). Se ha eliminado toda recompensa
por cantidades intermedias IMV/CAPI. Se mantiene el monitor de
recompensas sin variación.

La plantilla de **Qwen3 thinking** se aplica en SFT, GRPO e inferencia
a partir del tokenizer, con `enable_thinking=True`. Nunca se añade
manualmente `<think>` al prefijo de generación del alumno. Antes de
SFT se comprueba que el ejemplo conversacional tokenizado tiene
**exactamente el mismo prefijo de asistente** que se usa para
generación, y que su final supervisado está intacto. GRPO utiliza
ese prefijo ya renderizado en sus prompts para evitar discrepancias
entre plantillas. El EOS se toma del tokenizer (`<|im_end|>` en
Qwen3). No se han validado todavía los pesos en DGX con el nuevo
contrato; las comprobaciones de CPU no sustituyen a un smoke GPU.

**Los nuevos formatos de trazas, recompensas y splits requieren un
`--run-dir` nuevo** y no deben combinarse con adaptadores anteriores.

Los artefactos se guardan en el directorio indicado con `--run-dir`:

- `data/train.jsonl`, `data/test.jsonl`, `data/test_ood.jsonl`,
  `data/test_ood_threshold.jsonl`, `data/test_ood_composition.jsonl`
  y `data/test_ood_wording.jsonl`;
- `data/sft_traces.jsonl` con muestras aceptadas y rechazadas;
- `logs/distillation.jsonl` y `reports/distillation_validation.json`
  con la tasa de aceptación por familia;
- `weights/sft_lora/` y `weights/grpo_lora/`;
- `reports/eval_test.json`, `reports/eval_ood.json` y los informes
  `eval_ood_threshold.json`, `eval_ood_composition.json`,
  `eval_ood_wording.json`;
- `reports/grpo_curves.png`;
- `manifest.json` con configuración, comandos, resultados y fechas.

`rlm.validate_data` comprueba el cálculo de cada etiqueta, que no se
repiten parámetros ni preguntas entre splits y que las trazas aceptadas
son correctas. Si ninguna traza es aceptada, **SFT no se ejecuta**.
La validación compara cada ejemplo con `rlm/imv_engine.py` y el
fichero de reglas para 2026. También rechaza datos temporalmente imposibles:
ninguna persona puede tener residencia continuada anterior a su nacimiento,
ni constar conviviendo desde antes del nacimiento del integrante más joven.

La destilación informa `rejected_by_reason`, `hit_max_tokens`,
`verified_unique_problems` y `verified_families`. Se registran
mínimos orientativos (smoke: 6 trazas de 4 problemas y 3 familias;
entrenamiento completo: 100 trazas de 50 problemas y 5 familias).
Por defecto, `--quality-gates warn` convierte el incumplimiento de
estos mínimos en una **advertencia** en el log y en
`reports/distillation_validation.json` (`quality_warnings`).
El pipeline continúa a SFT si hay datos utilizables. Con
`--quality-gates strict` se conserva el bloqueo por calidad.

Los errores de integridad (etiquetas incorrectas, muestras ausentes,
trazas marcadas como verificadas que no coinciden con el oráculo)
siguen deteniendo el entrenamiento en ambos modos.

### Validaciones de SFT y entrada a GRPO

Antes de cargar los pesos del alumno, `rlm.train_sft` examina todas las
trazas **verificadas** con el tokenizador y la plantilla de chat reales.
Selecciona hasta dos trazas distintas y cortas por problema, siempre con
respuesta final completa, descartando las que superen **1024 tokens de
completion** o el máximo de prompt+completion (`--sft-max-length 8192`).
El informe `reports/sft_selection.json` cuenta descartes por longitud,
cobertura de problemas y familias. No se corta ninguna respuesta.

Para ampliar la cobertura, el entrenamiento completo genera
8 muestras del profesor por problema (el smoke usa 2).
La selección sigue usando una cobertura de referencia del 70 % de
problemas y un mínimo de 5 familias, pero por defecto
**no detiene el SFT** cuando no se alcancen: anota
`quality_passed=false` y `quality_warnings` en
`reports/sft_selection.json` y entrena con las trazas verificadas
disponibles. `--quality-gates strict` recupera el bloqueo.
No se aceptan trazas incorrectas, incompletas o mayores de
1024 tokens. Si ninguna traza supera el filtro, SFT no puede
entrenarse y el programa falla incluso en modo aviso.

Las trazas finalmente seleccionadas superan además el preflight
`reports/sft_input_validation.json` para descartar fallos de plantilla
y de tokenización.

Una vez entrenado SFT, el pipeline ejecuta **`SFT readiness gate` antes
de iniciar GRPO**, con hasta ocho preguntas del conjunto de *entrenamiento*
seleccionadas de distintas familias. Este control funcional exige al
menos **75 % de respuestas con formato válido y como máximo 25 % truncadas**
(`--sft-min-format-rate`, `--sft-max-truncated-rate`).
El presupuesto del control es también `--completion-tokens 1024`. Siempre se escribe
`reports/sft_readiness.json`. Por defecto, una tasa insuficiente
solo genera una advertencia y GRPO comienza igualmente. En modo
`--quality-gates strict` se bloquea GRPO si falla ese control. Este control
**no es una evaluación de generalización**, pues no emplea ni test ni OOD.
Los resultados de test y OOD siguen reservados para la fase final.

GRPO ahora utiliza **4 generaciones por problema incluso en smoke**;
la temperatura predeterminada es **0.6**, `top_p=0.95` y
`top_k=20`, recomendados para Qwen3 en modo thinking. Se pueden
ajustar con `--grpo-generations`, `--grpo-temperature`,
`--grpo-top-p` y `--grpo-top-k`. Cuatro generaciones consumen
más GPU que dos; en la DGX es preferible `--grpo-micro-batch-size 1`.
El smoke sigue teniendo solo **2 pasos** de GRPO: sirve para comprobar
el flujo, no para demostrar aprendizaje.

GRPO sigue registrando avisos si los grupos de recompensas no
presentan variación. Por defecto no aborta por ello (paciencia 0
en modo `warn`). Con `--quality-gates strict` se recupera
el corte tras cinco pasos consecutivos sin señal
(`--grpo-zero-signal-patience`, configurable).
Un smoke con 2 pasos puede acabar con recompensa nula: eso **no** demuestra
aprendizaje, aunque se hayan generado los informes.

Para iniciar una etapa que todavía no se haya ejecutado, sin repetir las
anteriores:

```bash
python -m rlm.pipeline --run-dir runs/phase1-2026-01 \
  --from-stage sft --through-stage evaluate
```

Para recuperar una destilación interrumpida (se escriben batches a disco)
o el último checkpoint de SFT/GRPO dentro de una etapa:

```bash
python -m rlm.pipeline --run-dir runs/phase1-2026-01 \
  --from-stage distill --through-stage evaluate --resume

python -m rlm.pipeline --run-dir runs/phase1-2026-01 \
  --from-stage grpo --through-stage evaluate --resume
```

`--resume` requiere que existan las trazas parciales o un checkpoint completo
`checkpoint-N/trainer_state.json` en el run. La destilación comprueba que
el prefijo de muestras guardado coincide con el mismo dataset, profesor y
configuración. Para recuperarse de una sesión de 24 horas, guarda el run
fuera del directorio temporal o copia los adaptadores/checkpoints al terminar.

Las salidas ya existentes se protegen. Para regenerar una etapa expresamente,
`--force` reemplaza sus salidas; para un run nuevo es preferible cambiar
`--run-dir`. Mantén el mismo modelo, semilla y configuración al continuar
una corrida; el fichero de manifiesto documenta la primera ejecución.

En caso de memoria insuficiente, reducir `--teacher-batch-size`,
`--sft-batch-size`, `--grpo-generations`,
`--grpo-micro-batch-size` y `--eval-batch-size`. El teacher incluye
reglas detalladas de 2026 que pueden alargar mucho los prompts; por ello el
pipeline reserva `--teacher-max-model-len 8192` y
`--teacher-max-new-tokens 2048` por defecto, **también en smoke**.
Si aún se truncan muchas trazas, se puede probar `--teacher-max-new-tokens
3072` en un run nuevo, sin exceder el contexto máximo y con el coste de
más tiempo de generación.

## Pruebas

```bash
make test-logic
make test
```

El primer comando ejecuta generador/oracle/verificador/validación/pipeline en
CPU. El segundo añade las pruebas que necesitan torch/datasets. Los tests de
entrenamiento no sustituyen a una ejecución GPU real.

## Límites y entrega

El oracle, el generador y la ficha del profesor incluyen el caso de otro
progenitor hospitalizado o encarcelado 12 meses o más; el caso explicita
los meses ininterrumpidos en el enunciado. Se ha corregido un parser de
destilación que confundía
`474,005` con `474005`. Por defecto solo las trazas verificadas llegan a SFT.

Tras el run, completar `EXPERIMENTS.md` con métricas observadas,
comparación pass@1, ablación y cinco fallos interpretados, evitando inventar
resultados. Para desplegar `/reasoning`, configurar `ARCA_RLM_ADAPTER` con
`weights/grpo_lora` o subir ese adaptador a Hugging Face; comprobar
`/health` y una petición real antes de exponer ngrok. Docker Compose levanta la API sin reservar GPU;
los servicios `smoke`, `train` y `check-gpu` sí requieren runtime NVIDIA.


## Adaptación a GPU locales

Si el ensayo no cabe en memoria, ajusta `--teacher-batch-size`,
`--teacher-gpu-memory`, `--grpo-micro-batch-size` y `--eval-batch-size`.
Utiliza otro `--run-dir` al cambiar modelos o prompts y conserva el modo
thinking nativo. Documenta las variables específicas de CUDA/vLLM que necesites.
