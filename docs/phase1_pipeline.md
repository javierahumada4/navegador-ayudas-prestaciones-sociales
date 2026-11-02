# Fase 1 reproducible — IMV + CAPI

Esta guía corresponde a la rama `fix/rlm-reproducible-pipeline`, derivada de
`feat/ngrok`. Genera **desde cero** problemas, trazas verificadas y adaptadores,
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
heurística `non_spanish_reasoning`. Para contrastar
una variante que desactiva ese pensamiento nativo y prefija `<think>`,
se puede usar `--teacher-no-thinking` en un run independiente. El prompt genérico
de GSM8K continúa en inglés, porque se evalúa por separado.

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

Como los prompts del alumno son ahora considerablemente más largos,
`--sft-max-length` pasa de 4096 a **8192** tokens. Esto aumenta la memoria
necesaria: en una MIG de 16 GiB puede ser necesario reducir el tamaño de
lote o revisar el presupuesto antes de repetir el smoke. Reducir
`--sft-max-length` indiscriminadamente puede truncar la respuesta
supervisada. Este cambio **invalida la comparación directa con los LoRA
anteriores**: empieza un `--run-dir` nuevo para las evaluaciones comparables.
Que los modelos vean las reglas no garantiza por sí solo una mejora en pass@1.

Los extras `distill` (vLLM y su versión de torch) y `train`
(TRL/Transformers/PEFT) tienen dependencias incompatibles. El lanzador prepara
dos entornos aislados bajo el run, `.venv-distill` y `.venv-train`, mediante
`UV_PROJECT_ENVIRONMENT`. No instalar ambos extras en un mismo entorno.

## Comandos

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
por importe contra el oracle. **SFT utiliza como máximo 2 trazas
verificadas, diferentes y preferentemente breves, por problema**
(`--sft-max-traces-per-problem`), sin cambiar el reporte bruto de
destilación. Antes de SFT se anteponen al pensamiento del profesor
las partidas verificadas por el oracle:
`<imv>importe_con_dos_decimales</imv>` y
`<capi>importe_con_dos_decimales</capi>`.
El alumno también recibe la instrucción de generar esos dos campos
dentro de `<think>`. Si una partida no corresponde, representa cero.

En GRPO se mantienen las recompensas anteriores de formato, importe
final y formato decimal, y se añaden
**`imv_component_reward` y `capi_component_reward` (peso 1 cada una)**.
Cada recompensa compara una etiqueta **única y estricta** con el
importe del oracle calculado a partir de `params`; no otorga puntos
por palabras clave ni por números sueltos. Se admiten las etiquetas
de componente en un `<think>` todavía sin cerrar, para dar señal
parcial cuando se agote el límite de tokens. Esto no garantiza que
haya variación entre generaciones; el monitor de señal cero sigue
activo. GRPO empieza desde el adaptador SFT validado.

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

Para recuperar trazas de runs anteriores rechazadas solo por idioma,
usa `python -m rlm.reverify_traces --train ... --input ... --output ...`.
No se modifica el archivo original: la herramienta vuelve a verificar
el formato y el importe de cada candidata y solo entonces cambia su
estado. No es necesario volver a generar las salidas del profesor.

La destilación informa `rejected_by_reason`, `hit_max_tokens`,
`verified_unique_problems` y `verified_families`. El modo `--smoke` exige
al menos **6 trazas de 4 problemas y 3 familias distintas**; un run normal,
100 trazas de 50 problemas y 5 familias, como mínimos de protección
(no garantizan razonamientos correctos). Si no se alcanzan, se guarda
`reports/distillation_validation.json` y el flujo se detiene **antes de SFT**.
Los límites se pueden ajustar con `--min-verified-traces`,
`--min-unique-trace-problems` y `--min-verified-families`,
pero no conviene reducirlos para ocultar una mala destilación.

### Validaciones de SFT y entrada a GRPO

Antes de cargar los pesos del alumno, `rlm.train_sft` tokeniza **todas**
las trazas verificadas con la plantilla de chat del modelo, y audita las
etiquetas `<think>` y `<answer>`, la longitud completa del prompt más
la respuesta, y una reserva de ocho tokens para el final de secuencia.
Cualquier ejemplo cuyo `</answer>` pueda quedar fuera de
`--sft-max-length` **bloquea SFT**: nunca se descarta ni recorta una
respuesta correcta silenciosamente. El informe se conserva en
`reports/sft_input_validation.json` incluso si el control falla. Si
tienes una MIG de 16 GiB, aumentar `--sft-max-length` puede causar OOM;
en ese caso hay que reducir la longitud de las trazas, no ocultar el
truncamiento.

Una vez entrenado SFT, el pipeline ejecuta **`SFT readiness gate` antes
de iniciar GRPO**, con hasta ocho preguntas del conjunto de *entrenamiento*
seleccionadas de distintas familias. Este control funcional exige al
menos **75 % de respuestas con formato válido y como máximo 25 % truncadas**
(`--sft-min-format-rate`, `--sft-max-truncated-rate`).
El presupuesto es `--sft-check-max-new-tokens 1536`. Siempre se escribe
`reports/sft_readiness.json`, aunque GRPO sea bloqueado. Este control
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

GRPO avisa al detectar recompensas sin variación y detiene un run largo
tras 5 pasos consecutivos sin señal (`--grpo-zero-signal-patience`).
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


## Ejemplo local en WSL2 (GPU NVIDIA de 12 GB)

Una vez comprobado `nvidia-smi` en Ubuntu/WSL2, ejecutar el ensayo con
**un nuevo directorio**, sin mezclar las trazas y adaptadores anteriores:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python3 -m rlm.pipeline --smoke --run-dir runs/phase1-smoke-v2 --teacher Qwen/Qwen3-1.7B --teacher-batch-size 1 --teacher-gpu-memory 0.75 --grpo-micro-batch-size 1 --eval-batch-size 1
```

`VLLM_USE_FLASHINFER_SAMPLER=0` evita el muestreador FlashInfer que
requiere compilar kernels con `nvcc` en algunas configuraciones WSL2;
no deshabilita CUDA ni FlashAttention. Si el profesor insiste en razonar
en inglés o genera cadenas interminables, probar `--teacher-no-thinking`
en **otro run-dir** y contrastar aceptación y cobertura antes de SFT.
El cambio de prompt o modelo no debe evaluarse reusando LoRA históricos.
