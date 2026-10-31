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

De forma predeterminada genera 800 casos de entrenamiento, 200 de test
y 100 OOD, reservados para la familia `capi_only`. Produce cuatro muestras
del profesor por caso, filtra con el oracle, entrena LoRA SFT sobre las
trazas válidas y luego GRPO **inicializado con el adaptador SFT**. Evalúa
base/SFT/GRPO separadamente en test y OOD.

Los artefactos se guardan en el directorio indicado con `--run-dir`:

- `data/train.jsonl`, `data/test.jsonl` y `data/test_ood.jsonl`;
- `data/sft_traces.jsonl` con muestras aceptadas y rechazadas;
- `logs/distillation.jsonl` y `reports/distillation_validation.json`
  con la tasa de aceptación por familia;
- `weights/sft_lora/` y `weights/grpo_lora/`;
- `reports/eval_test.json` y `reports/eval_ood.json`;
- `reports/grpo_curves.png`;
- `manifest.json` con configuración, comandos, resultados y fechas.

`rlm.validate_data` comprueba el cálculo de cada etiqueta, que no se
repiten parámetros ni preguntas entre splits y que las trazas aceptadas
son correctas. Si ninguna traza es aceptada, **SFT no se ejecuta**.
La validación compara cada ejemplo con `rlm/imv_engine.py` y el
fichero de reglas para 2026.

Para reanudar a partir de una fase sin reescribir las anteriores:

```bash
python -m rlm.pipeline --run-dir runs/phase1-2026-01 \
  --from-stage sft --through-stage evaluate
```

Las salidas ya existentes se protegen. Para regenerar una etapa expresamente,
`--force` reemplaza sus salidas; para un run nuevo es preferible cambiar
`--run-dir`. Mantén el mismo modelo, semilla y configuración al continuar
una corrida; el fichero de manifiesto documenta la primera ejecución.

En caso de memoria insuficiente, reducir `--teacher-batch-size`,
`--sft-batch-size`, `--grpo-generations`,
`--grpo-micro-batch-size` y `--eval-batch-size`. El teacher incluye
reglas detalladas de 2026 que pueden alargar mucho los prompts; por ello el
pipeline reserva `--teacher-max-model-len 6144` por defecto.

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
`/health` y una petición real antes de exponer ngrok. Docker Compose sigue
requiriendo runtime NVIDIA.
