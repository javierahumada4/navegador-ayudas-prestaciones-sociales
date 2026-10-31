# Datos IMV/CAPI de fase 1

El generador y oracle producen problemas sintéticos 2026 con `question`,
`answer`, `params`, `branches` y `split`. No requieren anotación humana.
El modelo profesor produce trazas de razonamiento que se aceptan solo después
de verificarlas contra el oracle. SFT utiliza exclusivamente `verified=true`.

Genera los tres conjuntos y comprueba sus etiquetas y separación:

```bash
python -m rlm.pipeline --through-stage generate --run-dir runs/phase1-rebuild
```

Rutas bajo ese directorio: `data/train.jsonl` (800),
`data/test.jsonl` (200) y `data/test_ood.jsonl` (100, familia CAPI-only).
La destilación escribirá `data/sft_traces.jsonl` con muestras aceptadas y
rechazadas. Todos estos artefactos son regenerables y se excluyen de Git.

Para generar un split manualmente:
`python -m rlm.generate_problems --n 800 --split train --seed 42 --out ruta.jsonl`.

Guía integral: [docs/phase1_pipeline.md](../../docs/phase1_pipeline.md).
