# Datos de la fase 1

Los conjuntos se generan con el motor IMV/CAPI; nadie los anota a mano. Esta carpeta es donde
`rlm.generate_problems` los escribe por defecto. Los ficheros generados no se guardan en Git:
los de la ejecución final están en
[`results/rlm_final/data/`](../../results/rlm_final/data/).

Cada línea es un problema en JSON:

| Campo | Contenido |
|---|---|
| `question` | Enunciado en lenguaje natural con todos los hechos del caso |
| `answer` | Total mensual IMV + CAPI calculado por el motor, con dos decimales |
| `params` | El caso normalizado que recibe el motor |
| `branches` | Ramas del motor que decidieron el resultado |
| `template_id` | Plantilla de redacción usada |
| `rule_context` | Hoja de reglas 2026 que precede al enunciado en el prompt |
| `split` | `train`, `test`, `ood`, `ood_threshold` u `ood_wording` |
| `label_source` | De dónde sale la etiqueta (siempre el motor) |

`sft_traces.jsonl` guarda cada traza del profesor, aceptada o rechazada, con el motivo del
rechazo.

Para generar un conjunto suelto:

```bash
uv run python -m rlm.generate_problems --n 800 --split train --seed 42 --out rlm/data/train.jsonl
```

Para generar y validar todos los conjuntos a la vez:

```bash
python -m rlm.pipeline --through-stage generate --run-dir runs/phase1
```
