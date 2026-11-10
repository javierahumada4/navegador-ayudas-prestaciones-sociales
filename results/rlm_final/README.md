# Resultados definitivos — RLM (IMV/CAPI)

Esta carpeta está preparada para **archivar los resultados de la ejecución final** del pipeline RLM. La creación de esta estructura **no implica que exista una ejecución final aquí**: los informes y figuras se añadirán una vez terminado y revisado el entrenamiento en la DGX.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `reports/` | Informes de evaluación (base, SFT, GRPO), validación del dataset, destilación y selección de trazas. |
| `figures/` | Curvas GRPO y otras figuras para el informe. |
| `metadata/` | `manifest.json`, configuración, semilla y referencia al commit usado. |
| `analysis/` | Interpretación de resultados y análisis de al menos cinco fallos concretos. |

### Informes esperados

- `dataset_validation.json`
- `distillation_validation.json`
- `sft_selection.json`
- `eval_test.json` y opcionalmente `eval_test.md`
- `eval_ood.json`, `eval_ood_threshold.json`, `eval_ood_wording.json` y sus resúmenes Markdown si se generaron
- `grpo_curves.png` (en `figures/`)

El pipeline actual genera **200 casos test** y **tres OOD de 50 casos** (CAPI, umbrales y redacción). Ya no utiliza `ood_composition`, `check_sft` ni `sft_preflight`.

## Copiar aquí los resultados desde una ejecución de la DGX

Desde la raíz del repositorio, en una sesión donde el directorio de ejecución sea accesible:

```bash
RUN="$HOME/clusters/dgx/arca-runs/NOMBRE-DE-LA-EJECUCION-FINAL"
DEST="results/rlm_final"

# Revisar el contenido antes de copiar
ls -lh "$RUN/reports" "$RUN/manifest.json"

# Copiar informes JSON y resúmenes Markdown que realmente existan
cp "$RUN"/reports/{dataset_validation,distillation_validation,sft_selection,eval_test,eval_ood,eval_ood_threshold,eval_ood_wording}.json "$DEST/reports/"
for file in "$RUN"/reports/eval_{test,ood,ood_threshold,ood_wording}.md; do
  if [ -f "$file" ]; then cp "$file" "$DEST/reports/"; fi
done

cp "$RUN/reports/grpo_curves.png" "$DEST/figures/"
cp "$RUN/manifest.json" "$DEST/metadata/"
git rev-parse HEAD > "$DEST/metadata/revision_de_copia.txt"
```

**Antes de declarar una ejecución como definitiva:** comprobar que todas las etapas acabaron correctamente, que los informes corresponden al mismo `--run-dir` y que `manifest.json` registra la configuración adecuada. Guardar la referencia al commit **realmente ejecutado**, que puede ser distinta del HEAD del momento de copiar.

Los pesos y checkpoints `weights/sft_lora/`, `weights/grpo_lora/`, las trazas completas y los entornos `uv` **no se guardan en Git**. Conservar los adaptadores en almacenamiento persistente y hacer una copia externa (por ejemplo, Hugging Face Hub u otro almacén de artefactos), documentando la ubicación y revisión exacta en `metadata/`.

Los resultados parciales, exploratorios o de ejecuciones anteriores deben permanecer fuera de esta carpeta para que los resultados finales sean inequívocos.
