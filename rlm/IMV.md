# Dominio de la fase 1: IMV + CAPI

Esta carpeta conserva el flujo de la plantilla ARCA y sustituye únicamente el ejemplo
de transporte por una tarea verificable del dominio: calcular la cuantía mensual total
de **Ingreso Mínimo Vital (IMV) + Complemento de Ayuda para la Infancia (CAPI)** a partir
de un caso normalizado.

## Por qué está así

La fase 1 necesita una respuesta comprobable automáticamente. En este dominio la respuesta
es un importe mensual con dos decimales. La implementación de referencia es
`rlm/imv_engine.py`; el generador la usa para crear las etiquetas y los tests la usan como
oracle. El modelo no ejecuta este motor durante inferencia.

```
caso normalizado
      |
      v
rlm/imv_engine.py       <- oracle determinista
      |
      +--> rlm/generate_problems.py --> train/test/test_ood JSONL
      |
      +--> tests/test_imv_engine.py
                         |
                         v
      distill.py -> train_sft.py -> train_grpo.py -> evaluate.py -> inference.py
```

Así SFT y GRPO entrenan exactamente la misma tarea que corrige el verificador.

## Ficheros del dominio

- `imv_engine.py`: reglas deterministas y cálculo de referencia.
- `rulesets/imv_2026.json`: cuantías y umbrales que pueden cambiar con la normativa.
- `generate_problems.py`: generador programático de casos.
- `verifier.py`: añade `IMVAmountVerifier`.
- `rewards.py`: comparación numérica compatible con decimal español.
- `train_grpo.py`: tercera recompensa de dominio: respuesta limpia con dos decimales.
- `distill.py`: usa el verificador IMV y puede enseñar las reglas al profesor sin
  incluirlas en el prompt del alumno.
- `evaluate.py`: pass@1 con el mismo verificador.
- `inference.py`: `ARCA_RLM_VERIFIER=imv`.

`train_sft.py` no necesita lógica específica del IMV: ya consume correctamente
`question + trace` verificados.

## Flujo reproducible del dominio

El generador utiliza `imv_engine.py` como oracle para train/test y para
el OOD de casos CAPI-only. El programa principal escribe y valida los JSONL
en un directorio aislado, sin reutilizar datasets históricos:

```bash
python -m rlm.pipeline --dry-run
python -m rlm.pipeline --through-stage generate --run-dir runs/phase1-rebuild
```

El generador muestra cobertura por familia, plantillas y fugas de respuesta.
El validador verifica de nuevo el importe calculado por el oracle y rechaza
parámetros o preguntas duplicados entre conjuntos.

## Destilación, SFT, GRPO y evaluación

`distill.py` envía `rule_context` solo al profesor, no al estudiante.
Las trazas aceptadas se usan en SFT LoRA; GRPO continúa desde el adaptador SFT
usando las respuestas del conjunto train. Los extras de vLLM (`distill`)
y TRL (`train`) utilizan entornos `uv` separados.

```bash
# Ensayo GPU pequeño de las cinco etapas
python -m rlm.pipeline --smoke --run-dir runs/phase1-smoke

# Generación, destilación, SFT, GRPO, evaluación y curvas
python -m rlm.pipeline --run-dir runs/phase1-rebuild
```

Se generan `runs/phase1-rebuild/weights/sft_lora`,
`runs/phase1-rebuild/weights/grpo_lora`, los informes
`reports/eval_test.json` y `reports/eval_ood.json`, y las curvas de GRPO.
Los pasos de ejecución, checkpoints y recuperación se explican en
[docs/phase1_pipeline.md](../docs/phase1_pipeline.md).

Las recompensas de GRPO son `format_reward`, `imv_accuracy_reward`
y `domain_reward`: estructura `<think>/<answer>`, importe monetario con
`ROUND_HALF_UP` y número limpio con dos decimales. Sus pesos iniciales son
`[1.0, 2.0, 0.5]`: se deben medir y justificar. `zero_answer_rate`
es un monitor con peso cero para detectar sesgo hacia `0.00`.

## API del dominio

Tras obtener el LoRA final:

```env
ARCA_RLM_ADAPTER=runs/phase1-rebuild/weights/grpo_lora
ARCA_RLM_VERIFIER=imv
```

También se puede publicar el adaptador en Hugging Face y configurar su repo.
`/reasoning` mantiene el contrato de `api/schemas.py`. La validación de
inferencia requiere un adaptador entrenado y una petición real a la API.

## Frontera de normalización

Los problemas de fase 1 usan hechos normalizados. Un cambio de umbral del IMV debe modificar
`rulesets/imv_2026.json` (o el ruleset futuro), no el nombre de los campos.

Ejemplo: se almacena `disability_percent=65`; no existe un campo
`disability_65_or_more`.

Del mismo modo, conceptos cuyo cálculo depende de otra normativa pueden llegar normalizados,
por ejemplo `countable_income_annual_eur`. El motor de esta fase se centra en aplicar la
normativa IMV/CAPI a esos hechos.

## Alcance de la referencia de fase 1

El oracle cubre el núcleo verificable necesario para un dataset rico de nuevas solicitudes:
unidad ordinaria por co-residencia/vínculo, edad/independencia ordinaria, residencia,
antigüedad, renta garantizada, discapacidad, monoparentalidad común, renta, patrimonio,
activos, administrador mercantil, tope por pensiones y CAPI.

No se presenta como simulador administrativo completo de todos los expedientes reales.
Revisión anual, incentivo al empleo, suspensión, extinción, reintegros y casuística documental
excepcional son procesos distintos y no deben inflar la tarea de razonamiento de fase 1.

## Fuentes del ruleset 2026

- Ley 19/2021, de 20 de diciembre, por la que se establece el ingreso mínimo vital,
  texto consolidado actualizado en abril de 2026.
- Seguridad Social, página oficial de Ingreso Mínimo Vital, cuantías 2026.

## `rule_context` para distillation

`generate_problems.py` construye `rule_context` automáticamente desde `rlm/rulesets/imv_2026.json`.
No es una segunda copia hardcodeada de los umbrales.

El contexto del profesor incluye explícitamente:

- cuantías ordinarias de renta garantizada;
- condición mínima de 10 €/mes para IMV;
- discapacidad >= 65% y complemento del 22%;
- todos los supuestos monoparentales usados por la Ley 19/2021;
- tabla completa de límites de patrimonio neto por composición;
- tabla completa del test de activos no societarios por composición;
- desigualdad exacta de cada test (`<` para patrimonio, `<=` para activos);
- límites CAPI del 300% de renta y 150% de patrimonio por composición;
- importes CAPI por edad;
- tope por pensiones contributivas/no contributivas y subsidio para mayores de 52 años: igualdad con la renta garantizada excluye IMV y, por debajo, `IMV = min(IMV_por_renta, renta_garantizada - pensiones)`;
- convención exacta de redondeo del oracle: `Decimal` + `ROUND_HALF_UP` a 2 decimales, indicando en qué pasos se redondea y en cuáles no.

Esto es deliberado: el profesor de distillation necesita los valores exactos para producir
trazas correctas en problemas de borde (`income_fail`, `patrimony_fail`, `capi_only`, etc.).
El estudiante de SFT/GRPO no necesita recibir esta hoja de reglas en inferencia salvo que
se ejecute el generador con `--with-rules` para un experimento específico.


## Convención monetaria del oracle

La convención forma parte del contrato de la tarea, no es un detalle de presentación:

1. La renta garantizada ordinaria por número de miembros se redondea a céntimos con `ROUND_HALF_UP`.
2. Los complementos de discapacidad/monoparentalidad se calculan sobre la base individual sin redondear cada complemento por separado; después se redondea la renta garantizada final.
3. `ingresos_anuales / 12` no se redondea antes de calcular el IMV.
4. El test de diferencia mínima de 10 € se hace sobre el IMV por renta antes del redondeo final.
5. El tope de pensiones/subsidio se aplica antes de redondear el IMV final.
6. El IMV final se redondea a céntimos `ROUND_HALF_UP`.
7. Se suman las cuantías CAPI de los menores y se redondea el total CAPI.
8. El total mensual suma IMV redondeado + CAPI redondeado y vuelve a cuantizar a céntimos.
9. Los límites monetarios calculados por el oracle (patrimonio, activos y límites CAPI) se cuantizan a céntimos antes de comparar.

`IMVAmountVerifier` e `imv_accuracy_reward` deben usar exactamente la misma cuantización. Así, por ejemplo, `474.005` se considera `474.01`, nunca `474.00`.

## Contrato de observabilidad del enunciado

El generador debe cumplir una regla estricta: **todo hecho que el oracle use para decidir la respuesta debe aparecer en `question`**.

En particular:

- Las edades se calculan con fecha completa. Para menores se muestran tanto la edad en la fecha de solicitud como la edad a 1 de enero del año de solicitud, que es la usada para el tramo CAPI.
- La residencia legal y efectiva continuada se muestra por miembro, con su fecha individual de inicio.
- En beneficiarios individuales se muestran la fecha desde la que vive en domicilio distinto al de progenitores/tutores y todos los periodos de alta en Seguridad Social usados por el test de independencia.
- Toda relación progenitor-hijo generada especifica si la custodia es `exclusiva` o `compartida (no exclusiva)`. Así, una unidad con un solo adulto y menores no se confunde automáticamente con una unidad monoparental.

Los tests de regresión y la auditoría del generador comprueban estas propiedades para train/test/OOD.
