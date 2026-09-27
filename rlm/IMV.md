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

## Generar datos

Desde la raíz del repositorio:

```bash
uv run python -m rlm.generate_problems --n 800 --split train --out rlm/data/train.jsonl
uv run python -m rlm.generate_problems --n 200 --split test --out rlm/data/test.jsonl
uv run python -m rlm.generate_problems --n 100 --split ood --out rlm/data/test_ood.jsonl
```

`train` y `test` comparten distribución. `test_ood` reserva una familia completa:
**CAPI-only** (sin derecho a IMV por renta, pero con derecho al CAPI). De ese modo existe
un experimento OOD claro para comparar base/SFT/GRPO.

El generador imprime:
- cobertura por familia y rama;
- número de plantillas;
- número de fugas de respuesta en el enunciado (debe ser 0).

## Destilación

El JSONL incluye `rule_context`. Por defecto `distill.py` lo añade solo al prompt del
modelo profesor. Esto es deliberado:

1. el profesor recibe las reglas exactas de 2026 y produce trazas fiables;
2. el alumno recibe el enunciado normal, no una chuleta con la respuesta;
3. SFT destila las reglas y el procedimiento de cálculo;
4. GRPO mejora la aplicación de las reglas mediante recompensa verificable.

```bash
uv run python -m rlm.distill \
  --data rlm/data/train.jsonl \
  --teacher Qwen/Qwen3-4B \
  --samples 4 \
  --verifier imv \
  --output rlm/data/sft_traces.jsonl
```

## SFT

Se mantiene el script de la plantilla:

```bash
uv run python -m rlm.train_sft \
  --data rlm/data/sft_traces.jsonl \
  --output rlm/weights/sft_lora
```

## GRPO

```bash
uv run python -m rlm.train_grpo \
  --data rlm/data/train.jsonl \
  --init-adapter rlm/weights/sft_lora \
  --output rlm/weights/final_rlm_lora
```

Las recompensas son:

1. `format_reward`: estructura `<think>...</think><answer>...</answer>`;
2. `accuracy_reward`: importe correcto;
3. `domain_reward`: `<answer>` contiene únicamente `123.45` con dos decimales.

Pesos iniciales propuestos: `[1.0, 2.0, 0.5]`. Deben validarse experimentalmente y
documentarse, no tratarse como hiperparámetros definitivos.

## Evaluación

```bash
uv run python -m rlm.evaluate \
  --data rlm/data/test.jsonl \
  --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora

uv run python -m rlm.evaluate \
  --data rlm/data/test_ood.jsonl \
  --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora
```

## API

Cuando exista el adaptador:

```env
ARCA_RLM_ADAPTER=rlm/weights/final_rlm_lora
ARCA_RLM_VERIFIER=imv
```

El contrato de `/reasoning` no se modifica; el repositorio de plantilla exige mantener
`api/schemas.py`.

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
