# Navegador IMV/CAPI — práctica ARCA

Asistente para orientar sobre el **Ingreso Mínimo Vital (IMV)** y el **Complemento de Ayuda
para la Infancia (CAPI)**. A partir de la descripción de un hogar, calcula la cuantía mensual
que le correspondería en 2026, explica qué requisito falla cuando no hay derecho, busca otras
convocatorias públicas en la Base de Datos Nacional de Subvenciones (BDNS) y guarda un informe
para el usuario.

Es la práctica final de Modelos Generativos Profundos (Máster en Inteligencia Artificial,
ICAI, 2026-2027). Está construida sobre la plantilla ARCA. El enunciado está en
[`docs/enunciado.pdf`](docs/enunciado.pdf), y la propuesta de tema aprobada en
[`docs/propuesta.md`](docs/propuesta.md).

> Las cuantías son una **simulación orientativa** sobre casos normalizados, no una resolución
> de la Seguridad Social.

## Estado

| Fase | Carpeta | Endpoint | Estado |
|---|---|---|---|
| 1. Razonamiento | [`rlm/`](rlm/README.md) | `POST /reasoning` | Hecha: Qwen3-1.7B destilado de Qwen3-4B, SFT + GRPO |
| 2. Herramientas | [`tool_use/`](tool_use/README.md) | `POST /tools` | Hecha: `calcular_imv`, `consultar_ayudas`, `guardar_informe` |
| 3. Conocimiento | `rag/` | `POST /rag` | Pendiente |
| 4. Agente | `agent/` | `POST /agent` | Pendiente |

## Resultados principales

**Fase 1.** Pass@1 sobre 200 casos de test con un importe exacto en euros, y sobre tres
conjuntos fuera de distribución de 50 casos:

| Modelo | Test | CAPI sin IMV | Umbrales | Redacción |
|---|---:|---:|---:|---:|
| Qwen3-1.7B base | 0,065 | 0,000 | 0,000 | 0,080 |
| + SFT | 0,375 | 0,460 | 0,220 | 0,560 |
| + SFT + GRPO | **0,490** | **0,500** | **0,300** | 0,540 |

**Fase 2.** Banco de 40 casos, éxito de principio a fin: herramienta, argumentos, importes,
fichero y explicación correctos.

| Cerebro | Con herramientas | Sin herramientas |
|---|---:|---:|
| Qwen3-1.7B base | **0,775** | 0,150 |
| Qwen3-1.7B + LoRA GRPO | 0,700 | 0,125 |

El análisis de cada resultado, con sus fallos, está en el README de cada fase. El registro
fechado de lo que hemos probado está en [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Estructura

```
rlm/                 Fase 1: motor IMV/CAPI, generador, verificador, destilación, SFT, GRPO
  imv_engine.py      Oráculo determinista de la cuantía (Ley 19/2021, cuantías 2026)
  rulesets/          Cuantías y umbrales por año
  pipeline.py        Lanza el pipeline completo, de los datos a la evaluación
tool_use/            Fase 2: herramientas, bucle de cinco pasos, banco y métricas
  domain_tools.py    calcular_imv, consultar_ayudas (BDNS) y guardar_informe
  executor.py        Bucle prompt → generación → parseo → ejecución → observación
  bench/             Banco de 40 casos y bancos por herramienta
api/                 FastAPI con los cuatro endpoints de la práctica
results/
  rlm_final/         Datos, trazas, informes y curvas de la ejecución final de la fase 1
  fase2/             Salidas completas de la evaluación de la fase 2
tests/               Tests sin GPU de motor, generador, recompensas, paso GRPO y herramientas
docs/                Enunciado, rúbrica, guías de la plantilla y nuestra propuesta
```

## Puesta en marcha

Hace falta Python 3.11 y [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/javierahumada4/navegador-ayudas-prestaciones-sociales.git
cd navegador-ayudas-prestaciones-sociales
cp .env.example .env
uv sync --extra train        # torch, transformers, TRL y PEFT
uv run --extra train pytest  # todos los tests; ninguno necesita GPU
make test-logic              # solo motor, generador, validación y recompensas
```

**Levantar la API:**

```bash
uv run arca-api              # http://localhost:8000/docs
```

Para que `/reasoning` y `/tools` respondan hay que configurar `.env`:

- `ARCA_RLM_ADAPTER` apunta al LoRA de la fase 1. Puede ser una carpeta local o el repositorio
  de Hugging Face `JES0406/imv-sft-lora/grpo_final`.
- `/tools` usa por defecto Qwen3-1.7B sin adaptador, que es el mejor cerebro de herramientas
  según nuestra evaluación (`ARCA_TOOLS_BRAIN=base`).

Cómo exponer la API para la corrección está en [`api/README.md`](api/README.md).

**Reproducir los resultados:** necesita una GPU; nosotros usamos una partición de la DGX.

```bash
make rlm-from-zero    # fase 1: datos, destilación, SFT, GRPO, evaluación y curvas (~12 h)
uv run --extra train python -m tool_use.compare --brain base \
    --out reports/phase2_base.json   # fase 2: con y sin herramientas (~10 min)
```

## Equipo

- Javier Escobar Serrano
- Javier Ahumada
- Marcos
