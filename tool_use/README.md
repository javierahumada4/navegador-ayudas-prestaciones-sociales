# Fase 2 — Herramientas IMV/CAPI (ARCA)

**Endpoint:** `POST /tools`. Implementación manual sin framework de agentes. Base: `main`
con motor IMV/CAPI de fase 1 intacto. La carpeta `rlm/` no se modifica.

## Herramientas y justificación

| Herramienta | Categoría del enunciado | Resultado |
|---|---|---|
| `calcular_imv(case: IMVCase)` | Cálculo local | IMV, CAPI, total, motivos y versión de reglas |
| `consultar_ayudas(tema, max_resultados)` | API HTTP externa | Convocatorias públicas registradas en BDNS |
| `guardar_informe(contenido)` | Acción observable | Ruta, tamaño y SHA256 de un TXT generado |

**El modelo escribe el informe en lenguaje natural.** `guardar_informe` únicamente crea
el archivo con su texto exacto; no usa ninguna plantilla de redacción. Si no se
ha solicitado explícitamente guardar el archivo, la ejecución lo rechaza. Requiere
una simulación IMV previa y un texto que refleje los tres importes devueltos
por el motor. El usuario debe solicitar explícitamente la generación del fichero;
el endpoint no ofrece consentimiento interactivo adicional.

**Datos oficiales:** el endpoint público de BDNS `/convocatorias/busqueda` recibe
`descripcion`, `descripcionTipoBusqueda=1`, `page`, `pageSize`, `order` y `direccion`.
Se valida la longitud de consulta, se acota la página y se aplica timeout de 15 s.
Los errores HTTP o de formato se comunican como observaciones, sin resultados
simulados. BDNS contiene convocatorias de subvenciones, no todas las
prestaciones sociales; las fechas de apertura y la elegibilidad no se infieren
de los resultados. Documentación:
https://www.infosubvenciones.es/bdnstrans/doc/swagger

**Rulesets:** `rules_for_date` busca los JSON `rlm/rulesets/imv_*.json` con
rangos de vigencia. No se generan reglas con el LLM. Si no existe exactamente
un ruleset para la fecha, se devuelve un error; el único existente en `main`
corresponde a 2026. No se inventan reglas para otros años.

## Implementación

- `registry.py`: infraestructura docente existente; Pydantic valida antes de ejecutar.
- `parser.py`: infraestructura docente existente; admite varios bloques `<tool_call>`.
- `imv_schema.py`: entradas tipadas, campos estrictos y comprobación de consistencia.
- `domain_tools.py`: motor 2026, llamada BDNS real y persistencia TXT acotada.
- `tools.py`: registro de tres herramientas del dominio; ejemplos docentes conservados.
- `executor.py`: bucle completo generación → parseo → ejecución → observación → respuesta.
- `bench/cases.jsonl`: 30 casos (14 cálculo, 10 HTTP, 6 sin herramienta; tres
  de los casos de cálculo incluyen petición de informe posterior).
- `evaluate.py`: exactitud de primera selección y de argumentos.
- `compare.py`: mismo modelo **con y sin herramientas**, guardando resultados reales.
- `tests/test_phase2_imv_tools.py`: pruebas offline del flujo y sus herramientas.

Las herramientas independientes pueden ejecutarse en un mismo turno, mientras que
`guardar_informe` necesita haber recibido el resultado de `calcular_imv` en un
turno previo. No se ejecutan procesos externos, ni se acepta una ruta suministrada
por el LLM, ni se sobrescriben ficheros. Los informes pueden contener datos personales:
se almacenan con permisos restrictivos en una ruta configurable mediante
`ARCA_IMV_REPORTS_DIR`, no deben subirse al repositorio ni guardarse sin autorización.

## Instalación, ejecución y verificación

```bash
uv sync --extra train
uv run pytest -q tests/test_phase2_imv_tools.py tests/test_tool_registry.py tests/test_tool_parser.py
uv run arca-api

# POST http://localhost:8000/tools
curl -X POST http://localhost:8000/tools \
  -H 'Content-Type: application/json' \
  -d '{"query":"¿Qué es el IMV?","max_turns":5}'

# En GPU, ejecutar la comparación REAL (no hay métricas precalculadas):
uv run --extra train python -m tool_use.compare \
  --bench tool_use/bench/cases.jsonl \
  --out reports/phase2_comparison.json

# O ejecutar el evaluador docente:
uv run --extra train python -m tool_use.evaluate \
  --bench tool_use/bench/cases.jsonl \
  --out reports/phase2_eval.json
```

Variables: `ARCA_TOOLS_MODEL_ID` (por defecto `Qwen/Qwen3-1.7B`),
`ARCA_TOOLS_BRAIN` (`base` o `rlm`), `ARCA_TOOLS_ADAPTER` / `ARCA_RLM_ADAPTER`,
`ARCA_IMV_REPORTS_DIR`. Para comparar el RLM con su base, utilizar **el mismo**
identificador que figura en `adapter_config.json`. La carga del modelo es diferida.
El modelo debe descargar los pesos con acceso a Hugging Face en el primer uso.

## Métricas e interpretación pendiente de ejecución GPU

El benchmark mide `tool_selection_accuracy` y `argument_accuracy`. La exactitud
de argumentos es un criterio de coincidencia *exacta* normalizada, y puede penalizar
JSON semánticamente equivalentes. En casos de informe se puntúa la **primera**
llamada (cálculo), y el flujo completo se mide aparte con tests multi-step.
Se registra también la traza de llamadas, las respuestas y los errores en el JSON
de comparación. No presentamos porcentajes sin haber ejecutado la inferencia.

En el informe de la práctica comparar: sin herramientas, con herramientas
en un solo turno, y con herramientas multi-step; documentar al menos cinco
fallos reales, especialmente omisión de relaciones, fechas, formato de llamadas,
parámetros BDNS y bloqueo de efectos secundarios.

## Limitaciones

El motor de `rlm/imv_engine.py` es un simulador de casos normalizados de 2026:
su exactitud en la batería de la fase 1 **no implica** elegibilidad administrativa
real. Los valores por defecto de algunos indicadores representan únicamente
los supuestos del benchmark y deben verificarse en consultas ciudadanas reales.
El adaptador RLM entrenado para salida numérica no garantiza buen tool calling:
hay que medirlo con la comparación incluida. El RAG normativo corresponde
a la fase 3, no se implementa aquí.
