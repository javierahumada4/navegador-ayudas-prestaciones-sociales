# Fase 2 — Herramientas y evaluación integral IMV/CAPI

**Endpoint docente:** POST /tools. **Rama:** feat/fase2-tool-use-imv-bdns-informes.
Motor determinista y ruleset IMV 2026 reutilizados de main, sin tocar la fase 1.

## Tres herramientas reales

| Nombre | Tipo | Trabajo |
| --- | --- | --- |
| calcular_imv(case: IMVCase) | Cálculo | Validar caso y calcular IMV/CAPI con JSON por fecha |
| consultar_ayudas(tema, max_resultados) | HTTP | Consultar convocatorias reales en la API BDNS |
| guardar_informe(contenido) | Efecto persistente | Escribir en TXT el texto íntegro compuesto por el LLM |

El modelo **redacta libremente** el informe. La herramienta guardar_informe solo
escribe el fichero con esos bytes; no se utiliza ninguna plantilla de texto.

El bucle manual (executor.py) soporta llamadas paralelas o encadenadas y
reinyecta observaciones de ejecución o errores de Pydantic. Si el usuario ha
solicitado explícitamente un informe:

1. El modelo debe calcular el IMV/CAPI y recibir la observación.
2. Se le recuerda que aún debe llamar a guardar_informe cuando intenta terminar
   antes. Si genera la redacción sin el marcador de tool-call, el orquestador
   puede enviar **ese texto exacto del modelo** a la herramienta, sin plantilla.
3. La herramienta solo se ejecuta si el texto cumple una comprobación mínima
   de importes. Se valida el permiso a partir de la solicitud expresa; **no hay
   interfaz de confirmación interactiva** en esta fase.
4. La respuesta confirma el guardado **solo tras un resultado ok de la tool**,
   usando la ruta real devuelta. Si no se pudo escribir, hay un error explícito:
   nunca se comunica un éxito inventado.
5. Los archivos viven en reports/imv/ o ARCA_IMV_REPORTS_DIR, con nombres
   aleatorios y permisos restrictivos. No se reciben rutas elegidas por el LLM
   ni se sobrescriben archivos. reports/imv/ está excluido de Git.

**Garantías y límites:** el motor simula casos normalizados (no resolución
administrativa). La comprobación previa de importes en el informe es numérica
y mínima; la auditoría de la precisión del lenguaje se realiza separadamente.
BDNS no engloba todas las prestaciones, no acredita elegibilidad ni plazos
abiertos. Endpoint: https://www.infosubvenciones.es/bdnstrans/doc/swagger.

Los rulesets se seleccionan por fecha desde rlm/rulesets/imv_*.json. Solo existe
el de 2026 en main; fechas sin reglas unívocas producen un error.

## Banco de 40 casos (bench/cases.jsonl)

| Categoría | Casos | Objetivo |
| --- | ---: | --- |
| structured_calculation | 4 | Conserva casos directos de JSON de la fase inicial |
| natural_calculation | 6 | Extraer fechas, miembros, relaciones e importes desde prosa |
| negative_eligibility | 6 | Ingresos, patrimonio, activos, residencia, empresa y edad |
| missing_facts | 5 | Pedir datos faltantes, no inventar y no guardar un fichero |
| bdns_natural | 8 | Filtrado HTTP en lenguaje natural, número de resultados |
| multi_step | 6 | IMV → informe, IMV + BDNS, IMV + BDNS → informe, 2 búsquedas |
| no_tool | 5 | Responder preguntas conceptuales sin invocar herramientas |

Los casos multi_step incluyen expected_tools; las búsquedas repetidas exigen
dos llamadas correctas. Para los cálculos existe expected_case (el JSON
normalizado de referencia) aunque la query sea texto natural. No es legítimo
tratar un caso con información desconocida como si sus valores fueran cero.

En el anterior benchmark de 30 casos casi todas las peticiones de cálculo
copiaban un JSON entero. Por tanto, **los porcentajes de los anteriores
phase2_base.json / phase2_grpo.json no son comparables** con los nuevos.

## Evaluación: primera llamada y flujo completo

El evaluador del docente (evaluate.py) se conserva sin cambios en sus métricas
de selección de la primera llamada y exactitud de sus argumentos.

compare.py añade mediante metrics.py, por caso y agregado:

- tool_sequence_success y multi_step_success: éxito de **todas** las herramientas
  exigidas (también si se repite la misma herramienta).
- all_tool_arguments_correct: compara argumentos de todas las acciones relevantes
  con el gold y no solo la primera llamada. La comparación es estricta y puede
  penalizar representaciones semánticamente equivalentes.
- amount_fidelity: exige los tres importes IMV/CAPI/total en la respuesta o
  en el informe escrito, cotejados con una ejecución independiente del motor.
  Soporta decimales con punto o coma y separadores de miles.
- report_created_and_verified: comprueba que un TXT existe de verdad dentro
  del directorio aislado, y que UTF-8, contenido, bytes y SHA-256 coinciden
  exactamente con el registro de guardar_informe.
- report_amount_fidelity: importes dentro del archivo efectivamente guardado.
- failed_reason_coverage y explanation_heuristic_accuracy: cobertura de los
  motivos de denegación, presencia de IMV/CAPI y aviso de simulación. Son
  **heurísticas léxicas**, no una certificación jurídica de las explicaciones.
- conceptual_fact_accuracy: presencia de expresiones correctas para las siglas
  IMV y CAPI cuando existe etiqueta gold; tampoco detecta negaciones por sí sola.
- clarification_accuracy: no ejecutar tools y solicitar la información que
  falta en solicitudes incompletas.
- unverified_save_claims: número de respuestas que afirman haber guardado algo
  sin constancia de ejecución satisfactoria.
- end_to_end_success: conjunto de llamadas correctas, argumentos esperados,
  verificación del archivo cuando procede, importes y comprobaciones de texto.

Cada métrica agrega su propio denominador (_n) y usa null cuando no aplica.
En particular, guardar_informe no se exige en casos sin solicitud de fichero.
El script escribe una fila por caso con la traza completa y el gold usado.

**Importante:** el directorio temporal de informes se elimina al finalizar
compare.py. Las comprobaciones de existencia se hacen durante la ejecución,
antes de borrarlo. Para auditar un TXT persistente, ejecutar POST /tools con
ARCA_IMV_REPORTS_DIR apuntando a un directorio privado.

## Comandos en la DGX

Desde el directorio raíz del proyecto:

```bash
git fetch origin
git switch feat/fase2-tool-use-imv-bdns-informes
git pull --ff-only
source smoke/dgx_env.sh
uv sync --extra train
uv run arca-check-gpu
uv run pytest -q tests/test_phase2_imv_tools.py tests/test_phase2_workflows.py tests/test_tool_registry.py tests/test_tool_parser.py
```

Smoke rápido (dos casos, sin GPU en los tests pero con inferencia en la comparación):

```bash
uv run --extra train python -m tool_use.compare --limit 2 \
  --model Qwen/Qwen3-1.7B --brain base --out reports/phase2_smoke.json
```

Evaluación completa del base, 40 casos **con** y **sin** herramientas:

```bash
uv run --extra train python -m tool_use.compare \
  --model Qwen/Qwen3-1.7B --brain base \
  --bench tool_use/bench/cases.jsonl --max-turns 5 \
  --out reports/phase2_base_v2.json
```

Evaluación del mismo modelo equipado con LoRA de la fase 1:

```bash
export ARCA_RLM_ADAPTER=/ruta/al/adaptador/grpo
export ARCA_TOOLS_MODEL_ID=$(uv run python -c \
  'import json,os,pathlib; print(json.loads((pathlib.Path(os.environ["ARCA_RLM_ADAPTER"])/"adapter_config.json").read_text())["base_model_name_or_path"])')
uv run --extra train python -m tool_use.compare \
  --model "$ARCA_TOOLS_MODEL_ID" --brain rlm --adapter "$ARCA_RLM_ADAPTER" \
  --bench tool_use/bench/cases.jsonl --max-turns 5 \
  --out reports/phase2_grpo_v2.json
```

Si el LoRA se entrenó sobre otro modelo base, ejecutar también la evaluación
base de esa misma arquitectura para comparar justamente.

## Qué analizar en la memoria de la práctica

Interpretar los falsos positivos del antiguo first-call accuracy, comparar
base frente a GRPO, señalar cinco fallos reales de extracción de argumentos,
residencia/ingresos, BDNS, nombres IMV/CAPI y redacción de informes. Probar
los fallos de red. Analizar las diferencias entre llamadas independientes,
paralelas y encadenadas. Ningún resultado de GPU se ha supuesto ni inventado;
los JSON v2 deben generarse nuevamente en la DGX.


## Evaluación por herramienta (aislamiento fase 2 → reutilización fase 4)

Para separar los errores de **extracción de datos**, **búsqueda HTTP** y
**redacción/persistencia**, el script `tool_use/stage_eval.py` evalúa **cuatro
contextos diferentes de conversación**, cada uno con su propio fichero JSONL.
No sustituye al evaluador docente, ni modifica `POST /tools` o `rlm/`.

| Etapa | Dataset | Casos | Únicas herramientas visibles |
| --- | --- | ---: | --- |
| IMV | `bench/imv.jsonl` | 21 | `calcular_imv` |
| BDNS | `bench/bdns.jsonl` | 10 | `consultar_ayudas` |
| Informe | `bench/report.jsonl` | 6 | `guardar_informe` |
| Integración | `bench/integration.jsonl` | 8 | Las tres |

**IMV:** consultas normalizadas y en español natural, denegaciones y
consultas incompletas. La herramienta recibe un único argumento `case`
con el esquema Pydantic, sin el contenedor extra `IMVCase`. El evaluador
compara el caso contra la referencia **después de aplicar los valores
por defecto del esquema a ambos**, distingue corrección de argumentos,
resultado del motor, fidelidad de la explicación y solicitud de aclaración.
Una falta de datos no autoriza a inventarlos; un error Pydantic se devuelve
como observación que el modelo puede corregir. Los errores recuperados
se registran en `repaired_validation_error` (presencia de error; el éxito
de recuperación se ve en las métricas de ejecución y argumentos).

**BDNS:** solo se expone `consultar_ayudas`, la herramienta ejecuta
**HTTP de verdad** y se puntúan por separado selección, argumentos,
éxito del HTTP y evidencia textual de las fuentes. Se comprueba si
la respuesta del modelo menciona resultados auténticamente devueltos,
en vez de conceder una prestación o inventar subvenciones. El resultado
depende de la disponibilidad de la API y su catálogo en el momento de
evaluar. Los tests de CPU usan una red simulada, **no** afirman conectar
realmente con la BDNS.

**Informe:** el usuario pide un archivo, y el modelo recibe dos
observaciones PREVIAS en la estructura de chat usada por el bucle Qwen:

```text
role=user: "Redacta y guarda un informe TXT..."
role=assistant: <tool_call>{"name":"calcular_imv",...}</tool_call>
role=tool (calcular_imv): {"ok":true,"result":{...cantidades...}}
role=assistant: <tool_call>{"name":"consultar_ayudas",...}</tool_call>
role=tool (consultar_ayudas): {"ok":true,"result":{...convocatorias...}}
role=assistant: ... SOLO puede llamar a guardar_informe ...
```

La observación IMV procede de ejecutar el motor real contra el caso de referencia,
**no** de introducir importes calculados por el modelo. La observación BDNS
usa, por defecto, una **fixture SINTÉTICA rotulada** (convocatoria
ficticia con código SIMULADA-001, respuesta vacía o error de red).
Esto permite comprobar que el modelo cita exclusivamente observaciones
recibidas, incluso ante fallos de BDNS, sin confundir datos ficticios con
convocatorias oficiales. Si se quiere comprobar el comportamiento con
la API real, pasar `--report-api live`. En ambos modos, el texto del
informe lo redacta el modelo: `guardar_informe` solamente lo escribe.
Se valida archivo, SHA-256, texto exacto, importes y referencias BDNS.
Las comprobaciones del texto son **heurísticas**, no auditoría jurídica.

**Integración:** se conservan los casos de dos búsquedas independientes,
cálculo y BDNS en un mismo turno, redacción en el turno siguiente y
solicitudes sin datos suficientes pero con búsqueda externa posible.
Se exige que el guardado espere a las observaciones requeridas.
Se conservan asimismo los 40 casos del evaluador global en
`bench/cases.jsonl` para la comparación con/sin herramientas del docente.

### Ejecutar en DGX

Desde la raíz del repositorio y en la rama `feat/fase2-tool-use-imv-bdns-informes`:

```bash
source smoke/dgx_env.sh
uv sync --extra train
uv run pytest -q tests/test_phase2_imv_tools.py \
  tests/test_phase2_workflows.py tests/test_phase2_stages.py \
  tests/test_tool_registry.py tests/test_tool_parser.py

# Prueba rápida de la etapa IMV (2 ejemplos, inferencia real)
uv run --extra train python -m tool_use.stage_eval \
  --stage imv --limit 2 --out reports/phase2_imv_smoke.json

# Cada etapa de forma independiente, modelo base
uv run --extra train python -m tool_use.stage_eval \
  --stage imv --brain base --out reports/phase2_imv_base.json
uv run --extra train python -m tool_use.stage_eval \
  --stage bdns --brain base --out reports/phase2_bdns_base.json
uv run --extra train python -m tool_use.stage_eval \
  --stage report --brain base --report-api fixture \
  --out reports/phase2_report_base.json
uv run --extra train python -m tool_use.stage_eval \
  --stage integration --brain base --out reports/phase2_integration_base.json
```

Alternativa: ejecutar las cuatro etapas en una sola invocación, reutilizando
los pesos cargados del modelo dentro del proceso:

```bash
uv run --extra train python -m tool_use.stage_eval \
  --stage all --brain base --model Qwen/Qwen3-1.7B \
  --out reports/phase2_stages_base.json
```

Para el adaptador LoRA de fase 1, el base seleccionado debe coincidir
exactamente con el identificador de `adapter_config.json`:

```bash
export ARCA_RLM_ADAPTER="/ruta/real/a/grpo_lora"
export ARCA_TOOLS_MODEL_ID="$(uv run python -c \
  'import json,os,pathlib; p=pathlib.Path(os.environ["ARCA_RLM_ADAPTER"])/"adapter_config.json"; print(json.loads(p.read_text())["base_model_name_or_path"])')"
uv run --extra train python -m tool_use.stage_eval \
  --stage all --brain rlm --adapter "$ARCA_RLM_ADAPTER" \
  --model "$ARCA_TOOLS_MODEL_ID" \
  --out reports/phase2_stages_grpo.json
```

**Limitaciones:** los resultados de BDNS en vivo pueden variar con la red;
las fixtures del informe son solo de evaluación. Los JSON de salida
contienen la traza, las métricas por etapa y sus denominadores;
los informes reales se verifican ANTES de que se borre el directorio
temporal aislado. Para comparar el base frente al GRPO científicamente,
usar el mismo modelo subyacente en ambos experimentos.
