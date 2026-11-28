# Fase 2 — Herramientas

El modelo de lenguaje no calcula prestaciones ni consulta bases de datos: genera llamadas a
herramientas. La aplicación las detecta, las valida, las ejecuta y le devuelve el resultado.
Es el bucle de cinco pasos, escrito a mano y sin frameworks de agentes:

```
prompt → generación → parseo de <tool_call> → ejecución validada → observación → …
```

Endpoint: `POST /tools`. Las salidas completas de la evaluación están en
[`results/fase2/`](../results/fase2/).

## Las tres herramientas

Están en [`domain_tools.py`](domain_tools.py) y se registran en
`tools.default_registry()`.

| Herramienta | Tipo | Qué hace |
|---|---|---|
| `calcular_imv(case)` | Cálculo | Valida el caso con Pydantic y ejecuta el motor de la fase 1 con el ruleset que corresponde a la fecha de solicitud. Devuelve IMV, CAPI, total, renta garantizada, miembros de la unidad y requisitos incumplidos |
| `consultar_ayudas(tema, max_resultados)` | API HTTP externa | Busca convocatorias reales en la API pública de la [BDNS](https://www.infosubvenciones.es/bdnstrans/doc/swagger). Devuelve código, título, fecha, administración, organismo y enlace de cada una, con un aviso: encontrar una convocatoria no implica que esté abierta ni que el usuario cumpla los requisitos |
| `guardar_informe(contenido)` | Acción | Escribe en un `.txt` el informe que ha redactado el modelo. Devuelve ruta, bytes y SHA-256 |

**Por qué estas tres:**

- **El cálculo no puede quedar en manos del modelo.** En la fase 1, ni el mejor modelo pasa
  de 0,49.
- **La BDNS cubre lo que el motor no sabe:** qué otras ayudas públicas hay para el caso
  (alquiler, discapacidad, familia numerosa…).
- **El informe es el entregable que se lleva el usuario.**

**El contrato de `calcular_imv`** está en [`imv_schema.py`](imv_schema.py):

- no admite campos desconocidos y los booleanos son estrictos;
- los porcentajes, grados y cantidades tienen límites;
- comprueba la coherencia del caso: ids únicos, que el solicitante esté en el domicilio y
  fechas no posteriores a la solicitud;
- si una persona aparece dos veces con exactamente los mismos datos, se queda una sola
  copia. Qwen3-1.7B lo hace con casi todas las personas que viven solas. En cambio, el
  mismo id con datos distintos sigue siendo un error, porque no se sabe cuáles son los
  ciertos.

Si el modelo se equivoca, el error de validación vuelve como observación y puede corregirse.

**Garantías de `guardar_informe`:**

- el modelo no elige la ruta: el nombre del fichero es aleatorio, dentro de `reports/imv/` o
  de `ARCA_IMV_REPORTS_DIR`;
- nunca sobrescribe un fichero y lo crea con permisos 600;
- la carpeta está excluida de Git, porque los informes contienen datos personales.

## El bucle

[`executor.py`](executor.py) implementa `ToolLoop`. Los esquemas de las herramientas entran
por la plantilla de chat de Qwen3 (`tools=`), y el parser ([`parser.py`](parser.py))
reconoce varias llamadas en una misma generación.

Varias llamadas en un mismo turno se ejecutan como independientes, en paralelo. Las que
dependen de un resultado anterior se encadenan en turnos sucesivos, como calcular y después
guardar el informe.

**El prompt de sistema:**

- define IMV y CAPI;
- prohíbe inventar datos: si falta uno obligatorio, el modelo debe preguntar;
- exige dar siempre los tres importes del motor y aclarar que es una simulación;
- incluye un ejemplo completo de llamada a `calcular_imv`.

**El parser tolera dos errores habituales de Qwen3-1.7B:** saltos de línea dentro de una
cadena (frecuentes en informes largos) y una llave de cierre de más al final de una llamada
anidada. Cualquier otro texto sobrante es un error, y el mensaje indica la posición.

**Las comprobaciones del orquestador no dependen del modelo:**

- **Solo se guarda un informe si el usuario lo pidió expresamente,** el texto contiene los
  importes de IMV, CAPI y total que dio el motor y, si se pidió una búsqueda, la BDNS ya ha
  respondido.
- **Si el modelo redacta el informe pero olvida la etiqueta `<tool_call>`,** el orquestador
  envía ese mismo texto a `guardar_informe`, tras las mismas comprobaciones y sin plantillas.
- **«Guardado» solo se dice con la ruta real que devolvió la herramienta.** Si el modelo
  afirma haber guardado algo que no existe, su respuesta se sustituye por una aclaración.
- **Si el modelo repite exactamente una salida que ya falló,** el bucle termina en lugar de
  agotar los turnos.

**Cerebro.** `ARCA_TOOLS_BRAIN=base` usa Qwen3-1.7B, que es el valor por defecto.
`ARCA_TOOLS_BRAIN=rlm` usa el LoRA de la fase 1 fusionado con el modelo, indicado en
`ARCA_TOOLS_ADAPTER` o `ARCA_RLM_ADAPTER`.

## El banco de pruebas

[`bench/cases.jsonl`](bench/cases.jsonl) tiene 40 casos. Cada uno indica la herramienta y los
argumentos esperados; en los casos de varios pasos, la secuencia completa.

| Categoría | Casos | Qué se comprueba |
|---|---:|---|
| Cálculo estructurado | 4 | Caso ya en JSON |
| Cálculo en lenguaje natural | 6 | Extraer fechas, miembros, relaciones e importes de un texto |
| Denegaciones | 6 | Ingresos, patrimonio, activos, residencia, sociedad y edad |
| Faltan datos | 5 | Preguntar por lo que falta, sin llamar a nada |
| Búsqueda BDNS | 8 | Tema y número de resultados a partir de la petición |
| Varios pasos | 6 | Cálculo → informe, cálculo + BDNS, cálculo + BDNS → informe, dos búsquedas |
| Sin herramienta | 5 | Preguntas conceptuales |

Además hay un banco por herramienta: [`imv.jsonl`](bench/imv.jsonl),
[`bdns.jsonl`](bench/bdns.jsonl), [`report.jsonl`](bench/report.jsonl) e
[`integration.jsonl`](bench/integration.jsonl). Los usa [`stage_eval.py`](stage_eval.py)
para evaluar cada herramienta por separado, con el mismo contexto que tendrá en la fase 4.

En la etapa de informe, las observaciones de la BDNS son ficticias y van marcadas como tales:
con resultados, vacías o con error de red. Así se mide la redacción sin depender de la red.

## Las métricas

[`evaluate.py`](evaluate.py) mide la primera llamada:

- **Selección:** la herramienta correcta, o ninguna cuando no hace falta.
- **Argumentos:** coinciden con los esperados tras normalizarlos. Para los campos de texto
  libre de búsqueda (`tema`), lo que cuenta es el conjunto de palabras con contenido: da
  igual mayúsculas, acentos, orden, preposiciones o un «subvenciones» redundante.
  «rehabilitación de vivienda» equivale a «rehabilitación vivienda», pero no a «vivienda».

[`metrics.py`](metrics.py) puntúa el flujo completo:

| Métrica | Qué comprueba |
|---|---|
| Secuencia | Se ejecutaron con éxito todas las herramientas necesarias, en orden válido, y ninguna sobrante |
| Argumentos de todas las llamadas | Todas las llamadas llevan los argumentos esperados |
| Importes | Los tres importes de la respuesta o del informe coinciden con un cálculo independiente del motor |
| Informe verificado | El fichero existe y su tamaño, SHA-256 y texto coinciden con la llamada |
| Explicación | Heurística, no revisión jurídica: nombra IMV y CAPI, aclara que es una simulación, cubre cada requisito incumplido y no aprueba a quien no tiene derecho |
| Aclaración | No llama a nada, pide datos (con una pregunta o un verbo de petición) y nombra los que faltan |
| Hechos conceptuales | La definición contiene el nombre correcto de la prestación |
| Afirmaciones de guardado sin verificar | Cuántas veces dice haber guardado algo que no existe |
| **Éxito de principio a fin** | Todas las anteriores que apliquen al caso |

## Resultados

[`compare.py`](compare.py) ejecuta el banco con y sin herramientas: el mismo modelo, que sin
ellas solo puede responder con texto.

**Configuración:** decodificación greedy, hasta 5 turnos y 1280 tokens por generación. La
BDNS se consulta en vivo.

**Ficheros:**
[`phase2_base.json`](../results/fase2/phase2_base.json) y
[`phase2_grpo.json`](../results/fase2/phase2_grpo.json) guardan, para cada caso, la consulta,
las llamadas con sus argumentos, la observación real, la respuesta y la puntuación. Las
trazas por turno se quedaron en la máquina de evaluación.

| Métrica (casos) | Base con herr. | Base sin herr. | GRPO con herr. | GRPO sin herr. |
|---|---:|---:|---:|---:|
| Selección de herramienta (40) | 0,950 | 0,250 | **0,975** | 0,250 |
| Argumentos de la primera llamada (40) | 0,900 | 0,250 | **0,925** | 0,250 |
| Secuencia correcta (40) | **0,900** | 0,250 | **0,900** | 0,250 |
| Argumentos de todas las llamadas (30) | **0,833** | 0,000 | **0,833** | 0,000 |
| Varios pasos (6) | **0,667** | 0,000 | 0,500 | 0,000 |
| Importes exactos (21) | **0,810** | 0,190 | 0,619 | 0,095 |
| Informe verificado en disco (4) | **0,500** | 0,000 | **0,500** | 0,000 |
| Explicación (21) | **0,714** | 0,286 | 0,571 | 0,286 |
| Aclaración cuando faltan datos (5) | **0,800** | 0,600 | 0,400 | 0,400 |
| Hechos conceptuales (2) | **1,000** | 0,000 | **1,000** | 0,000 |
| **Éxito de principio a fin (40)** | **0,750** | 0,150 | 0,650 | 0,125 |
| Tiempo medio por caso | 6,9 s | 5,9 s | 6,7 s | 4,9 s |

**Por categoría** (éxito de principio a fin, en número de casos):

| Categoría | Casos | Base con | Base sin | GRPO con | GRPO sin |
|---|---:|---:|---:|---:|---:|
| Cálculo estructurado | 4 | 4 | 0 | 4 | 0 |
| Cálculo en lenguaje natural | 6 | 2 | 0 | 2 | 0 |
| Denegaciones | 6 | 5 | 0 | 3 | 0 |
| Faltan datos | 5 | 4 | 3 | 2 | 2 |
| Búsqueda BDNS | 8 | 7 | 0 | 8 | 0 |
| Varios pasos | 6 | 3 | 0 | 2 | 0 |
| Sin herramienta | 5 | 5 | 3 | 5 | 3 |

**Lo que dicen los números:**

- **Las herramientas multiplican por cinco el éxito** (de 0,150 a 0,750 con el modelo base).
  Sin ellas no se puede completar ningún cálculo, búsqueda ni informe. El 0,25 de selección
  sin herramientas son los 10 casos que no las necesitan.
- **Sin el motor, el modelo inventa cuantías.** Solo acierta los tres importes en 4 de 21
  cálculos, y los cuatro son denegaciones donde basta con decir 0,00 €.
- **El modelo casi siempre elige bien la herramienta (38 de 40).** Los errores están en lo que
  viene después: completar el caso, recuperarse de un error y redactar.
- **Nunca afirma haber guardado un informe que no existe,** según el detector de la métrica.

## Revisión de las métricas

Al revisar los fallos caso a caso encontramos tres problemas del evaluador, no del modelo:

1. **Los argumentos de búsqueda se comparaban como cadenas exactas.** Pedir «rehabilitación
   de vivienda» en lugar de «rehabilitación vivienda» contaba como error, aunque la BDNS
   devolvía las mismas convocatorias. Ahora se compara el conjunto de palabras con contenido.
2. **La aclaración solo se reconocía con una lista cerrada de verbos.** «¿Puedes proporcionar
   el importe de la pensión y los activos?» o «¿Cuál es la edad exacta de los menores?» no
   contaban como pregunta. Ahora también valen el signo de interrogación, «puedes» y las
   formas de petición de «proporcionar». «Proporcionado» no cuenta: en un caso real, el
   modelo decía «no has proporcionado información» y no preguntaba nada.
3. **La referencia del caso P2-008 tenía un dato que la consulta no da**
   (`independent_from_parents_since: 2010-01-01`). Lo hemos quitado de `cases.jsonl` y de
   `imv.jsonl`.

La corrección se hizo sobre la evaluación anterior, la que no tenía la definición de IMV y
CAPI en el prompt. Sus salidas están en
[`results/fase2/sin_definicion/`](../results/fase2/sin_definicion/).

[`rescore.py`](rescore.py) recalcula las métricas sobre salidas guardadas, sin volver a
ejecutar el modelo. Para comprobar que es fiable, lo ejecutamos primero con las métricas
anteriores: reproduce exactamente las cuatro tablas originales. Con las métricas nuevas
cambian estos casos:

| Métrica | Base con herr. | GRPO con herr. | Casos que cambian |
|---|---|---|---|
| Argumentos de la primera llamada | 0,850 → 0,900 | 0,850 → 0,900 | P2-025, P2-027 |
| Argumentos de todas las llamadas | 0,800 → 0,867 | 0,733 → 0,800 | P2-025, P2-027 |
| Aclaración | 0,400 → 0,800 | 0,400 → 0,600 | Base: P2-019, P2-021. GRPO: P2-021 |
| **Éxito de principio a fin** | **0,675 → 0,775** | **0,625 → 0,700** | |

Sin herramientas no cambia nada. P2-008 sigue fallando con la referencia corregida, porque el
modelo se inventa la fecha de independencia. Comparativa completa en
[`phase2_rescored.json`](../results/fase2/sin_definicion/phase2_rescored.json).

## Efecto de definir IMV y CAPI en el prompt

En la evaluación anterior, el modelo se inventaba qué significan las siglas («Ingreso Mensual
de Vida», «Cálculo de Apoyos para la Vida y la Independencia»). Añadimos al prompt de sistema
una definición de una línea de cada prestación y repetimos la evaluación. Las dos se
comparan con las mismas métricas, las corregidas:

| | Sin definición | Con definición |
|---|---:|---:|
| Base, éxito de principio a fin | 0,775 (31/40) | 0,750 (30/40) |
| Base, hechos conceptuales | 0 / 2 | **2 / 2** |
| GRPO, éxito de principio a fin | 0,700 (28/40) | 0,650 (26/40) |
| GRPO, hechos conceptuales | 0 / 2 | **2 / 2** |

**El cambio resuelve lo que buscaba.** Las dos preguntas conceptuales pasan en los dos
modelos, y el IMV se define correctamente como «una prestación no contributiva de la
Seguridad Social regulada por la Ley 19/2021».

**Pero mueve otros casos.** Con decodificación greedy, una línea más en el prompt cambia la
generación de casos que no tienen nada que ver con las siglas:

| Modelo | Mejoran | Empeoran |
|---|---|---|
| Base | P2-014 (ahora sí llama al motor), P2-036, P2-037 | P2-009, P2-010, P2-024, P2-034 |
| GRPO | P2-014, P2-018, P2-036, P2-037 | P2-010, P2-012, P2-016, P2-020, P2-021, P2-032 |

Los casos que empeoran repiten tipos de error que ya aparecían en otros casos de la
evaluación anterior: no dar los tres importes, no explicar el requisito incumplido, no esperar
a la BDNS. No son errores nuevos que introduzca la definición. Mantenemos el cambio porque
corrige un error sistemático (alucinar qué es la prestación) a cambio de una variación de
pocos casos, del orden de lo que mueve cualquier cambio de prompt en un modelo de 1,7B.

## Los fallos del modelo base

Quedan 10 fallos, todos atribuibles al modelo:

| Caso | Categoría | Qué hace |
|---|---|---|
| P2-005 | Lenguaje natural | Pone a la única persona del caso dos veces en `persons`. La validación lo rechaza (`duplicate person ids`) y el modelo no corrige la llamada: le pide al usuario que corrija el id. Al probar la API vimos que pasa con casi todas las personas que viven solas, así que `calcular_imv` ahora se queda con una copia cuando los datos son idénticos. Repetida con ese cambio, esta misma llamada da el importe correcto (383,60 €). La tabla es anterior al cambio |
| P2-008 | Lenguaje natural | Importes correctos, pero se inventa una fecha de independencia que el usuario no dio |
| P2-009 | Lenguaje natural | Da el IMV y el total, pero en lugar de «CAPI 0,00 €» escribe «el CAPI no se aplica» |
| P2-010 | Lenguaje natural | Da bien los tres importes (IMV 0, CAPI 57,50), pero no explica por qué no hay IMV |
| P2-015 | Denegación | Llama al motor y da los importes, pero explica la denegación con el código interno `ACTIVE_COMPANY_DIRECTOR` en lugar de decir que administra una sociedad |
| P2-018 | Faltan datos | Pregunta, pero por la fecha de nacimiento, la residencia y los hijos; no por los ingresos, que es uno de los dos datos que faltaban |
| P2-024 | BDNS | Escribe una llamada correcta, pero no cierra la etiqueta `</tool_call>`, y el parser la rechaza como incompleta |
| P2-031 | Varios pasos | Recibe un caso JSON completo y la orden de guardar un informe, y no emite ninguna llamada. El orquestador no guarda nada y lo dice |
| P2-032 | Varios pasos | Calcula y guarda el informe con los importes exactos, pero **se inventa un umbral legal**: «excede el límite de 20,000 € establecido». Tampoco avisa de que es una simulación |
| P2-034 | Varios pasos | Intenta guardar el informe antes de buscar en la BDNS. El orquestador lo rechaza y no se guarda nada |

**Lo que tienen en común:**

- **Cuando le falta información, la rellena él:** el umbral de P2-032 y la fecha de P2-008.
- **Se salta pasos que conoce** (P2-031, P2-034).
- **Ante un error de validación pide ayuda en lugar de corregirse** (P2-005).
- **Las explicaciones se quedan cortas en las denegaciones** (P2-010, P2-015).

## Base frente al LoRA de la fase 1

El LoRA de GRPO elige la herramienta igual o mejor que el modelo base (0,975 frente a 0,950),
pero completa menos flujos: varios pasos 0,50 frente a 0,67, importes 0,62 frente a 0,81 y
explicaciones 0,57 frente a 0,71. En sus fallos se reconoce el entrenamiento de la fase 1,
que pedía un único número dentro de `<answer>`:

- **Da un solo número.** En P2-009 da solo el IMV (316,93 €), sin los importes de CAPI y
  total.
- **Omite los importes en las denegaciones.** En P2-012 y P2-016 explica bien por qué no hay
  derecho, pero no escribe «0,00 €».
- **Repite la instrucción en lugar de cumplirla.** En P2-017 y P2-021 contesta «Pregunta al
  usuario por…» o «Pregunta por…» en lugar de preguntar.
- **Se salta la BDNS.** En P2-033 no hace la búsqueda que se le pide; en P2-034 intenta
  guardar el informe dos veces sin haber buscado, y el orquestador lo rechaza las dos veces.

GRPO optimizó cálculos sueltos con respuesta en `<answer>`, sin herramientas, y esa
especialización estorba en un bucle de herramientas. Por eso `/tools` usa el modelo base.

## Limitaciones

- **Muestras pequeñas:** 4 informes, 5 aclaraciones, 6 casos de varios pasos y 2 preguntas
  conceptuales. Un caso mueve esas métricas entre 0,17 y 0,50.
- **Variación entre ejecuciones.** Un cambio de una línea en el prompt movió 7 casos del
  modelo base en un sentido u otro. Diferencias de 2 o 3 casos entre configuraciones no son
  concluyentes con una sola ejecución.
- **La explicación se mide con heurísticas,** no con revisión jurídica. P2-032 muestra que un
  informe puede tener los importes bien y afirmar algo falso.
- **La BDNS es un servicio en vivo.** Repetir la evaluación puede dar otras convocatorias y,
  si la red falla, otro resultado.
- **Detección de guardados falsos incompleta.** La expresión regular busca «se ha generado …
  informe», pero no el orden «informe … se ha generado». En la evaluación anterior, el LoRA
  dijo «El informe se ha generado» sin haberlo guardado (P2-013), y no lo detectó ni el
  orquestador ni la métrica.

## Cómo ejecutarlo

Desde la raíz del repositorio. Las evaluaciones necesitan GPU y `uv sync --extra train`.

```bash
# Con y sin herramientas, sobre el banco de 40 casos
uv run --extra train python -m tool_use.compare --brain base --out reports/phase2_base.json
uv run --extra train python -m tool_use.compare --brain rlm \
    --adapter JES0406/imv-sft-lora/grpo_final --out reports/phase2_grpo.json

# Cada herramienta por separado (--stage imv | bdns | report | integration | all)
uv run --extra train python -m tool_use.stage_eval --stage all --out reports/phase2_stages.json

# Recalcular las métricas de unas salidas ya guardadas (sin GPU)
uv run python -m tool_use.rescore results/fase2/sin_definicion/phase2_base.json \
    results/fase2/sin_definicion/phase2_grpo.json \
    --out results/fase2/sin_definicion/phase2_rescored.json

# Tests sin GPU ni red
uv run pytest tests/test_phase2_*.py tests/test_tool_*.py
```

**Trazas completas.** Cada evaluación guarda en `reports/phase2_artifacts/` un JSONL por caso
con el prompt, la salida en bruto, el parseo, las llamadas, las observaciones y la respuesta.
Cada evento se escribe en disco al momento. Para registrar también las peticiones reales a
`/tools`, se define `ARCA_TOOLS_TRACE_DIR`.

Ninguna de las dos carpetas se sube a Git: contienen los datos personales de los casos.

## Ficheros

| Fichero | Qué hace |
|---|---|
| `domain_tools.py`, `imv_schema.py` | Las tres herramientas y el contrato Pydantic del caso |
| `tools.py`, `registry.py` | Registro de herramientas. `get_weather` y `calculator` son los ejemplos de la plantilla, que usan los tests y las tareas de ejemplo de la fase 4 |
| `parser.py`, `executor.py` | Parseo de `<tool_call>` y bucle de cinco pasos con sus comprobaciones |
| `evaluate.py`, `metrics.py` | Métricas de primera llamada y de flujo completo |
| `compare.py`, `stage_eval.py`, `rescore.py` | Evaluación con y sin herramientas, por herramienta y recálculo de métricas |
| `tracing.py` | Trazas por caso |
| `bench/` | Bancos de pruebas |
