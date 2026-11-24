# Cuaderno de experimentos — Fase 3 (RAG)

Cuaderno de la rama `feat/fase3-rag`; se integrará en `EXPERIMENTS.md` al hacer merge. Las
entradas van fechadas y cuentan qué se probó, qué se esperaba y qué pasó, también lo que salió
mal. **Estado a 2026-11-22: las cifras locales (una RTX 4070 portátil) están medidas; las de la
DGX están en cola y cada sección dice qué job las producirá.** Donde falta un dato hay un hueco
con el nombre del job, no un resultado inventado.

## 1. Corpus (2026-11-19)

**Qué.** Marco legal del Ingreso Mínimo Vital y del CAPI, 7 documentos, 605 803 caracteres
(`rag/corpus/`, licencias en `rag/corpus/README.md`): Ley 19/2021 (texto consolidado del BOE),
RD 789/2022 (compatibilidad con rentas del trabajo), RD 635/2022 (Consejo consultivo), RD 64/2022
(Comisión de seguimiento), Orden ISM/1375/2021 (Registro de Mediadores Sociales), RDL 20/2020
(texto original, **derogado**) y la página del IMV de la Seguridad Social con las cuantías de 2026.

**Por qué estos.** La ley y la página de la Seguridad Social dicen muchas cosas con las mismas
palabras, y el RDL derogado repite casi toda la ley. Es adrede: da distractores realistas
(¿recupera el retriever la norma vigente?) y fragmentos igual de válidos, que es justo lo que el
enunciado pide anotar.

**Cómo se descarga.** `rag/fetch_corpus.py` (reproducible): para cada bloque del BOE se queda con
la versión vigente el día de la descarga, renderiza las tablas fila a fila y pega los encabezados
de capítulo al artículo siguiente. La lógica es una copia de `consultar_boe` (fase 2) para que
`rag/` no dependa de ramas sin integrar. `MANIFEST.json` guarda fecha, SHA-256 y licencia.

**Qué pasó.**
- La descarga del 2026-11-19 ya incluye reformas recientes (Ley 1/2026, RD 240/2026): el BOE
  consolidado cambia con el tiempo, y con él los ids de fragmento. **El dorado depende del
  snapshot commiteado; no hay que volver a descargar sin rehacer el dorado.**
- La página de la Seguridad Social vive en un portal con mucho ruido (menús, cookies, widgets);
  el extractor se queda con `<main>` y filtra el resto.
- Ids de fragmento: `<doc_id>-<NNNN>`, con `doc_id` = nombre del fichero (`ley-19-2021-0137`).
  Legibles y estables mientras el corpus no cambie. Antes eran un hash de la ruta: un modelo de
  1,7 B copia mejor un id legible.

## 2. Conjunto dorado (2026-11-20)

**Diseño.** 50 preguntas en `rag/gold/gold.jsonl`. Cada una guarda una frase **literal** del
corpus como evidencia (no un id de fragmento): `rag/gold.py` la convierte en ids para cada
troceado, así el mismo conjunto puntúa el recursivo y el semántico sin reescribir nada. Una
evidencia cortada por el límite de un fragmento se resuelve a los fragmentos que contienen una de
sus mitades y se cuenta (columna «split ev.»).

- 36 formales y 14 en lenguaje coloquial («tengo un piso alquilado, ¿cómo cuentan esos
  ingresos?»); 13 categorías (económico 9, procedimiento 7, cuantía 5, CAPI 5, requisitos 4,
  convivencia 4, incentivo al empleo 4, sanciones 3, mediadores 3, órganos 2, acreditación 2,
  definición 1, obligaciones 1).
- **Fragmentos igual de válidos.** 15 preguntas tienen más de una evidencia (la ley y la página
  de la Seguridad Social, o la ley y el RD 789/2022). Con `mode = any` basta recuperar una; con
  `all` (una pregunta de dos saltos, g32) se mide la fracción recuperada. Un texto derogado
  (RDL 20/2020) nunca cuenta como válido, aunque diga lo mismo.
- **Trampas deliberadas.** g14 (renta garantizada de 2026: la ley fija 5 538 € de 2020, solo la
  Seguridad Social da 733,6 €/mes), g17 y g18 (CAPI: 115 € en 2026 frente a los 100 € del texto de
  la ley), g20 (límite de patrimonio de 2026, 26 409,60 €, frente a los 16 614 € del anexo II),
  g34 y g35 (dos órganos con normas casi idénticas y 46 frente a 24 vocalías).
- 8 preguntas que el corpus no puede responder (`rag/gold/unanswerable.jsonl`), para medir «no lo
  sé»; algunas son cercanas a propósito (subsidio por desempleo, ayudas de vivienda).

**Limitación importante, a revisar por el equipo.** Este conjunto lo redactó un agente de código
leyendo el corpus, no una persona del equipo, y las preguntas heredan su vocabulario. Es probable
que favorezca a BM25 más de lo que lo haría un usuario real y que las preguntas coloquiales sean
más fáciles de lo que parecen. Antes de la entrega, el equipo debería leerlo, tocar al menos una
parte de las preguntas y señalar cuáles considera ambiguas. Con 50 preguntas, además, una
diferencia de 2-3 puntos en Recall@5 es un par de preguntas: no hay que sacar conclusiones de
ella.

## 3. Troceado: recursivo frente a semántico

### 3.1 La distancia entre frases (2026-11-20, local)

El troceado semántico corta cuando la distancia coseno entre frases consecutivas supera un umbral.
La plantilla propone 0,25. Con Qwen3-Embedding-0.6B sobre este corpus (7 documentos, ≈3 000 pares
de frases) la distribución es otra: percentiles 5/25/50/75/95 = 0,22 / 0,43 / 0,57 / 0,69 / 0,80.
Con 0,25 casi cada frase se queda sola.

| umbral (máx. 1500 car.) | fragmentos | media (car.) |
|---|---|---|
| 0,25 | 2 800 | 215 |
| 0,55 | 1 695 | 356 |
| 0,65 | 1 146 | 527 |
| 0,70 | 912 | 663 |
| 0,80 | 541 | 1 119 |

Para tener fragmentos del tamaño del recursivo (≈560 car.) hay que subir el umbral a ≈0,65-0,70.
*Figura con el umbral dibujado:* pendiente, job `130_rag_distance_plots`
(`docs/fase3_figs/`).

### 3.2 Recuperación por troceado (2026-11-21/22, **local, parcial**)

50 preguntas, λ = 0,5, Recall@5 / MRR (profundidad 20):

| troceado | fragmentos | media (car.) | BM25 | denso | híbrido |
|---|---|---|---|---|---|
| recursivo 400/50 | 2 373 | 262 | 0,43 / 0,306 | 0,54 / 0,431 | 0,58 / 0,403 |
| recursivo 800/100 | 1 096 | 559 | 0,51 / 0,326 | 0,70 / 0,444 | 0,70 / 0,463 |
| recursivo 1200/150 | 707 | 868 | 0,55 / 0,376 | 0,66 / 0,487 | 0,70 / 0,491 |
| semántico 0,50 (máx. 1000) | 2 053 | 294 | 0,35 / 0,277 | 0,56 / 0,407 | 0,49 / 0,410 |
| semántico 0,65 (máx. 1000) | 1 347 | 449 | 0,49 / 0,380 | 0,67 / 0,484 | 0,67 / 0,501 |

(JSON de tres de estas filas en `docs/fase3_data/local/`; las filas de recursivo 400/50 y
semántico 0,65 se leyeron de la salida de la sesión y se volverán a medir en el job 110.)

**Lectura provisional.**
- Los fragmentos pequeños (≈260-300 car.) rinden peor con los tres retrievers: una frase aislada
  pierde el contexto (el encabezado del artículo, el «esto» de la frase anterior).
- El semántico a 0,65 da el mejor MRR del híbrido (0,501) pero no mejora el Recall@5 del recursivo
  de 800 o 1200. Es un empate práctico: con 50 preguntas no se distingue.
- El híbrido no supera al denso de forma clara. BM25 solo es el peor en todas las filas.
- Esperábamos que el semántico ganara claramente (agrupa por tema). No ha sido así; una hipótesis
  es que el texto legal ya está estructurado por artículos y el recursivo, que corta por párrafos,
  hereda esa estructura gratis. No está comprobado.

**Pendiente (job `110_rag_chunking_sweep`, DGX):** barrido completo de umbral (0,5-0,8) y de
tamaño máximo (1000/1500), y recursivo 1600/200. Aquí irá la tabla definitiva y la elección del
troceado por defecto (hoy `recursive:800:100`, provisional).

## 4. Recuperadores

**Implementados.** BM25 (`rank_bm25`, la plantilla); denso con Qwen3-Embedding-0.6B y un almacén
NumPy exacto persistido en `rag/index/` (fuera de git, con clave por modelo y texto de los
fragmentos: cambiar el corpus o el troceado nunca reutiliza un índice viejo); híbrido
`s = λ·s_bm25 + (1-λ)·s_denso` con min-max sobre el top-20 de cada lista. Se eligió un almacén
exacto en NumPy en vez de Chroma: con ≈1 000-2 000 vectores la búsqueda exacta es instantánea y
un índice aproximado solo añadiría piezas.

**Pendiente (job `120_rag_retrieval_ablations`, DGX):**
- barrido de λ de 0 a 1 en pasos de 0,1 sobre cuatro troceados, con validación cruzada de 5
  particiones (elegir λ en 4 y puntuar en la 5.ª) para estimar cuánto se infla el MRR al elegir λ
  con las mismas 50 preguntas con que se evalúa. **λ por defecto hoy: 0,5, provisional.**
- instrucción de la consulta (ninguna, la de plantilla, genérica, jurídica en inglés, jurídica
  en español) con los pasajes sin cambios.
- ablación del tokenizador de BM25 (minúsculas + `\w+` frente a acentos plegados + palabras vacías).
- análisis de las preguntas que cada recuperador falla, con ejemplos.

## 5. Respuestas con citas

**Diseño (`rag/generate.py`).** Plantilla de la clase en español con ids de fragmento; generador
Qwen3-1.7B (HF, bf16, voraz, sin razonamiento). Cuatro salvaguardas, todas contadas en
`pipeline.stats`:
1. *Puerta de «no lo sé»*: si el mejor coseno denso queda por debajo de `min_dense_score` se
   responde «No lo sé» sin llamar al modelo. El umbral se calibra con datos (job 135); hoy está a
   0 (puerta apagada) y la abstención depende solo del modelo.
2. Citas que no están entre los fragmentos recuperados se eliminan del texto.
3. Una respuesta que no se abstiene y no cita nada válido se reintenta una vez con un recordatorio.
4. Si aun así no cita nada, se sustituye por «No lo sé»: una respuesta sin fragmento detrás no
   es una respuesta.

Los tests de CPU (`tests/test_rag_generate.py`) cubren las cuatro con un generador simulado.

**Pendiente (jobs `135_rag_calibrate_gate` y `140_rag_generation_by_retriever`, DGX):** coseno de
las 50 preguntas con respuesta frente a las 8 sin ella y el umbral resultante; para hybrid, dense y
BM25: porcentaje de respuestas con fragmento relevante citado, falsas abstenciones, abstenciones
correctas, cifras de la respuesta que aparecen en los fragmentos citados y cuántas veces actuó
cada salvaguarda. La puerta se calibra y se evalúa con las mismas 8 negativas: la cifra que salga
será optimista y así se dirá.

## 6. Incidencias y decisiones de proceso

- **2026-11-21/22 — WSL inestable.** Las ejecuciones locales de más de 3-4 minutos se cortaban y
  el acceso a `\\wsl.localhost` desaparecía durante minutos. Se perdieron dos lanzamientos del
  barrido de troceados. Desde entonces: una ejecución por especificación y salida a disco, y los
  experimentos largos pasan a la DGX (`jobs/1xx_rag_*.sh`).
- **El runner de la DGX en esta rama** (`smoke/dgx_runner.sh`, commit `chore:`) es una copia del de
  `feat/night-improvements`; hay que quitarlo antes de hacer merge a `main`.
- **Generación en lote:** no hace falta vLLM; son 58 preguntas de ≤ 400 tokens por retriever.
- `matplotlib` no está en el extra `rag`: las gráficas necesitan también `--extra train` (los jobs
  lo hacen). Añadirlo a `pyproject.toml` queda fuera del alcance de esta rama.

## 7. Qué falta para dar la fase por cerrada

- Resultados de los jobs 100-140 y su lectura en las secciones 3 a 5.
- Elegir con datos troceado, λ y umbral de la puerta, y fijarlos en `rag/generate.py`.
- Que el equipo revise el conjunto dorado (sección 2).
- Probar `POST /rag` con la API levantada (hoy solo está probado el pipeline).
- Quitar el runner de la DGX y los jobs antes del merge a `main`, y pasar este cuaderno a
  `EXPERIMENTS.md`.
