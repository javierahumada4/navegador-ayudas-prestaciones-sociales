# Fase 2 — Resultados (evaluación v7)

Resultados de `python -m tool_use.compare` sobre el banco principal
(`tool_use/bench/cases.jsonl`, 40 casos), con `--max-turns 5`, decodificación greedy y
`max_new_tokens=1280`. Se ejecutó en la DGX el 18-11-2026 desde el commit `d98ac8b`; el
código de `tool_use/` de ese commit es idéntico al de esta rama.

| Fichero | Cerebro |
| --- | --- |
| `phase2_base_v7.json` | `Qwen/Qwen3-1.7B` sin adaptador (`--brain base`) |
| `phase2_grpo_v7.json` | El mismo modelo con el LoRA final de la fase 1 fusionado (`--brain rlm --adapter JES0406/imv-sft-lora/grpo_final`) |

Cada fichero contiene un `summary` y las 40 filas (`rows`), tanto `with_tools` como
`without_tools`. Cada fila guarda la consulta, las llamadas con sus argumentos, la
observación real de cada herramienta, la respuesta final y la puntuación del flujo. Las
trazas completas por turno (prompt, salida en bruto, parseo) se quedaron en la DGX porque
`reports/phase2_artifacts/` está excluido de Git. La BDNS se consultó en vivo.

## 1. Con y sin herramientas

En el modo sin herramientas el mismo modelo responde solo con texto. Es la comparación que
pide el enunciado.

| Métrica (n) | Base con herr. | Base sin herr. | GRPO con herr. | GRPO sin herr. |
| --- | ---: | ---: | ---: | ---: |
| Selección de la primera herramienta (40) | **0,950** | 0,250 | **0,950** | 0,250 |
| Argumentos de la primera llamada (40) | **0,850** | 0,250 | **0,850** | 0,250 |
| Secuencia de herramientas correcta (40) | **0,925** | 0,250 | 0,875 | 0,250 |
| Todos los argumentos correctos (30) | **0,800** | 0,000 | 0,733 | 0,000 |
| Varios pasos completos (6) | **0,833** | 0,000 | 0,500 | 0,000 |
| Importes IMV/CAPI/total exactos (21) | **0,857** | 0,190 | 0,714 | 0,095 |
| Informe creado y verificado en disco (4) | **0,750** | 0,000 | 0,500 | 0,000 |
| Explicación correcta, heurística (21) | **0,762** | 0,286 | 0,714 | 0,286 |
| Pide aclaración cuando faltan datos (5) | 0,400 | **0,600** | 0,400 | 0,400 |
| Hechos conceptuales (2) | 0,000 | 0,000 | 0,000 | 0,000 |
| **Éxito de principio a fin (40)** | **0,675** | 0,150 | 0,625 | 0,125 |
| Tiempo medio por caso | 9,1 s | 6,0 s | 7,0 s | 5,0 s |

Cómo leer la tabla:

- **Con herramientas el éxito de principio a fin sube de 0,15 a 0,675 en el modelo base.**
  Sin herramientas no hay forma de completar un cálculo, una búsqueda ni un informe. El 0,25
  de selección son los 10 casos que no necesitan herramienta, que se aciertan "gratis".
- **Sin herramientas el modelo inventa importes.** Solo acierta los tres importes en 4 de
  21 cálculos (0,19), y además son casos de denegación donde basta con decir 0,00 €. Esa es
  la justificación de `calcular_imv`: el motor determinista acierta siempre que recibe un
  caso bien formado.
- **Pedir aclaración sale algo mejor sin herramientas** (0,60 frente a 0,40). Sin
  herramientas, preguntar es lo único que puede hacer el modelo. Con ellas, a veces pregunta
  con palabras que la métrica no reconoce (ver §3).
- **`unverified_save_claims` = 0 en las cuatro configuraciones.** El modelo nunca dice haber
  guardado un informe cuando no lo hizo, según el detector. Hay un caso que se le escapa
  (§4, GRPO P2-013).

## 2. Por categoría (éxito de principio a fin)

| Categoría | Casos | Base con | Base sin | GRPO con | GRPO sin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cálculo estructurado | 4 | 4 | 0 | 4 | 0 |
| Cálculo en lenguaje natural | 6 | 4 | 0 | 3 | 0 |
| Denegaciones | 6 | 4 | 0 | 4 | 0 |
| Faltan datos | 5 | 2 | 3 | 2 | 2 |
| Búsqueda BDNS | 8 | 6 | 0 | 6 | 0 |
| Varios pasos | 6 | 4 | 0 | 3 | 0 |
| Sin herramienta | 5 | 3 | 3 | 3 | 3 |
| **Total** | **40** | **27** | **6** | **25** | **5** |

La selección de herramienta es casi perfecta en las dos configuraciones con herramientas
(38/40). Los fallos están en lo que viene después: los argumentos, la redacción y las
preguntas.

## 3. Análisis de los 13 fallos del modelo base

Hemos revisado los 13 casos uno a uno. Se reparten en tres grupos.

### 3.1 Fallos reales del modelo (8)

| Caso | Categoría | Qué hizo el modelo |
| --- | --- | --- |
| P2-005 | Lenguaje natural | Puso a la única persona del caso dos veces en `persons`. Pydantic lo rechazó (`duplicate person ids`) y el error volvió como observación, pero el modelo no corrigió la llamada: le pidió al usuario que corrigiera el id. La validación funciona; lo que falla es la recuperación. |
| P2-014 | Denegación | Vio `legal_residence_in_spain: false` y concluyó él mismo que no tenía derecho, **sin llamar al motor**. La conclusión es correcta, pero no da importes ni pasa por el oráculo, que es precisamente lo que el prompt prohíbe. |
| P2-015 | Denegación | Llamó al motor bien y dio los importes exactos, pero explicó la denegación con el código interno `ACTIVE_COMPANY_DIRECTOR`, sin decir con palabras que es por ser administrador de una sociedad. |
| P2-018 | Faltan datos | No calculó (bien), pero respondió con una lista genérica de campos del esquema («ID de la solicitante», «custodia exclusiva»…), sin preguntar por los ingresos ni por la convivencia, que eran los datos que faltaban. |
| P2-031 | Varios pasos | Recibió un caso JSON completo y la orden de guardar un informe, y no emitió ninguna llamada: respondió que no había cálculo y pidió más información. El orquestador no guardó nada y lo dijo (bien), pero el flujo no se completó. |
| P2-032 | Varios pasos | Calculó, redactó y guardó el informe con los importes exactos, pero **se inventó un umbral legal**: «superior al límite de 20,000 € requerido para el IMV». Ese límite no sale del motor. Tampoco avisa de que es una simulación. |
| P2-036 | Sin herramienta | Definió el IMV como «Ingreso Mensual de Vida», vinculado a un inexistente «Programa de Ayudas a la Vida» para mayores de 65 años. Es una alucinación completa. |
| P2-037 | Sin herramienta | Definió el CAPI como «Cálculo de Apoyos para la Vida y la Independencia». En realidad es el Complemento de Ayuda para la Infancia. |

### 3.2 Falsos negativos de la métrica (4)

| Caso | Por qué la métrica lo cuenta como fallo |
| --- | --- |
| P2-025 | El modelo buscó `tema="rehabilitación de vivienda"` y se esperaba `"rehabilitación vivienda"`. La API devolvió convocatorias pertinentes. La comparación de argumentos es exacta tras normalizar, y una preposición basta para fallar. |
| P2-027 | Igual: `"subvenciones para empleo juvenil"` frente a `"empleo juvenil"`. |
| P2-019 | Respondió «¿Puedes proporcionar el importe de la pensión y los activos…?», justo los dos datos que faltaban. La métrica exige un verbo de petición de una lista cerrada (`necesito`, `falta`, `indica`, `dime`…) y «proporcionar» no está. |
| P2-021 | Respondió «¿Cuál es la edad exacta de los menores a cargo? ¿Cuál es el patrimonio familiar?». Es la aclaración correcta, pero no contiene ninguna palabra de esa lista. |

### 3.3 Caso ambiguo del banco (1)

- **P2-008:** los importes son exactos y la explicación es correcta. Falla
  `all_tool_arguments_correct` porque el modelo puso `independent_from_parents_since:
  2020-01-01` y la referencia dice `2010-01-01`. **Esa fecha no aparece en la consulta**:
  la referencia tiene un dato que el usuario no dio. Habría que corregir el caso del banco
  o que el modelo preguntara.

**Sensibilidad.** Si los 4 falsos negativos y el caso ambiguo se contaran como aciertos, el
modelo base llegaría a 32/40 = 0,80 con herramientas. **El número oficial sigue siendo
0,675.** No hemos tocado las métricas a posteriori; este recuento solo sirve para saber
cuánto del error es del modelo y cuánto del evaluador.

## 4. Por qué el LoRA de GRPO es peor como cerebro de herramientas

El LoRA de la fase 1 acierta la herramienta igual que el modelo base (0,95), pero completa
menos flujos: el éxito de principio a fin baja de 0,675 a 0,625 y los varios pasos de 0,83 a
0,50. Falla los casos P2-009, P2-013, P2-017, P2-033 y P2-034, que el modelo base resolvía.
En los cinco se reconoce el estilo con el que se entrenó en la fase 1:

- **Respuestas en formato de la fase 1.** En P2-033 la respuesta final es un JSON
  `{"imv": 865.07, "capi": 57.5, "total": 922.57}`, como el `<answer>` del entrenamiento.
  Además se salta la búsqueda en la BDNS que pedía el usuario.
- **Omite importes.** En P2-009 da solo el IMV (316,93 €/mes) y no da el CAPI ni el total.
  En P2-013 dice que «el importe del IMV y CAPI es cero» sin dar los tres importes.
- **Repite la instrucción en vez de cumplirla.** En P2-017 responde literalmente «Pregunta
  al usuario por la fecha de obtención de residencia y el patrimonio.».
- **No espera a la BDNS.** En P2-034 intenta guardar el informe dos veces antes de llamar a
  `consultar_ayudas`, y el orquestador lo rechaza las dos veces («Wait for the BDNS tool
  observation…»). Es la protección funcionando, no un fallo de ejecución.
- **Afirmación de guardado que no se detectó.** En P2-013 dice «El informe se ha generado
  con los datos proporcionados» sin haber llamado a `guardar_informe`. Ni el filtro de
  `executor._finish` ni la métrica lo detectan, porque las dos expresiones regulares buscan
  «se ha generado … informe» y aquí el orden es «informe se ha generado».

Interpretación: el GRPO de la fase 1 optimizó cálculos sueltos con respuesta en
`<answer>`, con la hoja de reglas en el prompt y sin herramientas. Esa especialización no se
traslada a un bucle de herramientas, y en parte lo estorba. Para la fase 4 esto justifica
usar el modelo base como cerebro de herramientas y dejar el LoRA para el razonamiento IMV.

## 5. Limitaciones

- **Muestras pequeñas** en varias métricas: 4 informes, 5 aclaraciones, 6 casos de varios
  pasos y 2 hechos conceptuales. Una diferencia de un caso mueve la métrica entre 0,17 y
  0,25.
- **La explicación se mide con heurísticas** de palabras clave y contradicciones
  (`metrics.explanation_checks`), no con una revisión jurídica. P2-032 muestra que un
  informe puede tener los importes bien y aun así afirmar algo legalmente falso.
- **La BDNS es un servicio en vivo**: repetir la evaluación puede dar otros resultados de
  búsqueda y, si la red falla, otro éxito.
- **Una sola ejecución con decodificación greedy**, sin semillas ni intervalos de confianza.
- **El motor es el de la fase 1**: cubre los casos del dataset sintético, no todas las
  excepciones del Real Decreto-ley 20/2020 ni de la Ley 19/2021.

## 6. Qué cambiaríamos

1. **Comparar `tema` sin preposiciones ni palabras vacías** en `evaluate.py`, y ampliar los
   verbos de petición de `metrics.py` (`puedes`, `proporciona`, `cuál es`). Publicaríamos
   las dos versiones de la métrica.
2. **Corregir P2-008**: añadir la fecha de independencia a la consulta o quitarla de la
   referencia.
3. **Añadir al prompt de sistema la definición de IMV y CAPI**, en una línea, para eliminar
   las alucinaciones de P2-036 y P2-037.
4. **Recuperación tras un error de validación**: con decodificación greedy, el modelo repite
   o abandona. La rama `eval/pipeline-fixes` reintenta una vez con muestreo; habría que
   medir su efecto con esta misma batería.
5. **Arreglar la expresión regular de afirmaciones de guardado** para que acepte también el
   orden «informe … se ha generado».
