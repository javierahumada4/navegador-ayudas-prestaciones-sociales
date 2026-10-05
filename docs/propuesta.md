# Propuesta de tema — Navegador del Ingreso Mínimo Vital (IMV + CAPI)

## Equipo

- Javier Escobar Serrano — javierescobarserrano@gmail.com — destilación y SFT.
- Javier Ahumada — 202213736@alu.comillas.edu — motor de reglas, generación de datos y verificación.
- Marcos — 202204116@alu.comillas.edu — GRPO y evaluación.

## El tema en una frase

Un agente especializado en el Ingreso Mínimo Vital (IMV) y el Complemento de Ayuda para la Infancia (CAPI) que ayuda a interpretar un caso familiar, estimar de forma reproducible las prestaciones que podrían corresponder, explicar los requisitos relevantes y fundamentar la respuesta con normativa y documentación oficial.

## El usuario y su problema

El proyecto está pensado principalmente para dos perfiles: profesionales de servicios sociales que necesitan hacer una primera orientación sobre un caso y personas que quieren entender si su situación encaja en el IMV o el CAPI antes de iniciar una solicitud.

El problema no consiste únicamente en encontrar información sobre la prestación. La cuantía y la elegibilidad dependen de varias reglas que interactúan entre sí: composición de la unidad de convivencia, edad de sus miembros, residencia legal y efectiva, independencia en beneficiarios individuales, ingresos, patrimonio, activos, discapacidad, monoparentalidad, determinadas incompatibilidades y los distintos umbrales aplicables al IMV y al CAPI.

Además, algunas condiciones tienen detalles fáciles de aplicar mal: desigualdades estrictas o inclusivas en los límites económicos, edades calculadas en fechas distintas, complementos porcentuales, topes por pensiones y redondeos monetarios.

Actualmente, para resolver un caso es necesario combinar información de la Seguridad Social, la Ley 19/2021 y cálculos propios. Esto hace difícil ofrecer una respuesta explicable y consistente, especialmente en situaciones próximas a los límites.

El objetivo final del agente no es sustituir a la Seguridad Social ni emitir una resolución administrativa, sino servir como herramienta de orientación: calcular sobre los supuestos que cubre el sistema, explicar por qué se obtiene un resultado y señalar la normativa o documentación relevante.

En la **fase 1** acotamos deliberadamente el problema a una tarea más pequeña y verificable: calcular el importe mensual de **IMV + CAPI** para nuevas solicitudes de 2026 a partir de un caso previamente normalizado.

## Diez preguntas o tareas reales

Las siguientes son las preguntas de demostración utilizadas para comprobar la tarea de razonamiento de la fase 1. Cubren casos ordinarios y varias ramas límite del motor.

### 1. Caso ordinario: 2 adultos + 1 menor

Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 3 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1988-05-10, 38 años a fecha de solicitud; p2: nacido el 1986-02-20, 40 años a fecha de solicitud; p3: nacido el 2010-06-01, 16 años a fecha de solicitud y 15 años a 1 de enero de 2026. Relaciones: p1 y p2 son cónyuges; p1 y p3 tienen relación progenitor-hijo con custodia compartida (no exclusiva). Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01; p3: sí, desde 2018-01-01. Ingresos computables anuales de la unidad: 9600.00 €. Patrimonio neto sin vivienda habitual: 5000.00 €. Activos no societarios sin vivienda habitual: 8000.00 €. Determina primero si hay derecho al IMV y/o al CAPI y calcula el total mensual a percibir. Responde solo con el importe total en euros, con dos decimales.

**Respuesta esperada:** `431.26`

### 2. Discapacidad >= 65 %

Caso para resolver sobre IMV 2026. Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 3 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1988-05-10, 38 años a fecha de solicitud; p2: nacido el 1986-02-20, 40 años a fecha de solicitud, discapacidad reconocida del 65 %; p3: nacido el 2012-04-10, 14 años a fecha de solicitud y 13 años a 1 de enero de 2026. Relaciones: p1 y p2 son cónyuges; p1 y p3 tienen relación progenitor-hijo con custodia compartida (no exclusiva). Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01; p3: sí, desde 2018-01-01. Ingresos computables anuales de la unidad: 12000.00 €. Patrimonio neto sin vivienda habitual: 6000.00 €. Activos no societarios sin vivienda habitual: 9000.00 €. ¿Cuál sería la suma mensual de IMV y CAPI? Da únicamente el número con dos decimales.

**Respuesta esperada:** `392.65`

### 3. Monoparental con dos menores

Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 3 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1990-04-12, 36 años a fecha de solicitud; p2: nacido el 2023-08-15, 3 años a fecha de solicitud y 2 años a 1 de enero de 2026; p3: nacido el 2016-05-20, 10 años a fecha de solicitud y 9 años a 1 de enero de 2026. Relaciones: p1 y p2 tienen relación progenitor-hijo con custodia exclusiva; p1 y p3 tienen relación progenitor-hijo con custodia exclusiva. Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01; p3: sí, desde 2018-01-01. Ingresos computables anuales de la unidad: 7200.00 €. Patrimonio neto sin vivienda habitual: 4000.00 €. Activos no societarios sin vivienda habitual: 6000.00 €. Aplica las reglas del IMV y del complemento de ayuda para la infancia. Si no corresponde ninguna de las dos prestaciones, indica cero euros. Indica el total mensual con dos decimales.

**Respuesta esperada:** `907.65`

### 4. Fallo por ingresos: diferencia inferior a 10 €

Analiza esta solicitud de IMV/CAPI: Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 2 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1991-01-10, 35 años a fecha de solicitud; p2: nacido el 1993-11-20, 32 años a fecha de solicitud. Relaciones: p1 y p2 son cónyuges. Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01. Ingresos computables anuales de la unidad: 11330.16 €. Patrimonio neto sin vivienda habitual: 5000.00 €. Activos no societarios sin vivienda habitual: 7000.00 €. Calcula el importe mensual final (IMV + CAPI) y contesta únicamente con la cifra.

**Respuesta esperada:** `0.00`

### 5. Fallo por residencia de un miembro

Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 3 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1985-05-10, 41 años a fecha de solicitud; p2: nacido el 1987-02-15, 39 años a fecha de solicitud; p3: nacido el 2017-03-01, 9 años a fecha de solicitud y 8 años a 1 de enero de 2026. Relaciones: p1 y p2 son cónyuges; p1 y p3 tienen relación progenitor-hijo con custodia compartida (no exclusiva). Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01; p3: sí, desde 2026-03-01. Ingresos computables anuales de la unidad: 8400.00 €. Patrimonio neto sin vivienda habitual: 3000.00 €. Activos no societarios sin vivienda habitual: 5000.00 €. Determina primero si hay derecho al IMV y/o al CAPI y calcula el total mensual a percibir. Responde solo con el importe total en euros, con dos decimales.

**Respuesta esperada:** `0.00`

### 6. Beneficiario individual + patrimonio exactamente en el límite

Caso para resolver sobre IMV 2026. Solicitud a fecha 2026-09-26. Solicitante: p1. Convive 1 persona en el domicilio desde 2023-01-01. Personas: p1: nacido el 1995-05-10, 31 años a fecha de solicitud. Relaciones: ninguna relevante. Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01. Independencia del solicitante: domicilio distinto al de sus progenitores/tutores desde 2022-01-01. Periodos de alta en Seguridad Social: 2023-01-01 a 2026-09-26. Ingresos computables anuales de la unidad: 0 €. Patrimonio neto sin vivienda habitual: 26409.60 €. Activos no societarios sin vivienda habitual: 5000.00 €. ¿Cuál sería la suma mensual de IMV y CAPI? Da únicamente el número con dos decimales.

**Respuesta esperada:** `0.00`

### 7. Administrador de sociedad mercantil activa

Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 3 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1989-06-10, 37 años a fecha de solicitud; p2: nacido el 1988-03-20, 38 años a fecha de solicitud; p3: nacido el 2021-08-10, 5 años a fecha de solicitud y 4 años a 1 de enero de 2026. Relaciones: p1 y p2 son cónyuges; p1 y p3 tienen relación progenitor-hijo con custodia compartida (no exclusiva). Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01; p3: sí, desde 2018-01-01. Ingresos computables anuales de la unidad: 9000.00 €. Patrimonio neto sin vivienda habitual: 5000.00 €. Activos no societarios sin vivienda habitual: 7000.00 €. Hay un administrador de una sociedad mercantil activa. Aplica las reglas del IMV y del complemento de ayuda para la infancia. Si no corresponde ninguna de las dos prestaciones, indica cero euros. Indica el total mensual con dos decimales.

**Respuesta esperada:** `0.00`

### 8. Tope por pensión

Analiza esta solicitud de IMV/CAPI: Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 3 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1988-05-10, 38 años a fecha de solicitud; p2: nacido el 1986-02-20, 40 años a fecha de solicitud; p3: nacido el 2012-04-10, 14 años a fecha de solicitud y 13 años a 1 de enero de 2026. Relaciones: p1 y p2 son cónyuges; p1 y p3 tienen relación progenitor-hijo con custodia compartida (no exclusiva). Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01; p3: sí, desde 2018-01-01. Ingresos computables anuales de la unidad: 9600.00 €. Patrimonio neto sin vivienda habitual: 5000.00 €. Activos no societarios sin vivienda habitual: 8000.00 €. Pensiones/subsidios sujetos al tope: 1000.00 €/mes. Calcula el importe mensual final (IMV + CAPI) y contesta únicamente con la cifra.

**Respuesta esperada:** `231.26`

### 9. CAPI sin IMV

Caso para resolver sobre IMV 2026. Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 2 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1991-04-10, 35 años a fecha de solicitud; p2: nacido el 2021-07-15, 5 años a fecha de solicitud y 4 años a 1 de enero de 2026. Relaciones: p1 y p2 tienen relación progenitor-hijo con custodia compartida (no exclusiva). Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01. Ingresos computables anuales de la unidad: 15000.00 €. Patrimonio neto sin vivienda habitual: 5000.00 €. Activos no societarios sin vivienda habitual: 7000.00 €. ¿Cuál sería la suma mensual de IMV y CAPI? Da únicamente el número con dos decimales.

**Respuesta esperada:** `80.50`

### 10. Caso de borde: activos exactamente en el máximo permitido

Solicitud a fecha 2026-09-26. Solicitante: p1. Conviven 4 personas en el mismo domicilio desde 2023-01-01. Personas: p1: nacido el 1987-05-10, 39 años a fecha de solicitud; p2: nacido el 1989-02-15, 37 años a fecha de solicitud; p3: nacido el 2024-03-10, 2 años a fecha de solicitud y 1 año a 1 de enero de 2026; p4: nacido el 2020-11-01, 5 años a fecha de solicitud y 5 años a 1 de enero de 2026. Relaciones: p1 y p2 son cónyuges; p1 y p3 tienen relación progenitor-hijo con custodia compartida (no exclusiva); p1 y p4 tienen relación progenitor-hijo con custodia compartida (no exclusiva). Residencia legal y efectiva continuada por miembro: p1: sí, desde 2018-01-01; p2: sí, desde 2018-01-01; p3: sí, desde 2018-01-01; p4: sí, desde 2018-01-01. Ingresos computables anuales de la unidad: 12000.00 €. Patrimonio neto sin vivienda habitual: 10000.00 €. Activos no societarios sin vivienda habitual: 116202.24 €. Determina primero si hay derecho al IMV y/o al CAPI y calcula el total mensual a percibir. Responde solo con el importe total en euros, con dos decimales.

**Respuesta esperada:** `589.34`

## La tarea verificable (fase 1)

### Tipo de problema

La fase 1 se centra en una tarea numérica verificable automáticamente.

Dado un **caso normalizado de nueva solicitud de IMV/CAPI en 2026**, el modelo debe calcular el **importe mensual total de IMV + CAPI**, expresado en euros con dos decimales.

El caso normalizado contiene los hechos necesarios para aplicar las reglas que cubre el motor:

- fecha de solicitud;
- miembros que conviven y sus fechas de nacimiento;
- relaciones familiares relevantes y tipo de custodia;
- discapacidad;
- residencia legal y efectiva continuada por miembro;
- datos necesarios para la independencia de un beneficiario individual;
- ingresos computables anuales;
- patrimonio neto excluyendo la vivienda habitual;
- activos no societarios excluyendo la vivienda habitual;
- existencia de un administrador de una sociedad mercantil activa;
- pensiones o subsidios sujetos al tope específico cuando proceda.

El resultado esperado es un único importe. Si no corresponde ni IMV ni CAPI, la respuesta es `0.00`.

Esta tarea **no pretende implementar todo el procedimiento administrativo del IMV**. El alcance de fase 1 es el núcleo determinista necesario para construir un dataset verificable de nuevas solicitudes: unidad de convivencia ordinaria, edad e independencia, residencia, renta garantizada, discapacidad, situaciones monoparentales contempladas por el motor, límites económicos, administrador mercantil, tope por pensiones y CAPI.

Quedan fuera de esta tarea procesos distintos como revisiones anuales, incentivo al empleo, suspensión, extinción, reintegros o casuística documental excepcional.

### Ejemplos

- Una unidad de dos adultos y un menor, con 9.600 € de ingresos anuales y por debajo de los límites de patrimonio y activos, obtiene **431.26 € mensuales** de IMV + CAPI.
- Un caso en el que uno de los miembros no cumple el periodo mínimo de residencia exigido obtiene **0.00 €**.

### Verificación con código

La referencia de verdad es el motor determinista `rlm/imv_engine.py`.

El generador de problemas y los tests utilizan ese mismo motor como *oracle*, pero el modelo **no puede llamarlo durante inferencia**. De esta forma, el alumno tiene que aprender a resolver los casos y no simplemente ejecutar la implementación de referencia.

`IMVAmountVerifier`:

1. extrae la respuesta del último bloque `<answer>...</answer>`;
2. acepta formatos decimales habituales en español o inglés;
3. convierte los importes usando `Decimal`;
4. cuantiza a céntimos con `ROUND_HALF_UP`;
5. compara el resultado con la etiqueta correcta.

El redondeo forma parte del contrato de la tarea. El motor especifica cuándo se redondean la renta garantizada, el IMV, el CAPI y el total final, evitando discrepancias de un céntimo entre generador, recompensa y verificador.

Para GRPO se utilizan tres señales:

- recompensa de formato para `<think>...</think><answer>...</answer>`;
- recompensa de exactitud IMV, basada en el mismo criterio monetario que el verificador;
- recompensa de dominio para que la respuesta final sea una cifra limpia con exactamente dos decimales.

### Construcción del dataset

Se utiliza la **estrategia de generador programático**.

`rlm/generate_problems.py` muestrea casos, construye un enunciado observable y obtiene su etiqueta ejecutando `rlm/imv_engine.py`.

El dataset previsto y soportado por el generador es:

- **800 casos de entrenamiento**;
- **200 casos de test** con la misma distribución;
- **100 casos OOD** (*out of distribution*).

El conjunto OOD reserva una familia completa que no aparece en train/test: **CAPI sin IMV**, es decir, hogares cuyos ingresos impiden recibir IMV pero todavía cumplen los umbrales más amplios del complemento de ayuda para la infancia.

Entre las familias de casos generadas se incluyen:

- casos ordinarios;
- beneficiarios individuales;
- unidades monoparentales;
- discapacidad;
- fallos por renta;
- fallos por residencia;
- límites de patrimonio o activos;
- administrador de sociedad mercantil activa;
- casos con tope por pensiones;
- CAPI sin IMV en el conjunto OOD.

El muestreo presta especial atención a valores cercanos a los umbrales para que el modelo tenga que aprender diferencias como `<` frente a `<=`, el mínimo de 10 € para el IMV y los distintos límites de patrimonio y activos.

El generador utiliza varias plantillas lingüísticas y garantiza un principio importante: **todo hecho utilizado por el oracle para calcular la respuesta aparece explícitamente en el enunciado**.

También se controla la deduplicación de casos y que la respuesta numérica no aparezca accidentalmente dentro de la pregunta.

### Destilación, SFT y GRPO

Las trazas para el arranque en frío se generan con `rlm/distill.py`. Un modelo profesor genera varias soluciones razonadas por problema y el verificador conserva únicamente aquellas cuya respuesta final coincide con el resultado del motor de referencia. Esas trazas verificadas se normalizan al formato `<think>...</think><answer>...</answer>` y se utilizan posteriormente para SFT.

El entrenamiento con **GRPO** está implementado en `rlm/train_grpo.py` utilizando `GRPOTrainer` de TRL y adaptadores LoRA. Puede comenzar directamente desde el modelo base o continuar desde un adaptador previo —por ejemplo, el obtenido mediante SFT— usando `--init-adapter`. Esto permite comparar el entrenamiento con arranque en frío frente a una variante más próxima a R1-Zero.

Para el dominio IMV/CAPI se combinan tres recompensas:

- **recompensa de formato (`format_reward`)**, con peso `1.0` por defecto, que exige la estructura `<think>...</think><answer>...</answer>`;
- **recompensa de exactitud (`imv_accuracy_reward`)**, con peso `2.0`, que utiliza exactamente la misma semántica monetaria que el oracle y `IMVAmountVerifier`: `Decimal` y `ROUND_HALF_UP` a céntimos;
- **recompensa de dominio (`domain_reward`)**, con peso `0.5`, que exige que la respuesta final sea una cifra limpia con exactamente dos decimales y punto decimal, por ejemplo `431.26`.

Los pesos pueden modificarse experimentalmente mediante `--reward-weights`.

Además se registra `zero_answer_rate`, una métrica con peso `0` que **no interviene en la optimización**. Su objetivo es detectar un posible caso de *reward hacking*: dado que una proporción relevante del dataset tiene `0.00` como respuesta correcta, un modelo podría aprender a responder cero con demasiada frecuencia y obtener cierta recompensa de exactitud sin resolver realmente los problemas.

GRPO utiliza grupos de varias generaciones para una misma pregunta (`--num-generations`, 8 por defecto), con temperatura configurable. El rango de *clipping* se controla mediante `epsilon` —`0.2` por defecto— y la penalización KL mediante `beta`, cuyo valor por defecto es `0.0`. El entrenamiento utiliza micro-batches para reducir el consumo de memoria durante el cálculo de la pérdida y guarda checkpoints periódicos.

Además del entrenamiento con TRL, `rlm/grpo_step.py` implementa explícitamente un paso del algoritmo para poder comprobar y explicar sus componentes: normalización de ventajas dentro de cada grupo,

`A_i = (r_i - mean(r)) / (std(r) + eps)`,

cálculo del *policy ratio* por token, objetivo recortado mediante *clipping* y penalización KL opcional respecto a una política de referencia. La implementación aplica una máscara sobre los tokens válidos, promedia primero dentro de cada respuesta y después sobre el grupo, y registra estadísticas como la fracción de tokens afectados por clipping, el KL medio y la ventaja media.

Durante el entrenamiento se conserva el histórico de métricas en `log_history.json`, lo que permite analizar posteriormente recompensas, longitudes, KL y comportamiento del clipping.

### Evaluación de fase 1

`rlm/evaluate.py` compara con las mismas preguntas y condiciones de generación:

- modelo base;
- modelo después de SFT;
- modelo después de GRPO.

La métrica principal es **pass@1**, calculada a partir de una o varias generaciones por problema. Cuando se utilizan varias muestras, el resultado corresponde a la precisión media sobre todas ellas, reduciendo la variabilidad de una única generación.

Además se registran:

- `pass@1` global y desglosado por familia de casos;
- porcentaje de respuestas con formato correcto;
- porcentaje de generaciones truncadas;
- proporción de respuestas `0.00`, comparada con la proporción real de ceros del dataset;
- longitud media de la generación;
- longitud media del razonamiento.

Los errores del dominio también se clasifican automáticamente utilizando el motor de referencia. Entre las categorías contempladas están negar incorrectamente una prestación, no detectar una causa de inelegibilidad, olvidar el CAPI, olvidar el IMV, no restar los ingresos o producir una cantidad cercana pero incorrecta.

Este desglose permite evaluar no solo si GRPO aumenta la exactitud global, sino **qué tipos de reglas aprende mejor o peor que SFT y el modelo base**, y comprobar si aparece comportamiento de *reward hacking*.

## Las herramientas (fase 2)

Las herramientas amplían el agente más allá del cálculo aprendido en fase 1. La propuesta es utilizar tres herramientas del dominio, una por cada categoría exigida:

### Consulta fuera del modelo — `consultar_boe`

Consulta la API de datos abiertos del BOE para recuperar el texto vigente de artículos relevantes de la Ley 19/2021 y metadatos de actualización.

Su utilidad es que el modelo no dependa exclusivamente de conocimiento memorizado y pueda contrastar una condición normativa con una fuente oficial.

### Cálculo — `calcular_imv`

Recibe un caso estructurado y validado y ejecuta el motor determinista del dominio.

Devuelve, además del total, información estructurada como:

- cuantía estimada de IMV;
- cuantía de CAPI;
- total;
- condiciones relevantes cumplidas o incumplidas.

Aquí sí tiene sentido permitir al agente ejecutar el motor: en fase 1 queremos comprobar que el modelo aprende a razonar; en fase 2 queremos construir una herramienta fiable que el agente pueda utilizar cuando lo apropiado sea calcular y no improvisar.

### Acción con efecto observable — `preparar_solicitud`

Genera un borrador estructurado con los datos recopilados del caso y la información necesaria para preparar los siguientes pasos de una solicitud.

La acción debe producir un artefacto verificable en disco y permitir comprobar automáticamente que:

- el fichero se ha creado;
- valida contra el esquema esperado;
- los datos básicos coinciden con el caso de entrada.

Se tratará de un **borrador de apoyo**, no de una presentación automática ante la Administración.

## El corpus (fase 3)

El corpus estará formado prioritariamente por documentación oficial necesaria para responder preguntas que el motor numérico de la fase 1 no cubre por sí solo.

### Fuentes previstas

- Ley 19/2021, de 20 de diciembre, por la que se establece el ingreso mínimo vital, en su texto consolidado.
- Documentación oficial de la Seguridad Social sobre IMV y CAPI.
- Preguntas frecuentes, requisitos, cuantías, documentación y obligaciones publicadas por organismos oficiales.
- Cuando aporte valor, guías públicas de administraciones o entidades que expliquen el procedimiento en lenguaje más accesible.

El objetivo es recopilar **decenas de documentos o varios cientos de fragmentos**, suficientes para que comparar BM25, recuperación densa e híbrida tenga sentido.

### Formatos

Principalmente:

- HTML;
- PDF con texto;
- Markdown o texto normalizado generado durante la ingesta.

### Licencia y reutilización

Se priorizará documentación pública y oficial cuya reutilización sea compatible con el proyecto, conservando siempre la fuente y los metadatos necesarios para citarla.

Los documentos de terceros solo se incorporarán si sus condiciones de uso permiten hacerlo.

### Preguntas que requieren el corpus

Ejemplos de preguntas que no son simplemente el cálculo numérico de fase 1:

1. «Me han concedido el IMV y ahora he encontrado trabajo. ¿Qué tengo que comunicar y cómo puede afectar a la prestación?»
2. «Presenté la solicitud hace meses y todavía no tengo respuesta. ¿Qué plazos y vías de actuación indica la normativa?»

El RAG deberá devolver fragmentos relevantes y producir una respuesta apoyada en citas a esos fragmentos.

## Integración en el agente (fase 4)

La versión final combinará las piezas desarrolladas en las fases anteriores mediante un agente ReAct.

Ante una consulta, el agente podrá decidir entre:

- razonar con el modelo entrenado;
- consultar información normativa;
- recuperar documentación del corpus;
- ejecutar el cálculo determinista;
- realizar una acción que genere un artefacto;
- o combinar varias de estas operaciones.

La separación es importante: no todas las preguntas del usuario son cálculos y no todas requieren consultar una herramienta.

El objetivo final es que el sistema pueda explicar qué está haciendo y apoyar la respuesta en resultados reproducibles o fuentes recuperadas, en lugar de depender únicamente de lo que recuerde el modelo.

## Qué puede salir mal

### 1. El modelo explota la distribución del dataset en vez de aprender las reglas

Hay varias familias cuya respuesta puede ser `0.00`. Si aparecen con demasiada frecuencia, un modelo que responda cero sistemáticamente podría obtener una exactitud aparentemente aceptable.

**Mitigación:**

- controlar la distribución de respuestas y familias;
- reportar resultados desglosados por tipo de caso;
- no limitar la evaluación a una métrica global;
- utilizar el conjunto OOD de CAPI sin IMV;
- analizar manualmente errores representativos.

### 2. El profesor de destilación genera razonamientos incorrectos

Aunque el profesor reciba las reglas, puede equivocarse en problemas largos o de borde.

**Mitigación:**

- generar varias trazas por pregunta;
- filtrar todas las trazas mediante `IMVAmountVerifier`;
- conservar métricas de aceptación y rechazo;
- entrenar SFT únicamente con soluciones cuya respuesta final haya sido verificada.

### 3. Limitaciones de cómputo

El entrenamiento se realiza con recursos limitados, por lo que modelos y longitudes demasiado grandes pueden impedir iterar de forma razonable.

**Mitigación:**

- modelos pequeños de la familia Qwen3;
- adaptadores LoRA;
- vLLM para acelerar la generación de trazas cuando sea posible;
- checkpoints periódicos;
- limitar y analizar las longitudes de razonamiento.

## Por qué este tema

El IMV y el CAPI son un buen dominio para esta práctica porque combinan las cuatro capacidades que se quieren estudiar.

La fase 1 tiene una respuesta numérica verificable y suficientes reglas y casos límite para que el razonamiento sea no trivial. La fase 2 permite incorporar consultas externas, cálculo determinista y acciones observables. La fase 3 dispone de abundante documentación real para construir y evaluar un sistema RAG. Finalmente, la fase 4 permite unir esas piezas en un agente que decide cuándo razonar, cuándo recuperar información y cuándo ejecutar una herramienta.

Además, es un problema con utilidad fuera de la asignatura: transformar reglas administrativas complejas en explicaciones y cálculos reproducibles puede ayudar tanto a profesionales como a personas que intentan orientarse sobre una prestación pública.
