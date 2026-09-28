# Propuesta de tema — Navegador del Ingreso Mínimo Vital (IMV + CAPI)

## Equipo

- Javier Escobar Serrano — javierescobarserrano@gmail.com (destilación y SFT)
- Javier Ahumada — 202213736@alu.comillas.edu (motor de reglas, generador y verificador)
- Marcos — 202204116@alu.comillas.edu (GRPO y evaluación)

## El tema en una frase

Un agente que, a partir de la situación de una familia contada en lenguaje llano, dice si tiene
derecho al Ingreso Mínimo Vital y al Complemento de Ayuda para la Infancia, cuánto cobraría al
mes, qué requisito falla si no tiene derecho y qué tiene que presentar para pedirlo, citando la
Ley 19/2021 y las guías de la Seguridad Social.

## El usuario y su problema

Dos perfiles: una trabajadora social de atención primaria que atiende a decenas de familias por
semana, y la propia persona en situación vulnerable que quiere saber si merece la pena pedirlo.

Hoy la trabajadora social consulta el simulador de la Seguridad Social (que solo da una
estimación y no explica qué requisito falla), la ley consolidada en el BOE y sus propias notas.
Los errores típicos son de cálculo de cuantía: escalas por número de miembros, complementos de
discapacidad y monoparentalidad del 22 %, límites de patrimonio y de activos que cambian con la
composición del hogar, tramos de edad del CAPI y tope por pensiones. Un error cuesta dinero a la
familia (no pide una ayuda a la que tiene derecho) o tiempo a todos (solicitud denegada que se
podía haber previsto). La persona afectada, además, no entiende el lenguaje de la norma.

## Diez preguntas o tareas reales

1. ¿Qué es el ingreso mínimo vital y quién lo puede pedir?
2. Vivo solo, tengo 40 años y no ingreso nada. ¿Cuánto me darían al mes?
3. Somos mi mujer, yo y dos niños de 4 y 7 años; entre los dos ganamos 16.000 € al año. ¿Nos corresponde algo?
4. Soy madre sola con custodia exclusiva de mi hija de 3 años y cobro 600 € al mes. ¿Cuánto me quedaría de IMV y de complemento por la niña?
5. Mi marido tiene una discapacidad del 80 %. ¿Eso cambia lo que nos dan?
6. Tengo 24 años y vivo con mis padres, pero trabajo desde hace un año. ¿Puedo pedirlo por mi cuenta?
7. Tenemos 30.000 € ahorrados en el banco. ¿Nos lo van a denegar por eso?
8. Llegamos a España hace seis meses mi pareja y yo, yo llevo aquí desde 2018. ¿Podemos pedirlo ya?
9. Cobro una pensión no contributiva y vivo con mi hijo de 15 años. ¿Puedo cobrar además el IMV o solo el complemento del niño?
10. Somos tres, cobro 900 € al mes, mi hijo tiene una discapacidad del 40 %, soy administradora de una sociedad sin actividad y vivimos de alquiler. ¿Qué me corresponde, qué me falla y qué papeles tengo que llevar? Prepárame la solicitud.

## La tarea verificable (fase 1)

**Tipo de problema.** Dado un caso normalizado (fecha de solicitud, miembros con fecha de
nacimiento, relaciones y tipo de custodia, discapacidad, residencia legal y efectiva por
miembro, ingresos computables anuales, patrimonio neto y activos no societarios sin vivienda
habitual, condición de administrador), calcular la **cuantía mensual total IMV + CAPI** en euros
con dos decimales. Si no corresponde ninguna prestación, la respuesta es `0.00`.

Dos ejemplos del conjunto de entrenamiento:

- Pareja casada con tres hijos de 3, 4 y 7 años a 1 de enero, custodia compartida, residencia
  desde 2018, ingresos anuales 16.300,94 €, patrimonio 8.740,08 €, activos 58.805,89 €.
  Respuesta: **474.01** (IMV por renta + CAPI de tres menores).
- Pareja casada, uno de los cónyuges con residencia continuada solo desde 2026-06-01, ingresos
  5.128,78 €. Respuesta: **0.00** (falla el requisito de un año de residencia de todos los miembros).

**Verificación con código.** `IMVAmountVerifier` extrae el último `<answer>`, acepta decimal
español o inglés, cuantiza a céntimos con `Decimal` y `ROUND_HALF_UP` (la misma convención que el
motor) y compara por igualdad exacta. La convención de redondeo es parte del contrato: el motor
redondea la renta garantizada, el IMV final y el total CAPI en pasos fijos documentados en
`rlm/IMV.md`. La recompensa de GRPO combina formato `<think>…</think><answer>…</answer>`,
exactitud y una tercera recompensa de dominio (respuesta limpia con dos decimales).

**Estrategia: 1, generador programático.** El motor de reglas `rlm/imv_engine.py` es a la vez
el generador de etiquetas y el verificador. Las cuantías y umbrales viven en
`rlm/rulesets/imv_2026.json` (fuentes: Ley 19/2021 consolidada y cuantías 2026 de la Seguridad
Social), no en el código. Parámetros muestreados:

- familia de caso: ordinaria, individual, monoparental, discapacidad, fallo por renta, fallo por
  residencia, fallo por patrimonio/activos, fallo por administrador societario (pesos
  20/13/14/10/12/8/12/6), más un 15 % de casos ordinarios con tope por pensiones;
- número de adultos y menores, fechas de nacimiento (edades a fecha de solicitud y a 1 de enero
  para el tramo CAPI), relaciones y custodia exclusiva o compartida;
- grado de discapacidad, fechas de residencia por miembro y de independencia del hogar familiar;
- ingresos, patrimonio y activos, muestreados alrededor de los umbrales para cubrir las ramas límite.

Tamaños: **800 train, 200 test** (misma distribución, de los que auditaremos ~50 a mano) y
**100 OOD** con una familia ausente de train: **solo CAPI** (sin derecho a IMV por renta pero con
derecho al complemento por infancia). Deduplicación por hash de parámetros e intersección
train/test vacía; el generador imprime la cobertura por familia y rama, el número de plantillas
(cuatro por ahora; queremos añadir una pasada de paráfrasis) y cuántas veces aparece la
respuesta como subcadena del enunciado (debe ser cero). Las trazas las genera un
profesor Qwen3-4B que sí recibe la hoja exacta de reglas (`rule_context`); el alumno solo ve el
enunciado.

## Las herramientas (fase 2)

- **Consulta fuera del modelo — `consultar_boe`:** consulta la [API de datos abiertos del BOE](https://www.boe.es/datosabiertos/api/api.php)
  para obtener el texto consolidado vigente de un artículo de la Ley 19/2021 (BOE-A-2021-21007)
  y su fecha de última actualización. Sirve para citar el requisito exacto y para detectar que
  la norma ha cambiado después del corte del modelo o del ruleset.
- **Cálculo — `calcular_imv`:** recibe el caso estructurado (validado con Pydantic) y devuelve
  la cuantía IMV, la CAPI, el total y la lista de requisitos cumplidos y fallidos, ejecutando el
  motor de reglas. El modelo no debe hacerlo de cabeza: son decenas de umbrales por composición
  del hogar, desigualdades estrictas o no según el test y un redondeo por pasos donde un céntimo
  de diferencia es un error.
- **Acción con efecto observable — `preparar_solicitud`:** genera en `runs/solicitudes/` un
  borrador de solicitud (JSON y PDF) con los datos del caso, la cuantía estimada y la lista de
  documentos a aportar según la situación (libro de familia, certificado de discapacidad,
  empadronamiento, resolución de custodia…). Se comprueba que el fichero existe, que valida
  contra su esquema y que sus campos coinciden con el caso de entrada.

## El corpus (fase 3)

- **Origen y tamaño:** Ley 19/2021 del IMV consolidada (BOE), los artículos del texto refundido
  de la Ley General de la Seguridad Social que remite, las páginas de la Seguridad Social sobre
  IMV y CAPI (requisitos, cuantías, preguntas frecuentes, documentación, obligaciones de los
  beneficiarios), la guía del IMV del Ministerio de Inclusión y las guías en lenguaje llano de
  servicios sociales municipales y de entidades del tercer sector. Del orden de 30–50 documentos
  y 400–600 páginas.
- **Formato:** HTML (BOE, seg-social.es) y PDF con texto (guías). Sin documentos escaneados.
- **Licencia:** las disposiciones legales y los actos oficiales no están sujetos a propiedad
  intelectual (art. 13 de la Ley de Propiedad Intelectual); la información del sector público es
  reutilizable según la Ley 37/2007 y el aviso legal de cada sede, citando la fuente. Las guías
  de terceros se usan solo si su licencia lo permite y siempre con cita.
- **Preguntas que solo se responden leyendo el corpus:**
  1. "Me han concedido el IMV y ahora he encontrado trabajo. ¿Tengo que avisar, en qué plazo y
     me lo quitan?"
  2. "Presenté la solicitud hace meses y nadie me contesta. ¿Qué significa eso y qué puedo hacer?"

## Qué puede salir mal

- **El motor de reglas no coincide con la ley en algún caso límite.** Ya lo hemos visto: la
  familia de fallo por residencia apenas la resuelve el profesor aunque tenga las reglas, lo que
  apunta a una discrepancia entre el enunciado de la regla y el motor. Mitigación: tests de
  regresión del motor contra ejemplos de la Seguridad Social, auditoría manual de ~50 casos de
  test y tabla de aciertos por familia en `EXPERIMENTS.md` para detectar familias anómalas.
- **Recompensa hackeable con `0.00`.** Casi un 30 % de los casos tiene respuesta cero, así que
  contestar siempre cero da una recompensa no trivial. Mitigación: vigilar la proporción de
  ceros en las respuestas durante GRPO, reequilibrar familias si hace falta y reportar pass@1
  por familia, no solo global.
- **Presupuesto de cómputo.** Una GPU de 16 GB con trazas de hasta 4.096 tokens limita el tamaño
  del alumno y del profesor. Mitigación: vLLM para destilar, LoRA para SFT/GRPO, alumno Qwen3
  pequeño (0,6B–1,7B) y filtrado de trazas por longitud.
- **La norma cambia o la API del BOE falla.** Mitigación: ruleset versionado por año, caché
  local de los artículos consultados y respuesta degradada citando la versión del corpus.
- **Consejo erróneo a una persona vulnerable.** Mitigación: cero datos personales reales, el
  agente nunca dice "no tienes derecho" sin citar el requisito que falla y siempre remite a
  servicios sociales o a la Seguridad Social como fuente final.

## Por qué este tema

Es un problema real con reglas publicadas, así que la tarea es verificable de verdad y no
depende de un juez humano. Y es útil: el IMV tiene una tasa de no solicitud muy alta entre
quienes tienen derecho, en buena parte porque calcular si te corresponde es difícil. Un agente
que lo explica en lenguaje llano y lo cita tiene sentido fuera de la asignatura.
