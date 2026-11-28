# Propuesta de tema — Navegador del Ingreso Mínimo Vital (IMV + CAPI)

## Equipo

- Javier Escobar Serrano — javierescobarserrano@gmail.com
- Javier Ahumada — 202213736@alu.comillas.edu
- Marcos — 202204116@alu.comillas.edu

## El tema en una frase

Un asistente que, a partir de la descripción de un hogar, calcula de forma reproducible
cuánto le correspondería de Ingreso Mínimo Vital (IMV) y de Complemento de Ayuda para la
Infancia (CAPI), explica por qué y lo apoya en la normativa.

## El usuario y su problema

Hay dos usuarios: el trabajador social que hace una primera orientación sobre un caso y la
persona que quiere saber si su situación encaja antes de pedir la prestación.

Hoy tienen que cruzar la Ley 19/2021, las cuantías publicadas por la Seguridad Social y
cálculos a mano. Es fácil equivocarse por los detalles:

- la composición de la unidad de convivencia y la edad de cada miembro;
- la residencia legal y efectiva de cada uno;
- los límites de patrimonio y de activos, que son distintos;
- los complementos por discapacidad y monoparentalidad;
- el mínimo de 10 € para cobrar IMV y el tope por pensiones;
- los tramos de edad del CAPI;
- desigualdades estrictas o inclusivas, y redondeos a céntimos.

Un error en uno de estos puntos cambia la respuesta de «sí» a «no».

## Diez preguntas o tareas reales

1. ¿Qué es el IMV y qué es el CAPI?
2. Vivo solo, tengo 46 años, cobro 4.200 € al año y no tengo patrimonio. ¿Cuánto me
   correspondería?
3. Somos un matrimonio sin hijos con 4.000 € de ingresos anuales. ¿Cuánto cobraríamos?
4. Soy madre sola con dos niños de 3 años y cobro 10.967 € al año. ¿Me toca IMV o solo CAPI?
5. Mi hijo de 16 años tiene un 70 % de discapacidad. ¿Cambia la cuantía?
6. Mi padre pensionista vive con nosotros. ¿Cómo afecta su pensión al cálculo?
7. Soy administradora de una sociedad que sigue activa. ¿Puedo pedirlo?
8. Mi pareja lleva solo seis meses de residencia legal en España. ¿Tenemos derecho?
9. Calcula mi caso, busca también ayudas de alquiler y guárdame un informe.
10. Tengo dos menores a cargo y no sé bien mi patrimonio. ¿Qué datos necesitas para calcularlo?

## La tarea verificable (fase 1)

**Tipo de problema.** Dado un caso normalizado de nueva solicitud en 2026, calcular el importe
mensual total de IMV + CAPI en euros con dos decimales. Si no hay derecho a ninguna de las dos
prestaciones, la respuesta es `0.00`.

**Dos ejemplos:**

- Dos adultos y un menor de 16 años, con residencia desde 2018, 9.600 € de ingresos anuales,
  5.000 € de patrimonio y 8.000 € de activos: **431.26**.
- Dos adultos y un menor con 8.400 € de ingresos anuales, en el que el menor solo tiene
  residencia legal y efectiva desde hace siete meses: **0.00**.

**Verificación con código.** Un motor determinista (`rlm/imv_engine.py`) aplica las reglas
con las cuantías de 2026 y es la referencia. El verificador extrae la respuesta de
`<answer>…</answer>`, la convierte a `Decimal`, la redondea a céntimos con `ROUND_HALF_UP` y
la compara con la del motor. El redondeo forma parte del contrato: el motor fija en qué paso
se redondea cada importe, para que generador, recompensa y verificador no difieran en un
céntimo.

**Estrategia de construcción: generador programático.** `rlm/generate_problems.py` muestrea
los hechos del hogar, redacta el enunciado y obtiene la etiqueta ejecutando el motor.

- **Parámetros que se muestrean:** número de adultos y menores, edades, relaciones y custodia,
  fechas de residencia, ingresos, patrimonio, activos, discapacidad, pensiones, administración
  de sociedades y progenitor ingresado en un centro.
- **Diez familias de casos con cuotas iguales:** ordinario, beneficiario individual,
  monoparental, monoparental con el otro progenitor en un centro, discapacidad, fallo por
  ingresos, fallo por residencia, fallo por patrimonio o activos, administrador de sociedad y
  casos en el borde de un umbral.
- **Tamaño previsto:** 800 casos de entrenamiento y 200 de test.
- **Tres conjuntos fuera de distribución de 50 casos:**
  - una familia que nunca se ve al entrenar (CAPI sin IMV);
  - casos con dos condiciones a la vez cerca del límite;
  - enunciados con otra redacción.
- **Garantías:** todo hecho que use el motor aparece en el enunciado, la respuesta nunca
  aparece dentro de la pregunta y no hay casos repetidos entre conjuntos.

## Las herramientas (fase 2)

- **Consulta externa: `consultar_ayudas`.** Busca convocatorias reales en la API pública de la
  BDNS ([documentación](https://www.infosubvenciones.es/bdnstrans/doc/swagger)). Sirve para lo
  que el motor no cubre: qué otras ayudas públicas hay para el caso, como alquiler,
  discapacidad o familia numerosa.
- **Cálculo: `calcular_imv`.** Ejecuta el motor de la fase 1 sobre un caso validado con
  Pydantic. El modelo no debe hacer este cálculo de cabeza: la fase 1 muestra que, incluso
  entrenado, falla la mitad de los casos.
- **Acción: `guardar_informe`.** Escribe en disco el informe que redacta el modelo. Que lo ha
  hecho se comprueba porque el fichero existe y su tamaño y su SHA-256 coinciden con lo que
  devolvió la herramienta.

## El corpus (fase 3)

- **Fuentes:** el texto consolidado de la Ley 19/2021 del IMV, obtenido de la API de datos
  abiertos del BOE, y las páginas de la Seguridad Social sobre requisitos, cuantías,
  obligaciones y preguntas frecuentes. Prevemos decenas de documentos y varios cientos de
  fragmentos.
- **Formatos:** XML y HTML del BOE, HTML y PDF con texto de la Seguridad Social.
- **Licencia:** la normativa del BOE es reutilizable; la documentación de la Seguridad Social
  es información pública. Conservamos la fuente de cada fragmento para citarlo.
- **Dos preguntas que solo se responden leyendo el corpus:**
  - «Me han concedido el IMV y he encontrado trabajo: ¿qué tengo que comunicar?»
  - «Pedí el IMV hace meses y no me han contestado: ¿qué dice la ley?»

## Qué puede salir mal

1. **El modelo aprende a contestar `0.00`.** Un tercio de los casos de test no tiene derecho a
   la prestación, así que contestar siempre cero da una exactitud aparente. Para detectarlo:
   - un monitor `zero_answer_rate` que compara la tasa de ceros con la esperada;
   - resultados por familia;
   - el conjunto CAPI sin IMV, donde la respuesta nunca es cero.
2. **El modelo profesor se equivoca en los casos de borde.** Generamos varias trazas por caso,
   solo conservamos las que verifica el motor y publicamos la tasa de aceptación por familia.
3. **Poco cómputo.** Para que quepa en la partición de la DGX usamos modelos Qwen3 pequeños
   (4B de profesor, 1.7B de alumno), LoRA, respuestas de 1024 tokens como máximo y puntos de
   control para poder reanudar.

## Por qué este tema

Porque tiene las cuatro piezas que pide la práctica:

- **Fase 1:** una respuesta numérica exacta con reglas que se cruzan.
- **Fase 2:** una API pública real, un cálculo que no conviene dejar al modelo y una acción
  comprobable.
- **Fase 3:** normativa oficial abundante.

Y porque es útil fuera de la asignatura: la complejidad de estas reglas hace que haya
personas con derecho al IMV que no lo piden.
