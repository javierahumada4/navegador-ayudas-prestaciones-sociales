# EXPERIMENTS.md — cuaderno de experimentos

Una entrada por ejecución o decisión, de la más antigua a la más reciente. El detalle de cada
fase está en [`rlm/README.md`](rlm/README.md) y [`tool_use/README.md`](tool_use/README.md).

---

### 2026-11-10 · Fase 1 · Dataset generado con el motor

**Qué queríamos saber.** Si el generador produce un dataset equilibrado, sin fugas y con
etiquetas que el motor confirma.

**Qué hicimos.** Etapa `generate` de `python -m rlm.pipeline` con los valores por defecto y
semilla 42. Es la ejecución que se publica en `results/rlm_final/`:

- 800 casos de `train` y 200 de `test`, de diez familias con cuotas iguales;
- tres conjuntos fuera de distribución de 50 casos: CAPI sin IMV, dos condiciones en el
  umbral y otra redacción.

`rlm.validate_data` vuelve a calcular cada etiqueta con el motor.

**Qué pasó.** Las cinco particiones pasan la validación
([`dataset_validation.json`](results/rlm_final/reports/dataset_validation.json)):

- 80 casos por familia en `train` y 20 en `test`;
- ningún caso repetido entre conjuntos;
- ninguna respuesta dentro de su pregunta.

El 32 % de los casos de test tienen respuesta `0.00`.

**Qué concluimos.** Contestar siempre `0.00` acertaría un tercio del test. Por eso vigilamos
la tasa de ceros durante GRPO y en la evaluación.

---

### 2026-11-10 · Fase 1 · Destilación con Qwen3-4B

**Qué queríamos saber.** Cuántas trazas correctas consigue el profesor y en qué familias
falla.

**Qué hicimos.** Qwen3-4B en modo thinking, 8 muestras por problema de `train`, 1024 tokens
como máximo. Profesor y alumno reciben la hoja de reglas 2026 delante del enunciado. Se
conservan las trazas cuyo importe acepta `IMVAmountVerifier`. Tardó 4 h 36 min.

**Qué pasó.**

- **Se aceptan 1430 de 6400 trazas (22,3 %),** que cubren 420 problemas.
- **El 63 % de las generaciones llega al máximo de tokens.** 3971 trazas se rechazan por
  `<think>` sin cerrar y solo 868 por importe incorrecto.
- **La aceptación por familia va del 77 % (beneficiario individual) al 0,8 % (fallo por
  residencia, 5 de 640).**
- **El profesor razona en inglés,** aunque el prompt pide español.

([`distillation_validation.json`](results/rlm_final/reports/distillation_validation.json))

**Qué concluimos.**

- **El límite es la longitud, no la exactitud.** Con más tokens el profesor terminaría más
  trazas, pero la ejecución sería varias horas más larga.
- **No filtramos por idioma:** perderíamos trazas correctas.
- **Residencia apenas tendrá ejemplos.** Lo esperamos ver reflejado en la evaluación.

---

### 2026-11-10 · Fase 1 · SFT con LoRA

**Qué queríamos saber.** Si el alumno aprende el formato y a terminar dentro de 1024 tokens.

**Qué hicimos.** LoRA de rango 16 sobre Qwen3-1.7B:

- 2 épocas, tasa de aprendizaje 2e-4, lote efectivo de 8;
- contexto de 8192 tokens, por la hoja de reglas;
- hasta 2 trazas distintas por problema, preferiblemente cortas.

**Qué pasó.**

- **708 trazas sobre 418 problemas.** La cobertura es del 52 %, por debajo del 70 % que exige
  el pipeline en modo estricto. Lo ejecutamos en modo `warn`.
- **La pérdida baja de 0,50 a 0,23** y la exactitud por token sube de 0,85 a 0,92
  ([`sft_curves.png`](results/rlm_final/reports/sft_curves.png)).
- **Tardó 22 minutos.**

**Qué concluimos.** Seguimos con GRPO sobre este adaptador. La baja cobertura queda
registrada como limitación: es consecuencia directa de la tasa de aceptación de la
destilación.

---

### 2026-11-10 · Fase 1 · GRPO desde el adaptador SFT

**Qué queríamos saber.** Si las recompensas verificables mejoran la exactitud sobre SFT.

**Qué hicimos.** `GRPOTrainer` de TRL:

- 300 pasos de un problema cada uno, 4 generaciones;
- temperatura 0,6, tasa de aprendizaje 5e-6;
- `epsilon` 0,2 y `beta` 0;
- recompensas de formato (1,0), exactitud (2,0) y dominio (0,5), y el monitor
  `zero_answer_rate` con peso 0.

Tardó 2 h 48 min.

**Qué pasó** ([`grpo_curves.png`](results/rlm_final/reports/grpo_curves.png)):

| Curva | Evolución |
|---|---|
| Recompensa total | De ~1,1 a ~2,4 de media, muy ruidosa |
| Exactitud | De 0 a ~0,5 |
| Longitud | De ~870 a ~650 tokens |
| Truncadas | Del ~25 % al ~5 % |
| Grupos sin señal | Entre el 20 % y el 60 % |
| Fracción recortada | **0 en todos los pasos** |

**Qué concluimos.**

- **El recorte es 0 por construcción, no por un error.** TRL actualiza una sola vez por lote
  de generaciones (`num_iterations=1`), así que el ratio de políticas vale 1. Sin KL
  (`beta=0`), el algoritmo queda en un gradiente de política con la media del grupo como
  línea base. El recorte y el KL están implementados y probados en `rlm/grpo_step.py`.
- **Muchos grupos no tienen señal.** Con 4 generaciones, a menudo todas aciertan o todas
  fallan, y la ventaja es 0. Más generaciones por grupo darían señal en las familias
  difíciles.
- **La recompensa de dominio se satura hacia el paso 40.** Se solapa con la de formato.

---

### 2026-11-11 · Fase 1 · Evaluación de base, SFT y GRPO

**Qué queríamos saber.** El pass@1 de cada modelo en test y fuera de distribución.

**Qué hicimos.** `rlm.evaluate`: una muestra por problema, temperatura 0,6, 1024 tokens y el
mismo verificador. Base, SFT y GRPO sobre `test` y los tres conjuntos fuera de distribución.

**Qué pasó**
([`eval_test.md`](results/rlm_final/reports/eval_test.md) y `eval_ood*.md`):

| Modelo | Test | Truncadas | CAPI sin IMV | Umbrales | Redacción |
|---|---:|---:|---:|---:|---:|
| Base | 0,065 | 0,89 | 0,00 | 0,00 | 0,08 |
| SFT | 0,375 | 0,32 | 0,46 | 0,22 | 0,56 |
| GRPO | 0,490 | 0,08 | 0,50 | 0,30 | 0,54 |

- **GRPO convierte respuestas truncadas en terminadas:** de 64 a 16 truncadas, de 75 a 98
  aciertos.
- **Por familia,** va de 1,00 (beneficiario individual) a **0,00 en residencia en los tres
  modelos**.
- **GRPO contesta `0.00` el 14,5 % de las veces, frente al 32 % esperado.** Su error más
  frecuente después del importe equivocado es dar importe a quien no tiene derecho (31
  casos).

**Qué concluimos.**

- **No hay reward hacking hacia el cero.** El sesgo es el contrario.
- **Residencia es 0 porque el profesor no dio trazas** y GRPO no puede aprender sin aciertos
  en el grupo.
- **El modelo generaliza a la familia que no vio** (0,50 en CAPI sin IMV).
- **Los cinco fallos que analizamos son de lookup en tablas,** no de aritmética: renta por
  número de miembros, tramos del CAPI. Están en `rlm/README.md`.

---

### 2026-11-18 · Fase 2 · Con y sin herramientas, modelo base y LoRA GRPO

**Qué queríamos saber.** Cuánto aportan las herramientas y qué cerebro usar en `/tools`.

**Qué hicimos.** `tool_use.compare` sobre el banco de 40 casos:

- Qwen3-1.7B sin adaptador y con el LoRA GRPO de la fase 1;
- cada uno con herramientas (hasta 5 turnos) y sin ellas;
- decodificación greedy y la BDNS en vivo.

**Qué pasó**
([`phase2_base.json`](results/fase2/sin_definicion/phase2_base.json),
[`phase2_grpo.json`](results/fase2/sin_definicion/phase2_grpo.json)):

| Configuración | Éxito de principio a fin |
|---|---:|
| Base con herramientas | 0,675 |
| Base sin herramientas | 0,150 |
| GRPO con herramientas | 0,625 |
| GRPO sin herramientas | 0,125 |

- **La selección de herramienta es 0,95 en los dos cerebros.**
- **Con GRPO, los casos de varios pasos caen de 0,83 a 0,50.**

**Qué concluimos.**

- **Las herramientas son imprescindibles:** sin ellas el modelo inventa cuantías.
- **Para `/tools` usamos el modelo base.** El LoRA de la fase 1 arrastra el formato
  `<answer>`, da un solo número y se salta pasos.

---

### 2026-11-28 · Fase 2 · Revisión de las métricas

**Qué queríamos saber.** Si los 13 fallos del modelo base eran todos del modelo.

**Qué hicimos.** Revisamos cada fallo a mano.

- **4 eran del evaluador:**
  - dos búsquedas en la BDNS penalizadas por una preposición en `tema`;
  - dos aclaraciones correctas que no usaban ningún verbo de la lista.
- **Un caso del banco (P2-008) tenía en la referencia un dato que la consulta no da.**

Corregimos las tres cosas y escribimos `tool_use.rescore`, que recalcula las métricas sobre
las salidas guardadas sin volver a ejecutar el modelo. Con las métricas anteriores, reproduce
exactamente las cuatro tablas originales.

**Qué pasó**
([`phase2_rescored.json`](results/fase2/sin_definicion/phase2_rescored.json)):

- **Éxito de principio a fin:** el base pasa de 0,675 a 0,775 y el GRPO de 0,625 a 0,700.
- **Sin herramientas no cambia nada.**
- **Una versión intermedia de la lista de verbos aceptaba «proporcionado».** Daba por buena
  una respuesta que no preguntaba nada y se inventaba un «Impuesto sobre el Móvil». La
  restringimos a las formas de petición.

**Qué concluimos.** Quedan 9 fallos y todos son del modelo:

- alucina definiciones cuando no hay herramienta;
- a veces se salta el motor;
- se inventa un umbral en un informe;
- ante un error de validación pide ayuda en lugar de corregirse.

---

### 2026-11-28 · Fase 2 · Definición de IMV y CAPI en el prompt

**Qué queríamos saber.** Cómo evitar que el modelo invente qué significan IMV y CAPI. Lo hace
en los dos casos conceptuales del banco: «Ingreso Mensual de Vida», «Cálculo de Apoyos para
la Vida y la Independencia».

**Qué hicimos.** Añadimos al prompt de sistema de `tool_use/executor.py` una definición de
una línea de cada prestación. Repetimos `tool_use.compare` en la DGX con los dos cerebros,
con las métricas corregidas.

**Qué pasó**
([`phase2_base.json`](results/fase2/phase2_base.json),
[`phase2_grpo.json`](results/fase2/phase2_grpo.json)):

| | Sin definición | Con definición |
|---|---:|---:|
| Base, éxito de principio a fin | 0,775 | 0,750 |
| Base, hechos conceptuales | 0 / 2 | 2 / 2 |
| GRPO, éxito de principio a fin | 0,700 | 0,650 |
| GRPO, hechos conceptuales | 0 / 2 | 2 / 2 |

- **P2-036 y P2-037 pasan en los dos modelos.** El IMV se define bien.
- **En el base mejora además P2-014,** que ahora sí llama al motor.
- **Empeoran P2-009, P2-010, P2-024 y P2-034:**
  - no da el CAPI como 0,00 €;
  - no explica por qué no hay IMV;
  - deja una llamada sin cerrar;
  - intenta guardar el informe antes de buscar en la BDNS.
- **En GRPO mejoran 4 casos y empeoran 6.**
- **Sin herramientas no cambia nada.**

**Qué concluimos.**

- **La definición arregla el error que buscábamos,** que era sistemático.
- **Los casos que empeoran no son errores nuevos.** Repiten tipos de fallo que ya aparecían
  en otros casos: con decodificación greedy, cualquier cambio de prompt redistribuye los
  casos dudosos de un modelo de 1,7B.
- **Mantenemos el cambio y publicamos esta ejecución como la principal.** La anterior queda
  en [`results/fase2/sin_definicion/`](results/fase2/sin_definicion/).
- **Una diferencia de uno o dos casos entre configuraciones no es concluyente** con una sola
  ejecución.
