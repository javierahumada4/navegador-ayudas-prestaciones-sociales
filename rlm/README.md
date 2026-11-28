# Fase 1 — Razonamiento: calcular IMV + CAPI

Convertimos Qwen3-1.7B en un modelo que razona entre `<think>` y `</think>` y da el importe
mensual de IMV + CAPI entre `<answer>` y `</answer>`. El camino es el de DeepSeek-R1 a escala
pequeña:

1. un motor de reglas que sirve de verificador;
2. trazas de un modelo profesor filtradas con ese verificador;
3. SFT con LoRA sobre esas trazas;
4. GRPO con recompensas verificables.

Los datos, informes y curvas de la ejecución final están en
[`results/rlm_final/`](../results/rlm_final/).

## La tarea

**Entrada:** un caso de nueva solicitud en 2026 redactado en lenguaje natural:

- fecha de solicitud;
- miembros del hogar, con fecha de nacimiento y edad;
- relaciones y tipo de custodia;
- residencia legal y efectiva de cada miembro;
- ingresos anuales computables, patrimonio neto y activos no societarios;
- y, cuando aplican, discapacidad, pensiones, administración de una sociedad o un progenitor
  ingresado en un centro.

**Salida:** un único número, el total mensual IMV + CAPI con dos decimales. Si no hay derecho
a ninguna de las dos prestaciones, la salida es `0.00`.

```
<think>… razonamiento …</think><answer>431.26</answer>
```

## El motor de referencia

[`imv_engine.py`](imv_engine.py) aplica la Ley 19/2021 con las cuantías de 2026, que están en
[`rulesets/imv_2026.json`](rulesets/imv_2026.json): si cambian las cifras, se cambia el JSON y
no el código.

Comprueba estos requisitos:

- unidad de convivencia, edad e independencia del solicitante;
- residencia de cada miembro y antigüedad de la convivencia;
- renta garantizada, con los complementos por discapacidad ≥ 65 % (+22 %) y por
  monoparentalidad;
- mínimo de 10 € para cobrar IMV;
- límites de patrimonio (`<`) y de activos (`<=`) según la composición del hogar;
- exclusión por administrar una sociedad activa;
- tope por pensiones;
- CAPI por tramos de edad, con sus propios límites de renta (300 %) y patrimonio (150 %).

**El redondeo forma parte del contrato:**

- todos los importes se calculan con `Decimal` y se redondean a céntimos con `ROUND_HALF_UP`;
- `ingresos / 12` no se redondea;
- el mínimo de 10 € se comprueba antes de redondear el IMV;
- el total es la suma del IMV redondeado y el CAPI redondeado.

Así el generador, la recompensa y el verificador nunca discrepan en un céntimo. Los 22 tests
de [`tests/test_imv_engine.py`](../tests/test_imv_engine.py) cubren cada rama del motor.

**Fuera de alcance:** revisiones anuales, incentivo al empleo, suspensión, extinción,
reintegros y excepciones documentales. Son procedimientos distintos del cálculo de una nueva
solicitud. Tampoco implementamos la exención de residencia previa de los recién nacidos
(art. 10.1.a). Según nuestra revisión, el generador no produce casos que la necesiten, pero
`calcular_imv` sí podría recibirlos en la fase 2.

## El dataset

[`generate_problems.py`](generate_problems.py) muestrea los hechos de un hogar, redacta el
enunciado con una de varias plantillas y obtiene la etiqueta ejecutando el motor. No hay
anotación manual.

| Conjunto | Casos | Contenido |
|---|---:|---|
| `train` | 800 | Diez familias, 80 casos cada una |
| `test` | 200 | Las mismas diez familias, 20 casos cada una |
| `ood` | 50 | CAPI sin IMV: ingresos que impiden el IMV pero no el CAPI. No aparece en train |
| `ood_threshold` | 50 | Dos condiciones a la vez cerca de su umbral |
| `ood_wording` | 50 | Los mismos hechos con otra redacción |

**Las diez familias:**

- ordinario;
- beneficiario individual;
- monoparental;
- monoparental con el otro progenitor en prisión u hospital;
- discapacidad;
- fallo por ingresos;
- fallo por residencia;
- fallo por patrimonio o activos;
- administrador de sociedad;
- borde: renta, patrimonio, activos o residencia a un céntimo o un día del umbral.

El 32 % de los casos de test tienen respuesta `0.00`.

[`validate_data.py`](validate_data.py) vuelve a calcular cada etiqueta con el motor y
comprueba tres cosas:

- **No hay casos repetidos entre conjuntos**, ni en los parámetros ni en el texto.
- **Todo hecho que usa el motor aparece en el enunciado.** Por ejemplo, la edad de cada
  menor a 1 de enero, que decide el tramo del CAPI.
- **La respuesta nunca aparece dentro de la pregunta.**

El informe está en
[`dataset_validation.json`](../results/rlm_final/reports/dataset_validation.json).

## El verificador

`IMVAmountVerifier` en [`verifier.py`](verifier.py):

1. extrae la respuesta de `<answer>`;
2. acepta formatos decimales españoles e ingleses;
3. la convierte a `Decimal`;
4. la redondea a céntimos con `ROUND_HALF_UP`;
5. la compara con la etiqueta.

`imv_accuracy_reward` usa exactamente la misma comparación.

## Destilación

**Profesor:** Qwen3-4B en modo thinking. Genera 8 trazas por cada uno de los 800 problemas de
entrenamiento, con un máximo de 1024 tokens ([`distill.py`](distill.py)).

**Hoja de reglas:** profesor y alumno reciben, delante del enunciado, la misma hoja de reglas
generada desde el ruleset: cuantías, límites por composición, desigualdades exactas y
convención de redondeo. **Es una decisión de diseño:** la tarea es *aplicar* las reglas a un
hogar concreto, no memorizar tablas de umbrales que cambian cada año. El alumno la recibe
también en SFT, GRPO, evaluación y `/reasoning`.

**Resultado:** se aceptan **1430 de 6400 trazas (22,3 %)**, que cubren 420 problemas
distintos de las diez familias.

| Causa de rechazo | Trazas |
|---|---:|
| `<think>` sin cerrar (se acabaron los 1024 tokens) | 3971 |
| Importe incorrecto | 868 |
| Sin respuesta | 131 |

**El cuello de botella es la longitud, no la exactitud:** el 63 % de las generaciones llegó
al máximo de tokens.

| Familia | Aceptación | | Familia | Aceptación |
|---|---:|---|---|---:|
| individual | 76,6 % | | ordinario | 17,8 % |
| administrador | 56,1 % | | ingresos | 17,5 % |
| discapacidad | 13,8 % | | patrimonio | 13,8 % |
| monoparental | 9,5 % | | monoparental en centro | 8,9 % |
| borde | 8,8 % | | **residencia** | **0,8 %** (5 de 640) |

## SFT

[`train_sft.py`](train_sft.py) entrena un LoRA de rango 16 (alpha 32, todas las capas
lineales) sobre Qwen3-1.7B:

- 2 épocas, tasa de aprendizaje 2e-4, lote efectivo de 8;
- contexto de hasta 8192 tokens, porque la hoja de reglas es larga;
- respuestas de como máximo 1024 tokens.

**Selección de trazas.** [`sft_selection.py`](sft_selection.py) se queda con 2 trazas
verificadas distintas por problema como máximo, preferiblemente las cortas: **708 trazas
sobre 418 problemas**.

**Cobertura.** Es el **52 % de los problemas**, por debajo del 70 % que exige el pipeline en
modo estricto. Seguimos en modo `warn` y lo dejamos registrado en
[`sft_selection.json`](../results/rlm_final/reports/sft_selection.json).

**Curvas.** La pérdida baja de 0,50 a 0,23 y la exactitud por token sube de 0,85 a 0,92
([`sft_curves.png`](../results/rlm_final/reports/sft_curves.png)).

## GRPO

[`train_grpo.py`](train_grpo.py) parte del adaptador SFT y usa `GRPOTrainer` de TRL:

- 300 pasos;
- 4 generaciones por problema;
- temperatura 0,6, `top_p` 0,95 y `top_k` 20;
- tasa de aprendizaje 5e-6;
- `epsilon` 0,2 y `beta` 0.

**Recompensas:**

| Recompensa | Peso | Qué mide |
|---|---:|---|
| `imv_format_reward` | 1,0 | Un único `<think>…</think>` seguido de un único `<answer>` con un importe `123.45` |
| `imv_accuracy_reward` | 2,0 | Importe correcto según el verificador |
| `domain_reward` | 0,5 | Respuesta limpia para una máquina: exactamente dos decimales, punto decimal, sin símbolo ni texto |
| `zero_answer_rate` | 0 | Monitor, no recompensa: fracción de respuestas `0.00` |

**Por qué estas recompensas:**

- **La exactitud pesa el doble** porque es lo que queremos. El formato es condición para poder
  leer la respuesta.
- **La recompensa de dominio** exige el contrato de salida que necesitan el verificador y la
  fase 2. En la práctica se solapa con la de formato, que ya exige un importe de dos
  decimales; solo añade señal cuando el formato falla pero la respuesta está limpia. Se
  satura en torno a 0,9-1,0 hacia el paso 40.
- **`zero_answer_rate`** vigila el *reward hacking* más barato de este dominio: contestar
  siempre `0.00` acierta un tercio de los casos.

**El paso de GRPO a mano** está en [`grpo_step.py`](grpo_step.py):

- ventajas normalizadas dentro del grupo;
- ratio de políticas por token;
- objetivo recortado;
- KL opcional respecto a la política de referencia;
- media por respuesta y luego por grupo, aplicando la máscara de tokens válidos;
- estadísticas de recorte.

Los 10 tests de [`tests/test_grpo_step.py`](../tests/test_grpo_step.py) comprueban cada pieza,
incluidos los casos límite: grupo de un solo elemento, máscara vacía y `beta > 0` sin
política de referencia.

**Curvas** ([`grpo_curves.png`](../results/rlm_final/reports/grpo_curves.png)), con media
móvil de 5 pasos:

| Curva | Evolución |
|---|---|
| Recompensa total | De ~1,1 a una media de ~2,4. Muy ruidosa |
| Recompensa de exactitud | De 0 a ~0,5 |
| Longitud media | De ~870 a ~650 tokens |
| Respuestas truncadas | Del ~25 % al ~5 % |

**La fracción recortada es exactamente 0 en todo el entrenamiento, por construcción.** TRL
hace una sola actualización por lote de generaciones (`num_iterations=1`), así que la
política que se actualiza es la misma que generó y el ratio vale 1. Con `beta=0` tampoco hay
KL. En esta configuración GRPO se reduce a un gradiente de política con la media del grupo
como línea base. El recorte y el KL están implementados y probados en `grpo_step.py`, pero
esta ejecución no los usa.

**Grupos sin señal.** Entre el 20 % y el 60 % de los grupos no aportan señal: con 4
generaciones, a menudo las cuatro aciertan o las cuatro fallan, y todas las ventajas valen 0.
Con 300 pasos de un problema cada uno, GRPO ve 300 de los 800 problemas, una sola vez.

## Evaluación

[`evaluate.py`](evaluate.py) evalúa los tres modelos con las mismas preguntas: una muestra por
problema, temperatura 0,6, semilla 42 y 1024 tokens como máximo. Resultados completos en
[`eval_test.md`](../results/rlm_final/reports/eval_test.md).

| Modelo | pass@1 | Formato válido | Truncadas | Responde `0.00` (esperado 0,32) | Tokens medios |
|---|---:|---:|---:|---:|---:|
| Base | 0,065 | 0,115 | 0,890 | 0,015 | 1001 |
| SFT | 0,375 | 0,685 | 0,320 | 0,130 | 783 |
| SFT + GRPO | **0,490** | **0,920** | **0,080** | 0,145 | 677 |

**Por familia:**

| Familia | Base | SFT | GRPO |
|---|---:|---:|---:|
| individual | 0,50 | 0,95 | 1,00 |
| administrador | 0,05 | 0,90 | 0,95 |
| ingresos | 0,05 | 0,30 | 0,70 |
| ordinario | 0,00 | 0,35 | 0,55 |
| borde | 0,00 | 0,20 | 0,35 |
| discapacidad | 0,00 | 0,25 | 0,35 |
| monoparental | 0,00 | 0,25 | 0,35 |
| patrimonio | 0,05 | 0,25 | 0,35 |
| monoparental en centro | 0,00 | 0,30 | 0,30 |
| **residencia** | 0,00 | 0,00 | **0,00** |

**Fuera de distribución:**

| Conjunto (50 casos) | Base | SFT | GRPO |
|---|---:|---:|---:|
| CAPI sin IMV | 0,00 | 0,46 | 0,50 |
| Dos condiciones en el umbral | 0,00 | 0,22 | 0,30 |
| Otra redacción | 0,08 | 0,56 | 0,54 |

**Cómo leer las tablas:**

- **El modelo base casi nunca termina.** El 89 % de sus respuestas se trunca a los 1024
  tokens.
- **SFT le enseña el formato y a ser breve.** El pass@1 pasa de 0,065 a 0,375.
- **GRPO aporta sobre todo respuestas terminadas.** Las truncadas bajan de 64 a 16 y los
  aciertos suben de 75 a 98. Los errores
  `wrong_amount` (de 36 a 51) y `missed_ineligibility` (de 19 a 31) suben, porque ahora hay
  más respuestas completas que pueden equivocarse.
- **No hay reward hacking hacia el cero.** El modelo contesta `0.00` menos de lo que debería
  (0,145 frente a 0,32). Su error típico es el contrario: dar importe a quien no tiene
  derecho.
- **Lo aprendido generaliza.** En CAPI sin IMV, una familia que nunca vio, GRPO llega a 0,50.
  El cambio de redacción apenas afecta. Los casos con dos condiciones en el umbral son los
  más difíciles.
- **Residencia: 0 en los tres modelos.** Es consecuencia directa de la destilación: el
  profesor solo acertó 5 de 640 casos y el SFT apenas tiene ejemplos. GRPO no lo arregla
  porque, sin aciertos, sus grupos no tienen señal.

## Cinco fallos analizados

Son fallos de GRPO sobre el conjunto de test. Están en `failures_to_analyse` de
[`eval_test.json`](../results/rlm_final/reports/eval_test.json).

1. **Ignora la residencia** (caso 8, `residence_fail`; esperado 0.00, responde 526.10).
   - **El caso:** un menor nacido en 2025 solo tiene residencia legal desde junio de 2026.
   - **Lo que hace el modelo:** comprueba patrimonio, activos y edades, pero no la residencia
     de cada miembro, y calcula IMV y CAPI como si cumpliera.
   - **Por qué:** es el patrón de toda la familia, y viene de no tener trazas correctas de
     ella.
2. **Usa la renta garantizada de un adulto solo para una unidad de dos** (caso 20,
   `monoparental_institution`; esperado 410.68, responde 190.61).
   - **Lo que hace el modelo:** reconoce bien el supuesto monoparental (el otro progenitor
     lleva 19 meses hospitalizado), pero parte de 733,60 € en lugar de 953,68 €.
   - **Consecuencia:** el IMV le sale 110,11 € en lugar de 330,18 €.
3. **Se contradice con el tramo del CAPI** (caso 16, `monoparental`; esperado 582.23,
   responde 489.84).
   - **Lo que hace el modelo:** escribe que a dos niños de 4 años les corresponden 57,50 €
     cada uno, y luego suma 115 + 115.
   - **Lo correcto:** 80,50 € cada uno.
   - **Además:** olvida el complemento de monoparentalidad en la renta garantizada.
4. **Resta al revés** (caso 25, `ordinary`; esperado 244.87, responde 136.21).
   - **Lo que hace el modelo:** para un adulto con un menor, usa la renta garantizada de un
     adulto solo y calcula «ingresos − renta garantizada» (789,31 − 733,60 = 55,71) en lugar
     de «renta − ingresos».
   - **Lo correcto:** con la renta de 953,68 €, el IMV es 164,37 €.
5. **Tramo de edad del CAPI y comparación incoherente** (caso 36, `disability`; esperado
   301.30, responde 358.80).
   - **Acierta:** el complemento del 22 % por discapacidad del 70 %, y el IMV (243,80).
   - **Falla:** asigna 115 € de CAPI a un menor de 16 años, cuando le corresponden 57,50 €.
   - **Además:** antes escribe que el patrimonio «está por encima del límite, así que está
     por debajo».

**Lo que tienen en común:**

- **Los errores son de lookup en tablas** (renta por número de miembros, tramos del CAPI) y
  de omitir un requisito. Casi nunca son de aritmética.
- **El razonamiento sale en inglés**, aunque el prompt pide español. No filtramos las trazas
  por idioma y el alumno heredó el idioma del pensamiento nativo del profesor.

## Limitaciones conocidas

- **Cobertura del SFT:** queda en el 52 %. Para subirla haría falta dar más tokens al
  profesor (el 63 % se trunca) o cambiar de profesor.
- **GRPO ve poco:** 300 problemas, una vez cada uno, con 4 generaciones. Más pasos o grupos
  más grandes darían más señal, sobre todo en las familias difíciles.
- **Verificador y recompensa no extraen la respuesta igual:** el verificador toma el último
  `<answer>` y la recompensa exige que haya uno solo. En las cuatro evaluaciones no hay
  ninguna respuesta correcta con formato inválido, así que la diferencia no cambia ningún
  resultado.

## Reproducirlo

[`pipeline.py`](pipeline.py) encadena las etapas y escribe todo en un directorio de ejecución:

1. generación y validación;
2. destilación;
3. SFT;
4. GRPO;
5. evaluación y curvas.

Se puede reanudar desde cualquier etapa. La destilación y el entrenamiento necesitan
dependencias incompatibles, vLLM y TRL, así que el pipeline prepara dos entornos `uv`
separados dentro de la ejecución.

```bash
make rlm-plan                                       # imprime los comandos, sin GPU
python -m rlm.pipeline --smoke --run-dir runs/smoke # ensayo corto en GPU
python -m rlm.pipeline --run-dir runs/phase1        # ejecución completa (~12 h en la DGX)
python -m rlm.pipeline --run-dir runs/phase1 --from-stage grpo --resume
```

La ejecución publicada en `results/rlm_final/` usó exactamente los valores por defecto. Su
`manifest.json` guarda cada comando con su duración.

**Sale de la ejecución:**

- `data/`: conjuntos y trazas;
- `weights/sft_lora` y `weights/grpo_lora`;
- `reports/`: validaciones, `eval_*.json` y `.md`, y curvas.

Los adaptadores no se guardan en Git. Están publicados en Hugging Face en
`JES0406/imv-sft-lora`, subcarpetas `sft_final` y `grpo_final`.

**El endpoint.** `/reasoning` usa [`inference.py`](inference.py), con
`ARCA_RLM_ADAPTER=JES0406/imv-sft-lora/grpo_final` y `ARCA_RLM_VERIFIER=imv`. Si la petición
trae `expected_answer`, la respuesta incluye el veredicto del verificador.

## Ficheros

| Fichero | Qué hace |
|---|---|
| `imv_engine.py`, `rulesets/` | Motor de referencia y cuantías 2026 |
| `generate_problems.py`, `validate_data.py` | Generación y validación del dataset |
| `verifier.py`, `rewards.py` | Verificador y recompensas de formato y exactitud |
| `data.py`, `chat_template.py` | Prompts del alumno y plantilla de chat de Qwen3 |
| `distill.py` | Trazas del profesor filtradas por el verificador |
| `sft_selection.py`, `train_sft.py` | Selección de trazas y SFT con LoRA |
| `train_grpo.py`, `grpo_step.py` | GRPO con TRL y paso de GRPO escrito a mano |
| `evaluate.py`, `curves.py` | pass@1 por modelo y familia, clasificación de fallos y curvas |
| `pipeline.py` | Orquestación reproducible de todas las etapas |
| `inference.py` | Modelo detrás de `/reasoning` |
