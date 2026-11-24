# Cuaderno de experimentos — Fase 4 (agente)

Rama `feat/fase4-agent` (encima de `feat/fase3-rag-boe`, que a su vez parte de
`release/fases-1-2-definitiva`). Mismo criterio que el cuaderno de fase 3: entradas fechadas, lo
que se probó, lo que se esperaba y lo que pasó. **Estado a 2026-11-24: implementación y banco
hechos y probados en CPU con un modelo guionizado; ninguna cifra de GPU todavía.** Cada hueco
lleva el nombre del job que lo rellenará (`jobs/200_agent_bench.sh`).

## 1. Qué hay (2026-11-24)

| Pieza | Fichero | Qué hace |
|---|---|---|
| Bucle ReAct | `agent/react_agent.py` | Pensamiento → una acción → observación, hasta `final_answer` o `max_steps` |
| Cerebros | `agent/models.py` | `base`, `rlm` (LoRA GRPO de fase 1), `thinking` (Qwen3 con razonamiento nativo) |
| Memoria | `agent/memory.py` | Dentro de una ejecución y entre turnos (`session_id`), con resumen automático |
| Base de conocimiento | `search_knowledge_base` | El recuperador híbrido de fase 3, fragmentos con id citable |
| Herramientas | fase 2 | `calcular_imv`, `consultar_ayudas` (BDNS), `guardar_informe`, `consultar_boe` |
| Agente de código | `agent/code_agent.py` | `CodeAgent` de smolagents con el mismo modelo, herramientas y timeouts |
| Banco | `agent/bench/build_tasks.py` → `tasks.jsonl` | 20 tareas encadenadas + 3 imposibles |
| Evaluación | `agent/evaluate.py` | Éxito, pasos, tokens y tiempo por agente y cerebro; ablaciones |
| Visor | `agent/ui.py` | Tarjetas de colores por paso contra `POST /agent` |

Tests de CPU: `tests/test_agent.py`, `tests/test_agent_bench.py` y `tests/test_code_agent.py`
(smolagents 1.26 de verdad, modelo guionizado).

## 2. Decisiones de diseño y por qué

- **Los tres cerebros comparten pesos.** `base` y `thinking` son el mismo Qwen3-1.7B (uno con
  `enable_thinking=False` y otro con `True`), y `rlm` es ese modelo más nuestro LoRA GRPO. Así la
  comparación aísla una sola variable: sin razonamiento, con razonamiento nativo o con nuestro
  razonamiento entrenado. Con un 4B «thinking» compararíamos además tamaños.
- **Una copia de cada modelo en la GPU.** La API ya tiene el modelo base de `/tools` y el GRPO de
  `/reasoning`, y el agente carga a través de ellos (`ModelPool._borrow`). Un error que solo
  apareció al revisar el código: `ReasoningModel` guarda la ruta *resuelta* del adaptador y el
  agente lo comparaba con el id del Hub. No coincidían y el agente habría cargado otra copia de
  3,4 GB.
- **Una acción por paso.** Si el modelo escribe varias llamadas, se ejecuta la primera y la
  observación lo dice. La traza queda limpia y cada paso se puede comprobar.
- **Timeouts por herramienta.** Cada llamada va en su propio hilo, con un límite de 10 a 180 s
  según la herramienta (la primera búsqueda carga el embedder). Python no puede matar el hilo, así
  que una llamada colgada termina en segundo plano y se descarta. Además hay un tope de 16 llamadas
  colgadas: al principio usábamos un pool de 4 hilos, y la revisión de seguridad automática
  detectó que 4 herramientas colgadas lo bloqueaban para siempre.
- **Recuperación de errores.** JSON roto, herramienta desconocida, argumentos inválidos,
  excepciones y timeouts vuelven como observación; nada tumba el bucle.
- **Detección de bucles.** Una llamada idéntica a otra anterior no se ejecuta otra vez. A la
  tercera repetición la ejecución se da por fallida en vez de gastar los pasos que quedan. Con
  decodificación voraz es lo esperable: en fase 2 vimos reintentos byte a byte idénticos.
- **Herramientas con efectos.** `guardar_informe` (`requires_confirmation`) solo se ejecuta si la
  tarea pide un informe (el mismo detector que fase 2). La API no tiene un segundo turno para
  preguntar al usuario.
- **Respuesta sin `final_answer`.** Se acepta y se marca en el pensamiento del paso. Los modelos
  pequeños responden a menudo en texto plano, y descartarlo mediría el formato y no la tarea.

**Memoria (política).**
- *Dentro de una ejecución:* la tarea del usuario se conserva siempre literal (perder una fecha o
  una cifra de la petición es el peor fallo de un resumen) y los últimos 6 mensajes también. Lo de
  en medio se resume con el mismo cerebro cuando el contexto pasa de 6000 tokens
  (`ARCA_AGENT_TOKEN_BUDGET`). Antes de resumir, cada observación se recorta a 1500 caracteres: un
  artículo del BOE o una lista de convocatorias ocupan casi todo el contexto, y el modelo ya usó lo
  que necesitaba. Si la llamada de resumen falla, queda un digest mecánico. El resumen tiene un
  tope de 4000 caracteres.
- *Entre turnos (`session_id`, emitido por el servidor: se pide con `"new"` y un id no emitido da 404, para que nadie lea otra conversación adivinando su id):* de cada turno se guardan la tarea y la respuesta final, no el
  tráfico de herramientas. Pasados 1500 tokens (`ARCA_AGENT_SESSION_BUDGET`), los turnos más
  antiguos se resumen salvo los dos últimos. Las sesiones viven solo en memoria (100 como máximo,
  se descarta la menos usada) porque pueden contener datos personales.

## 3. El banco (2026-11-24)

`agent/bench/build_tasks.py` genera `tasks.jsonl`. Los importes esperados los calcula el motor
determinista al generar el banco, no se escriben a mano; los datos legales salen del conjunto
dorado de fase 3 (evidencia literal del corpus).

- **20 tareas encadenadas (T01-T20), cada una con al menos dos herramientas distintas:** base de
  conocimiento + cálculo (9), BOE + cálculo (3), cálculo + BDNS (2), cálculo + informe (1), base
  de conocimiento + cálculo + informe (1), BDNS + informe (1), BOE + base de conocimiento (1),
  base de conocimiento + BDNS (1) y cálculo + base de conocimiento (1). Tres (T13, T14, T20) dan
  el caso en prosa y no en JSON, para medir el fallo de extracción que fase 2 vio con 0/6.
- **3 imposibles (I01-I03):** datos ausentes, una prestación que el corpus no cubre (Renta
  Valenciana de Inclusión) y una acción sin herramienta (presentar la solicitud). Se considera
  acierto si el agente dice que no puede.
- **Comprobación automática:** números con formato español o inglés (`733,60`, `26.409,60`),
  expresiones regulares, y combinaciones `all` / `any`. Una denegación se acepta con «0» o con un
  «no tiene derecho».
- **Importes del motor:** A 733,60 · B 922,57 · C 1008,84 · E 1063,40 €/mes; D (ingresos) y F
  (patrimonio) denegados.
- **Limitación:** las tareas de BDNS y BOE dependen de servicios externos en vivo; una caída
  durante la ejecución cuenta como fallo del agente. Por eso hay una ablación con la BDNS rota a
  propósito.

## 4. Hipótesis (escritas antes de medir)

1. `thinking` gana a `base` en tareas de varios pasos, pero gasta bastantes más tokens por paso
   (2048 de presupuesto frente a 1024).
2. `rlm` puede **perder** frente a `base`: se entrenó para cerrar con `<answer>` sobre problemas de
   cálculo, no para emitir `<tool_call>`. Si es así, la respuesta a «¿ayuda haber entrenado el
   modelo?» es que no ayuda fuera de su tarea. Es un resultado tan válido como el contrario.
3. Las tareas en prosa (T13, T14, T20) fallan en los tres cerebros, como en fase 2.
4. **Código frente a JSON:** CodeAct promete menos pasos. Hemos visto una trampa concreta en los
   tests: el banco da los casos en JSON y, copiados tal cual en Python, `true` no existe. El agente
   de código tiene que usar `json.loads` o traducirlos. Esperamos menos pasos pero también errores
   de este tipo.
5. Con la BDNS rota, un buen agente responde con lo que sí obtuvo (el cálculo) e indica el fallo;
   uno malo repite la llamada hasta el tope o inventa convocatorias.
6. Con el presupuesto de memoria a 1500 tokens el éxito baja poco y los tokens totales suben (las
   llamadas de resumen cuestan).

## 5. Resultados — pendiente (job `200_agent_bench`)

Banco completo, `max_steps = 8`, decodificación voraz, DGX MIG 1g.18gb.

| agente | cerebro | éxito (T) | imposibles OK | cadena completa | pasos | tokens | s/tarea |
|---|---|---:|---:|---:|---:|---:|---:|
| json | base | *job 200* | | | | | |
| json | rlm | *job 200* | | | | | |
| json | thinking | *job 200* | | | | | |
| code | base | *job 200* | | | | | |

Ablaciones (cerebro base): BDNS rota en T03/T09/T12/T18 (`phase4_broken.json`) y memoria a
1500 tokens (`phase4_memory.json`).

**Qué analizar cuando lleguen:** qué cerebro gana y a qué coste; cinco fallos concretos con su
traza (qué hizo el agente cuando falló); si las tareas en prosa fallan por extracción o por el
bucle; qué hace ante la BDNS rota; cuántas veces actuó cada salvaguarda (repeticiones, «no
ejecutado», timeouts) y si la memoria cambió el resultado.

## 6. Cómo reproducir

```bash
uv run python -m agent.bench.build_tasks          # regenera tasks.jsonl con el motor
uv run --extra train --extra rag --extra agent python -m agent.evaluate \
    --agents json --brains base rlm thinking --out reports/phase4_eval.json
uv run --extra agent python -m agent.ui           # visor contra la API (ARCA_API_URL)
```
