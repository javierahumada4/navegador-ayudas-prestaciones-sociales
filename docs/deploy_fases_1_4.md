# Despliegue de las fases 1 a 4

Rama `feat/fase4-agent`. Es `release/fases-1-2-definitiva` (fases 1 y 2, ver
[`deploy_fases_1_2.md`](deploy_fases_1_2.md)) más el RAG del BOE de fase 3 (`feat/fase3-rag-boe`)
y el agente de fase 4. Endpoints: `/health`, `/reasoning`, `/tools`, `/rag`, `/agent` y `/docs`.

## En la DGX (sin Docker)

El contenedor OnDemand de la DGX no ejecuta Docker. La API se levanta con `smoke/serve_ngrok.sh`,
que en esta rama instala los extras `train` y `rag` (configurable con `ARCA_API_EXTRAS`):

```bash
git fetch origin && git switch -C fase4 origin/feat/fase4-agent
# .env: NGROK_AUTHTOKEN, HF_TOKEN, ARCA_RLM_ADAPTER=JES0406/imv-sft-lora/grpo_final,
#       ARCA_TOOLS_BRAIN=base, ARCA_TOOLS_ENABLE_BOE=1, ARCA_IMV_REPORTS_DIR=$HOME/arca-reports
export ARCA_WORK_DIR=$HOME          # si no existe ~/clusters/dgx
source smoke/dgx_env.sh && bash smoke/serve_ngrok.sh
```

El primer `/health` descarga Qwen3-Embedding-0.6B e indexa el corpus (`rag/index/`, se reutiliza
en los arranques siguientes): puede tardar unos minutos. Caliéntalo en local antes de compartir
la URL de ngrok, que corta a los 300 s.

**Memoria de GPU (16 GB).** `/reasoning` (Qwen3-1.7B + LoRA GRPO), `/tools` (Qwen3-1.7B base), el
embedder de `/rag` (0,6B) y el generador de `/rag` (1.7B, solo tras la primera llamada a `/rag`).
`/agent` no carga copias propias: reutiliza el modelo base de `/tools` (cerebros `base` y
`thinking`) y el de `/reasoning` (cerebro `rlm`). Unos 11-12 GB en total.

## Pruebas rápidas

```bash
P=8888   # ARCA_API_PORT
curl -s localhost:$P/health
curl -s localhost:$P/rag -H 'Content-Type: application/json' \
  -d '{"question":"¿Qué edad mínima hay que tener para ser titular del IMV?","top_k":5}'
curl -s localhost:$P/agent -H 'Content-Type: application/json' \
  -d '{"task":"Busca en la base de conocimiento la edad mínima para ser titular del IMV y dime cuántos días hay para comunicar un cambio a la Seguridad Social.","brain":"base","max_steps":6}'
# conversación de varios turnos: "new" y luego el id que devuelve la respuesta
curl -s localhost:$P/agent -H 'Content-Type: application/json' \
  -d '{"task":"Hola, ¿qué es el IMV?","brain":"base","session_id":"new"}'
```

Visor de trazas: `ARCA_API_URL=http://localhost:$P uv run --extra agent python -m agent.ui`.

## Variables nuevas (fases 3 y 4)

| Variable | Uso |
|---|---|
| `ARCA_RAG_CHUNKING`, `ARCA_RAG_LAMBDA`, `ARCA_RAG_MIN_SCORE`, `ARCA_RAG_MODEL` | Troceado, peso BM25 del híbrido, puerta de «no lo sé» y generador de `/rag` (por defecto `recursive:800:100`, 0,5, 0 y Qwen3-1.7B: provisionales hasta los jobs 110-135) |
| `ARCA_AGENT_BASE_MODEL`, `ARCA_THINKING_MODEL` | Modelos de los cerebros `base` / `rlm` y `thinking` (por defecto Qwen3-1.7B) |
| `ARCA_AGENT_TOKEN_BUDGET`, `ARCA_AGENT_SESSION_BUDGET` | Presupuestos de memoria dentro de una ejecución (6000) y entre turnos (1500) |
| `ARCA_AGENT_RETRIEVER`, `ARCA_AGENT_TOOL_TIMEOUT` | Recuperador de `search_knowledge_base` (`hybrid`) y timeout único para todas las herramientas |

## Medidas pendientes (runner de la DGX)

Con la API parada (los jobs necesitan la GPU), en un clon aparte:
`bash smoke/dgx_runner.sh feat/fase4-agent`. Ejecuta los jobs 100-140 (fase 3: troceado, λ,
puerta, generación con citas) y 200 (fase 4: banco × 3 cerebros, agente de código, BDNS rota,
memoria reducida). Los resultados vuelven a `jobs/done/` y se resumen en
`docs/experiments_fase3.md` y `docs/experiments_fase4.md`. Antes de hacer merge a `main` hay que
quitar `smoke/dgx_runner.sh` y `jobs/`.

## Seguridad

Lo mismo que en fases 1-2 (sin autenticación: no exponer la URL sin control). Además, las
sesiones de `/agent` guardan las preguntas y respuestas en memoria del proceso (100 como máximo,
sin escribir a disco). Sus ids los emite el servidor, así que no se pueden adivinar.
`guardar_informe` solo se ejecuta si la tarea pide un informe.
