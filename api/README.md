# `api/` — La API

Una FastAPI con un endpoint por fase. Los contratos de entrada y salida están en
[`schemas.py`](schemas.py), y [`backends.py`](backends.py) carga cada fase la primera vez que
se llama y la reutiliza en las siguientes.

```bash
uv run arca-api          # http://localhost:8000/docs
docker compose up api
```

| Método y ruta | Fase | Estado | Detrás |
|---|---|---|---|
| `GET /health` | — | Listo | Estado de cada fase, equipo y dominio |
| `POST /reasoning` | 1 | Listo | `rlm.inference.ReasoningModel`: Qwen3-1.7B con el LoRA GRPO final |
| `POST /tools` | 2 | Listo | `tool_use.executor.ToolLoop` con `calcular_imv`, `consultar_ayudas` y `guardar_informe` |
| `POST /rag` | 3 | Pendiente (`501`) | — |
| `POST /agent` | 4 | Pendiente (`501`) | — |

`/reasoning` recibe `question` y, opcionalmente, `expected_answer` y `max_new_tokens`. Devuelve:

- `thinking` y `answer`, por separado;
- `has_valid_format`;
- el veredicto del verificador IMV, si la petición trae `expected_answer`;
- `tokens_generated`;
- `truncated`, si la generación llegó al máximo de tokens. El máximo por defecto es 1024, el
  mismo presupuesto con el que se entrenó.

`/tools` recibe `query` y `max_turns`. Devuelve la respuesta, cada llamada (nombre,
argumentos, resultado, error y latencia) y el número de turnos.

## Configuración

Todo se configura en `.env`. Hay una plantilla en [`.env.example`](../.env.example).

| Variable | Valor | Para qué |
|---|---|---|
| `ARCA_RLM_ADAPTER` | `JES0406/imv-sft-lora/grpo_final` o una carpeta local | LoRA de `/reasoning`. Si está vacía, la fase 1 queda pendiente |
| `ARCA_RLM_VERIFIER` | `imv` (por defecto) | Verificador de `/reasoning` |
| `ARCA_TOOLS_BRAIN` | `base` (por defecto) o `rlm` | Cerebro de `/tools`. `base` es el que mejor resultado da |
| `ARCA_IMV_REPORTS_DIR` | `reports/imv` (por defecto) | Dónde escribe `guardar_informe` |
| `ARCA_TOOLS_TRACE_DIR` | vacía | Si se define, guarda una traza completa de cada petición a `/tools` |
| `ARCA_TEAM`, `ARCA_DOMAIN` | — | Aparecen en `/health` |
| `HF_TOKEN`, `HF_HOME` | — | Descarga de modelos de Hugging Face |

## Exponerla para la corrección

Las dos formas usan [ngrok](https://ngrok.com/) y leen `NGROK_AUTHTOKEN` de `.env`. Con
`NGROK_ARGS=--url=<dominio>.ngrok-free.app` la URL no cambia entre arranques.

- **Con Docker:**

  ```bash
  docker compose up api ngrok
  ```

  La URL pública aparece en http://localhost:4040.
- **En la DGX,** que no tiene Docker ni puertos públicos, pero ngrok solo necesita conexión
  saliente:

  ```bash
  source smoke/dgx_env.sh && bash smoke/serve_ngrok.sh
  ```

Antes de pasar la URL hay que comprobar en `/health` que las fases 1 y 2 aparecen como
`ready`.

Los tests de [`tests/test_api.py`](../tests/test_api.py) comprueban que la API arranca sin
modelos y que valida las entradas.
