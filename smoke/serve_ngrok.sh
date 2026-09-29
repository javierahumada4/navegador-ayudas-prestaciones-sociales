#!/usr/bin/env bash
# Levanta la API en la DGX y la expone con ngrok (la DGX no abre puertos, pero ngrok solo
# necesita conexión saliente). Desde la raíz del repo:
#
#     source smoke/dgx_env.sh
#     bash smoke/serve_ngrok.sh
#
# Lee de .env: NGROK_AUTHTOKEN (obligatorio), ARCA_RLM_ADAPTER, HF_TOKEN y opcionalmente
# NGROK_ARGS (p. ej. --url=<vuestro-dominio>.ngrok-free.app para una URL fija).
# Ctrl+C cierra el túnel y la API.

set -euo pipefail

if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

: "${NGROK_AUTHTOKEN:?Falta NGROK_AUTHTOKEN en .env (https://dashboard.ngrok.com/get-started/your-authtoken)}"
export ARCA_RLM_ADAPTER="${ARCA_RLM_ADAPTER:-JES0406/imv-sft-lora}"
export ARCA_API_PORT="${ARCA_API_PORT:-8000}"
# El venv con torch cu128 de la fase 1; el .venv por defecto puede estar mezclado con vLLM.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/arca-train}"

WORK_DIR="${ARCA_WORK_DIR:-$HOME/clusters/dgx}"
NGROK_DIR="$WORK_DIR/.ngrok/bin"
if ! command -v ngrok >/dev/null 2>&1 && [ ! -x "$NGROK_DIR/ngrok" ]; then
    echo "Instalando ngrok en $NGROK_DIR ..."
    mkdir -p "$NGROK_DIR"
    curl -fsSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
        | tar -xz -C "$NGROK_DIR"
fi
export PATH="$NGROK_DIR:$PATH"

if curl -s "localhost:$ARCA_API_PORT/health" >/dev/null; then
    echo "Algo ya escucha en el puerto $ARCA_API_PORT; exporta ARCA_API_PORT con otro." >&2
    exit 1
fi

mkdir -p runs
uv run --extra train arca-api > runs/api.log 2>&1 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

echo "Esperando a la API en el puerto $ARCA_API_PORT (log en runs/api.log) ..."
until curl -s "localhost:$ARCA_API_PORT/health" >/dev/null; do
    kill -0 "$API_PID" 2>/dev/null || { echo "La API no arrancó:" >&2; tail -20 runs/api.log >&2; exit 1; }
    sleep 2
done
echo "API lista. Abriendo túnel; la URL pública sale en la línea 'Forwarding'."

# shellcheck disable=SC2086
ngrok http "$ARCA_API_PORT" --authtoken "$NGROK_AUTHTOKEN" ${NGROK_ARGS:-}
