# Despliegue de las fases 1 y 2 (rama consolidada)

Rama: `release/fases-1-2-definitiva`. La fase 1 utiliza el pipeline de razonamiento verificado de `main`. La fase 2 unifica el bucle manual, cálculo IMV/CAPI, consulta BDNS, informe persistente y consulta BOE opcional.

## Arranque

```bash
git fetch origin
git switch release/fases-1-2-definitiva
cp .env.example .env
# Configurar ARCA_RLM_ADAPTER con la ruta local o el repositorio del LoRA GRPO entrenado.
# El adaptador no está versionado en GitHub.
docker compose build
docker compose up -d api
curl -fsS http://localhost:8000/health
```

La imagen usa CUDA 12.8 y puede consumir mucha RAM/disco. Para inferencia de calidad y velocidad, ejecutar sobre una GPU NVIDIA compatible. En CPU el modelo Qwen3-1.7B puede ser lento. La primera carga puede necesitar conexión a Hugging Face y espacio para la caché. Ver también `docs/dgx.md`.

El servicio `api` expone `GET /health`, `POST /reasoning`, `POST /tools` y `/docs`. Las fases 3 y 4 quedan fuera de esta entrega y no se declaran completas.

**Configuración esencial:**

| Variable | Uso |
| --- | --- |
| `ARCA_RLM_ADAPTER` | Obligatoria para activar fase 1: carpeta del LoRA GRPO o repo HF autorizado |
| `ARCA_RLM_BASE_MODEL` | Opcional: normalmente se infiere del adaptador |
| `ARCA_TOOLS_BRAIN` | `base` (por defecto), o `rlm` para reutilizar el LoRA |
| `ARCA_TOOLS_MODEL_ID` | Qwen3 base; por defecto `Qwen/Qwen3-1.7B` |
| `ARCA_TOOLS_ADAPTER` | Adaptador de fase 2 con brain `rlm`; debe coincidir con el modelo base |
| `ARCA_TOOLS_ENABLE_BOE` | `1`: incluye `consultar_boe`; `0`: conserva las tres tools del benchmark |
| `HF_TOKEN` | Solo cuando Hugging Face exija acceso autorizado |
| `HF_CACHE_DIR` | Caché persistente para modelos |
| `ARCA_REPORTS_HOST_DIR` | Carpeta local para informes, montada en `/data/reports` |
| `ARCA_API_PORT` | Puerto público local (8000 por defecto) |

No fijar `ARCA_TOOLS_BRAIN=rlm` con el modelo 1.7B si el LoRA fue entrenado sobre 0.6B. El código comprueba la arquitectura indicada en `adapter_config.json`.

## Comprobación

```bash
python -m compileall -q api rlm tool_use tests
python -m pytest -q tests/test_phase2_imv_tools.py tests/test_phase2_stages.py \
    tests/test_phase2_tracing.py tests/test_phase2_workflows.py \
    tests/test_boe_tool.py tests/test_release_phase12.py \
    tests/test_tool_registry.py tests/test_tool_parser.py tests/test_api.py
# Evaluación real del modelo: requiere pesos, recursos y APIs externas.
uv run --extra train python -m tool_use.stage_eval \
    --stage all --brain base --out reports/phase2_stages_base.json
```

Ejemplo de llamada:

```bash
curl -s http://localhost:8000/tools -H 'Content-Type: application/json' \
  -d '{"query":"Busca convocatorias de ayudas para alquiler en la BDNS.","max_turns":5}'
```

Las pruebas con redes simuladas verifican orquestación y errores; **no** acreditan que BDNS y BOE estén disponibles ni que un informe emitido sea una resolución administrativa. La legislación y el ruleset se deben volver a verificar cuando cambien los importes o las normas.

## Seguridad

No exponer esta API de prueba directamente en Internet sin autenticación, controles de cuota, políticas de retención y revisión de datos personales. `guardar_informe` escribe archivos privados tras una petición explícita del usuario; la API de ejemplo no autentica identidades. Los reportes pueden contener ingresos y circunstancias familiares: mantener `ARCA_REPORTS_HOST_DIR` protegido y fuera de Git; no habilitar trazas completas por defecto. El puerto 4040 de ngrok solo se publica en localhost. Para producción, poner HTTPS + autenticación delante del servicio y evitar montar el código del host con permisos de escritura.

## Alcance y verificación

- Se mantiene intacto el motor de fase 1 de `main` y su CI.
- La fase 2 aporta cálculo determinista, consulta HTTP BDNS, informes bajo ruta del servidor y BOE optativo.
- El benchmark por defecto conserva tres tools, compatible con las evaluaciones de 40 casos y cuatro etapas.
- Tests CPU automatizados validan contratos y lógica; **no sustituyen** la evaluación de inferencia sobre el adaptador GRPO real ni el smoke test de Docker/GPU.
- Sin `ARCA_RLM_ADAPTER`, `/reasoning` sigue indicando que la fase 1 no está cargada: los pesos no están en el repositorio.
