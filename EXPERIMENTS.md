# EXPERIMENTS — Registro de fase 1 IMV/CAPI

**Estado:** el pipeline está implementado; no se ha entrenado ni probado
de extremo a extremo con GPU en esta rama. No se atribuyen métricas a
ejecuciones inexistentes.

## 2026-10-31 — Preparación del run reproducible

- Rama: `fix/rlm-reproducible-pipeline`.
- Código base: `feat/ngrok`, SHA `e9ca7c50accf5a3e22e08b2f746e0fac0d0a2267`.
- Dominio IMV/CAPI 2026 y oracle `rlm/imv_engine.py`.
- Datos por defecto: 800 train, 200 test, 100 OOD de CAPI únicamente.
- Teacher por defecto: `Qwen/Qwen3-4B`; cuatro muestras por enunciado.
- Student: `Qwen/Qwen3-0.6B`.
- SFT: LoRA con trazas verificadas.
- GRPO: LoRA inicializado desde SFT; pesos recompensa formato 1.0,
  acierto monetario 2.0, formato de respuesta IMV 0.5; monitor de ceros 0.0.
- Evaluación: base vs SFT vs GRPO en test y OOD, curvas y clasificación
  automática de errores. Semilla predeterminada: 42.

## Datos pendientes tras la ejecución

| Evidencia | Ubicación relativa al run | Estado |
|---|---|---|
| Validación del oracle y splits | `reports/dataset_validation.json` | Pendiente |
| Número y porcentaje de trazas aceptadas, por familia | `reports/distillation_validation.json` | Pendiente |
| Training loss y checkpoints SFT | `weights/sft_lora/` | Pendiente |
| Reward, KL, clipping, longitudes GRPO | `weights/grpo_lora/log_history.json` | Pendiente |
| Curvas | `reports/grpo_curves.png` | Pendiente |
| Pass@1 base/SFT/GRPO en test | `reports/eval_test.json` | Pendiente |
| Generalización OOD | `reports/eval_ood.json` | Pendiente |
| Cinco fallos con causas y ejemplos | Este documento | Pendiente |
| Ablación reproducible (variar un peso de recompensa) | Run alternativo | Pendiente |
| Validación /health y /reasoning | Despliegue final | Pendiente |

Los datos históricos de `traces2` no se mezclan con las generaciones del
nuevo run. En esta rama se han dejado fuera de Git los antiguos
`rlm/data/{train,test}.jsonl`, `rlm/test_ood/test.jsonl` y
`reports/phase1_eval.json`; permanecen en las ramas/commits anteriores.
La destilación guarda cada batch y el entrenamiento puede reanudar desde
checkpoints. La ejecución GPU integral sigue pendiente. Tras el smoke local
inicial se detectaron 45 de 48 generaciones del profesor truncadas (solo 3
verificadas, de 2 problemas), y dos pasos GRPO con 100% de grupos sin
señal. Por ello el pipeline incorporó límites de calidad y corte de GRPO,
un presupuesto de profesor de 2048 tokens en smoke y coherencia
cronológica en el generador. Los resultados anteriores no son comparables
con los nuevos y no se presentan como mejoras verificadas. Anotar la fecha, GPU/VRAM, SHA, duración, hiperparámetros,
aceptación, ablation y resultados de cada ejecución real.
