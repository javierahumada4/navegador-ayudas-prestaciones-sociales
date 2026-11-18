# EXPERIMENTS — Cuaderno de experimentos ARCA (IMV/CAPI)

Registro fechado y honesto: qué probamos, qué esperábamos, qué pasó y qué concluimos. Las cifras
salen de los ficheros citados; lo que todavía no se ha ejecutado está marcado como pendiente.

## 2026-10-31 — Preparación del run reproducible

Pipeline `rlm/pipeline.py` (generate → distill → sft → grpo → evaluate), oracle determinista
`rlm/imv_engine.py`, semilla 42. Un primer smoke mostró 45 de 48 trazas del profesor truncadas y
pasos GRPO con el 100 % de grupos sin señal; se añadieron puertas de calidad, corte de GRPO y
coherencia cronológica en el generador. Los resultados de ramas anteriores (`traces2`, student
0.6B) no son comparables y no se presentan.

## 2026-11-10 — Run final de fase 1 (`results/rlm_final`, DGX MIG 1g.16gb)

**Configuración** (`manifest.json`): 800 train / 200 test / 3×50 OOD; profesor Qwen3-4B (HF,
8 muestras, 1024 tokens); student Qwen3-1.7B; SFT LoRA 2 épocas; GRPO 300 pasos, 4 generaciones,
T 0.6, recompensas formato 1.0 + exactitud 2.0 + dominio 0.5, β = 0; evaluación T 0.6, 1 muestra.
Duración ≈ 11.9 h: destilación 4.6 h, SFT 0.4 h, GRPO 2.8 h, evaluación 4.1 h.

**Destilación.** 1430 de 6400 trazas verificadas (22 %); 708 seleccionadas cubren el 52 % de los
problemas (umbral 70 %, solo aviso). El 80 % de los rechazos son razonamientos cortados a 1024
tokens. La aceptación por familia va de 77 % (`individual`) a **0.8 % (`residence_fail`: 5 trazas)**.

**Resultados** (`reports/eval_*.md`, pass@1):

| Modelo | test | OOD capi_only | OOD threshold | OOD wording | formato test | truncadas test |
|---|---:|---:|---:|---:|---:|---:|
| base | .065 | .000 | .000 | .080 | .115 | .890 |
| SFT | .375 | .460 | .220 | .560 | .685 | .320 |
| SFT+GRPO | **.490** | **.500** | **.300** | .540 | .920 | .080 |

Por familia (GRPO): individual 1.00, director 0.95, income 0.70, ordinary 0.55, boundary /
disability / monoparental / patrimony 0.35, monoparental_institution 0.30, **residence 0.00 en los
tres modelos**.

**Qué esperábamos y qué pasó.** Esperábamos que el SFT enseñara el formato y el GRPO la
aritmética. El SFT aporta casi todo el salto (×6 sobre base), sobre todo porque deja de
truncar (.89 → .32). El GRPO añade +11.5 puntos en test y sobre todo **fiabilidad de formato**
(.685 → .920, truncadas .32 → .08): aprende a cerrar antes (longitud media 783 → 677 tokens).
En OOD de redacción el GRPO no mejora al SFT (.54 frente a .56): lo aprendido es formato y
longitud más que comprensión de la regla.

**Curvas SFT** (`reports/sft_curves.png`): pérdida .50 → .23 y exactitud por token .85 → .92 en
178 pasos; meseta desde el paso ~120 (segunda época). Sin validación, no podemos separar
aprendizaje de memorización; la mejora en test sugiere que no es solo memorización.

**Curvas GRPO** (`reports/grpo_curves.png`):
- La exactitud por paso oscila entre 0.1 y 0.9 **sin tendencia clara**: cada paso ve un solo
  prompt (4 completions), así que la curva mide la dificultad del prompt tanto como la política.
  300 pasos son solo ~30 prompts por familia.
- **Grupos sin señal entre 20 % y 60 %**: con 4 generaciones, muchos grupos son todo aciertos o
  todo fallos y su ventaja es 0. Es gradiente desperdiciado; más generaciones por prompt lo reducen.
- La recompensa de dominio satura en 0.9–1.0 desde el paso 40: es casi un subconjunto de la de formato
  (`imv_format_reward` ya exige dos decimales en `<answer>`), así que **no aporta señal propia**. Decisión honesta: fue
  redundante; la alternativa útil es una recompensa de elegibilidad (ver abajo).
- **Fracción recortada siempre 0**, y es lo esperado: con `num_iterations=1` y
  `steps_per_generation` igual a la acumulación de gradiente, TRL toma la política antigua como
  la actual (`old_logps = logps.detach()`), el cociente vale exactamente 1 y el recorte de PPO
  nunca se activa. Solo se activaría reutilizando cada lote (`num_iterations > 1`).
- Entropía baja y estable (.10–.13): no hay colapso ni explosión; β = 0 no desestabilizó.

**Cinco fallos con ejemplos** (82 errores del GRPO en test, clasificados contra el oracle):
1. *Residencia (19 de 31 «no detecta inelegibilidad»).* El enunciado decía `p2: sí, desde
   2026-01-15` y la hoja de reglas solo «≥ 12 meses», sin fecha límite. GRPO: «All members have
   legal residence, so that's covered». Hasta el profesor 4B cae («la residencia… sí»). Además la
   hoja decía que el CAPI «puede concederse aunque el IMV sea 0» sin aclarar que la residencia
   también lo elimina: ids 61 y 176 responden exactamente 115.00 (solo CAPI). **Es un fallo de
   especificación, no del modelo.**
2. *Patrimonio en el límite.* id 40: «patrimonio 58101.12, exactamente el límite… así que está
   bien», pero la regla es estricta (<). Además usa la fila de composición equivocada (id 1).
3. *Franja de edad del CAPI (~16 errores, deltas ±23, ±34.50, ±57.50).* id 47: «p3 tiene 5 años
   (6-17), así que 57.50»; id 76 trata a un niño de 4 años como menor de 3.
4. *Número de miembros (7 errores, delta −220.08 = 30 % de la base).* id 54: «1 adulto y 1 menor…
   la renta garantizada es 733.60»: cuenta solo adultos, mezclando la tabla por miembros con las
   tablas «N adultos + M menores».
5. *Complemento monoparental omitido.* id 172: custodia «exclusiva», pero el modelo dice «no se
   menciona monoparentalidad» (−161.39).

## 2026-11-18 — Hipótesis: la hoja de reglas v2 arregla fallos de especificación

**Cambio** (`873b2c6`, solo texto del prompt y del enunciado; mismos casos, misma semilla):
fecha límite explícita de residencia (`<= 2025-09-26`), «si falla un requisito del apartado 7 no hay
CAPI», franjas del CAPI enumeradas (0-2 / 3-5 / 6-17), «cuenta todos los miembros (1 adulto + 1
menor = 2 miembros)», y residencia redactada como «legal y efectiva, continuada desde …». Se ha
comprobado que la semilla 42 reproduce los 200 problemas de test idénticos (respuesta y
parámetros); solo cambia el texto.

**Esperamos:** `residence_fail` de 0 hacia el nivel de `director_fail` en el reentreno, y una
mejora menor sin reentrenar (el adaptador actual nunca vio la regla).

**Ablaciones encoladas en la DGX** (`jobs/`, ejecutadas por `smoke/dgx_runner.sh`):

| Job | Qué cambia | Para medir |
|---|---|---|
| 010 | mismos adaptadores (base/SFT/GRPO), test v1 frente a v2 | efecto del prompt sin entrenar; velocidad de evaluación |
| 030 | pipeline completo con hoja v2, resto igual | efecto de la especificación en destilación, SFT y GRPO |

Resultados: *pendientes*.

**Velocidad** (mismas salidas, distinta ejecución): evaluación con batch 16 y LoRA fusionada
(antes batch 2 sin fusionar, 2.1 h solo el test); destilación con vLLM batch 64 (antes HF batch
1, 4.6 h). Tiempos medidos: *pendientes*.

**Siguiente ablación propuesta** (no ejecutada): sustituir la recompensa de dominio, redundante,
por una de elegibilidad (acierta si responde 0 cuando el oracle da 0), y subir a 8 generaciones
por prompt para reducir los grupos sin señal.

## 2026-11-17 — Fase 2: herramientas, base frente a GRPO, con y sin tools (v4)

- Rama: `feat/fase2-tool-use-imv-bdns-informes` (Ahumada). DGX, MIG 1g.16gb, greedy, `max_turns=5`.
- Herramientas: `calcular_imv` (cálculo, motor determinista), `consultar_ayudas` (HTTP real, API BDNS),
  `guardar_informe` (efecto: TXT, `requires_confirmation`). Esquemas desde Pydantic.
- Banco: 40 casos (4 JSON estructurado, 6 cálculo en prosa, 6 denegaciones, 5 datos faltantes,
  8 BDNS, 6 multi-step, 5 sin herramienta). Trazas completas por caso (JSONL).

| Métrica | base + tools | GRPO + tools | base sin | GRPO sin |
|---|---:|---:|---:|---:|
| Selección (1.ª llamada) | .750 | .875 | .250 | .250 |
| Argumentos (1.ª llamada) | .725 | .800 | .250 | .250 |
| Secuencia completa | .675 | .775 | .250 | .250 |
| Multi-step (6) | .50 | .50 | 0 | 0 |
| Importes correctos (21) | .381 | .286 | .143 | .143 |
| Éxito extremo a extremo | **.500** | .425 | .150 | .125 |

**Qué esperábamos.** Que el GRPO de fase 1, entrenado en el mismo dominio, ganase al base también
con herramientas.

**Qué pasó.** Las herramientas dominan: el éxito extremo a extremo se multiplica por 3–3,5. Entre
modelos hay empate: en pareado el base gana 5 casos y el GRPO 2 (test de signos p ≈ 0,45). El GRPO
**decide** mejor (selección .875 frente a .750; no llama a la calculadora con datos faltantes,
P2-019) pero **redacta** peor: en 3 denegaciones explica bien el motivo y omite los importes a 0 €.
Encaja con su entrenamiento: una sola cantidad final dentro de `<answer>`, sin desglose.

**Cinco fallos con traza.**
1. *JSON con una `}` de más* (P2-005…009, ambos modelos): 7 llaves abiertas, 8 cerradas. Con
   decodificación greedy cada reintento es byte a byte idéntico, así que la observación de error no
   sirve y se gastan los 5 turnos (hasta 341 s, por encima del límite de 300 s de ngrok).
2. *Identificadores inventados* (P2-006…010): `"123456789"` en lugar de `adult`, aunque la consulta
   lo dice literalmente; no aparece en el prompt. Además omite a la segunda persona (hijo o pareja).
3. *Argumento con pérdida* (P2-028): busca "familia" en vez de "familia numerosa"; resultados
   irrelevantes presentados como pertinentes.
4. *Siglas alucinadas* sin herramienta (P2-036/037): "IMV = Inscripción de Migración y Vivienda",
   CAPI "para mayores de 65". 0/2 en ambos modelos: el RLM aprendió a calcular, no el dominio.
5. *Cadena incompleta* (P2-034): el modelo emite `guardar_informe` en el mismo turno que
   `calcular_imv`, antes de ver el resultado; la herramienta lo rechaza y se recupera en el turno 2.

**Single-step frente a multi-step.** Las llamadas paralelas aparecen solas (P2-035: dos búsquedas
BDNS en un turno). El multi-step falla cuando el modelo intenta resolver la cadena entera en un turno.

**Limitaciones honestas.** Seis respuestas finales son plantillas del ejecutor ("Informe guardado
correctamente en…"), no texto del modelo. Las métricas de explicación y aclaración son heurísticas
léxicas. n=40: diferencias de menos de ~15 puntos no son significativas.

## 2026-11-18 — Fase 2 v5: velocidad y extracción (rama `feat/fase2-speed`)

**Cambios** a partir de los fallos anteriores: LoRA fusionada (40 → ~75 tok/s medidos en el
base); corte cuando el modelo repite byte a byte una salida cuyas llamadas fallaron (en v4 se
perdían 12–17 turnos, unos 350–440 s por modelo); límite de 1280 tokens (el turno legítimo más
largo fue de 1115); el esquema se envía una sola vez por `tools=` (el bloque de sistema pasa de
~18.5k a ~10k caracteres); regla «usa los id literales de la consulta e incluye a todas las
personas» con un ejemplo de dos personas distinto del banco; el parser perdona solo llaves `}`
sobrantes al final y saltos de línea dentro de cadenas.

**Esperamos:** menos tiempo por caso sin perder aciertos, y algún cálculo en prosa resuelto.
**Riesgo:** el ejemplo puede sesgar al modelo hacia dos personas.

**Resultados (job 020, 2026-11-18 02:39, mismos 40 casos):**

| Métrica (con tools) | base v4 | base v5 | GRPO v4 | GRPO v5 |
|---|---:|---:|---:|---:|
| Selección | .750 | **.975** | .875 | **.975** |
| Argumentos (1.ª llamada) | .725 | .800 | .800 | .825 |
| Secuencia completa | .675 | **.900** | .775 | .850 |
| Multi-step (6) | .50 | **.83** | .50 | .50 |
| Importes correctos | .381 | .381 | .286 | .238 |
| Éxito extremo a extremo | .500 | .400 | .425 | .400 |
| Tiempo con tools | 1376 s | 1278 s | 1335 s | 1234 s |

**Qué pasó.** La parte de *herramientas* mejora claramente: el JSON del cálculo en prosa ya se
parsea y ejecuta (selección .975 en ambos), y el base encadena mejor (multi-step .50 → .83).
Pero el éxito extremo a extremo **baja**, por dos causas que vemos en las filas:
1. *El ejemplo sesgó los ids*: ya no inventa `123456789`, pero copia `titular` (la palabra del
   ejemplo y de la consulta «el titular adult») en vez de `adult`. P2-007 y P2-009 fallan
   **solo** por el id; P2-008 por una fecha. Era el riesgo anotado, en otra forma.
2. *Las denegaciones no dan importes*: el motivo es correcto pero la respuesta no dice
   «IMV 0,00 €, CAPI 0,00 €, total 0,00 €» y la métrica lo exige (base 3/6 → 0/6).
La velocidad apenas cambia (−7 %): hay turnos únicos de 245 s que no explican ni los tokens ni
el prompt; el cuello está en la decodificación HF en el slice (ver fase 1, vLLM).

**v6 (job 021):** ejemplo «rol id → id» con nombres que no aparecen en el banco (`ana`, `leo`)
y regla «da siempre IMV, CAPI y total, también si son 0,00 €». Resultados: *pendientes*.
