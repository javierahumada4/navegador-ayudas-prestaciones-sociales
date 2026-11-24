# Scorecard — baseline

API: `http://localhost:8899` · 2026-11-24 09:41

**Health:** 4/4 fases ready (reasoning=ready, tools=ready, rag=ready, agent=ready)

## phase1

| pass@1 | format | truncated | mean_tokens | n | errores | >300 s | p50 s | p90 s |
|---|---|---|---|---|---|---|---|---|
| 0.625 | 0.975 | 0.025 | 694.800 | 40 | 0 | 1 | 20.950 | 43.500 |

## phase2

| selection | arguments | tool_sequence | end_to_end | amount_fidelity | n | errores | >300 s | p50 s | p90 s |
|---|---|---|---|---|---|---|---|---|---|
| 0.950 | 0.850 | 0.900 | 0.725 | 0.810 | 40 | 0 | 0 | 6.200 | 18.000 |

## phase3

| recall@5 | mrr | cited_relevant | citations_exist | false_abstentions | unanswerable_abstained | n | errores | >300 s | p50 s | p90 s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.500 | 0.326 | 0.320 | 1.000 | 0.100 | 0.750 | 58 | 15 | 0 | 0.650 | 3.300 |

## phase4

| cerebro | success | impossible_ok | chain_complete | mean_steps | mean_tokens | errores | >300 s | p50 s | p90 s |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 23 | 0 | 0.000 | 0.000 |
| rlm | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 23 | 0 | 0.000 | 0.000 |
| thinking | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 23 | 0 | 0.000 | 0.000 |

**Memoria multi-turno:** 0/2 conversaciones recuerdan el turno anterior.
