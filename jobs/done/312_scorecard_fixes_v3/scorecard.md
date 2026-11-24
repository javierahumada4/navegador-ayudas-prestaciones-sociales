# Scorecard — fixes-v3

API: `http://localhost:8899` · 2026-11-24 12:44

**Health:** 4/4 fases ready (reasoning=ready, tools=ready, rag=ready, agent=ready)

## phase1

| pass@1 | format | truncated | mean_tokens | n | errores | >300 s | p50 s | p90 s |
|---|---|---|---|---|---|---|---|---|
| 0.575 | 0.925 | 0.075 | 702.200 | 40 | 0 | 0 | 19.400 | 28.000 |

## phase2

| selection | arguments | tool_sequence | end_to_end | amount_fidelity | n | errores | >300 s | p50 s | p90 s |
|---|---|---|---|---|---|---|---|---|---|
| 0.950 | 0.850 | 0.900 | 0.725 | 0.810 | 40 | 0 | 0 | 5.900 | 17.700 |

## phase3

| recall@5 | mrr | cited_relevant | citations_exist | false_abstentions | unanswerable_abstained | n | errores | >300 s | p50 s | p90 s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.740 | 0.581 | 0.480 | 1.000 | 0.200 | 0.875 | 58 | 0 | 0 | 0.950 | 3.400 |

## phase4

| cerebro | success | impossible_ok | chain_complete | mean_steps | mean_tokens | errores | >300 s | p50 s | p90 s |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.600 | 0.333 | 0.550 | 3.217 | 16968.739 | 0 | 0 | 13.300 | 25.000 |
| rlm | 0.450 | 1.000 | 0.750 | 3.000 | 16660.609 | 0 | 0 | 19.500 | 30.300 |
| thinking | 0.500 | 0.667 | 0.300 | 2.783 | 16103.348 | 0 | 0 | 21.700 | 50.800 |

**Memoria multi-turno:** 0/2 conversaciones recuerdan el turno anterior.
