# baseline → fixes

| fase | métrica | antes | después | Δ | | ¿> 2σ? |
|---|---|---:|---:|---:|---|---|
| phase1 | pass@1 | 0.625 | 0.575 | -0.050 | peor | no |
| phase1 | format | 0.975 | 0.925 | -0.050 | peor | no |
| phase1 | truncated | 0.025 | 0.075 | 0.050 | peor | no |
| phase1 | over_300s | 1 | 0 | -1 | mejor |  |
| phase1 | p50_s | 20.950 | 19.400 | -1.550 | mejor |  |
| phase2 | selection | 0.950 | 0.950 | 0.000 |  | no |
| phase2 | arguments | 0.850 | 0.850 | 0.000 |  | no |
| phase2 | tool_sequence | 0.900 | 0.900 | 0.000 |  | no |
| phase2 | end_to_end | 0.725 | 0.725 | 0.000 |  | no |
| phase2 | amount_fidelity | 0.810 | 0.810 | 0.000 |  | no |
| phase2 | over_300s | 0 | 0 | 0 |  |  |
| phase2 | p50_s | 6.200 | 5.900 | -0.300 | mejor |  |
| phase3 | recall@5 | 0.700 | 0.740 | 0.040 | mejor | no |
| phase3 | mrr | 0.455 | 0.581 | 0.126 | mejor | no |
| phase3 | cited_relevant | 0.420 | 0.480 | 0.060 | mejor | no |
| phase3 | citations_exist | 1.000 | 1.000 | 0.000 |  |  |
| phase3 | false_abstentions | 0.120 | 0.200 | 0.080 | peor | no |
| phase3 | unanswerable_abstained | 0.875 | 0.875 | 0.000 |  | no |
| phase3 | p50_s | 1.400 | 1.000 | -0.400 | mejor |  |
| phase4 base | success | 0.600 | 0.600 | 0.000 |  | no |
| phase4 base | impossible_ok | 0.000 | 0.333 | 0.333 | mejor | no |
| phase4 base | chain_complete | 0.650 | 0.550 | -0.100 | peor | no |
| phase4 base | mean_steps | 3.087 | 3.217 | 0.130 | peor |  |
| phase4 base | mean_tokens | 15810.435 | 16970.957 | 1160.522 | peor |  |
| phase4 base | over_300s | 0 | 0 | 0 |  |  |
| phase4 base | p50_s | 12.900 | 14.400 | 1.500 | peor |  |
| phase4 rlm | success | 0.450 | 0.450 | 0.000 |  | no |
| phase4 rlm | impossible_ok | 1.000 | 1.000 | 0.000 |  |  |
| phase4 rlm | chain_complete | 0.750 | 0.750 | 0.000 |  | no |
| phase4 rlm | mean_steps | 3.087 | 3.000 | -0.087 | mejor |  |
| phase4 rlm | mean_tokens | 15967.217 | 16661.000 | 693.783 | peor |  |
| phase4 rlm | over_300s | 0 | 0 | 0 |  |  |
| phase4 rlm | p50_s | 20.200 | 20.400 | 0.200 | peor |  |
| phase4 thinking | success | 0.550 | 0.450 | -0.100 | peor | no |
| phase4 thinking | impossible_ok | 1.000 | 0.667 | -0.333 | peor | no |
| phase4 thinking | chain_complete | 0.350 | 0.300 | -0.050 | peor | no |
| phase4 thinking | mean_steps | 2.696 | 2.783 | 0.087 | peor |  |
| phase4 thinking | mean_tokens | 15289.217 | 16099.783 | 810.565 | peor |  |
| phase4 thinking | over_300s | 0 | 0 | 0 |  |  |
| phase4 thinking | p50_s | 22.700 | 21.800 | -0.900 | mejor |  |
| memoria | conversaciones OK | 1/2 | 0/2 | | | |

n por fase: {'phase1': 40, 'phase2': 40, 'phase3': 58}; agente 23 tareas por cerebro. Una o dos preguntas de diferencia son ruido con estos tamaños.
