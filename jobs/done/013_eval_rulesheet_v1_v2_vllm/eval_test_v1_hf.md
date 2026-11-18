| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |
|---|---|---|---|---|---|
| base | 0.075 | 0.130 | 0.870 | 0.005 (esperado 0.320) | 989 |
| sft | 0.405 | 0.695 | 0.305 | 0.135 (esperado 0.320) | 790 |
| grpo | 0.510 | 0.910 | 0.090 | 0.180 (esperado 0.320) | 687 |

| Familia | Generaciones | base | sft | grpo |
|---|---|---|---|---|
| boundary | 20 | 0.000 | 0.200 | 0.250 |
| director_fail | 20 | 0.000 | 0.800 | 1.000 |
| disability | 20 | 0.000 | 0.200 | 0.550 |
| income_fail | 20 | 0.050 | 0.600 | 0.700 |
| individual | 20 | 0.700 | 1.000 | 1.000 |
| monoparental | 20 | 0.000 | 0.250 | 0.250 |
| monoparental_institution | 20 | 0.000 | 0.200 | 0.350 |
| ordinary | 20 | 0.000 | 0.400 | 0.600 |
| patrimony_fail | 20 | 0.000 | 0.400 | 0.400 |
| residence_fail | 20 | 0.000 | 0.000 | 0.000 |

| Fallo | base | sft | grpo |
|---|---|---|---|
| missed_imv | 0 | 5 | 3 |
| missed_ineligibility | 8 | 20 | 28 |
| near_miss | 1 | 1 | 2 |
| truncated | 174 | 61 | 18 |
| wrong_amount | 2 | 31 | 42 |
| wrongly_denied | 0 | 1 | 5 |
