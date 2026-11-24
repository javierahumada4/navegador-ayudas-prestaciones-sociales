| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |
|---|---|---|---|---|---|
| base | 0.045 | 0.125 | 0.875 | 0.020 (esperado 0.320) | 994 |
| sft | 0.410 | 0.700 | 0.300 | 0.160 (esperado 0.320) | 782 |
| grpo | 0.480 | 0.900 | 0.100 | 0.155 (esperado 0.320) | 682 |

| Familia | Generaciones | base | sft | grpo |
|---|---|---|---|---|
| boundary | 20 | 0.000 | 0.250 | 0.400 |
| director_fail | 20 | 0.000 | 1.000 | 0.950 |
| disability | 20 | 0.000 | 0.350 | 0.450 |
| income_fail | 20 | 0.050 | 0.350 | 0.550 |
| individual | 20 | 0.400 | 0.900 | 0.900 |
| monoparental | 20 | 0.000 | 0.200 | 0.400 |
| monoparental_institution | 20 | 0.000 | 0.400 | 0.350 |
| ordinary | 20 | 0.000 | 0.300 | 0.500 |
| patrimony_fail | 20 | 0.000 | 0.350 | 0.300 |
| residence_fail | 20 | 0.000 | 0.000 | 0.000 |

| Fallo | base | sft | grpo |
|---|---|---|---|
| missed_imv | 0 | 5 | 4 |
| missed_ineligibility | 10 | 21 | 30 |
| near_miss | 1 | 2 | 4 |
| truncated | 175 | 60 | 20 |
| wrong_amount | 2 | 29 | 43 |
| wrongly_denied | 3 | 1 | 3 |
