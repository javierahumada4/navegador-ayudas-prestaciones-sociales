| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |
|---|---|---|---|---|---|
| base | 0.065 | 0.115 | 0.890 | 0.015 (esperado 0.320) | 1001 |
| sft | 0.375 | 0.685 | 0.320 | 0.130 (esperado 0.320) | 783 |
| grpo | 0.490 | 0.920 | 0.080 | 0.145 (esperado 0.320) | 677 |

| Familia | Generaciones | base | sft | grpo |
|---|---|---|---|---|
| boundary | 20 | 0.000 | 0.200 | 0.350 |
| director_fail | 20 | 0.050 | 0.900 | 0.950 |
| disability | 20 | 0.000 | 0.250 | 0.350 |
| income_fail | 20 | 0.050 | 0.300 | 0.700 |
| individual | 20 | 0.500 | 0.950 | 1.000 |
| monoparental | 20 | 0.000 | 0.250 | 0.350 |
| monoparental_institution | 20 | 0.000 | 0.300 | 0.300 |
| ordinary | 20 | 0.000 | 0.350 | 0.550 |
| patrimony_fail | 20 | 0.050 | 0.250 | 0.350 |
| residence_fail | 20 | 0.000 | 0.000 | 0.000 |

| Fallo | base | sft | grpo |
|---|---|---|---|
| missed_imv | 0 | 4 | 3 |
| missed_ineligibility | 6 | 19 | 31 |
| near_miss | 2 | 1 | 1 |
| truncated | 177 | 63 | 16 |
| wrong_amount | 2 | 36 | 51 |
| wrongly_denied | 0 | 2 | 0 |
