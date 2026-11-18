| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |
|---|---|---|---|---|---|
| base | 0.045 | 0.125 | 0.875 | 0.020 (esperado 0.320) | 994 |
| sft | 0.445 | 0.700 | 0.300 | 0.145 (esperado 0.320) | 787 |
| grpo | 0.475 | 0.890 | 0.110 | 0.160 (esperado 0.320) | 687 |

| Familia | Generaciones | base | sft | grpo |
|---|---|---|---|---|
| boundary | 20 | 0.000 | 0.450 | 0.350 |
| director_fail | 20 | 0.000 | 0.950 | 1.000 |
| disability | 20 | 0.000 | 0.300 | 0.500 |
| income_fail | 20 | 0.050 | 0.400 | 0.350 |
| individual | 20 | 0.400 | 0.950 | 1.000 |
| monoparental | 20 | 0.000 | 0.350 | 0.200 |
| monoparental_institution | 20 | 0.000 | 0.300 | 0.400 |
| ordinary | 20 | 0.000 | 0.400 | 0.500 |
| patrimony_fail | 20 | 0.000 | 0.350 | 0.400 |
| residence_fail | 20 | 0.000 | 0.000 | 0.050 |

| Fallo | base | sft | grpo |
|---|---|---|---|
| missed_imv | 0 | 2 | 3 |
| missed_ineligibility | 10 | 24 | 29 |
| near_miss | 1 | 0 | 2 |
| truncated | 175 | 60 | 22 |
| wrong_amount | 2 | 24 | 49 |
| wrongly_denied | 3 | 1 | 0 |
