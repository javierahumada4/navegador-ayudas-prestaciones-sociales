| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |
|---|---|---|---|---|---|
| base | 0.040 | 0.085 | 0.915 | 0.000 (esperado 0.320) | 1006 |
| sft | 0.350 | 0.735 | 0.265 | 0.135 (esperado 0.320) | 769 |
| grpo | 0.375 | 0.865 | 0.135 | 0.160 (esperado 0.320) | 702 |

| Familia | Generaciones | base | sft | grpo |
|---|---|---|---|---|
| boundary | 20 | 0.000 | 0.100 | 0.150 |
| director_fail | 20 | 0.000 | 1.000 | 0.950 |
| disability | 20 | 0.000 | 0.150 | 0.400 |
| income_fail | 20 | 0.000 | 0.200 | 0.300 |
| individual | 20 | 0.400 | 1.000 | 1.000 |
| monoparental | 20 | 0.000 | 0.150 | 0.250 |
| monoparental_institution | 20 | 0.000 | 0.350 | 0.100 |
| ordinary | 20 | 0.000 | 0.350 | 0.300 |
| patrimony_fail | 20 | 0.000 | 0.200 | 0.200 |
| residence_fail | 20 | 0.000 | 0.000 | 0.100 |

| Fallo | base | sft | grpo |
|---|---|---|---|
| missed_imv | 0 | 3 | 2 |
| missed_ineligibility | 5 | 20 | 27 |
| near_miss | 1 | 1 | 1 |
| truncated | 183 | 53 | 27 |
| wrong_amount | 3 | 52 | 65 |
| wrongly_denied | 0 | 1 | 3 |
