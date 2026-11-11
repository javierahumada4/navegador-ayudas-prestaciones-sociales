| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |
|---|---|---|---|---|---|
| base | 0.000 | 0.020 | 0.980 | 0.000 (esperado 0.380) | 1020 |
| sft | 0.220 | 0.480 | 0.520 | 0.100 (esperado 0.380) | 920 |
| grpo | 0.300 | 0.900 | 0.100 | 0.100 (esperado 0.380) | 774 |

| Familia | Generaciones | base | sft | grpo |
|---|---|---|---|---|
| threshold_ood | 50 | 0.000 | 0.220 | 0.300 |

| Fallo | base | sft | grpo |
|---|---|---|---|
| missed_imv | 0 | 1 | 3 |
| missed_ineligibility | 1 | 8 | 15 |
| near_miss | 0 | 0 | 2 |
| truncated | 49 | 26 | 5 |
| wrong_amount | 0 | 2 | 6 |
| wrongly_denied | 0 | 2 | 4 |
