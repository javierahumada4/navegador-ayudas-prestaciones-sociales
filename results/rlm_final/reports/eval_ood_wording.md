| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |
|---|---|---|---|---|---|
| base | 0.080 | 0.120 | 0.880 | 0.000 (esperado 0.000) | 1005 |
| sft | 0.560 | 0.660 | 0.340 | 0.000 (esperado 0.000) | 848 |
| grpo | 0.540 | 0.920 | 0.080 | 0.000 (esperado 0.000) | 738 |

| Familia | Generaciones | base | sft | grpo |
|---|---|---|---|---|
| wording_ood | 50 | 0.080 | 0.560 | 0.540 |

| Fallo | base | sft | grpo |
|---|---|---|---|
| missed_imv | 0 | 0 | 2 |
| near_miss | 0 | 1 | 2 |
| truncated | 44 | 17 | 4 |
| wrong_amount | 2 | 4 | 15 |
