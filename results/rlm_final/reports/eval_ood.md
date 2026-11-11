| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |
|---|---|---|---|---|---|
| base | 0.000 | 0.020 | 0.980 | 0.000 (esperado 0.000) | 1024 |
| sft | 0.460 | 0.660 | 0.340 | 0.000 (esperado 0.000) | 871 |
| grpo | 0.500 | 0.900 | 0.100 | 0.000 (esperado 0.000) | 765 |

| Familia | Generaciones | base | sft | grpo |
|---|---|---|---|---|
| capi_only | 50 | 0.000 | 0.460 | 0.500 |

| Fallo | base | sft | grpo |
|---|---|---|---|
| truncated | 49 | 17 | 5 |
| wrong_amount | 1 | 10 | 20 |
