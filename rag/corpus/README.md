# Corpus de la fase 3: marco legal del IMV y el CAPI

Siete documentos en Markdown, descargados de fuentes públicas oficiales. Los ficheros están en
git (≈600 000 caracteres) y se regeneran con:

```powershell
uv run python -m rag.fetch_corpus
```

El script escribe los `.md` y `MANIFEST.json` (título, URL, fecha de descarga, SHA-256 y licencia
de cada documento). **La legislación consolidada del BOE se actualiza con el tiempo**: si volvéis
a descargar, el texto puede cambiar y con él los ids de fragmento del conjunto dorado. El
snapshot que usa el dorado es el que está commiteado (`retrieved` en el manifiesto); para
reproducirlo exactamente, usad el corpus del repositorio sin volver a descargar.

| Fichero | Documento | Fuente | Vigente |
|---|---|---|---|
| `ley-19-2021.md` | Ley 19/2021, por la que se establece el ingreso mínimo vital (incluye el CAPI) | BOE-A-2021-21007 | sí |
| `rd-789-2022.md` | RD 789/2022: compatibilidad del IMV con rentas del trabajo (incentivo al empleo) | BOE-A-2022-15764 | sí |
| `rd-635-2022.md` | RD 635/2022: Consejo consultivo del IMV | BOE-A-2022-12508 | sí |
| `rd-64-2022.md` | RD 64/2022: Comisión de seguimiento del IMV | BOE-A-2022-1275 | sí |
| `orden-ism-1375-2021.md` | Orden ISM/1375/2021: Registro de Mediadores Sociales del IMV | BOE-A-2021-20316 | sí |
| `rdl-20-2020.md` | RDL 20/2020, texto original del IMV | BOE-A-2020-5493 | **derogado** |
| `ss-imv.md` | Seguridad Social: Ingreso mínimo vital (requisitos, cuantías 2026, solicitud) | seg-social.es | sí |

El RDL 20/2020 está derogado por la Ley 19/2021 y se incluye adrede como distractor de versión:
sus fragmentos se parecen mucho a los de la ley vigente, y un buen retriever (y un buen
generador) deben preferir la norma en vigor. Su primer párrafo lo indica.

## Licencias

- **BOE** (`ley-…`, `rd-…`, `orden-…`, `rdl-…`): datos abiertos del BOE a través de la API de
  legislación consolidada. Reutilización permitida con atribución a la Agencia Estatal Boletín
  Oficial del Estado, sin alteración del sentido de la información (Orden PRE/2065/2014 y aviso
  legal de [boe.es](https://www.boe.es/informacion/aviso_legal/)). Los textos normativos son
  además documentos oficiales sin protección de propiedad intelectual (art. 13 del TRLPI).
- **Seguridad Social** (`ss-imv.md`): contenido público de seg-social.es, reutilizable con
  atribución según la Ley 37/2007 de reutilización de la información del sector público. Se
  guarda solo el texto de la página, sin menús ni cookies.

El código del repositorio es MIT (ver `LICENSE`); estas licencias solo afectan a los datos.
