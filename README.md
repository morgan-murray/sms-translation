# Private translation workbench

An offline-capable web UI and JSON API for bidirectional translation between English and Russian, Simplified Chinese, and Korean.

The deployed workbench is available at:

`https://translation.157-180-36-232.sslip.io`

## API

`POST /api/v1/translate`

```json
{
  "source": "english",
  "dest": "russian",
  "corpus": "Text to translate",
  "modelset": "default"
}
```

Model-set values are `default`, `opus_hplt`, `hy_mt2_1_8b_q4`, and `translategemma_4b_q4`. Only English-to-target and target-to-English combinations are accepted.

The deployed `default` route uses Tencent Hy-MT2. The specialist OPUS/HPLT set and TranslateGemma remain available as explicit choices.

## Security and privacy

- HTTP Basic authentication is verified in the application using a PBKDF2-SHA256 password hash.
- The plaintext password is never stored.
- Translation contents and results are not logged.
- Responses use `Cache-Control: no-store` and a restrictive Content Security Policy.
- Model workers are connected only to an internal Docker network.

## Models

The exact source revisions and checksums are recorded in `model-manifest.json`. Run:

```sh
docker compose --profile tools run --rm model-preparer
```

This downloads and converts the specialist Marian models to CTranslate2 INT8 and downloads the two verified GGUF files. Temporary source archives are removed after successful conversion.

Model licences remain the licences of their respective upstream projects. The TranslateGemma deployment requires prior acceptance of Google's Gemma Terms of Use.

No Hugging Face token is required for the artifact set recorded in the manifest. Once the models and container images have been downloaded, translation inference itself does not require internet access.

## Deployment

Create `.env` from `.env.example`, then run:

```sh
docker compose up -d --build
```

The API joins the existing private Caddy network using the alias `translation-api`; no host port is published.

The public proxy target is `translation-api:8000`. Caddy provides TLS; the application enforces authentication on every route except `/healthz`.

## Verification

Run the unit tests in the same container environment as production:

```sh
docker build --target test -t translation-test .
docker run --rm translation-test
```

Run all 18 public model-and-language checks without writing the plaintext password to disk:

```sh
TRANSLATION_SMOKE_PASSWORD='your-password' python3 scripts/smoke_public.py \
  --url https://translation.157-180-36-232.sslip.io \
  --username your-username
```
