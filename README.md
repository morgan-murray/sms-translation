# Embedded translation service

TranslateGemma-only sidecar for a host application. The React/MUI UI calls this
service before submitting translated text to its existing SDS middleware.
Middleware REST/WebSockets and the OCPI gRPC driver remain unchanged.
This branch does not transmit radio messages or enforce SDS encoding limits.

## Install

1. Clone and checkout `embedded-service`.
2. Copy `.env.example` to `.env`. Set a username and generate a password hash with
   `python -m scripts.hash_password`. Keep the hash single-quoted in `.env`.
3. Create `models/gguf` and copy `translategemma-4b-it.Q4_K_M.gguf` there from the
   existing deployment. Alternatively run `docker compose --profile tools run
   --rm model-preparer` after creating the writable `models` directory. Set
   MODEL_PREPARER_UID/GID to its owner's numeric IDs if they differ from 1000.
   The preparer verifies the pinned size and SHA-256, including cached files.
4. Run `docker compose up -d --build`.

API binds to localhost:8001 (TRANSLATION_PORT overrides the port). Put it behind
the host application's HTTPS proxy for remote browsers. The model worker has no
published port and uses an internal network. This Compose project is independent
of the original VPS deployment. `/healthz` is API liveness only, not model readiness.
Allow a warm-up period; translation may return 503 until the worker is ready.

The sole model is Google's TranslateGemma 4B, using the existing third-party
GGUF Q4_K_M artifact pinned in `model-manifest.json` (2,489,909,760 bytes).
The client must review the applicable Gemma terms and artifact provenance before
distribution. No Tencent or specialist model is downloaded or selected in this branch.
Only copy the Gemma weights when packaging this deployment. For an air-gapped
system also export/import the built API and pinned llama.cpp container images.
No internet connection is needed for inference once artifacts are installed.

## Host UI integration

Prefer a same-origin reverse proxy, e.g. `/translation/` stripping that prefix
before forwarding to localhost:8001. For Caddy on the same machine:

```caddy
handle_path /translation/* {
    reverse_proxy 127.0.0.1:8001
}
```

Merge this into the host's routing; retain its middleware/WebSocket routes.
If the proxy is containerized, use a shared network and the API's container port
8000 instead of the proxy container's localhost.
Keep application Basic authentication enabled. The host proxy can inject its
Authorization header server-side **only after authenticating the host user**;
overwrite any incoming Authorization header on that route. Never embed shared
credentials in the React bundle. Alternatively, a PoC operator can supply separate
translation credentials at runtime. For direct cross-origin calls, set
`CORS_ALLOWED_ORIGINS` to exact UI origins (comma separated, no wildcard), use HTTPS,
and send Basic Authorization. CORS is not authentication.

```http
POST /translation/api/v1/translate
Content-Type: application/json

{"source":"english","dest":"russian","corpus":"Return to base.","modelset":"default"}
```

`dest` accepts `russian`, `mandarin` (Simplified Chinese), or `korean` for English
input. Reverse translations to English also work. Omit `modelset` or use `default`;
both always select TranslateGemma. Explicit `translategemma_4b_q4` is also accepted;
other model names are rejected with 422. The response contains `translation`,
`model_used`, `timing_ms`, `request_id`, and the echoed `request`.

For the composer:

- Default “Send in” to English and bypass translation in that mode.
- For other languages, translate the draft and let the operator review it.
- Invalidate translated text on any draft/language edit. Ignore stale in-flight
  results; see `examples/host-translation.ts` for an abortable request helper.
- Send the reviewed text using the existing SDS REST payload and recipient.
- Validate the translated text against the middleware's actual encoding and byte
  limits. Do not assume 2,000 API characters fit an SDS or truncate silently.
- Keep drafts on failure; never silently send the English text instead.

API failures: 401 authentication, 422 invalid text/language/model, 503 worker
unavailable, 502 empty or truncated model output. The model timeout is 180 seconds;
configure the proxy timeout accordingly. Client abort stops waiting but does not
guarantee inference cancellation. The service's 2,000-character input cap and model
context/output limits are independent of SDS limits.

## Verification

```sh
docker build --target test -t translation-embedded-test .
docker run --rm translation-embedded-test
```

For real inference, set `TRANSLATION_SMOKE_PASSWORD` in the environment and run:

```sh
python scripts/smoke_public.py --url http://localhost:8001 --username YOUR_USERNAME
```

This probes all six Gemma translation directions. Check quality with representative
radio messages on target hardware; automated script checks are not semantic review.
