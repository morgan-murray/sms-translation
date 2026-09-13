import base64
import os

import pytest
from fastapi.testclient import TestClient

from app.adapters import LlamaAdapter
from app.domain import Language, ModelSet, pair_key
from app.main import app
from app.security import hash_password, verify_password


PASSWORD = "correct-horse-battery"
os.environ["AUTH_USERNAME"] = "tester"
os.environ["AUTH_PASSWORD_HASH"] = hash_password(PASSWORD, salt="00" * 16, iterations=10_000)


class FakeService:
    async def translate(self, request):
        return {
            "request": request.model_dump(),
            "translation": "Привет",
            "model_used": {
                "modelset": "translategemma_4b_q4",
                "model_id": "test/model",
                "revision": "abc123",
                "quantization": "INT8",
            },
            "timing_ms": 12,
            "request_id": "00000000-0000-0000-0000-000000000000",
        }


@pytest.fixture(autouse=True)
def fake_service():
    original = app.state.service
    app.state.service = FakeService()
    yield
    app.state.service = original


def auth_header():
    token = base64.b64encode(f"tester:{PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_password_round_trip():
    encoded = hash_password("secret-value", salt="11" * 16, iterations=10_000)
    assert verify_password("secret-value", encoded)
    assert not verify_password("wrong-value", encoded)


def test_health_is_available_without_credentials():
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200


def test_ui_requires_authentication():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Basic realm="Private translation"'


def test_translation_response_echoes_request_and_model():
    payload = {
        "source": "english",
        "dest": "russian",
        "corpus": "Hello",
        "modelset": "default",
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/translate", headers=auth_header(), json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["request"] == payload
    assert body["translation"] == "Привет"
    assert body["model_used"]["model_id"] == "test/model"
    assert response.headers["cache-control"] == "no-store"


def test_invalid_non_english_pair_is_rejected():
    assert pair_key(Language.ENGLISH, Language.KOREAN) == "en-ko"
    with pytest.raises(ValueError):
        pair_key(Language.RUSSIAN, Language.KOREAN)


def test_translategemma_prompt_uses_model_control_tokens_and_language_codes():
    adapter = LlamaAdapter(
        endpoint="http://example.invalid",
        modelset=ModelSet.TRANSLATEGEMMA,
        model_id="test/translategemma",
        revision="abc123",
        prompt_style="translategemma",
    )
    prompt = adapter._translategemma_prompt(
        Language.ENGLISH, Language.MANDARIN, "Hello there"
    )
    assert prompt.startswith("<start_of_turn>user\n")
    assert "English (en) to Chinese (zh-CN) translator" in prompt
    assert "Hello there<end_of_turn>" in prompt
    assert prompt.endswith("<start_of_turn>model\n")


def test_removed_models_are_rejected():
    with TestClient(app) as client:
        for model in ("hy_mt2_1_8b_q4", "opus_hplt"):
            response = client.post('/api/v1/translate', headers=auth_header(), json={
                'source': 'english', 'dest': 'russian', 'corpus': 'Hello', 'modelset': model,
            })
            assert response.status_code == 422


@pytest.mark.anyio
async def test_default_routes_only_to_gemma():
    from app.service import TranslationService
    from app.domain import TranslationRequest, AdapterResult, ModelUsed
    service = TranslationService()
    calls = []
    async def translate(*args):
        calls.append(args)
        return AdapterResult(translation='Привет', model_used=ModelUsed(
            modelset=ModelSet.TRANSLATEGEMMA, model_id='gemma', revision='test', quantization='Q4'))
    service._adapter.translate = translate
    for model in (ModelSet.DEFAULT, ModelSet.TRANSLATEGEMMA):
        result = await service.translate(TranslationRequest(source='english', dest='russian', corpus='Hello', modelset=model))
        assert result.model_used.modelset == ModelSet.TRANSLATEGEMMA
    assert len(calls) == 2


@pytest.fixture
def anyio_backend():
    return 'asyncio'


def test_cors_preflight_and_auth(monkeypatch):
    import importlib
    import app.main as main
    monkeypatch.setenv('CORS_ALLOWED_ORIGINS', 'https://host.example')
    importlib.reload(main)
    try:
        with TestClient(main.app) as client:
            headers = {'Origin': 'https://host.example', 'Access-Control-Request-Method': 'POST',
                       'Access-Control-Request-Headers': 'authorization,content-type'}
            response = client.options('/api/v1/translate', headers=headers)
            assert response.status_code == 200
            assert response.headers['access-control-allow-origin'] == 'https://host.example'
            response = client.post('/api/v1/translate', headers={'Origin': 'https://host.example'}, json={})
            assert response.status_code == 401
            assert response.headers['access-control-allow-origin'] == 'https://host.example'
            headers['Origin'] = 'https://untrusted.example'
            assert client.options('/api/v1/translate', headers=headers).status_code == 400
    finally:
        monkeypatch.delenv('CORS_ALLOWED_ORIGINS')
        importlib.reload(main)


@pytest.mark.anyio
@pytest.mark.parametrize('body,expected', [
    ({'content': 'Привет'}, None),
    ({'content': 'partial', 'stopped_limit': True}, 'execution'),
    ({'content': 'partial', 'truncated': True}, 'execution'),
    ({'content': '  '}, 'execution'),
    ({}, 'unavailable'),
])
async def test_gemma_worker_response(monkeypatch, body, expected):
    import httpx
    from app.adapters import TranslationExecutionError, ModelUnavailableError
    from app.service import TranslationService
    original_client = httpx.AsyncClient
    def worker(request):
        import json
        assert request.url.path == '/completion'
        assert 'English (en) to Russian (ru)' in json.loads(request.content)['prompt']
        return httpx.Response(200, json=body)
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original_client(
        transport=httpx.MockTransport(worker), **kwargs))
    adapter = TranslationService()._adapter
    if expected:
        error = TranslationExecutionError if expected == 'execution' else ModelUnavailableError
        with pytest.raises(error):
            await adapter.translate('en-ru', Language.ENGLISH, Language.RUSSIAN, 'Hello')
    else:
        result = await adapter.translate('en-ru', Language.ENGLISH, Language.RUSSIAN, 'Hello')
        assert result.translation == 'Привет'
        assert result.model_used.modelset == ModelSet.TRANSLATEGEMMA
