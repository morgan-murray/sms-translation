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
                "modelset": "opus_hplt",
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
