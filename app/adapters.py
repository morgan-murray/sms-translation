from __future__ import annotations

import json
import unicodedata
from typing import Any

import httpx

from app.domain import AdapterResult, Language, ModelSet, ModelUsed


class ModelUnavailableError(RuntimeError):
    pass


class TranslationExecutionError(RuntimeError):
    pass


class LlamaAdapter:
    def __init__(
        self,
        *,
        endpoint: str,
        modelset: ModelSet,
        model_id: str,
        revision: str,
        prompt_style: str,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._modelset = modelset
        self._model_id = model_id
        self._revision = revision
        self._prompt_style = prompt_style

    @staticmethod
    def _code(language: Language) -> str:
        return {
            Language.ENGLISH: "en",
            Language.RUSSIAN: "ru",
            Language.MANDARIN: "zh-CN",
            Language.KOREAN: "ko",
        }[language]

    @staticmethod
    def _language_name(language: Language) -> str:
        return {
            Language.ENGLISH: "English",
            Language.RUSSIAN: "Russian",
            Language.MANDARIN: "Chinese",
            Language.KOREAN: "Korean",
        }[language]

    def _translategemma_prompt(self, source: Language, dest: Language, corpus: str) -> str:
        """Render TranslateGemma's text-only template for llama.cpp /completion.

        llama.cpp cannot currently construct its automatic response grammar from
        TranslateGemma's strict structured-content Jinja template. Rendering the
        model's documented text prompt here keeps the intended control tokens and
        translation instruction while avoiding chat-template reinterpretation.
        """
        normalized = unicodedata.normalize("NFC", corpus)
        source_name = self._language_name(source)
        target_name = self._language_name(dest)
        source_code = self._code(source)
        target_code = self._code(dest)
        instruction = (
            f"You are a professional {source_name} ({source_code}) to "
            f"{target_name} ({target_code}) translator. Your goal is to accurately "
            f"convey the meaning and nuances of the original {source_name} text while "
            f"adhering to {target_name} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {target_name} translation, without any additional "
            f"explanations or commentary. Please translate the following {source_name} "
            f"text into {target_name}:\n\n\n{normalized}"
        )
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

    async def translate(self, pair: str, source: Language, dest: Language, corpus: str) -> AdapterResult:
        del pair
        path = "/completion"
        payload = {
            "prompt": self._translategemma_prompt(source, dest, corpus),
            "temperature": 0,
            "top_p": 1,
            "n_predict": 1024,
            "stop": ["<end_of_turn>", "<eos>"],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
                response = await client.post(f"{self._endpoint}{path}", json=payload)
                response.raise_for_status()
                data = response.json()
            if data.get("truncated") or data.get("stopped_limit"):
                raise TranslationExecutionError("translation exceeded the model context or output limit; shorten the message")
            translation = data["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError("translation worker is unavailable") from exc
        if not translation:
            raise TranslationExecutionError(f"{self._modelset.value} returned an empty translation")
        return AdapterResult(
            translation=translation,
            model_used=ModelUsed(
                modelset=self._modelset,
                model_id=self._model_id,
                revision=self._revision,
                quantization="GGUF Q4_K_M",
            ),
        )
