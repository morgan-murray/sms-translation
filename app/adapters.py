from __future__ import annotations

import asyncio
import json
import os
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ctranslate2
import httpx
import sentencepiece as spm

from app.domain import AdapterResult, Language, ModelSet, ModelUsed


class ModelUnavailableError(RuntimeError):
    pass


class TranslationExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpecialistSpec:
    model_id: str
    revision: str
    source_prefix: str | None = None


SPECIALISTS = {
    "en-ru": SpecialistSpec(
        "Helsinki-NLP/opus-mt-tc-big-en-zle",
        "708be1d372fe4c358a352f404e6dc9ca0126ba48",
        ">>rus<<",
    ),
    "ru-en": SpecialistSpec(
        "Helsinki-NLP/opus-mt-tc-big-zle-en",
        "09a40f722d6d8b76aaad6fe51a06c914622a13d1",
    ),
    "en-zh": SpecialistSpec(
        "Helsinki-NLP/opus-mt-en-zh",
        "408d9bc410a388e1d9aef112a2daba955b945255",
        ">>cmn_Hans<<",
    ),
    "zh-en": SpecialistSpec(
        "Helsinki-NLP/opus-mt-zh-en",
        "cf109095479db38d6df799875e34039d4938aaa6",
    ),
    "en-ko": SpecialistSpec(
        "HPLT/translate-en-ko-v2.0-hplt_opus",
        "97103c8e1819f4accfc9e94e618124e980876d4b",
    ),
    "ko-en": SpecialistSpec(
        "HPLT/translate-ko-en-v2.0-hplt_opus",
        "e8ff1422b2d2b181b2d4119920e50bcdaf573d05",
    ),
}


class SpecialistAdapter:
    def __init__(self, model_root: str | Path):
        self._root = Path(model_root) / "specialists"
        self._loaded: dict[str, tuple[ctranslate2.Translator, Any, Any]] = {}
        self._lock = threading.Lock()

    def _load(self, pair: str):
        with self._lock:
            loaded = self._loaded.get(pair)
            if loaded is not None:
                return loaded
            model_dir = self._root / pair
            required = (model_dir / "model.bin", model_dir / "source.spm", model_dir / "target.spm")
            if not all(path.is_file() for path in required):
                raise ModelUnavailableError(f"specialist model {pair} is not installed")
            translator = ctranslate2.Translator(
                str(model_dir),
                device="cpu",
                compute_type="int8",
                inter_threads=1,
                intra_threads=max(1, min(os.cpu_count() or 1, 8)),
            )
            source_sp = spm.SentencePieceProcessor(model_file=str(model_dir / "source.spm"))
            target_sp = spm.SentencePieceProcessor(model_file=str(model_dir / "target.spm"))
            loaded = (translator, source_sp, target_sp)
            self._loaded[pair] = loaded
            return loaded

    def _translate_sync(self, pair: str, corpus: str) -> AdapterResult:
        spec = SPECIALISTS[pair]
        translator, source_sp, target_sp = self._load(pair)
        normalized = unicodedata.normalize("NFC", corpus)
        source_tokens = source_sp.encode(normalized, out_type=str)
        if spec.source_prefix:
            # Marian target-language labels are vocabulary tokens but are not
            # part of the SentencePiece model, so they must be inserted after
            # tokenization instead of being split into ordinary text pieces.
            source_tokens.insert(0, spec.source_prefix)
        results = translator.translate_batch(
            [source_tokens],
            beam_size=2,
            max_decoding_length=1024,
            repetition_penalty=1.05,
        )
        translation = target_sp.decode(results[0].hypotheses[0]).strip()
        if not translation:
            raise TranslationExecutionError("the specialist model returned an empty translation")
        return AdapterResult(
            translation=translation,
            model_used=ModelUsed(
                modelset=ModelSet.OPUS_HPLT,
                model_id=spec.model_id,
                revision=spec.revision,
                quantization="CTranslate2 INT8",
            ),
        )

    async def translate(self, pair: str, source: Language, dest: Language, corpus: str) -> AdapterResult:
        del source, dest
        return await asyncio.to_thread(self._translate_sync, pair, corpus)


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

    def _messages(self, source: Language, dest: Language, corpus: str) -> list[dict[str, Any]]:
        normalized = unicodedata.normalize("NFC", corpus)
        target_name = {
            Language.ENGLISH: "English",
            Language.RUSSIAN: "Russian",
            Language.MANDARIN: "Simplified Chinese",
            Language.KOREAN: "Korean",
        }[dest]
        prompt = (
            f"Translate the following text into {target_name}. "
            "Return only the translated text without explanation:\n\n"
            f"{normalized}"
        )
        return [{"role": "user", "content": prompt}]

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
        if self._prompt_style == "translategemma":
            path = "/completion"
            payload = {
                "prompt": self._translategemma_prompt(source, dest, corpus),
                "temperature": 0,
                "top_p": 1,
                "n_predict": 1024,
                "stop": ["<end_of_turn>", "<eos>"],
                "stream": False,
            }
        else:
            path = "/v1/chat/completions"
            payload = {
                "model": self._model_id,
                "messages": self._messages(source, dest, corpus),
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 1024,
                "stream": False,
            }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
                response = await client.post(f"{self._endpoint}{path}", json=payload)
                response.raise_for_status()
                data = response.json()
            if self._prompt_style == "translategemma":
                translation = data["content"].strip()
            else:
                translation = data["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError(f"{self._modelset.value} worker is unavailable: {exc}") from exc
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
