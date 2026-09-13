from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Language(str, Enum):
    ENGLISH = "english"
    RUSSIAN = "russian"
    MANDARIN = "mandarin"
    KOREAN = "korean"


class ModelSet(str, Enum):
    DEFAULT = "default"
    TRANSLATEGEMMA = "translategemma_4b_q4"


LANGUAGE_CODES = {
    Language.ENGLISH: "en",
    Language.RUSSIAN: "ru",
    Language.MANDARIN: "zh",
    Language.KOREAN: "ko",
}


class TranslationRequest(BaseModel):
    source: Language
    dest: Language
    corpus: str = Field(min_length=1, max_length=2000)
    modelset: ModelSet = ModelSet.DEFAULT

    @field_validator("corpus")
    @classmethod
    def corpus_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("corpus must contain non-whitespace text")
        return value


class ModelUsed(BaseModel):
    modelset: ModelSet
    model_id: str
    revision: str
    quantization: str


class TranslationResponse(BaseModel):
    request: TranslationRequest
    translation: str
    model_used: ModelUsed
    timing_ms: int
    request_id: str


class AdapterResult(BaseModel):
    translation: str
    model_used: ModelUsed


def pair_key(source: Language, dest: Language) -> str:
    if source == dest:
        raise ValueError("source and destination languages must differ")
    if Language.ENGLISH not in (source, dest):
        raise ValueError("only translations between English and a target language are supported")
    return f"{LANGUAGE_CODES[source]}-{LANGUAGE_CODES[dest]}"
