from __future__ import annotations

import os
import time
import uuid

from app.adapters import LlamaAdapter, SpecialistAdapter
from app.domain import ModelSet, TranslationRequest, TranslationResponse, pair_key


class TranslationService:
    def __init__(self):
        model_root = os.environ.get("MODEL_ROOT", "/models")
        self._default = ModelSet(os.environ.get("DEFAULT_MODELSET", ModelSet.OPUS_HPLT.value))
        self._adapters = {
            ModelSet.OPUS_HPLT: SpecialistAdapter(model_root),
            ModelSet.HY_MT2: LlamaAdapter(
                endpoint=os.environ.get("HY_MT2_ENDPOINT", "http://hy-mt2:8080"),
                modelset=ModelSet.HY_MT2,
                model_id="tencent/Hy-MT2-1.8B-GGUF",
                revision="1cd5208700acedef4ef93019b6cfc148b8522d45",
                prompt_style="hy_mt2",
            ),
            ModelSet.TRANSLATEGEMMA: LlamaAdapter(
                endpoint=os.environ.get("TRANSLATEGEMMA_ENDPOINT", "http://translategemma:8080"),
                modelset=ModelSet.TRANSLATEGEMMA,
                model_id="mradermacher/translategemma-4b-it-GGUF",
                revision="35a7486e128b19642cdc72d7b91b21ba388aaf42",
                prompt_style="translategemma",
            ),
        }

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        pair = pair_key(request.source, request.dest)
        selected = self._default if request.modelset == ModelSet.DEFAULT else request.modelset
        started = time.perf_counter()
        result = await self._adapters[selected].translate(
            pair,
            request.source,
            request.dest,
            request.corpus,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return TranslationResponse(
            request=request,
            translation=result.translation,
            model_used=result.model_used,
            timing_ms=elapsed_ms,
            request_id=str(uuid.uuid4()),
        )
