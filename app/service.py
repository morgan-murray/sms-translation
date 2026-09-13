from __future__ import annotations

import os
import time
import uuid

from app.adapters import LlamaAdapter
from app.domain import ModelSet, TranslationRequest, TranslationResponse, pair_key


class TranslationService:
    def __init__(self):
        # This distribution deliberately has a single inference backend.
        self._adapter = LlamaAdapter(
            endpoint=os.environ.get("TRANSLATEGEMMA_ENDPOINT", "http://translategemma:8080"),
            modelset=ModelSet.TRANSLATEGEMMA,
            model_id="mradermacher/translategemma-4b-it-GGUF",
            revision="35a7486e128b19642cdc72d7b91b21ba388aaf42",
            prompt_style="translategemma",
        )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        pair = pair_key(request.source, request.dest)
        started = time.perf_counter()
        result = await self._adapter.translate(pair, request.source, request.dest, request.corpus)
        return TranslationResponse(
            request=request,
            translation=result.translation,
            model_used=result.model_used,
            timing_ms=round((time.perf_counter() - started) * 1000),
            request_id=str(uuid.uuid4()),
        )
