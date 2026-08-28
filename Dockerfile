FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv/translation \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/translation

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY model-manifest.json ./model-manifest.json

RUN useradd --create-home --uid 10001 translation \
    && chown -R translation:translation /srv/translation

FROM base AS runtime

USER translation

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM base AS test

USER root
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests ./tests
USER translation
CMD ["python", "-m", "pytest", "-q"]
