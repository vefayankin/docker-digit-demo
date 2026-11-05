# Multi-stage build
#   base -> deps -> train -> serve   (the image we ship)
#                         \-> test   (CI: build fails if a test fails)

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models
WORKDIR /app

# Dependencies first: this layer stays cached while only the code changes
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Train the model during the build
FROM deps AS train
COPY app/ app/
ARG MODEL_TYPE=mlp
RUN python -m app.train --model ${MODEL_TYPE} --out /models

# Final image: code + trained model, non-root, with a health check
FROM deps AS serve
RUN useradd --create-home appuser
COPY app/ app/
COPY --from=train /models /models
RUN chown -R appuser /models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Tests run inside the real image
FROM serve AS test
USER root
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests/ tests/
COPY examples/ examples/
RUN python -m pytest -q
