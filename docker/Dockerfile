FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# `.env` is excluded via .dockerignore (#20) so it's never present in the
# build context and can't be baked into this (or any) layer by `COPY . .`.
# Runtime config should be injected via `docker run --env-file`/orchestrator
# secrets instead of a committed/copied .env file.
COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "ml.inference:app", "--host", "0.0.0.0", "--port", "8000"]
