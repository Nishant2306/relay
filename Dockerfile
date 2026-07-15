FROM python:3.11-slim

WORKDIR /app

# layer-cache the dependency install
COPY pyproject.toml ./
RUN mkdir -p gateway verifier mockprovider admin \
    && touch gateway/__init__.py verifier/__init__.py \
             mockprovider/__init__.py admin/__init__.py \
    && pip install --no-cache-dir -e .

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# fastembed downloads bge-small on first use; persist it in a volume
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache

EXPOSE 8080
CMD ["uvicorn", "gateway.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
