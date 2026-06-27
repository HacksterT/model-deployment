# Local API image. Build from repo root: docker build -f api/Dockerfile -t superkart-api .
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ ./src/
COPY models/ ./models/
COPY api/ ./api/
EXPOSE 7860
CMD ["uv", "run", "gunicorn", "-b", "0.0.0.0:7860", "--chdir", "api", "app:app"]
