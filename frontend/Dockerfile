# Local UI image. Build from repo root: docker build -f frontend/Dockerfile -t superkart-ui .
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY frontend/ ./frontend/
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0"]
