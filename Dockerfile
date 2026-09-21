FROM python:3.11-slim

# Configuration de l'environnement Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dépendance système obligatoire pour LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Récupération du binaire uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Dépendances du projet
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Code et artefacts
COPY api/ ./api/
COPY data/mlflow/ ./data/mlflow/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]