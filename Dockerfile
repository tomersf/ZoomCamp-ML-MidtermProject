# ---- Stage 1: install dependencies and make sure a trained model exists ----
FROM python:3.11-slim AS build

# libgomp is needed by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.12.19

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# Runtime dependencies only (no jupyter / plotting), exactly as pinned in uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    # CPU-only XGBoost (same version): drops ~600MB of CUDA/GPU libraries
    && uv pip uninstall xgboost nvidia-nccl-cu12 \
    && uv pip install --no-cache xgboost-cpu==3.2.0

COPY . .

# Train the model only if the repo doesn't already ship one
# (uses data/*.parquet, or downloads the dataset from Kaggle if it is missing)
RUN [ -f models/wildfire_model.pkl ] || python train.py

# Serving doesn't read data files or talk to Kaggle, so leave those libraries out of the final image
RUN uv pip uninstall pyarrow kagglehub


# ---- Stage 2: small runtime image with just the app and the model ----
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PORT=8000

COPY --from=build /app/.venv .venv
COPY --from=build /app/models models
COPY predict.py .
COPY static static

EXPOSE 8000
CMD ["python", "predict.py"]
