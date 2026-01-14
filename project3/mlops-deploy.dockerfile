#   docker build -t litserve-onnx-app:latest .
#   docker run -p 8000:8000 litserve-onnx-app

FROM python:3.12-slim
# https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for faster startup
# or comment for smaller image (bytecode can add 100-150 MB)
#ENV UV_COMPILE_BYTECODE=1
# Use copy instead of hardlink to support cache mounts
ENV UV_LINK_MODE=copy

# https://docs.astral.sh/uv/guides/integration/docker/#installing-requirements
#  uv pip install torch --torch-backend=cu126
# - either venv or --system
COPY requirements-min.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache --system --torch-backend=cpu -r requirements-min.txt

COPY . .
# TODO only required files for this project

WORKDIR /app/project2
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# CMD  - allow more args passed after "docker run"
ENTRYPOINT ["uv", "run", "server.py"]
