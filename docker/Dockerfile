# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Multi-stage build.
#
# Stage 1 (builder): install dependencies into a virtualenv using `uv`, which
#   is dramatically faster than pip and produces reproducible installs.
# Stage 2 (runtime): copy only the venv + source into a slim image. Keeping
#   build tools out of the final image reduces size and attack surface.
#
# A non-root user runs the service (least privilege). The CMD launches uvicorn
# against the application-factory module `api.app:app`.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Install uv (static binary, no extra deps).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./

# Create the venv and install runtime dependencies only (no dev extras).
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python .

# ---------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create an unprivileged user.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness check hits the health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
