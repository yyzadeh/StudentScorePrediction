ARG PYTHON_VERSION=3.11.0
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# uv normally hardlinks from its cache, which fails across filesystem
# boundaries (the --mount cache is a separate fs). Copy mode avoids that.
ENV UV_LINK_MODE=copy

WORKDIR /app

# Pull the uv binary from its official image — no pip needed at all
RUN pip install --no-cache-dir uv

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy only the dependency files first — layer stays cached until they change
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen
# Activate the venv for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
