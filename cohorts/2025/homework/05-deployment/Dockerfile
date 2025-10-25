FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN mv /code/pipeline_v2.bin /app/

ENV PATH="/app/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

COPY "predict_fastapi.py" "pipeline_v1.bin" ./

EXPOSE 8000

ENTRYPOINT ["uvicorn", "predict_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
