FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /app

RUN pip install --no-cache-dir uv==0.10.8

# Зависимости ставятся отдельным слоем: он пересобирается, только когда меняется uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

CMD ["uvicorn", "assistant.infra.api:app", "--host", "0.0.0.0", "--port", "8000"]
