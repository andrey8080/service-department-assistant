# Команды проекта. Файл остаётся как есть.
# По умолчанию всё выполняется в контейнере. Чтобы работать без Docker, передайте RUN="uv run":
#   make test RUN="uv run"
RUN ?= docker compose run --rm app
MODE ?= agent
ARGS ?=
Q ?= Ошибка E-17 на AR-210. Какую деталь заказать, есть ли она на складе в Казани и сколько она стоит?

include .env
export

.PHONY: up up-llm pull-models index ask eval eval-all eval-extraction test lint down

.env:
	cp .env.example .env

up: .env ## собрать и запустить сервисы
	docker compose up -d --build app erp

up-llm: .env ## запустить модельный сервер в контейнере
	docker compose --profile local-llm up -d llm

pull-models: ## скачать модели в контейнер llm
	docker compose --profile local-llm exec llm ollama pull $(LLM_MODEL)
	docker compose --profile local-llm exec llm ollama pull $(EMBED_MODEL)

index: ## построить граф, сообщества и сводки
	$(RUN) python -m assistant index

ask: ## задать вопрос: make ask MODE=local Q="..."
	$(RUN) python -m assistant ask "$(Q)" --mode $(MODE)

eval: ## прогнать открытые вопросы в одном режиме: make eval MODE=baseline
	$(RUN) python -m assistant eval --mode $(MODE)

eval-all: ## прогнать открытые вопросы в четырёх режимах
	$(RUN) sh -c 'for mode in baseline local global agent; do python -m assistant eval --mode $$mode || exit 1; done'

eval-extraction: ## качество извлечения на эталоне
	$(RUN) python -m assistant eval-extraction

test: ## все тесты; один файл: make test ARGS=tests/contract/test_tools_and_agent.py
	$(RUN) pytest -q $(ARGS)

lint: ## линтер
	$(RUN) ruff check .

down: ## остановить сервисы
	docker compose --profile local-llm down
