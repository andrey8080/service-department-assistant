"""Инструменты поиска и ERP."""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Question(_Arguments):
    question: StrictStr = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Вопрос не должен быть пустым")
        return value


class _Part(_Arguments):
    part_id: StrictStr = Field(
        description="Номер детали P-… из вопроса или фактов поиска; код оборудования не подходит."
    )

    @field_validator("part_id")
    @classmethod
    def part_code(cls, value):
        code = normalize_code(value)
        if code is None or code_type(code) != "Part":
            raise ValueError("Нужен номер детали вида P-1042 или P-1042B")
        return code


class _Stock(_Part):
    warehouse: Literal["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


class _Order(_Stock):
    qty: StrictInt = Field(gt=0)
    model: StrictStr

    @field_validator("model")
    @classmethod
    def model_code(cls, value):
        code = normalize_code(value)
        if code is None or code_type(code) != "Model":
            raise ValueError("Нужен код модели оборудования, например AR-210")
        return code


ARGUMENT_MODELS = {
    "search_knowledge": _Question,
    "summarize_topic": _Question,
    "check_stock": _Stock,
    "get_price": _Part,
    "create_service_order": _Order,
}
DESCRIPTIONS = {
    "search_knowledge": "Поиск точных фактов, детали и её актуальной замены по документам. Начинай с него.",
    "summarize_topic": "Обзор отказов, узлов и поставщиков по сводкам сообществ графа.",
    "check_stock": "Остаток известной детали на выбранном складе и discontinued. Остаток 0 — известное значение.",
    "get_price": "Текущая цена известной детали в рублях и discontinued, из ERP.",
    "create_service_order": "Заказать указанное пользователем количество; требуется подтверждение человека.",
}
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {"name": name, "description": DESCRIPTIONS[name], "parameters": model.model_json_schema()},
    }
    for name, model in ARGUMENT_MODELS.items()
]


def validate_arguments(name: str, arguments: dict) -> dict:
    """Одна и та же строгая проверка используется обёртками инструментов и агентом."""
    if name not in ARGUMENT_MODELS or not isinstance(arguments, dict):
        raise ValueError("Неизвестный инструмент или нечитаемые аргументы")
    return ARGUMENT_MODELS[name].model_validate(arguments).model_dump()


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Оборачивает поиск и ERP; временная недоступность передаётся агенту для повторов."""

    def search_knowledge(question):
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question):
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    functions = {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": erp.get_stock,
        "get_price": erp.get_price,
        "create_service_order": erp.create_order,
    }

    def wrap(name, function):
        def checked(**arguments):
            try:
                args = validate_arguments(name, arguments)
            except ValueError as exc:
                return {"error": f"Некорректные аргументы: {exc}"}
            try:
                return function(**args)
            except ErpNotFound as exc:
                return {"error": f"Данных нет в ERP: {exc}"}
            except ErpRejected as exc:
                return {"error": f"ERP отклонила запрос: {exc}"}

        return checked

    return {name: wrap(name, function) for name, function in functions.items()}
