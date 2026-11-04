from unittest.mock import Mock

import pytest

from assistant.agent.agent import run_agent
from assistant.agent.tools import build_tools, validate_arguments
from assistant.agent.verification import requested_quantities
from assistant.infra.erp_client import ErpNotFound, ErpRejected, ErpUnavailable
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.schemas import Answer
from assistant.infra.testing import FakeLLM

ORDER = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}


def call(name, **arguments):
    return LLMResponse("", [ToolCall("test-call", name, arguments)])


@pytest.mark.parametrize("qty", [True, 1.5, "2", 0, -1])
def test_order_rejects_invalid_quantity(qty):
    erp = Mock()
    result = build_tools(None, FakeLLM([]), erp)["create_service_order"](**{**ORDER, "qty": qty})
    assert "error" in result
    erp.create_order.assert_not_called()


def test_arguments_are_normalized_and_extra_fields_are_rejected():
    assert validate_arguments("get_price", {"part_id": "p-1042b"}) == {"part_id": "P-1042B"}
    with pytest.raises(ValueError):
        validate_arguments("get_price", {"part_id": "P-1042B", "admin": True})


@pytest.mark.parametrize("question", ["", "   ", None])
def test_empty_question_never_reaches_search(monkeypatch, question):
    search = Mock()
    monkeypatch.setattr("assistant.agent.tools.local_search", search)
    tools = build_tools(None, FakeLLM([]), Mock())
    assert "error" in tools["search_knowledge"](question=question)
    search.assert_not_called()


@pytest.mark.parametrize("tool,search", [("search_knowledge", "local_search"), ("summarize_topic", "global_search")])
def test_search_tools_keep_sources_and_refusal(monkeypatch, tool, search):
    answer = Answer("В базе знаний не найдено.", sources=["manual.md"], facts=[], refused=True)
    query = Mock(return_value=answer)
    monkeypatch.setattr(f"assistant.agent.tools.{search}", query)
    kb, llm = object(), FakeLLM([])
    result = build_tools(kb, llm, Mock())[tool](question="Ошибка E-17")
    query.assert_called_once_with("Ошибка E-17", kb, llm)
    assert result["answer"] == answer.text and result["sources"] == answer.sources and result["refused"]


def test_erp_errors_and_zero_stock_are_distinct():
    erp = Mock()
    erp.get_stock.return_value = {"qty": 0, "discontinued": False}
    erp.get_price.side_effect = ErpNotFound("деталь не найдена")
    erp.create_order.side_effect = ErpRejected("деталь снята с производства")
    tools = build_tools(None, FakeLLM([]), erp)
    assert tools["check_stock"](part_id="P-1042B", warehouse="Казань")["qty"] == 0
    assert "Данных нет" in tools["get_price"](part_id="P-1042B")["error"]
    assert "отклонила" in tools["create_service_order"](**ORDER)["error"]
    erp.get_price.side_effect = ErpUnavailable("нет связи")
    with pytest.raises(ErpUnavailable):
        tools["get_price"](part_id="P-1042B")


def test_codes_and_prices_are_not_order_quantities():
    assert requested_quantities("Закажи P-1042B для AR-210, бюджет 2 руб.") == set()
    assert requested_quantities("Закажи две штуки P-1042B") == {2}


def test_batch_of_tool_calls_respects_limit_and_cache():
    price = Mock(return_value={"part_id": "P-1042B", "price_rub": 2630})
    batch = LLMResponse("", [ToolCall(str(i), "get_price", {"part_id": "P-1042B"}) for i in range(4)])
    result = run_agent("Цена P-1042B?", FakeLLM([batch]), {"get_price": price}, max_steps=2)
    assert len(result.steps) == 2 and price.call_count == 1
    assert "2630" in result.answer and "лимит" in result.answer


def test_repeated_confirmed_order_is_created_once():
    create = Mock(return_value={"order_id": "SO-000001", **ORDER})
    llm = FakeLLM([call("create_service_order", **ORDER), call("create_service_order", **ORDER), "Готово."])
    result = run_agent(
        "Закажи 2 штуки P-1042B для AR-210 в Казань", llm, {"create_service_order": create}, confirm=True
    )
    create.assert_called_once_with(**ORDER)
    assert "SO-000001" in result.answer


def test_codes_in_errors_do_not_authorize_tool_calls():
    price = Mock()
    tools = {"search_knowledge": lambda **kw: {"error": "Нет данных для P-7777"}, "get_price": price}
    llm = FakeLLM(
        [call("search_knowledge", question="Нужна деталь"), call("get_price", part_id="P-7777"), "Не найдено."]
    )
    result = run_agent("Деталь для AR-210?", llm, tools)
    price.assert_not_called()
    assert "search_knowledge" in result.steps[-1].error


def test_order_with_extra_arguments_is_rejected_before_confirmation():
    create = Mock()
    llm = FakeLLM([call("create_service_order", **ORDER, admin=True), "Некорректный заказ."])
    result = run_agent("Закажи 2 штуки P-1042B", llm, {"create_service_order": create})
    create.assert_not_called()
    assert result.steps[0].error and not result.needs_confirmation


def test_erp_failure_is_reported_even_if_model_omits_it():
    price = Mock(side_effect=ErpUnavailable("нет связи"))
    llm = FakeLLM([call("get_price", part_id="P-1042B"), "Попробуйте позже."])
    result = run_agent("Сколько стоит P-1042B?", llm, {"get_price": price})
    assert price.call_count == 3 and result.steps[0].error
    assert "Данные ERP получить не удалось" in result.answer
