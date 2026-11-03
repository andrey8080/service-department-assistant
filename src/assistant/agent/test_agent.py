from unittest.mock import Mock

import pytest

from assistant.agent.tools import build_tools, validate_arguments
from assistant.infra.erp_client import ErpNotFound, ErpRejected, ErpUnavailable
from assistant.infra.schemas import Answer
from assistant.infra.testing import FakeLLM

ORDER = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}


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
