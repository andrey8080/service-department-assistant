import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from assistant.agent.agent import run_agent
from assistant.agent.tools import build_tools, validate_arguments
from assistant.agent.verification import requested_quantities, unsupported_erp_claims, unsupported_facts
from assistant.graphrag.graph_builder import build_graph
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected, ErpUnavailable
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.schemas import Answer, Entity, ExtractionResult, KnowledgeBase, Relation
from assistant.infra.testing import FakeLLM
from erp.app import create_app

ORDER = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}


def call(name, **arguments):
    return LLMResponse("", [ToolCall("test-call", name, arguments)])


def test_search_preserves_current_part_requirement_even_without_a_supplier_question():
    search = Mock(return_value={
        "answer": "P-9070B", "facts": ["P-9070 -REPLACED_BY-> P-9070B"], "sources": ["bulletin.md"],
    })
    llm = FakeLLM([
        call("search_knowledge", question="деталь насоса для GS-610"),
        call("search_knowledge", question="актуальная деталь насоса для GS-610"),
        "Актуальная деталь — P-9070B.",
    ])

    result = run_agent("Какая деталь сейчас ставится как насос для GS-610?", llm, {"search_knowledge": search})

    search.assert_called_once_with(question="актуальная деталь насоса для GS-610")
    assert result.answer == "Актуальная деталь — P-9070B." and not result.refused
    assert len(result.steps) == 2 and "актуальности" in result.steps[0].error
    assert "1–3 предложениях" not in llm.requests[0]["messages"][0]["content"]
    assert "1–3 предложениях" in llm.requests[-1]["messages"][0]["content"]


def test_verified_current_part_can_be_used_in_a_followup_without_repeating_the_time_qualifier():
    search = Mock(side_effect=[
        {"answer": "P-9070B", "facts": ["P-9070 -REPLACED_BY-> P-9070B"]},
        {"answer": "Новый", "facts": ["P-9070B -SUPPLIED_BY-> Новый"]},
    ])
    llm = FakeLLM([
        call("search_knowledge", question="актуальная деталь насоса для GS-610"),
        call("search_knowledge", question="поставщик P-9070B"),
        "Актуальная деталь — P-9070B.",
    ])

    result = run_agent("Какая деталь сейчас ставится как насос для GS-610?", llm, {"search_knowledge": search})

    assert not result.refused
    assert search.call_count == 2 and all(not step.error for step in result.steps)


def test_recovered_argument_error_does_not_disable_review_of_an_unjustified_refusal():
    search = Mock(return_value={
        "answer": "P-9070B", "facts": ["P-9070 -REPLACED_BY-> P-9070B"], "refused": False,
    })
    llm = FakeLLM([
        call("search_knowledge", question="деталь насоса для GS-610"),
        call("search_knowledge", question="актуальная деталь насоса для GS-610"),
        "В базе знаний не найдено.",
        "Актуальная деталь — P-9070B.",
    ])

    result = run_agent("Какая деталь сейчас ставится как насос для GS-610?", llm, {"search_knowledge": search})

    assert not result.refused and "P-9070B" in result.answer
    assert search.call_count == 1 and len(llm.requests) == 4
    assert llm.requests[-1]["messages"][-1]["role"] == "user"
    assert result.steps[0].error and result.steps[0].result["error"]
    assert any("потерял" in message["content"] for message in llm.requests[1]["messages"] if message["role"] == "tool")
    assert any("resolved_error" in message["content"] for message in llm.requests[2]["messages"])


def test_a_refused_overview_does_not_block_exact_procedure_search():
    overview = Mock(return_value={"answer": "В базе знаний не найдено", "refused": True})
    search = Mock(return_value={"answer": "SP-151", "facts": ["E-51 -FIXED_BY-> SP-151"], "refused": False})
    llm = FakeLLM([
        call("summarize_topic", question="регламент для насоса"),
        "В базе знаний не найдено.",
        call("search_knowledge", question="Какой регламент устраняет отказ насоса?"),
        "SP-151.",
    ])

    result = run_agent(
        "Какой регламент устраняет отказ насоса?", llm,
        {"summarize_topic": overview, "search_knowledge": search},
    )

    assert not result.refused and result.answer == "SP-151."
    assert overview.call_count == 1 and search.call_count == 1


@pytest.mark.parametrize("answer,wrong", [
    ("Текущий поставщик — Новый (P-9070).", ["P-9070"]),
    ("Текущий поставщик — Новый (P-9070B).", ["P-9070B"]),
    ("Текущий поставщик — Новый (P-9070C).", []),
    ("Текущий поставщик — Новый.", []),
    ("Вместо P-9070 используется P-9070C, поставщик — Новый.", []),
])
def test_current_supplier_cannot_be_labelled_with_only_an_obsolete_part_number(answer, wrong):
    evidence = [("search_knowledge", {"facts": [
        "P-9070 -REPLACED_BY-> P-9070B", "P-9070B -REPLACED_BY-> P-9070C",
        "P-9070C -SUPPLIED_BY-> Новый",
    ]})]
    assert unsupported_facts(answer, "Кто сейчас поставляет насос для GS-610?", evidence) == wrong


@pytest.mark.parametrize("repaired", [True, False])
def test_current_supplier_part_mismatch_gets_one_correction(repaired):
    search = Mock(return_value={
        "answer": "Новый", "facts": ["P-9070 -REPLACED_BY-> P-9070B", "P-9070B -SUPPLIED_BY-> Новый"],
    })
    wrong = "Текущий поставщик — Новый (P-9070)."
    llm = FakeLLM([
        call("search_knowledge", question="Кто сейчас поставляет насос для GS-610?"),
        wrong, "Новый." if repaired else wrong,
    ])

    result = run_agent("Кто сейчас поставляет насос для GS-610?", llm, {"search_knowledge": search})

    assert result.refused is not repaired
    assert len(llm.requests) == 3 and search.call_count == 1
    assert "старой версии" in llm.requests[-1]["messages"][-1]["content"]


def test_overview_preserves_the_original_catalog_part_constraint():
    question = "Какие детали из руководств линейки «Север» сняты с производства?"
    overview = Mock(return_value={"answer": "P-9070", "sources": ["manual.md", "bulletin.md"]})
    llm = FakeLLM([
        call("summarize_topic", question="снятые детали линейки Север"),
        call("summarize_topic", question=question),
        "P-9070.",
    ])

    result = run_agent(question, llm, {"summarize_topic": overview})

    overview.assert_called_once_with(question=question)
    assert not result.refused and result.answer == "P-9070."
    assert len(result.steps) == 2 and "исходных" in result.steps[0].error


@pytest.mark.parametrize(
    "text",
    [
        "Для точного поставщика требуется проверка в ERP по детали P-4800B.",
        "Поставщика можно узнать через ERP.",
        "ERP хранит сведения о поставщике детали.",
        "Получите информацию о поставщике из ERP.",
        "Поставщик не указан в документах, но его можно узнать через ERP.",
        "Поставщика не знаем, но ERP может вернуть эти сведения.",
    ],
)
def test_supplier_lookup_is_not_an_erp_capability(text):
    assert unsupported_erp_claims(text)


@pytest.mark.parametrize(
    "text",
    [
        "ERP не содержит сведений о поставщике.",
        "ERP не может вернуть сведения о поставщике.",
        "Не стоит проверять поставщика через ERP.",
        "Узнать поставщика через ERP нельзя.",
        "Поставщик СеверПрибор; ERP вернула цену 2630 руб.",
        "Получите остаток из ERP, поставщик указан в бюллетене.",
        "Поставщик СеверПрибор. ERP временно недоступна.",
        "Поставщика нужно проверить по документам.",
    ],
)
def test_erp_limitations_and_independent_document_facts_are_allowed(text):
    assert not unsupported_erp_claims(text)


def test_capability_review_removes_unrequested_supplier_advice_without_calling_erp():
    search = Mock(
        return_value={
            "answer": "P-4800B.",
            "facts": ["P-4800 -REPLACED_BY-> P-4800B"],
            "sources": ["bulletin.md"],
            "refused": False,
        }
    )
    price = Mock()
    llm = FakeLLM(
        [
            call("search_knowledge", question="Актуальная замена P-4800?"),
            "Актуальная деталь P-4800B. Поставщика можно узнать через ERP.",
            "Актуальная деталь P-4800B.",
        ]
    )

    result = run_agent("Актуальная замена P-4800?", llm, {"search_knowledge": search, "get_price": price})

    assert result.answer == "Актуальная деталь P-4800B." and not result.refused
    assert result.sources == ["bulletin.md"] and len(result.steps) == 1
    assert "не поддерживает" in llm.requests[2]["messages"][-1]["content"]
    assert len(llm.requests) == 3
    price.assert_not_called()


def test_capability_review_can_request_the_supplier_from_documents():
    question = "Кто поставляет P-4800?"
    search = Mock(return_value={"answer": "СеверПрибор.", "sources": ["manual.md"], "refused": False,
                                "facts": ["P-4800 -SUPPLIED_BY-> СеверПрибор"]})
    llm = FakeLLM(
        [
            "Сведения о поставщике можно получить в ERP.",
            call("search_knowledge", question=question),
            "Поставщик СеверПрибор.",
        ]
    )

    result = run_agent(question, llm, {"search_knowledge": search})

    assert result.answer == "Поставщик СеверПрибор." and not result.refused
    search.assert_called_once_with(question=question)


@pytest.mark.parametrize(
    "second",
    ["Уточните поставщика в ERP.", "Цена 999999 руб.", "Поставщик P-8800."],
)
def test_capability_review_shares_one_correction_with_grounding(second):
    llm = FakeLLM(["Уточните поставщика в ERP.", second])

    result = run_agent("Кто поставляет P-4800?", llm, {})

    assert result.refused and "В базе знаний не найдено" in result.answer
    assert len(llm.requests) == 2 and not result.steps


def test_tool_observation_omits_source_paths_but_keeps_all_evidence_and_public_sources():
    value = {
        "answer": "Нужна P-4800B.",
        "facts": ["P-4800 -REPLACED_BY-> P-4800B"],
        "sources": ["bulletin.md", "manual.md"],
        "refused": False,
    }
    llm = FakeLLM([call("search_knowledge", question="Замена P-4800?"), "Нужна P-4800B."])

    result = run_agent("Замена P-4800?", llm, {"search_knowledge": Mock(return_value=value)})

    observation = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert observation == {key: item for key, item in value.items() if key != "sources"}
    assert result.steps[0].result == value and result.sources == value["sources"]


@pytest.mark.parametrize("confirm", [False, True])
def test_capability_correction_cannot_bypass_order_confirmation(confirm):
    create = Mock(return_value={"order_id": "SO-000042", **ORDER})
    llm = FakeLLM(
        [
            "Перед заказом поставщика можно проверить в ERP.",
            call("create_service_order", **ORDER),
            "Заказ SO-000042 создан.",
        ]
    )

    result = run_agent(
        "Закажи 2 шт. P-1042B для AR-210 в Казани.",
        llm,
        {"create_service_order": create},
        confirm=confirm,
    )

    assert result.needs_confirmation == (not confirm) and not result.refused
    if confirm:
        create.assert_called_once_with(**ORDER)
        assert result.pending_action is None and "SO-000042" in result.answer
    else:
        create.assert_not_called()
        assert result.pending_action == {"tool": "create_service_order", "args": ORDER}
        assert "SO-000042" not in result.answer


@pytest.mark.parametrize(
    ("requested", "clarified"),
    [("актуальную", "актуальный"), ("нынешнюю", "текущий"), ("последнюю", "нынешний")],
)
def test_supplier_query_keeps_the_requested_replacement_scope(requested, clarified):
    question = f"Назови {requested} деталь для GS-610 и её поставщика."
    short = "поставщик детали для GS-610"
    full = f"{clarified} поставщик детали для GS-610 с учётом замены"
    search = Mock(return_value={"answer": "P-4800B: Новый завод.", "refused": False, "sources": ["bulletin.md"],
                                "facts": ["P-4800B -SUPPLIED_BY-> Новый завод"]})
    llm = FakeLLM(
        [call("search_knowledge", question=short), call("search_knowledge", question=full), "P-4800B: Новый завод."]
    )

    result = run_agent(question, llm, {"search_knowledge": search})

    search.assert_called_once_with(question=full)
    assert result.steps[0].error and result.steps[0].args == {"question": short}
    assert not result.steps[1].error and not result.refused
    assert result.answer == "P-4800B: Новый завод." and result.sources == ["bulletin.md"]


def test_original_supplier_query_does_not_require_a_current_replacement():
    question = "Кто поставляет исходную деталь P-4800 из руководства?"
    search = Mock(return_value={"answer": "Старый завод.", "refused": False,
                                "facts": ["P-4800 -SUPPLIED_BY-> Старый завод"]})
    llm = FakeLLM([call("search_knowledge", question="поставщик P-4800"), "Старый завод."])

    result = run_agent(question, llm, {"search_knowledge": search})

    search.assert_called_once_with(question="поставщик P-4800")
    assert not result.refused and not result.steps[0].error


def test_repeated_supplier_scope_errors_stop_without_executing_search():
    search = Mock()
    query = call("search_knowledge", question="поставщик детали GS-610")
    result = run_agent(
        "Нужна актуальная деталь GS-610 и её поставщик.",
        FakeLLM([query, query]),
        {"search_knowledge": search},
        max_steps=2,
    )

    search.assert_not_called()
    assert len(result.steps) == 2 and all(step.error for step in result.steps)
    assert "лимит" in result.answer


@pytest.mark.parametrize("confirm", [False, True])
def test_text_confirmation_must_be_followed_by_the_order_tool(confirm):
    order = {"part_id": "P-4800B", "warehouse": "Москва", "qty": 3, "model": "GS-610"}
    create = Mock(return_value={"order_id": "SO-000042", **order})
    llm = FakeLLM(["Подтвердите заказ на 3 шт. P-4800B.", call("create_service_order", **order), "Готово."])

    result = run_agent(
        "Оформи заказ на 3 насоса P-4800B для GS-610, склад Москва.",
        llm,
        {"create_service_order": create},
        confirm=confirm,
    )

    assert not result.refused and result.needs_confirmation == (not confirm)
    assert len(result.steps) == 1
    if confirm:
        create.assert_called_once_with(**order)
        assert "SO-000042" in result.answer and result.pending_action is None
    else:
        create.assert_not_called()
        assert result.pending_action == {"tool": "create_service_order", "args": order}
        descriptions = {
            schema["function"]["name"]: schema["function"]["description"]
            for schema in llm.requests[1]["tools"]
        }
        assert "ДО запроса подтверждения" in descriptions["create_service_order"]


def test_repeated_text_confirmation_does_not_claim_a_prepared_order():
    create = Mock()
    draft = "Подтвердите заказ на 3 шт. P-4800B."
    llm = FakeLLM([draft, draft])

    result = run_agent("Закажи 3 шт. P-4800B для GS-610, склад Москва.", llm, {"create_service_order": create})

    create.assert_not_called()
    assert len(llm.requests) == 2
    assert result.refused and not result.needs_confirmation and result.pending_action is None
    assert "Не удалось подготовить заказ" in result.answer and "не создан" in result.answer


@pytest.mark.parametrize(
    "question",
    [
        "Сколько стоят 3 шт. P-4800B?",
        "Не оформи заказ на 3 шт. P-4800B.",
        "Как оформить заказ на 3 шт. P-4800B?",
        "Оформи заказ P-4800B, количество уточним позже.",
    ],
)
def test_confirmation_reminder_does_not_invent_an_order_request_or_quantity(question):
    create = Mock()
    llm = FakeLLM(["Подтвердите данные."])

    result = run_agent(question, llm, {"create_service_order": create})

    create.assert_not_called()
    assert len(llm.requests) == 1 and result.pending_action is None


def test_confirmation_reminder_cannot_authorize_an_unknown_part():
    create = Mock()
    order = {"part_id": "P-8800", "warehouse": "Москва", "qty": 3, "model": "GS-610"}
    llm = FakeLLM(["Подтвердите заказ.", call("create_service_order", **order), "Подтвердите заказ."])

    result = run_agent(
        "Создайте заказ на 3 шт. P-4800B для GS-610, склад Москва.", llm, {"create_service_order": create}
    )

    create.assert_not_called()
    assert result.refused and result.pending_action is None
    assert result.steps[0].error and result.steps[0].args["part_id"] == "P-8800"


def test_agent_recovers_after_model_code_is_used_as_part_id():
    model, old, new, component, error = [
        Entity(kind, name)
        for kind, name in [
            ("Model", "AR-210"),
            ("Part", "P-1042"),
            ("Part", "P-1042B"),
            ("Component", "пусковое реле"),
            ("ErrorCode", "E-17"),
        ]
    ]
    graph = build_graph(
        [
            ExtractionResult(
                "manual.md#0",
                "manual.md",
                relations=[
                    Relation(model, "HAS_PART", old),
                    Relation(old, "IS_A", component),
                    Relation(error, "INDICATES", component),
                ],
            ),
            ExtractionResult("bulletin.md#0", "bulletin.md", relations=[Relation(old, "REPLACED_BY", new)]),
        ]
    )
    kb = KnowledgeBase(graph, [], {"Component": [component.name]})
    question = "Ошибка E-17 на AR-210. Какую деталь заказать, есть ли она в Казани и сколько стоит?"
    llm = FakeLLM(
        [
            call("search_knowledge", question=question),
            "Пусковое реле P-1042 заменено на P-1042B.",
            call("check_stock", part_id="AR-210", warehouse="Казань"),
            call("check_stock", part_id="P-1042B", warehouse="Казань"),
            call("get_price", part_id="P-1042B"),
            "Закажите P-1042B: в Казани остаток 3 шт., цена 2630 руб.",
        ]
    )
    with TestClient(create_app(fail_rate=0)) as http:
        erp = ErpClient("http://testserver", http=http)
        result = run_agent(question, llm, build_tools(kb, llm, erp))

    assert not result.refused and not result.needs_confirmation
    assert result.steps[1].error and "Некорректные аргументы" in result.steps[1].error
    assert result.steps[2].result["qty"] == 3
    assert result.steps[3].result["price_rub"] == 2630
    assert "AR-210 -HAS_PART-> P-1042" in result.steps[0].result["facts"]
    assert result.sources == ["bulletin.md", "manual.md"]
    assert "P-1042B" in result.answer and "2630" in result.answer


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


def test_supplier_question_can_recover_after_an_incomplete_search_argument():
    question = "Какая актуальная деталь контроллера для GS-610 и кто её поставляет?"
    complete = "Актуальная деталь контроллера GS-610 и её поставщик"
    search = Mock(
        return_value={
            "answer": "P-4800B, поставщик СеверПрибор.",
            "facts": [
                "GS-610 -HAS_PART-> P-4800",
                "P-4800 -REPLACED_BY-> P-4800B",
                "P-4800B -SUPPLIED_BY-> СеверПрибор",
            ],
            "sources": ["bulletin.md"],
            "refused": False,
        }
    )
    llm = FakeLLM(
        [
            call("search_knowledge", question="контроллер GS-610"),
            call("search_knowledge", question=complete),
            "Актуальная деталь P-4800B, поставщик СеверПрибор.",
        ]
    )

    result = run_agent(question, llm, {"search_knowledge": search})

    search.assert_called_once_with(question=complete)
    assert result.steps[0].error and "поставщик" in result.steps[0].error
    assert result.steps[0].result == {"error": result.steps[0].error}
    assert not result.steps[1].error and not result.refused
    assert result.sources == ["bulletin.md"] and "P-4800B" in result.answer
    assert len(llm.requests) == 3
    feedback = llm.requests[1]["messages"][-1]
    assert feedback["role"] == "tool" and "поставщик" in feedback["content"]


def test_repeated_incomplete_supplier_searches_stop_at_the_step_limit():
    search = Mock()
    llm = FakeLLM([call("search_knowledge", question="контроллер GS-610")] * 2)

    result = run_agent(
        "Кто сейчас поставляет контроллер для GS-610?",
        llm,
        {"search_knowledge": search},
        max_steps=2,
    )

    search.assert_not_called()
    assert len(result.steps) == len(llm.requests) == 2
    assert all(step.error and "поставщик" in step.error for step in result.steps)
    assert "лимит" in result.answer and not result.needs_confirmation


@pytest.mark.parametrize(
    ("question", "query"),
    [
        ("Кто поставляет контроллер для GS-610?", "Поставщик контроллера GS-610"),
        ("Кто ПОСТАВЛЯЕТ контроллер GS-610?", "Кто ПОСТАВЛЯЕТ контроллер GS-610?"),
        ("Какой контроллер используется в GS-610?", "контроллер GS-610"),
        ("Какую деталь поставить при ошибке E-41 на GS-610?", "Деталь GS-610 при ошибке E-41"),
        ("Что поставить вместо P-4800?", "Замена P-4800"),
        ("Поставь 3 шт. P-4800 для GS-610", "Деталь GS-610"),
    ],
)
def test_complete_supplier_queries_and_other_document_questions_are_not_blocked(question, query):
    search = Mock(return_value={"answer": "P-4800 — Новый завод.", "refused": False,
                                "facts": ["P-4800 -SUPPLIED_BY-> Новый завод"]})
    result = run_agent(
        question,
        FakeLLM([call("search_knowledge", question=query), "P-4800 — Новый завод."]),
        {"search_knowledge": search},
    )

    search.assert_called_once_with(question=query)
    assert len(result.steps) == 1 and not result.steps[0].error and not result.refused


def test_supplier_argument_check_does_not_change_overview_calls():
    overview = Mock(return_value={"answer": "Поставщик СеверПрибор.", "refused": False})
    query = "Обзор деталей линейки Север"
    result = run_agent(
        "Какие поставщики у линейки Север?",
        FakeLLM([call("summarize_topic", question=query), "Поставщик СеверПрибор."]),
        {"summarize_topic": overview},
    )

    overview.assert_called_once_with(question=query)
    assert len(result.steps) == 1 and not result.steps[0].error and not result.refused


def test_corrected_supplier_query_keeps_erp_values_and_order_confirmation():
    question = (
        "Уточни актуальную деталь и поставщика контроллера для GS-610 "
        "и оформи заказ на 3 штуки со склада Москва."
    )
    query = "Актуальная деталь и поставщик контроллера для GS-610"
    order = {"part_id": "P-4800B", "warehouse": "Москва", "qty": 3, "model": "GS-610"}
    search = Mock(
        return_value={
            "answer": "P-4800B, поставщик СеверПрибор.",
            "facts": ["GS-610 -HAS_PART-> P-4800", "P-4800 -REPLACED_BY-> P-4800B"],
            "refused": False,
        }
    )
    price = Mock(return_value={"part_id": "P-4800B", "price_rub": 12340})
    stock = Mock(return_value={"part_id": "P-4800B", "warehouse": "Москва", "qty": 5})
    create = Mock()
    llm = FakeLLM(
        [
            call("search_knowledge", question="контроллер GS-610"),
            call("search_knowledge", question=query),
            call("get_price", part_id="P-4800B"),
            call("check_stock", part_id="P-4800B", warehouse="Москва"),
            call("create_service_order", **order),
        ]
    )

    result = run_agent(
        question,
        llm,
        {"search_knowledge": search, "get_price": price, "check_stock": stock, "create_service_order": create},
    )

    search.assert_called_once_with(question=query)
    price.assert_called_once_with(part_id="P-4800B")
    stock.assert_called_once_with(part_id="P-4800B", warehouse="Москва")
    create.assert_not_called()
    assert result.needs_confirmation and not result.refused
    assert result.pending_action == {"tool": "create_service_order", "args": order}
    assert len(result.steps) == 5 and result.steps[-1].error == "Ожидается подтверждение человека"
    assert all(value in result.answer for value in ["3 шт.", "P-4800B", "GS-610", "Москва", "12340", "остаток 5"])


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


@pytest.mark.parametrize("quantity", ["2.5", "2,5", "2 500,5"])
def test_fractional_quantity_never_authorizes_its_digits(quantity):
    question = f"Закажи {quantity} шт. P-1042B для AR-210 в Казань"
    assert requested_quantities(question) == set()
    create = Mock()
    result = run_agent(
        question, FakeLLM([call("create_service_order", **ORDER)]), {"create_service_order": create}, confirm=True
    )
    create.assert_not_called()
    assert "количество" in result.answer and not result.needs_confirmation


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Закажи 2 500 шт. P-1042B", {2500}),
        ("Нужно 2\u00a0500 единиц P-1042B", {2500}),
        ("Количество: 2\u202f500", {2500}),
        ("Заказать 2500 P-1042B", {2500}),
        ("Закажи 2 штуки P-1042B", {2}),
        ("Закажи две P-1042B", {2}),
        ("Нужно две штуки P-1042B", {2}),
    ],
)
def test_quantity_is_parsed_as_one_number(question, expected):
    assert requested_quantities(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Оформи заказ на 3 компрессора", {3}),
        ("Создай заказ на 4 вентилятора", {4}),
        ("Заказ на 12 деталей", {12}),
        ("Заказ на три датчика", {3}),
        ("Оформи заказ на две штуки", {2}),
        ("Количество: семь", {7}),
        ("Заказ на 2 500 деталей", {2500}),
        ("Сумма заказа на 3000 рублей", set()),
        ("Заказ на три рубля", set()),
        ("Заказ на 5% от бюджета", set()),
        ("Заказ на 3 дня", set()),
        ("Заказ на 6:00", set()),
        ("Заказ на 3.5 датчика", set()),
        ("Заказ на -3 детали", set()),
        ("Заказ на 0 деталей", set()),
        ("Номер заказа: 3", set()),
        ("Заказ на P-4800 для GS-610", set()),
    ],
)
def test_order_quantity_with_preposition_does_not_include_money_time_or_codes(question, expected):
    assert requested_quantities(question) == expected


@pytest.mark.parametrize("confirm", [False, True])
def test_order_with_preposition_still_requires_confirmation(confirm):
    args = {"part_id": "P-4800", "warehouse": "Москва", "qty": 3, "model": "GS-610"}
    create = Mock(return_value={"order_id": "SO-000042", **args})
    result = run_agent(
        "Оформи заказ на 3 компрессора P-4800 для GS-610 со склада Москва.",
        FakeLLM([call("create_service_order", **args), "Заказ SO-000042 создан."]),
        {"create_service_order": create},
        confirm=confirm,
    )
    assert not result.refused
    if confirm:
        create.assert_called_once_with(**args)
        assert not result.needs_confirmation and result.pending_action is None
    else:
        create.assert_not_called()
        assert result.needs_confirmation
        assert result.pending_action == {"tool": "create_service_order", "args": args}
        assert "3 шт." in result.answer


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


@pytest.mark.parametrize("with_erp", [False, True])
def test_confirmation_shows_validated_order_and_erp_values_without_search_paraphrase(with_erp):
    search = Mock(
        return_value={
            "answer": "Лишнее пояснение поиска, которое не нужно для подтверждения.",
            "facts": ["AR-210 -HAS_PART-> P-1042B"],
            "sources": ["manual.md"],
            "refused": False,
        }
    )
    create = Mock()
    script = [call("search_knowledge", question="Деталь для AR-210?")]
    tools = {"search_knowledge": search, "create_service_order": create}
    if with_erp:
        tools["check_stock"] = Mock(return_value={"part_id": "P-1042B", "warehouse": "Казань", "qty": 3})
        tools["get_price"] = Mock(return_value={"part_id": "P-1042B", "price_rub": 2630})
        script.extend(
            [call("check_stock", part_id="P-1042B", warehouse="Казань"), call("get_price", part_id="P-1042B")]
        )
    script.append(call("create_service_order", **ORDER))

    result = run_agent("Закажи 2 штуки детали для AR-210, склад Казань.", FakeLLM(script), tools)

    assert result.needs_confirmation and not result.refused
    assert result.pending_action == {"tool": "create_service_order", "args": ORDER}
    assert all(value in result.answer for value in ["P-1042B", "AR-210", "Казань", "2 шт."])
    assert "Лишнее пояснение" not in result.answer
    assert result.sources == ["manual.md"]
    assert ("2630" in result.answer and "остаток 3" in result.answer) == with_erp
    create.assert_not_called()


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


def test_stock_and_part_digits_cannot_justify_price():
    results = [("check_stock", {"part_id": "P-1042", "qty": 3})]
    assert unsupported_facts("Цена 1042 руб.", "Сколько стоит P-1042?", results)
    assert unsupported_facts("Цена 3 руб.", "Сколько стоит P-1042?", results)
    assert not unsupported_facts("Остаток 3 шт.", "Есть ли P-1042?", results)


@pytest.mark.parametrize("price", ["2 630 руб.", "2\u00a0630,00 ₽", "2630.0 рублей"])
def test_price_format_does_not_change_its_value(price):
    results = [("get_price", {"part_id": "P-1042B", "price_rub": 2630})]
    assert not unsupported_facts(price, "Цена P-1042B?", results)


def test_wrong_stock_without_units_requires_correction():
    tools = {"check_stock": lambda **kw: {"qty": 0, "part_id": "P-1042B", "warehouse": "Казань"}}
    llm = FakeLLM([call("check_stock", part_id="P-1042B", warehouse="Казань"), "Остаток: 8", "Остаток: 0"])
    result = run_agent("Есть ли P-1042B в Казани?", llm, tools)
    assert result.answer == "Остаток: 0" and len(llm.requests) == 3


def test_invented_order_number_is_not_returned():
    llm = FakeLLM(["Заказ SO-123456 создан.", "Заказ SO-123456 создан."])
    result = run_agent("Проверь заказ", llm, {})
    assert result.refused and "SO-123456" not in result.answer


def test_missing_record_is_refusal_but_zero_stock_is_a_value():
    missing = FakeLLM([call("get_price", part_id="P-9999"), "Такой детали нет."])
    tools = {"get_price": lambda **kw: {"error": "Данных нет в ERP: деталь не найдена"}}
    result = run_agent("Сколько стоит P-9999?", missing, tools)
    assert result.refused and "В базе знаний не найдено" in result.answer
    zero = FakeLLM([call("check_stock", part_id="P-1042B", warehouse="Москва"), "Остаток 0 шт."])
    tools = {"check_stock": lambda **kw: {"qty": 0, "part_id": "P-1042B", "warehouse": "Москва"}}
    assert not run_agent("Есть ли P-1042B в Москве?", zero, tools).refused


def test_refused_search_cannot_turn_into_an_unverified_answer():
    llm = FakeLLM([call("search_knowledge", question="Ошибка E-99"), "Поменяйте компрессор."])
    tools = {"search_knowledge": lambda **kw: {"answer": "Не найдено.", "facts": [], "sources": [], "refused": True}}
    result = run_agent("Ошибка E-99", llm, tools)
    assert result.refused and "В базе знаний не найдено" in result.answer


def test_refused_search_keeps_its_status_at_step_limit():
    llm = FakeLLM([call("search_knowledge", question="Ошибка E-99")])
    tools = {"search_knowledge": lambda **kw: {"answer": "В базе знаний не найдено.", "refused": True}}
    result = run_agent("Ошибка E-99", llm, tools, max_steps=1)
    assert result.refused and "В базе знаний не найдено" in result.answer and "лимит" in result.answer
    assert len(result.steps) == 1


def test_step_limit_keeps_successful_facts_alongside_refused_search():
    llm = FakeLLM([call("get_price", part_id="P-1042B"), call("search_knowledge", question="Ошибка E-99")])
    tools = {
        "get_price": lambda **kw: {"part_id": "P-1042B", "price_rub": 2630},
        "search_knowledge": lambda **kw: {"answer": "В базе знаний не найдено.", "refused": True},
    }
    result = run_agent("Цена P-1042B и ошибка E-99?", llm, tools, max_steps=2)
    assert not result.refused and "2630" in result.answer and "лимит" in result.answer


def test_early_refusal_gets_one_chance_to_search_before_answering():
    question = "Кто поставляет деталь P-2126?"
    search = Mock(return_value={"answer": "ХолодКомплект", "sources": ["manual_BZ-220.md"], "refused": False})
    llm = FakeLLM(
        ["В базе знаний не найдено.", call("search_knowledge", question=question), "Поставщик — ХолодКомплект."]
    )
    result = run_agent(question, llm, {"search_knowledge": search})
    assert not result.refused and "ХолодКомплект" in result.answer
    search.assert_called_once_with(question=question)
    assert len(result.steps) == 1 and len(llm.requests) == 3
    assert result.sources == ["manual_BZ-220.md"]
    assert "Ты ещё не проверял" in llm.requests[1]["messages"][-1]["content"]


def test_repeated_early_refusal_stops_without_running_a_tool_in_code():
    search = Mock()
    llm = FakeLLM(["В базе знаний не найдено.", "В базе знаний не найдено."])
    result = run_agent("Что означает ошибка E-28?", llm, {"search_knowledge": search})
    assert result.refused and result.steps == [] and len(llm.requests) == 2
    search.assert_not_called()


def test_unknown_code_still_refuses_after_checking_the_knowledge_base():
    question = "Какую деталь заказать для AR-987?"
    search = Mock(return_value={"answer": "В базе знаний не найдено.", "facts": [], "refused": True})
    llm = FakeLLM(["В базе знаний не найдено.", call("search_knowledge", question=question), "Не знаю."])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert result.refused and "В базе знаний не найдено" in result.answer
    assert len(result.steps) == 1
    search.assert_called_once_with(question=question)


@pytest.mark.parametrize(
    ("question", "tools"),
    [("Какие узлы чаще ломаются?", {}), ("Что означает ошибка E-28?", {})],
)
def test_early_refusal_is_not_retried_without_a_knowledge_tool(question, tools):
    llm = FakeLLM(["В базе знаний не найдено."])
    result = run_agent(question, llm, tools)
    assert result.refused and result.steps == [] and len(llm.requests) == 1


def test_early_refusal_without_codes_can_recover_through_overview_tool():
    question = "Какие узлы чаще ломаются в линейке Север?"
    overview = Mock(return_value={"answer": "Насос: 4 отказа.", "sources": ["reports.md"], "refused": False})
    llm = FakeLLM(["В базе знаний не найдено.", call("summarize_topic", question=question), "Чаще ломается насос."])

    result = run_agent(question, llm, {"summarize_topic": overview})

    assert not result.refused and result.answer == "Чаще ломается насос."
    assert result.sources == ["reports.md"]
    assert len(result.steps) == 1 and len(llm.requests) == 3
    overview.assert_called_once_with(question=question)


def test_early_refusal_without_codes_is_retried_only_once():
    search = Mock()
    llm = FakeLLM(["В базе знаний не найдено.", "В базе знаний не найдено."])

    result = run_agent("Что известно о линейке Север?", llm, {"search_knowledge": search})

    assert result.refused and result.steps == [] and len(llm.requests) == 2
    search.assert_not_called()


def test_search_reminder_keeps_the_single_correction_for_unsupported_facts():
    llm = FakeLLM(["В базе знаний не найдено.", "Вам нужна P-7777.", "Вам нужна P-7777."])
    search = Mock()
    result = run_agent("Деталь для GS-410?", llm, {"search_knowledge": search})
    assert result.refused and "P-7777" not in result.answer and len(llm.requests) == 3
    search.assert_not_called()


def test_search_reminder_does_not_add_an_extra_tool_step():
    question = "Кто поставляет P-2126?"
    search = Mock(return_value={"answer": "ХолодКомплект", "refused": False})
    llm = FakeLLM(["В базе знаний не найдено.", call("search_knowledge", question=question)])
    result = run_agent(question, llm, {"search_knowledge": search}, max_steps=1)
    assert len(result.steps) == 1 and len(llm.requests) == 2 and "лимит" in result.answer
    search.assert_called_once_with(question=question)


@pytest.mark.parametrize("tool", ["search_knowledge", "summarize_topic"])
def test_successful_search_allows_the_model_to_remove_an_unrequested_refusal(tool):
    question = "Какой узел неисправен при E-41?"
    search = Mock(
        return_value={
            "answer": "Неисправен мотор насоса.",
            "facts": ["E-41 -INDICATES-> мотор насоса"],
            "sources": ["diagnostics.md"],
            "refused": False,
        }
    )
    llm = FakeLLM(
        [
            call(tool, question=question),
            "Неисправен мотор насоса. В базе знаний не найдено сведений о цене.",
            "Неисправен мотор насоса.",
        ]
    )

    result = run_agent(question, llm, {tool: search})

    assert result.answer == "Неисправен мотор насоса." and not result.refused
    assert result.sources == ["diagnostics.md"]
    assert len(llm.requests) == 3 and len(result.steps) == 1
    search.assert_called_once_with(question=question)


@pytest.mark.parametrize("partial_question", [False, True])
def test_answer_review_stops_after_one_attempt_and_keeps_an_honest_refusal(partial_question):
    question = "Какой узел неисправен при E-41?"
    if partial_question:
        question += " Кто его поставляет?"
    answer = "Неисправен мотор насоса. В базе знаний не найдено данных о поставщике."
    search = Mock(return_value={"answer": "Мотор насоса.", "refused": False, "sources": ["diagnostics.md"]})
    llm = FakeLLM([call("search_knowledge", question=question), answer, answer])

    result = run_agent(question, llm, {"search_knowledge": search})

    assert result.answer == answer and result.refused
    assert len(llm.requests) == 3 and len(result.steps) == 1
    assert result.sources == ["diagnostics.md"]
    search.assert_called_once_with(question=question)


@pytest.mark.parametrize(
    ("error", "attempts"), [(ErpUnavailable, 3), (ErpNotFound, 1), (ErpRejected, 1)]
)
def test_partial_erp_failure_does_not_trigger_answer_review(error, attempts):
    question = "Какой узел неисправен при E-41 и сколько стоит P-4800?"
    search = Mock(return_value={"answer": "Мотор насоса.", "refused": False, "sources": ["diagnostics.md"]})
    price = Mock(side_effect=error("Нет данных"))
    answer = "Неисправен мотор насоса. В базе знаний не найдено данных о цене: ERP не дала ответ."
    llm = FakeLLM(
        [call("search_knowledge", question=question), call("get_price", part_id="P-4800"), answer]
    )

    result = run_agent(question, llm, {"search_knowledge": search, "get_price": price})

    assert result.answer == answer and result.refused
    assert price.call_count == attempts and result.steps[-1].error
    assert len(llm.requests) == 3 and len(result.steps) == 2
    assert result.sources == ["diagnostics.md"]


def test_refused_second_search_keeps_a_justified_partial_refusal_without_review():
    question = "Что означают E-41 и E-42?"
    search = Mock(
        side_effect=[
            {"answer": "E-41: мотор насоса.", "refused": False, "sources": ["diagnostics.md"]},
            {"answer": "В базе знаний не найдено.", "refused": True, "sources": []},
        ]
    )
    answer = "E-41: мотор насоса. В базе знаний не найдено данных об E-42."
    llm = FakeLLM(
        [call("search_knowledge", question="E-41"), call("search_knowledge", question="E-42"), answer]
    )

    result = run_agent(question, llm, {"search_knowledge": search})

    assert result.answer == answer and result.refused
    assert search.call_count == 2 and len(llm.requests) == 3
    assert result.sources == ["diagnostics.md"]


@pytest.mark.parametrize("invented", ["Нужна деталь P-8800.", "Цена 12345 руб."])
def test_answer_review_does_not_bypass_the_single_grounding_correction(invented):
    question = "Что означает E-41?"
    search = Mock(return_value={"answer": "Мотор насоса.", "refused": False})
    llm = FakeLLM(
        [
            call("search_knowledge", question=question),
            "Мотор насоса. В базе знаний не найдено сведений о цене.",
            invented,
            invented,
        ]
    )

    result = run_agent(question, llm, {"search_knowledge": search})

    assert result.refused and invented not in result.answer
    assert len(llm.requests) == 4 and len(result.steps) == 1
    search.assert_called_once_with(question=question)


@pytest.mark.parametrize("source", ["overview.md", "procedure_SP-141.md"])
def test_procedure_review_lets_the_model_request_local_search_and_keeps_its_sources(source):
    question = "По какому регламенту устраняют отказ мотора насоса?"
    overview = Mock(return_value={"answer": "Замените P-4800.", "sources": [source], "refused": False})
    search = Mock(
        return_value={
            "answer": "Регламент SP-141.",
            "facts": ["E-41 -FIXED_BY-> SP-141"],
            "sources": ["repair_rules.md"],
            "refused": False,
        }
    )
    llm = FakeLLM(
        [
            call("summarize_topic", question=question),
            "Замените P-4800.",
            call("search_knowledge", question=question),
            "Используйте регламент SP-141.",
        ]
    )

    result = run_agent(question, llm, {"summarize_topic": overview, "search_knowledge": search})

    assert result.answer == "Используйте регламент SP-141." and not result.refused
    assert result.sources == sorted([source, "repair_rules.md"])
    assert [step.tool for step in result.steps] == ["summarize_topic", "search_knowledge"]
    assert len(llm.requests) == 4
    search.assert_called_once_with(question=question)


@pytest.mark.parametrize("field", ["answer", "facts"])
def test_procedure_in_search_evidence_does_not_trigger_another_lookup(field):
    question = "Какую процедуру применяют при отказе мотора насоса?"
    evidence = {"answer": "Регламент найден.", "refused": False, "sources": ["repair_rules.md"]}
    evidence[field] = "Регламент SP-141." if field == "answer" else ["E-41 -FIXED_BY-> SP-141"]
    overview = Mock(return_value=evidence)
    search = Mock()
    llm = FakeLLM([call("summarize_topic", question=question), "Регламент SP-141."])

    result = run_agent(question, llm, {"summarize_topic": overview, "search_knowledge": search})

    assert result.answer == "Регламент SP-141." and not result.refused
    assert len(llm.requests) == 2 and len(result.steps) == 1
    search.assert_not_called()


@pytest.mark.parametrize("case", ["already_local", "no_tool", "not_a_procedure", "step_limit"])
def test_procedure_review_respects_previous_search_available_tools_and_step_limit(case):
    question = "Какой регламент применяют при отказе мотора насоса?"
    if case == "not_a_procedure":
        question = "Какую деталь используют в моторе насоса?"
    answer = "Деталь P-4800."
    value = {"answer": answer, "refused": False}
    overview, search = Mock(return_value=value), Mock(return_value=value)
    tools = {"summarize_topic": overview}
    if case != "no_tool":
        tools["search_knowledge"] = search
    script = [call("summarize_topic", question=question)]
    if case == "already_local":
        script.append(call("search_knowledge", question=question))
    if case != "step_limit":
        script.append(answer)
    if case == "already_local":
        script.append(answer)
    llm = FakeLLM(script)

    result = run_agent(question, llm, tools, max_steps=1 if case == "step_limit" else 6)

    if case == "already_local":
        assert result.refused and "В базе знаний не найдено" in result.answer
    else:
        assert answer in result.answer and not result.refused
    assert len(llm.requests) == len(script)
    assert len(result.steps) == (2 if case == "already_local" else 1)
    assert search.call_count == (1 if case == "already_local" else 0)
    assert ("лимит" in result.answer) == (case == "step_limit")


def test_repeated_wrong_procedure_answer_stops_without_executing_search_in_code():
    question = "Какой регламент применяют при отказе мотора насоса?"
    answer = "Замените P-4800."
    overview = Mock(return_value={"answer": answer, "refused": False})
    search = Mock()
    llm = FakeLLM([call("summarize_topic", question=question), answer, answer])

    result = run_agent(question, llm, {"summarize_topic": overview, "search_knowledge": search})

    assert result.refused and result.answer != answer and len(result.steps) == 1 and len(llm.requests) == 3
    search.assert_not_called()


def test_tool_batch_after_answer_review_still_respects_step_limit_and_cache():
    question = "Какой регламент применяют при отказе мотора насоса?"
    overview = Mock(return_value={"answer": "Замените P-4800.", "refused": False})
    search = Mock(return_value={"answer": "Регламент SP-141.", "sources": ["repair_rules.md"], "refused": False})
    batch = LLMResponse(
        "", [ToolCall(str(i), "search_knowledge", {"question": question}) for i in range(4)]
    )
    llm = FakeLLM([call("summarize_topic", question=question), "Замените P-4800.", batch])

    result = run_agent(question, llm, {"summarize_topic": overview, "search_knowledge": search}, max_steps=3)

    assert len(result.steps) == 3 and len(llm.requests) == 3
    assert [step.tool for step in result.steps] == ["summarize_topic", "search_knowledge", "search_knowledge"]
    assert "SP-141" in result.answer and "лимит" in result.answer
    assert result.sources == ["repair_rules.md"]
    search.assert_called_once_with(question=question)


@pytest.mark.parametrize("confirm", [False, True])
def test_order_after_answer_review_still_requires_confirmation(confirm):
    question = "Подберите деталь по регламенту для GS-610 и закажите 2 шт. на склад Москва."
    order = {"part_id": "P-4800", "warehouse": "Москва", "qty": 2, "model": "GS-610"}
    overview = Mock(return_value={"answer": "Нужна P-4800.", "refused": False})
    search = Mock(return_value={"answer": "Деталь P-4800, регламент SP-141.", "refused": False})
    create = Mock(return_value={"order_id": "SO-987654", **order})
    script = [
        call("summarize_topic", question=question),
        "Нужна P-4800.",
        call("search_knowledge", question=question),
        call("create_service_order", **order),
    ]
    if confirm:
        script.append("Регламент SP-141. Заказ SO-987654 создан.")
    llm = FakeLLM(script)

    result = run_agent(
        question,
        llm,
        {"summarize_topic": overview, "search_knowledge": search, "create_service_order": create},
        confirm=confirm,
    )

    assert not result.refused and len(result.steps) == 3
    assert result.needs_confirmation == (not confirm)
    if confirm:
        create.assert_called_once_with(**order)
        assert "SO-987654" in result.answer and not result.pending_action
    else:
        create.assert_not_called()
        assert result.pending_action == {"tool": "create_service_order", "args": order}
        assert all(value in result.answer for value in ["P-4800", "GS-610", "Москва", "2 шт."])


@pytest.mark.parametrize("question, lost", [
    ("Для GS-610 нужны артикул и регламент при E-51", "ошибка E-51 на GS-610"),
    ("Назови поставщика P-9070", "postavshchik P-9070"),
])
def test_search_retry_preserves_the_original_requested_fields(question, lost):
    found = "P-9070, SP-151" if "регламент" in question else "Новый завод"
    facts = (["GS-610 -HAS_PART-> P-9070", "E-51 -FIXED_BY-> SP-151"]
             if "регламент" in question else ["P-9070 -SUPPLIED_BY-> Новый завод"])
    search = Mock(return_value={"answer": found, "refused": False, "facts": facts})
    llm = FakeLLM([call("search_knowledge", question=lost),
                   call("search_knowledge", question=question), found])

    result = run_agent(question, llm, {"search_knowledge": search})

    search.assert_called_once_with(question=question)
    assert not result.refused
    assert question in result.steps[0].error


def test_a_missing_procedure_is_checked_separately_after_finding_the_part():
    question = "На GS-610 ошибка E-51. Назови артикул и регламент."
    search = Mock(side_effect=[
        {"answer": "P-9070", "facts": ["GS-610 -HAS_PART-> P-9070"], "sources": ["manual.md"]},
        {"answer": "SP-151", "facts": ["E-51 -FIXED_BY-> SP-151"], "sources": ["procedure.md"]},
    ])
    llm = FakeLLM([call("search_knowledge", question=question), "Артикул P-9070.",
                   call("search_knowledge", question="Какой регламент при E-51?"), "P-9070, SP-151."])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert not result.refused and result.answer == "P-9070, SP-151."
    assert search.call_count == 2 and len(result.steps) == 2
    assert result.sources == ["manual.md", "procedure.md"]


@pytest.mark.parametrize("partial", ["Актуальная деталь P-9070B.", "Поставщик — Север."])
def test_a_supplier_is_checked_separately_for_the_already_found_replacement(partial):
    question = "Укажи актуальный номер детали и её поставщика для GS-610."
    search = Mock(side_effect=[
        {"answer": "P-9070B", "facts": ["P-9070 -REPLACED_BY-> P-9070B"], "sources": ["notice.md"]},
        {"answer": "Новый завод", "facts": ["P-9070B -SUPPLIED_BY-> Новый завод"], "sources": ["supplier.md"]},
    ])
    llm = FakeLLM([call("search_knowledge", question=question), partial,
                   call("search_knowledge", question="поставщик P-9070B"), "P-9070B — Новый завод."])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert not result.refused and result.answer == "P-9070B — Новый завод."
    assert search.call_count == 2 and len(result.steps) == 2
    assert all(not step.error for step in result.steps)
    assert result.sources == ["notice.md", "supplier.md"]
    assert "все поля" in llm.requests[1]["messages"][0]["content"]


def test_a_failed_composite_search_can_be_split_into_diagnosis_part_and_supplier():
    question = "На GS-610 ошибка E-51. Нужны артикул контроллера и поставщик."
    search = Mock(side_effect=[
        {"answer": "В базе знаний не найдено", "facts": [], "refused": True},
        {"answer": "контроллер", "facts": ["E-51 -INDICATES-> контроллер"]},
        {"answer": "P-9070", "facts": ["GS-610 -HAS_PART-> P-9070", "P-9070 -IS_A-> контроллер"]},
        {"answer": "Новый завод", "facts": ["P-9070 -SUPPLIED_BY-> Новый завод"]},
    ])
    llm = FakeLLM([
        call("search_knowledge", question=question), "В базе знаний не найдено.",
        call("search_knowledge", question="Что означает E-51?"),
        call("search_knowledge", question="деталь контроллера GS-610"),
        call("search_knowledge", question="поставщик P-9070"), "P-9070 — Новый завод.",
    ])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert result.answer == "P-9070 — Новый завод." and not result.refused
    assert len(result.steps) == 4 and search.call_count == 4
    assert all(not step.error for step in result.steps)


def test_a_partial_search_cannot_introduce_an_unseen_code():
    question = "Кто поставляет деталь для GS-610?"
    search = Mock(return_value={"answer": "В базе знаний не найдено", "facts": [], "refused": True})
    llm = FakeLLM([call("search_knowledge", question=question),
                   call("search_knowledge", question="деталь для GS-999")])
    result = run_agent(question, llm, {"search_knowledge": search}, max_steps=2)
    assert search.call_count == 1 and result.steps[-1].error


def test_new_evidence_allows_reviewing_a_missing_field_again():
    question = "Укажи артикул и поставщика для GS-610."
    search = Mock(side_effect=[
        {"answer": "В базе знаний не найдено", "facts": [], "refused": True},
        {"answer": "P-9070", "facts": ["GS-610 -HAS_PART-> P-9070"]},
        {"answer": "Новый завод", "facts": ["P-9070 -SUPPLIED_BY-> Новый завод"]},
    ])
    llm = FakeLLM([
        call("search_knowledge", question=question), "В базе знаний не найдено.",
        call("search_knowledge", question="деталь GS-610"), "Артикул P-9070.",
        call("search_knowledge", question="поставщик P-9070"), "P-9070 — Новый завод.",
    ])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert not result.refused and result.answer == "P-9070 — Новый завод."
    assert len(result.steps) == 3


def test_a_repeated_incomplete_answer_is_not_a_successful_supplier_answer():
    question = "Назови поставщика P-9070."
    search = Mock(return_value={"answer": "SUPPLIED_BY: P-9070",
                                "facts": ["P-9070 -IS_A-> контроллер"]})
    llm = FakeLLM([call("search_knowledge", question=question),
                   "SUPPLIED_BY: P-9070", "SUPPLIED_BY: P-9070"])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert result.refused and result.answer == "В базе знаний не найдено достаточно подтверждённых данных для ответа."
    assert len(llm.requests) == 3


def test_repeated_cached_search_does_not_reset_the_same_missing_field_review():
    question = "Назови поставщика P-9070."
    search = Mock(return_value={"answer": "P-9070", "facts": ["P-9070 -IS_A-> контроллер"]})
    llm = FakeLLM([call("search_knowledge", question=question), "P-9070",
                   call("search_knowledge", question=question), "P-9070"])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert result.refused and len(llm.requests) == 4 and len(result.steps) == 2
    assert search.call_count == 1


@pytest.mark.parametrize("company", ["Старый завод", "ХолодКомплект"])
def test_current_supplier_cannot_be_confirmed_by_the_original_part(company):
    evidence = [("search_knowledge", {
        "refused": True,
        "facts": ["P-9070 -REPLACED_BY-> P-9070B", f"P-9070 -SUPPLIED_BY-> {company}"],
    })]
    assert company in unsupported_facts(company, "Кто сейчас поставляет P-9070?", evidence)


@pytest.mark.parametrize("warehouse, question", [
    ("Санкт-Петербург", "Проверь P-9070 именно на складе Пенза."),
    ("Москва", "Есть ли P-9070 на складе в Казани?"),
    ("Казань", "Закажи 2 шт. P-9070 для GS-610 со склада Москва."),
])
def test_explicit_warehouse_cannot_be_replaced_by_a_different_valid_one(warehouse, question):
    tool = "create_service_order" if question.startswith("Закажи") else "check_stock"
    args = {"part_id": "P-9070", "warehouse": warehouse}
    if tool == "create_service_order":
        args.update(qty=2, model="GS-610")
    function = Mock()
    llm = FakeLLM([call(tool, **args), "В базе знаний не найдено данных об указанном складе."])
    result = run_agent(question, llm, {tool: function})
    function.assert_not_called()
    assert result.refused and result.steps[0].error and not result.needs_confirmation


@pytest.mark.parametrize("question, warehouse", [
    ("Есть ли P-9070 в Казани?", "Казань"),
    ("Есть ли P-9070 на складе Москвы?", "Москва"),
    ("Сколько P-9070 в Санкт-Петербурге?", "Санкт-Петербург"),
    ("Проверь склад Новосибирск для P-9070.", "Новосибирск"),
    ("На каких складах есть P-9070?", "Москва"),
])
def test_warehouse_validation_keeps_inflections_and_unspecified_locations(question, warehouse):
    stock = Mock(return_value={"part_id": "P-9070", "warehouse": warehouse, "qty": 4})
    llm = FakeLLM([call("check_stock", part_id="P-9070", warehouse=warehouse), "Остаток 4 шт."])
    result = run_agent(question, llm, {"check_stock": stock})
    assert not result.refused and not result.steps[0].error
    stock.assert_called_once_with(part_id="P-9070", warehouse=warehouse)


def test_refused_local_answers_do_not_discard_a_supplier_proven_by_accumulated_facts():
    question = "Кто сейчас поставляет контроллер GS-610?"
    search = Mock(side_effect=[
        {"answer": "В базе знаний не найдено", "refused": True, "facts": [
            "GS-610 -HAS_PART-> P-9070", "P-9070 -IS_A-> контроллер", "P-9070 -REPLACED_BY-> P-9070B",
        ]},
        {"answer": "В базе знаний не найдено", "refused": True,
         "facts": ["P-9070B -SUPPLIED_BY-> Новый завод"]},
    ])
    llm = FakeLLM([call("search_knowledge", question=question),
                   call("search_knowledge", question="поставщик P-9070B"), "Новый завод."])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert not result.refused and result.answer == "Новый завод."
    assert search.call_count == 2


@pytest.mark.parametrize("question, answer, facts", [
    ("Какой артикул детали нужен при E-51?", "Нужен насос.", ["E-51 -INDICATES-> насос"]),
    ("Кто поставляет деталь для AR-999?", "ТермоПрибор.", ["P-9070 -SUPPLIED_BY-> ТермоПрибор"]),
    ("Какой регламент при E-51?", "SP-151.", ["E-51 -INDICATES-> насос", "E-52 -FIXED_BY-> SP-151"]),
])
def test_partial_facts_cannot_turn_a_failed_lookup_into_an_unproven_answer(question, answer, facts):
    search = Mock(return_value={"answer": "В базе знаний не найдено", "refused": True, "facts": facts})
    llm = FakeLLM([call("search_knowledge", question=question), answer])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert result.refused and "В базе знаний не найдено" in result.answer


def test_a_part_request_keeps_the_article_after_a_supplier_lookup():
    question = "Назови деталь и её поставщика для устранения E-51 на GS-610."
    search = Mock(return_value={
        "answer": "контроллер, Новый завод", "facts": [
            "E-51 -INDICATES-> контроллер", "GS-610 -HAS_PART-> P-9070",
            "P-9070 -IS_A-> контроллер", "P-9070 -SUPPLIED_BY-> Новый завод",
        ],
    })
    llm = FakeLLM([call("search_knowledge", question=question),
                   "Контроллер, поставщик Новый завод.", "P-9070, поставщик Новый завод."])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert not result.refused and "P-9070" in result.answer
    assert len(result.steps) == 1 and len(llm.requests) == 3


@pytest.mark.parametrize("attribute", ["телефон", "контактный адрес", "e-mail"])
@pytest.mark.parametrize("refused", [True, False])
def test_a_supplier_name_does_not_answer_a_contact_question(attribute, refused):
    question = f"Какой {attribute} указан у поставщика ГидроТех?"
    search = Mock(return_value={"answer": "ГидроТех", "refused": refused,
                                "facts": ["P-9070 -SUPPLIED_BY-> ГидроТех"]})
    llm = FakeLLM([call("search_knowledge", question=question), "ГидроТех", "ГидроТех"])
    result = run_agent(question, llm, {"search_knowledge": search})
    assert result.refused and "В базе знаний не найдено" in result.answer


@pytest.mark.parametrize("confirmation", ["Подтвердите заказ.", "Требуется ваше подтверждение заказа."])
@pytest.mark.parametrize("recovered_error", [True, False])
def test_both_confirmation_wordings_prepare_a_pending_order(confirmation, recovered_error):
    question = "Оформи заказ на 2 шт. P-9070 для GS-610 со склада Москва."
    order = {"part_id": "P-9070", "model": "GS-610", "warehouse": "Москва", "qty": 2}
    price = Mock(return_value={"part_id": "P-9070", "price_rub": 5500})
    create = Mock()
    script = [call("get_price", part_id="GS-610")] if recovered_error else []
    script.extend([call("get_price", part_id="P-9070"), confirmation,
                   call("create_service_order", **order)])
    llm = FakeLLM(script)
    result = run_agent(question, llm, {"get_price": price, "create_service_order": create})
    assert result.needs_confirmation and result.pending_action == {"tool": "create_service_order", "args": order}
    create.assert_not_called()
