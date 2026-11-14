import json
from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import pytest

from assistant.agent import tools as agent_tools
from assistant.graphrag.communities import _compact_lines
from assistant.graphrag.search import OverviewAnswer, global_search
from assistant.infra.schemas import MAP_SCHEMA, Answer, Community, KnowledgeBase
from assistant.infra.testing import FakeLLM


@pytest.fixture
def kb():
    graph = nx.DiGraph()

    def connect(start, relation, end, sources, weight=1):
        for node in (start, end):
            kind, name = node.split(":", 1)
            graph.add_node(node, type=kind, name=name, sources=[])
        graph.add_edge(start, end, type=relation, weight=weight, sources=sources)

    for model, parts in (("AR-751", ("P-7510", "P-7514")), ("AR-752", ("P-7520", "P-7524"))):
        source = [f"manual_{model}.md#0"]
        connect(f"Model:{model}", "BELONGS_TO", "ProductLine:Лагуна", source)
        for part, component in zip(parts, ("двигатель", "датчик"), strict=True):
            connect(f"Model:{model}", "HAS_PART", f"Part:{part}", source)
            connect(f"Part:{part}", "IS_A", f"Component:{component}", source)
    for model, component, reports in (
        ("AR-751", "двигатель", (1, 2)),
        ("AR-751", "датчик", (3,)),
        ("AR-752", "двигатель", (4,)),
        ("AR-752", "датчик", (5, 6, 7)),
    ):
        connect(f"Model:{model}", "FAILED", f"Component:{component}",
                [f"visit_{number:02}.md#0" for number in reports], weight=len(reports))
    communities = [
        Community(0, ["Model:AR-751", "Model:AR-752", "ProductLine:Лагуна"]),
        Community(1, ["Component:двигатель", "Part:P-7510", "Part:P-7520"]),
        Community(2, ["Component:датчик", "Part:P-7514", "Part:P-7524"]),
    ]
    for community in communities:
        community.summary = "\n".join(_compact_lines(community, graph))
        assert 0 < len(community.summary.split()) <= 150
    return KnowledgeBase(graph, communities, {"ProductLine": ["Лагуна"], "Component": ["двигатель", "датчик"]})


def map_answer(score=8):
    return json.dumps({"answer": "Есть частичные сведения для сравнения.", "score": score}, ensure_ascii=False)


def run(kb, question, selection, maps=None):
    maps = maps if maps is not None else [map_answer() for c in kb.communities if c.summary.strip()]
    response = json.dumps({"row_ids": selection}) if isinstance(selection, list) else selection
    llm = FakeLLM([*maps, response])
    answer = global_search(question, kb, llm)
    return answer, llm


def reduce_payload(llm):
    return json.loads(llm.requests[-1]["messages"][-1]["content"])


def test_every_summary_has_one_map_and_exactly_one_reduce_preserves_deduplicated_matrix(kb):
    question = "Сравни AR-751 и AR-752 по отказам узлов: распределение и общее число случаев."
    answer, llm = run(kb, question, [1, 2, 3])

    assert not answer.refused
    assert len(llm.requests) == len(kb.communities) + 1
    for community, request in zip(kb.communities, llm.requests[:-1], strict=True):
        assert request["json_schema"] == MAP_SCHEMA
        assert request["messages"][-1]["content"] == f"Вопрос: {question}\nСводка: {community.summary}"
    assert "row_ids" in llm.requests[-1]["json_schema"]["properties"]
    payload = reduce_payload(llm)
    assert payload["complete"]
    assert payload["verified_rows"] == [
        {"id": 1, "text": "AR-751 — датчик: 1; двигатель: 2; всего: 3."},
        {"id": 2, "text": "AR-752 — датчик: 3; двигатель: 1; всего: 4."},
        {"id": 3, "text": "Общее число отказов: 7."},
    ]
    assert answer.text == "\n".join(row["text"] for row in payload["verified_rows"])
    assert set(answer.sources) == {"manual_AR-751.md", "manual_AR-752.md"} | {
        f"visit_{number:02}.md" for number in range(1, 8)
    }
    assert all("#" not in source for source in answer.sources)


@pytest.mark.parametrize("selection", [
    [], [1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [1, 1, 2, 3], ["1", 2, 3],
    "не JSON", '{"row_ids": [1, 2, 3], "answer": "лишний текст"}',
])
def test_invalid_or_incomplete_row_selection_cannot_bypass_reduce(kb, selection):
    answer, llm = run(kb, "Сравни AR-751 и AR-752 по отказам узлов, укажи общий итог.", selection)

    assert answer.refused
    assert "в базе знаний не найдено" in answer.text.lower()
    assert len(llm.requests) == 4


def test_zero_usefulness_for_all_maps_does_not_read_the_graph_as_an_alternative(kb):
    llm = FakeLLM([map_answer(0) for _ in kb.communities])
    answer = global_search("Сравни AR-751 и AR-752 по отказам узлов.", kb, llm)

    assert answer.refused
    assert len(llm.requests) == len(kb.communities)
    assert all(request["json_schema"] == MAP_SCHEMA for request in llm.requests)


@pytest.mark.parametrize("first_map", [map_answer(0), "это не JSON"])
def test_ignored_model_summary_does_not_silently_supply_coverage(kb, first_map):
    answer, llm = run(kb, "Сравни AR-751 и AR-752 по отказам узлов.", [],
                      maps=[first_map, map_answer(), map_answer()])

    assert answer.refused
    assert len(llm.requests) == 4
    payload = reduce_payload(llm)
    assert not payload["complete"] and payload["verified_rows"] == []


def test_missing_summary_cannot_be_reconstructed_from_remaining_graph(kb):
    kb.communities[0].summary = ""
    answer, llm = run(kb, "Сравни AR-751 и AR-752 по отказам узлов.", [])

    assert kb.graph.has_edge("Model:AR-751", "Component:двигатель")
    assert answer.refused
    assert len(llm.requests) == 3
    assert not reduce_payload(llm)["complete"]


def test_missing_failure_record_is_not_recovered_from_graph_or_other_complete_totals(kb):
    graph_before = deepcopy(kb.graph)
    for community in kb.communities:
        community.summary = "\n".join(
            line for line in community.summary.splitlines()
            if not (line.startswith("Отказы |") and "AR-752" in line and "двигатель=1" in line)
        )
    answer, llm = run(kb, "Сравни AR-751 и AR-752 по отказам узлов, укажи общий итог.", [])

    assert nx.utils.graphs_equal(kb.graph, graph_before)
    assert kb.graph.edges["Model:AR-752", "Component:двигатель"]["weight"] == 1
    assert answer.refused
    payload = reduce_payload(llm)
    assert not payload["complete"] and payload["verified_rows"] == []


@pytest.mark.parametrize("question", [
    "По «Лагуне» распиши отказы двигателя по моделям.",
    "Распиши отказы двигателя для каждой модели «Лагуны».",
])
def test_indirect_cases_keep_line_and_soft_sign_component_scope(kb, question):
    answer, llm = run(kb, question, [1, 2])

    assert not answer.refused
    assert answer.text == "AR-751 — двигатель: 2; всего: 2.\nAR-752 — двигатель: 1; всего: 1."
    payload = reduce_payload(llm)
    assert payload["complete"] and len(payload["verified_rows"]) == 2
    assert "датчик" not in answer.text


def test_overview_does_not_turn_failure_counts_into_erp_prices(kb):
    llm = FakeLLM([*[map_answer() for _ in kb.communities],
                   '{"answer": "Цена двигателя — 123 рублей.", "evidence_ids": [1]}'])
    answer = global_search("Сколько стоит деталь двигателя для моделей линейки Лагуна?", kb, llm)

    assert answer.refused
    assert "123" not in answer.text
    assert "row_ids" not in llm.requests[-1]["json_schema"]["properties"]


@pytest.mark.parametrize("kind, refused, complete", [
    (Answer, False, False), (OverviewAnswer, False, True), (OverviewAnswer, True, False),
])
def test_only_verified_complete_overview_gets_the_completion_marker(kb, monkeypatch, kind, refused, complete):
    answer = kind("Подтверждённый ответ", ["manual.md"], ["подтверждённая связь"], refused)
    monkeypatch.setattr(agent_tools, "global_search", lambda *_: answer)
    erp = SimpleNamespace(get_stock=lambda **_: {}, get_price=lambda **_: {}, create_order=lambda **_: {})
    result = agent_tools.build_tools(kb, FakeLLM([]), erp)["summarize_topic"](question="Обзор")

    assert (result.get("complete") is True) == complete
    assert (result.get("facts") == answer.facts) == complete
