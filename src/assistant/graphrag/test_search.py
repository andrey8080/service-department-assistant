import json

import networkx as nx
import pytest

from assistant.graphrag.communities import detect_communities, summarize_community
from assistant.graphrag.search import global_search, local_search
from assistant.infra.schemas import MAP_SCHEMA, Community, KnowledgeBase
from assistant.infra.testing import FakeLLM


def connect(graph, start, relation, end, *, weight=1, source="manual.md#0"):
    for node in (start, end):
        if node not in graph:
            kind, name = node.split(":", 1)
            graph.add_node(node, type=kind, name=name, sources=[source])
    graph.add_edge(start, end, type=relation, weight=weight, sources=[source])


def test_empty_graph_and_isolated_nodes():
    assert detect_communities(nx.DiGraph()) == []
    graph = nx.DiGraph()
    graph.add_nodes_from(["c", "a", "b"])
    assert detect_communities(graph) == [Community(0, ["a"]), Community(1, ["b"]), Community(2, ["c"])]


def test_partition_does_not_depend_on_insertion_order():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("a", "b", 8), ("b", "a", 4), ("b", "c", 1), ("c", "d", 12)])
    graph.add_node("e")
    reordered = nx.DiGraph()
    reordered.add_nodes_from(reversed(list(graph)))
    reordered.add_edges_from(reversed(list(graph.edges(data=True))))
    before = list(graph.edges(data=True))

    assert detect_communities(graph) == detect_communities(reordered)
    assert detect_communities(graph) == [Community(0, ["a", "b"]), Community(1, ["c", "d"]), Community(2, ["e"])]
    assert list(graph.edges(data=True)) == before


def test_opposite_edge_weights_are_added_before_louvain(monkeypatch):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("a", "b", 3), ("b", "a", 5)])

    def inspect_weights(undirected, *, weight, seed):
        assert not undirected.is_directed()
        assert undirected.edges["a", "b"][weight] == 8
        assert seed == 7
        return [{"a", "b"}]

    monkeypatch.setattr(nx.community, "louvain_communities", inspect_weights)
    assert detect_communities(graph) == [Community(0, ["a", "b"])]


def test_summary_counts_failures_for_each_model_and_line():
    graph = nx.DiGraph()
    for model, weight in (("AR-210", 37), ("AR-220", 5)):
        connect(graph, f"Model:{model}", "FAILED", "Component:пусковое реле", weight=weight)
        connect(graph, f"Model:{model}", "BELONGS_TO", "ProductLine:Арктика")
    connect(graph, "Model:BZ-310", "FAILED", "Component:компрессор", weight=100)
    llm = FakeLLM([" ".join(["слово"] * 180)])

    summary = summarize_community(Community(0, ["Component:пусковое реле"]), graph, llm)

    assert len(llm.requests) == 1
    assert len(summary.split()) == 150
    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert payload["failure_counts_by_model"] == {"AR-210: пусковое реле": 37, "AR-220: пусковое реле": 5}
    assert payload["failure_counts_by_line"] == {"Арктика": {"пусковое реле": 42}}
    assert "компрессор" not in json.dumps(payload, ensure_ascii=False)


def test_summary_model_totals_include_failures_outside_community():
    graph = nx.DiGraph()
    connect(graph, "Model:BZ-110", "FAILED", "Component:пусковое реле", weight=5)
    connect(graph, "Model:BZ-110", "FAILED", "Component:компрессор", weight=5)
    connect(graph, "Model:BZ-120", "FAILED", "Component:пусковое реле", weight=9)
    for model in ("BZ-110", "BZ-120"):
        connect(graph, f"Model:{model}", "BELONGS_TO", "ProductLine:Бриз")
    llm = FakeLLM(["BZ-110: 10 отказов; BZ-120: 9 отказов. Линейка Бриз.", "Отказы компрессора."])

    summarize_community(Community(0, ["Model:BZ-110", "Model:BZ-120", "Component:пусковое реле"]), graph, llm)
    summarize_community(Community(1, ["Component:компрессор"]), graph, llm)

    totals = json.loads(llm.requests[0]["messages"][-1]["content"])["model_failure_totals"]
    assert totals == {
        "BZ-110": {"total": 10, "product_lines": ["Бриз"]},
        "BZ-120": {"total": 9, "product_lines": ["Бриз"]},
    }
    assert json.loads(llm.requests[1]["messages"][-1]["content"])["model_failure_totals"] == {}


def test_summary_contains_internal_facts_and_not_unrelated_edges():
    graph = nx.DiGraph()
    connect(graph, "Part:P-1042", "REPLACED_BY", "Part:P-1042B")
    connect(graph, "Part:P-1042B", "SUPPLIED_BY", "Supplier:ХолодКомплект")
    llm = FakeLLM(["Старая деталь заменена новой."])

    summarize_community(Community(0, ["Part:P-1042", "Part:P-1042B"]), graph, llm)

    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert payload["facts"] == ["P-1042 -REPLACED_BY-> P-1042B"]
    assert payload["failure_counts_by_model"] == {}


def test_local_search_includes_all_short_paths_and_preserves_edge_direction():
    graph = nx.DiGraph()
    connect(graph, "ErrorCode:E-17", "INDICATES", "Component:пусковое реле")
    connect(graph, "Model:AR-210", "FAILED", "Component:пусковое реле")
    connect(graph, "Model:AR-210", "HAS_PART", "Part:P-1042")
    connect(graph, "Part:P-1042", "IS_A", "Component:пусковое реле")
    connect(graph, "Model:AR-210", "HAS_PART", "Part:P-1003", source="long_path.md#0")
    connect(graph, "Part:P-1003", "REPLACED_BY", "Part:P-1004", source="long_path.md#0")
    connect(graph, "Part:P-1004", "IS_A", "Component:пусковое реле", source="long_path.md#0")
    llm = FakeLLM(["Пусковое реле P-1042."])

    answer = local_search("Что означает E-17 на AR-210?", KnowledgeBase(graph, [], {}), llm)

    assert set(answer.facts) == {
        "E-17 -INDICATES-> пусковое реле",
        "AR-210 -FAILED-> пусковое реле",
        "AR-210 -HAS_PART-> P-1042",
        "P-1042 -IS_A-> пусковое реле",
    }
    assert answer.sources == ["manual.md"]


@pytest.mark.parametrize("cycle", [False, True])
def test_local_search_follows_long_replacement_chains_without_looping(cycle):
    graph = nx.DiGraph()
    parts = [f"Part:P-100{i}" for i in range(1, 7)]
    for start, end in zip(parts, parts[1:], strict=False):
        connect(graph, start, "REPLACED_BY", end, source="replacements.md#1")
    if cycle:
        connect(graph, parts[-1], "REPLACED_BY", parts[0], source="replacements.md#2")

    answer = local_search("Чем заменить P-1001?", KnowledgeBase(graph, [], {}), FakeLLM(["Цепочка замен."]))

    assert len(answer.facts) == 5 + int(cycle)
    assert "P-1005 -REPLACED_BY-> P-1006" in answer.facts
    assert answer.sources == ["replacements.md"]


def test_local_search_does_not_expand_ordinary_edges_beyond_two_hops():
    graph = nx.DiGraph()
    connect(graph, "Model:AR-210", "HAS_PART", "Part:P-1042")
    connect(graph, "Part:P-1042", "IS_A", "Component:пусковое реле")
    connect(graph, "ErrorCode:E-17", "INDICATES", "Component:пусковое реле")

    answer = local_search("Детали AR-210", KnowledgeBase(graph, [], {}), FakeLLM(["Пусковое реле P-1042."]))

    assert len(answer.facts) == 2
    assert not any("E-17" in fact for fact in answer.facts)


@pytest.mark.parametrize("question", ["E-17 на AR-210", "AR-999", "Что нового?"])
def test_local_search_refuses_disconnected_or_missing_anchors(question):
    graph = nx.DiGraph()
    for kind, name in [("Model", "AR-210"), ("ErrorCode", "E-17")]:
        graph.add_node(f"{kind}:{name}", type=kind, name=name, sources=[])
    llm = FakeLLM([])

    answer = local_search(question, KnowledgeBase(graph, [], {}), llm)

    assert answer.refused and "В базе знаний не найдено" in answer.text
    assert not llm.requests


def test_local_search_bounds_numbered_context_and_cites_only_selected_facts():
    graph = nx.DiGraph()
    for i in range(500):
        connect(graph, "Part:P-1042", "IS_A", f"Component:узел{i}", source=f"doc{i}.md#0")
    llm = FakeLLM(["Узлы из фактов."])

    answer = local_search("Что за P-1042?", KnowledgeBase(graph, [], {}), llm)

    context = llm.requests[0]["messages"][-1]["content"].split("Факты:\n", 1)[1]
    assert len(context) <= 8000
    assert 0 < len(answer.facts) < 500
    assert answer.sources == sorted(f"doc{fact.rsplit('узел', 1)[1]}.md" for fact in answer.facts)


def test_local_search_can_find_supplier_of_replacement_outside_anchor_path():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-520", "HAS_PART", "Part:P-3126")
    connect(graph, "Part:P-3126", "IS_A", "Component:реле давления")
    connect(graph, "Part:P-3126", "REPLACED_BY", "Part:P-3126B")
    connect(graph, "Part:P-3126B", "SUPPLIED_BY", "Supplier:АвтоматикаСевер", source="suppliers.md#0")
    kb = KnowledgeBase(graph, [], {"Component": ["реле давления"]})

    answer = local_search("Кто поставляет реле давления для GS-520?", kb, FakeLLM(["АвтоматикаСевер"]))

    assert "P-3126B -SUPPLIED_BY-> АвтоматикаСевер" in answer.facts
    assert answer.sources == ["manual.md", "suppliers.md"]


@pytest.mark.parametrize("response", ["", "   ", "В базе знаний не найдено нужного регламента."])
def test_local_search_marks_empty_answers_and_model_refusals(response):
    graph = nx.DiGraph()
    connect(graph, "Part:P-1042", "IS_A", "Component:пусковое реле")

    answer = local_search("Как ремонтировать P-1042?", KnowledgeBase(graph, [], {}), FakeLLM([response]))

    assert answer.refused
    assert "В базе знаний не найдено" in answer.text


@pytest.mark.parametrize(
    "partial",
    [
        "это не JSON",
        '{"answer": "ответ", "score": 11}',
        '{"answer": "ответ", "score": -1}',
        '{"answer": 7, "score": 5}',
        '{"answer": "ответ", "score": "5"}',
        '{"answer": "ответ", "score": true}',
        '{"answer": "ответ"}',
        '{"answer": " ", "score": 5}',
        '{"answer": "ответ", "score": 0}',
    ],
)
def test_global_search_skips_invalid_or_useless_partial_answers(partial):
    kb = KnowledgeBase(nx.DiGraph(), [Community(0, [], "Сводка")], {})
    llm = FakeLLM([partial])

    answer = global_search("Обзор неисправностей", kb, llm)

    assert answer.refused and not answer.sources
    assert len(llm.requests) == 1
    assert llm.requests[0]["json_schema"] == MAP_SCHEMA


def test_global_search_cites_only_useful_communities_and_deduplicates_documents():
    graph = nx.DiGraph()
    connect(graph, "Model:AR-210", "FAILED", "Component:пусковое реле", source="arctic.md#1")
    connect(graph, "Model:AR-210", "BELONGS_TO", "ProductLine:Арктика", source="arctic.md#2")
    connect(graph, "Model:BZ-310", "FAILED", "Component:компрессор", source="unrelated.md#0")
    communities = [
        Community(0, ["Model:AR-210", "Component:пусковое реле", "ProductLine:Арктика"], "Отказы Арктики"),
        Community(1, ["Model:BZ-310", "Component:компрессор"], "Отказы другой линейки"),
        Community(2, [], " "),
    ]
    llm = FakeLLM(
        [
            '{"answer": "Пусковое реле: 37 отказов.", "score": 8}',
            '{"answer": "НЕ НУЖНО", "score": 0}',
            "У пускового реле 37 отказов.",
        ]
    )

    answer = global_search("Что ломается в Арктике?", KnowledgeBase(graph, communities, {}), llm)

    assert not answer.refused and answer.sources == ["arctic.md"]
    assert len(llm.requests) == 3
    assert all(request["json_schema"] == MAP_SCHEMA for request in llm.requests[:2])
    merged = llm.requests[-1]["messages"][-1]["content"]
    assert "Пусковое реле: 37 отказов." in merged
    assert "НЕ НУЖНО" not in merged


@pytest.mark.parametrize("communities", [[], [Community(0, [], " ")]])
def test_global_search_without_summaries_refuses_without_calling_model(communities):
    llm = FakeLLM([])

    answer = global_search("Обзор", KnowledgeBase(nx.DiGraph(), communities, {}), llm)

    assert answer.refused and not llm.requests


@pytest.mark.parametrize("response", ["", "В базе знаний не найдено нужных сведений."])
def test_global_search_marks_empty_final_answer_and_model_refusal(response):
    kb = KnowledgeBase(nx.DiGraph(), [Community(0, [], "Известны отказы реле")], {})
    llm = FakeLLM(['{"answer": "Отказывает реле.", "score": 2}', response])

    answer = global_search("Что ломается?", kb, llm)

    assert answer.refused
    assert "В базе знаний не найдено" in answer.text
