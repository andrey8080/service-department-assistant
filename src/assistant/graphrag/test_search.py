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


def summary_response(*, original_parts=None, replacements=None, failure_counts=None, facts=None):
    return json.dumps(
        {
            "original_parts": original_parts or [],
            "replacements": replacements or [],
            "failure_counts": failure_counts or [],
            "facts": facts or [],
        },
        ensure_ascii=False,
    )


@pytest.fixture
def summary_graph():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070")
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север")
    connect(graph, "Model:BZ-610", "HAS_PART", "Part:P-9070B")
    connect(graph, "Model:BZ-610", "BELONGS_TO", "ProductLine:Юг")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9070B", "REPLACED_BY", "Part:P-9070C")
    connect(graph, "Model:GS-610", "FAILED", "Component:насос", weight=4)
    connect(graph, "Model:GS-610", "FAILED", "Component:контроллер", weight=1)
    return graph


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
    assert summary == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert 0 < len(summary.split()) <= 150
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


def test_summary_has_complete_counts_for_relevant_line_without_mixing_other_lines():
    graph = nx.DiGraph()
    connect(graph, "Model:AR-410", "FAILED", "Component:контроллер", weight=3)
    connect(graph, "Model:AR-420", "FAILED", "Component:компрессор", weight=9)
    for model in ("AR-410", "AR-420"):
        connect(graph, f"Model:{model}", "BELONGS_TO", "ProductLine:Север")
    connect(graph, "Model:BZ-410", "FAILED", "Component:вентилятор", weight=90)
    connect(graph, "Model:BZ-410", "BELONGS_TO", "ProductLine:Юг")
    llm = FakeLLM(["Север: контроллер=3, компрессор=9; AR-410=3, AR-420=9."])

    summarize_community(Community(0, ["Component:контроллер"]), graph, llm)

    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert payload["failure_counts_by_line"] == {"Север": {"контроллер": 3}}
    assert payload["model_failure_totals"] == {}
    assert payload["full_failure_counts_by_line"] == {
        "Север": {"by_component": {"контроллер": 3, "компрессор": 9}, "by_model": {"AR-410": 3, "AR-420": 9}},
    }
    assert "Юг" not in json.dumps(payload, ensure_ascii=False)
    assert "AR-410 -BELONGS_TO-> Север" in payload["boundary_facts"]


def test_summary_renders_confirmed_model_records_without_mixing_them(summary_graph):
    response = summary_response(
        original_parts=[
            {"part": "P-9070", "model": "GS-610", "product_lines": ["Север"]},
            {"part": "P-9070B", "model": "BZ-610", "product_lines": ["Юг"]},
        ],
        replacements=[{"old": "P-9070", "new": "P-9070B"}, {"old": "P-9070B", "new": "P-9070C"}],
        failure_counts=[
            {"product_line": "Север", "by_component": {"насос": 4, "контроллер": 1}, "by_model": {"GS-610": 5}},
        ],
        facts=["GS-610 -HAS_PART-> P-9070"],
    )
    llm = FakeLLM([response])

    summary = summarize_community(Community(0, sorted(summary_graph)), summary_graph, llm)

    lines = summary.splitlines()
    assert any("Север" in line and "GS-610" in line and "| P-9070" in line for line in lines)
    assert any("Юг" in line and "BZ-610" in line and "| P-9070B" in line for line in lines)
    assert "P-9070 снята; замена P-9070B" in summary
    assert "P-9070B снята; замена P-9070C" in summary
    assert any("Отказы узлов" in line and "Север" in line and "насос: 4" in line for line in lines)
    assert any("Всего отказов моделей" in line and "Север" in line and "GS-610: 5" in line for line in lines)
    assert "GS-610 -HAS_PART-> P-9070" in lines
    assert 0 < len(summary.split()) <= 150
    assert len(llm.requests) == 1
    assert llm.requests[0]["json_schema"]


@pytest.mark.parametrize(
    "record",
    [
        {"part": "P-9070B", "model": "GS-610", "product_lines": ["Север"]},
        {"part": "P-9070", "model": "GS-610", "product_lines": ["Юг"]},
    ],
)
def test_summary_rejects_wrong_original_part_or_line(summary_graph, record):
    llm = FakeLLM([summary_response(original_parts=[record])])

    summary = summarize_community(Community(0, sorted(summary_graph)), summary_graph, llm)

    assert summary == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert len(llm.requests) == 1


def test_summary_rejects_a_shortcut_instead_of_two_replacements(summary_graph):
    llm = FakeLLM([summary_response(replacements=[{"old": "P-9070", "new": "P-9070C"}])])

    summary = summarize_community(Community(0, sorted(summary_graph)), summary_graph, llm)

    assert summary == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert len(llm.requests) == 1


@pytest.mark.parametrize(
    "record",
    [
        {"product_line": "Север", "by_component": {"насос": 5, "контроллер": 1}, "by_model": {"GS-610": 4}},
        {"product_line": "Север", "by_component": {"насос": 4}, "by_model": {"GS-610": 5}},
        {"product_line": "Юг", "by_component": {"насос": 4, "контроллер": 1}, "by_model": {"GS-610": 5}},
        {"product_line": "Север", "by_component": {"насос": 4, "контроллер": True}, "by_model": {"GS-610": 5}},
    ],
)
def test_summary_rejects_changed_or_incomplete_failure_counts(summary_graph, record):
    llm = FakeLLM([summary_response(failure_counts=[record])])

    summary = summarize_community(Community(0, sorted(summary_graph)), summary_graph, llm)

    assert summary == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert len(llm.requests) == 1


def test_summary_rejects_an_unsupported_fact_even_when_codes_exist(summary_graph):
    llm = FakeLLM([summary_response(facts=["BZ-610 -HAS_PART-> P-9070"])])

    summary = summarize_community(Community(0, sorted(summary_graph)), summary_graph, llm)

    assert summary == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert len(llm.requests) == 1


def test_summary_does_not_add_rows_that_the_model_omitted(summary_graph):
    llm = FakeLLM([summary_response(replacements=[{"old": "P-9070", "new": "P-9070B"}])])

    summary = summarize_community(Community(0, sorted(summary_graph)), summary_graph, llm)

    assert "P-9070 снята; замена P-9070B" in summary
    assert "P-9070C" not in summary
    assert "GS-610" not in summary
    assert "Север" not in summary
    assert len(llm.requests) == 1


def test_summary_rejects_a_long_valid_record_without_cutting_it():
    graph = nx.DiGraph()
    supplier = "Поставщик " + "длинное " * 160
    connect(graph, "Part:P-9070", "SUPPLIED_BY", f"Supplier:{supplier}")
    fact = f"P-9070 -SUPPLIED_BY-> {supplier}"
    llm = FakeLLM([summary_response(facts=[fact])])

    summary = summarize_community(Community(0, ["Part:P-9070"]), graph, llm)

    assert summary == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert "P-9070" not in summary
    assert 0 < len(summary.split()) <= 150
    assert len(llm.requests) == 1


def test_summary_limits_extra_facts_without_losing_parts_replacements_or_counts(summary_graph):
    facts = []
    for number in range(30):
        supplier = f"Поставщик оборудования номер {number}"
        connect(summary_graph, "Part:P-9070C", "SUPPLIED_BY", f"Supplier:{supplier}")
        facts.append(f"P-9070C -SUPPLIED_BY-> {supplier}")
    llm = FakeLLM(
        [
            summary_response(
                original_parts=[{"part": "P-9070", "model": "GS-610", "product_lines": ["Север"]}],
                replacements=[{"old": "P-9070", "new": "P-9070B"}, {"old": "P-9070B", "new": "P-9070C"}],
                failure_counts=[
                    {
                        "product_line": "Север",
                        "by_component": {"насос": 4, "контроллер": 1},
                        "by_model": {"GS-610": 5},
                    }
                ],
                facts=facts,
            )
        ]
    )

    summary = summarize_community(Community(0, sorted(summary_graph)), summary_graph, llm)

    assert "Исходная деталь | Север | GS-610 | P-9070" in summary.splitlines()
    assert "P-9070 снята; замена P-9070B." in summary.splitlines()
    assert "P-9070B снята; замена P-9070C." in summary.splitlines()
    assert "Отказы узлов, Север | контроллер: 1; насос: 4" in summary.splitlines()
    assert "Всего отказов моделей, Север | GS-610: 5" in summary.splitlines()
    returned_facts = [line for line in summary.splitlines() if " -SUPPLIED_BY-> " in line]
    assert 0 < len(returned_facts) < len(facts)
    assert set(returned_facts) <= set(facts)
    assert len(summary.split()) <= 150
    assert len(llm.requests) == 1


def test_summary_rejects_essential_counts_over_budget_without_shortening_the_table():
    graph = nx.DiGraph()
    component = "Узел " + "длинное " * 160
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север")
    connect(graph, "Model:GS-610", "FAILED", f"Component:{component}", weight=4)
    llm = FakeLLM(
        [
            summary_response(
                failure_counts=[{"product_line": "Север", "by_component": {component: 4}, "by_model": {"GS-610": 4}}]
            )
        ]
    )

    summary = summarize_community(Community(0, sorted(graph)), graph, llm)

    assert summary == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert len(llm.requests) == 1


def test_summary_still_checks_unsupported_extra_facts_that_would_not_fit(summary_graph):
    unsupported = "BZ-610 -HAS_PART-> P-9070 " + "неподтверждённое " * 160
    llm = FakeLLM(
        [
            summary_response(
                replacements=[{"old": "P-9070", "new": "P-9070B"}],
                facts=[unsupported],
            )
        ]
    )

    summary = summarize_community(Community(0, sorted(summary_graph)), summary_graph, llm)

    assert summary == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert len(llm.requests) == 1


def test_summary_keeps_boundary_relations_and_model_of_replacement_chain():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070")
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9070B", "REPLACED_BY", "Part:P-9070C")
    connect(graph, "Part:P-9070B", "IS_A", "Component:насос")
    connect(graph, "Part:P-9070C", "SUPPLIED_BY", "Supplier:Поставщик")
    llm = FakeLLM(["P-9070B заменена на P-9070C для GS-610 линейки Север."])

    summarize_community(Community(0, ["Part:P-9070B", "Part:P-9070C"]), graph, llm)

    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert payload["replacement_context"] == [
        {"old": "P-9070", "new": "P-9070B"},
        {"old": "P-9070B", "new": "P-9070C"},
    ]
    assert payload["original_parts"] == [{"part": "P-9070", "model": "GS-610", "product_lines": ["Север"]}]
    assert set(payload["boundary_facts"]) == {
        "P-9070 -REPLACED_BY-> P-9070B",
        "P-9070B -IS_A-> насос",
        "P-9070C -SUPPLIED_BY-> Поставщик",
        "GS-610 -HAS_PART-> P-9070",
        "GS-610 -BELONGS_TO-> Север",
    }
    assert len(llm.requests) == 1


def test_summary_distinguishes_original_parts_of_different_models():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070")
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север")
    connect(graph, "Model:BZ-610", "HAS_PART", "Part:P-9070B")
    connect(graph, "Model:BZ-610", "BELONGS_TO", "ProductLine:Юг")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9070B", "REPLACED_BY", "Part:P-9070C")
    llm = FakeLLM(["В Севере исходная P-9070, в Юге исходная P-9070B."])

    summarize_community(Community(0, ["Part:P-9070B"]), graph, llm)

    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert sorted(payload["original_parts"], key=lambda item: item["part"]) == [
        {"part": "P-9070", "model": "GS-610", "product_lines": ["Север"]},
        {"part": "P-9070B", "model": "BZ-610", "product_lines": ["Юг"]},
    ]
    assert payload["replacement_context"] == [
        {"old": "P-9070", "new": "P-9070B"},
        {"old": "P-9070B", "new": "P-9070C"},
    ]


def test_summary_keeps_replacement_branches_as_separate_pairs():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070C")
    llm = FakeLLM(["Для P-9070 есть две замены: P-9070B и P-9070C."])

    summarize_community(Community(0, ["Part:P-9070B"]), graph, llm)

    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert payload["replacement_context"] == [
        {"old": "P-9070", "new": "P-9070B"},
        {"old": "P-9070", "new": "P-9070C"},
    ]
    assert payload["original_parts"] == [{"part": "P-9070", "model": "GS-610", "product_lines": []}]


def test_summary_restores_full_chain_from_its_last_part():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070")
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9070B", "REPLACED_BY", "Part:P-9070C")
    connect(graph, "Model:BZ-610", "HAS_PART", "Part:P-9080")
    connect(graph, "Part:P-9080", "REPLACED_BY", "Part:P-9080B")
    llm = FakeLLM(["P-9070 заменена на P-9070B, затем на P-9070C."])

    summarize_community(Community(0, ["Part:P-9070C"]), graph, llm)

    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert payload["replacement_context"] == [
        {"old": "P-9070", "new": "P-9070B"},
        {"old": "P-9070B", "new": "P-9070C"},
    ]
    assert payload["original_parts"] == [{"part": "P-9070", "model": "GS-610", "product_lines": ["Север"]}]
    assert "P-9080" not in json.dumps(payload, ensure_ascii=False)
    assert "BZ-610" not in json.dumps(payload, ensure_ascii=False)
    assert len(llm.requests) == 1


def test_summary_visits_replacement_cycle_once_without_inventing_original_parts():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9070B", "REPLACED_BY", "Part:P-9070C")
    connect(graph, "Part:P-9070C", "REPLACED_BY", "Part:P-9070")
    llm = FakeLLM(["В заменах найден замкнутый цикл."])

    summarize_community(Community(0, ["Part:P-9070B"]), graph, llm)

    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert payload["replacement_context"] == [
        {"old": "P-9070", "new": "P-9070B"},
        {"old": "P-9070B", "new": "P-9070C"},
        {"old": "P-9070C", "new": "P-9070"},
    ]
    assert payload["original_parts"] == []
    assert len(llm.requests) == 1


def test_summary_does_not_guess_line_from_model_code():
    graph = nx.DiGraph()
    connect(graph, "Model:AR-410", "FAILED", "Component:контроллер", weight=2)
    connect(graph, "Model:AR-410", "HAS_PART", "Part:P-9070")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    llm = FakeLLM(["AR-410: контроллер=2; P-9070 заменена на P-9070B."])

    summarize_community(Community(0, ["Model:AR-410", "Part:P-9070"]), graph, llm)

    payload = json.loads(llm.requests[0]["messages"][-1]["content"])
    assert payload["full_failure_counts_by_line"] == {}
    assert payload["original_parts"] == [{"part": "P-9070", "model": "AR-410", "product_lines": []}]
    assert payload["model_failure_totals"]["AR-410"]["product_lines"] == []


def test_summary_bounds_boundary_context():
    graph = nx.DiGraph()
    for number in range(300):
        connect(graph, f"Part:P-{9000 + number}", "SUPPLIED_BY", "Supplier:Поставщик длинное название")
    llm = FakeLLM(["Поставщик деталей."])

    summarize_community(Community(0, ["Supplier:Поставщик длинное название"]), graph, llm)

    context = json.loads(llm.requests[0]["messages"][-1]["content"])["boundary_facts"]
    assert 0 < sum(map(len, context)) <= 8000
    assert len(context) < 300


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


def test_local_search_repairs_a_code_missing_from_selected_facts():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9988", "IS_A", "Component:контроллер")
    llm = FakeLLM(["Закажите p 9988.", "Закажите P-9070B."])

    answer = local_search("Чем заменить P-9070?", KnowledgeBase(graph, [], {}), llm)

    assert not answer.refused and answer.text == "Закажите P-9070B."
    assert len(llm.requests) == 2
    assert len(llm.requests[0]["messages"]) == 2
    repair = llm.requests[1]["messages"]
    assert len(repair) == 2
    assert all(message["role"] != "assistant" for message in repair)
    assert "9988" not in str(repair)
    assert "Чем заменить P-9070?" in repair[1]["content"]
    assert "P-9070 -REPLACED_BY-> P-9070B" in repair[1]["content"]


@pytest.mark.parametrize("response", ["Нужна P-8800.", "В базе знаний не найдено.", "Нужна P-4800B."])
def test_local_retry_uses_the_same_evidence_without_the_incorrect_explanation(response):
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-4800", source="manual.md#0")
    connect(graph, "Part:P-4800", "REPLACED_BY", "Part:P-4800B", source="bulletin.md#0")
    llm = FakeLLM(["Для GS-610 нужна P-4800B, потому что P-4800 заменена на P-8800.", response])

    answer = local_search("Что заказать для GS-610?", KnowledgeBase(graph, [], {}), llm)

    assert len(llm.requests) == 2
    first = llm.requests[0]["messages"]
    retry = llm.requests[1]["messages"]
    assert first[0] == retry[0]
    assert all(message["role"] != "assistant" for message in retry)
    assert "8800" not in str(retry)
    assert all(fact in retry[1]["content"] for fact in answer.facts)
    assert answer.sources == ["bulletin.md", "manual.md"]
    assert answer.refused == (response != "Нужна P-4800B.")
    assert "P-8800" not in answer.text


def test_local_search_refuses_after_one_unsuccessful_code_repair():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    llm = FakeLLM(["Нужна P-9988.", "Тогда P-9989."])

    answer = local_search("Чем заменить P-9070?", KnowledgeBase(graph, [], {}), llm)

    assert answer.refused and answer.text == "В базе знаний не найдено достаточно данных для ответа."
    assert len(llm.requests) == 2
    assert "P-9988" not in answer.text and "P-9989" not in answer.text


def test_local_search_accepts_existing_codes_without_extra_call():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    llm = FakeLLM(["Для P-9070 используйте P-9070B."])

    answer = local_search("Чем заменить P-9070?", KnowledgeBase(graph, [], {}), llm)

    assert not answer.refused
    assert len(llm.requests) == 1


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


@pytest.mark.parametrize(
    "question",
    [
        "Что означает E-99 на AR-610?",
        "Какую деталь заказать при E-28 на AR-999?",
        "Сколько стоит P-9999?",
    ],
)
def test_global_search_refuses_unknown_codes_despite_plausible_other_summaries(question):
    graph = nx.DiGraph()
    connect(graph, "Model:AR-610", "HAS_PART", "Part:P-9070")
    connect(graph, "ErrorCode:E-28", "INDICATES", "Component:насос")
    summary = "В модели AR-610 ошибка E-28 требует детали P-9070."
    llm = FakeLLM([])

    answer = global_search(question, KnowledgeBase(graph, [Community(0, sorted(graph), summary)], {}), llm)

    assert answer.refused and not answer.sources and not answer.facts
    assert not llm.requests


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


def test_global_search_keeps_summary_evidence_that_the_partial_answer_lost():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9080")
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9070B", "REPLACED_BY", "Part:P-9070C")
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север")
    connect(graph, "Model:GS-610", "FAILED", "Component:скрытый узел", weight=9999)
    summary = (
        "Исходная деталь | Север | GS-610 | P-9070\n"
        "P-9070 снята; замена P-9070B.\n"
        "P-9070B снята; замена P-9070C.\n"
        "Отказы узлов, Север | насос: 4; контроллер: 1\n"
        "Всего отказов моделей, Север | GS-610: 5"
    )
    partial = "P-9070 и P-9070B сняты с производства."
    llm = FakeLLM(
        [
            json.dumps({"answer": partial, "score": 8}, ensure_ascii=False),
            "Исходная деталь из руководства GS-610 — P-9070; она снята с производства.",
        ]
    )

    answer = global_search(
        "Какие детали руководств Севера сняты?",
        KnowledgeBase(graph, [Community(7, ["Model:GS-610"], summary)], {}),
        llm,
    )

    assert not answer.refused
    assert len(llm.requests) == 2
    assert llm.requests[0]["json_schema"] == MAP_SCHEMA
    content = llm.requests[-1]["messages"][-1]["content"]
    partials, rows = content.split("Частичные ответы:\n", 1)[1].split("\nПодтверждения:\n", 1)
    useful = json.loads(partials)
    evidence = {int(line.split(". ", 1)[0]): line.split(". ", 1)[1] for line in rows.splitlines()}
    assert len(useful) == 1
    assert useful[0]["answer"] == partial
    assert "\n".join(evidence[number] for number in useful[0]["evidence_ids"]) == summary
    assert "P-9080" not in content
    assert "9999" not in content
    assert "скрытый узел" not in content


def test_global_search_keeps_supporting_summaries_only_for_valid_positive_answers():
    accepted = "Север: насос — 4 отказа."
    rejected = "Неподходящая сводка: P-9099 заменена на P-9099B."
    unreadable = "Сводка с нечитаемым ответом map: P-9098 заменена на P-9098B."
    communities = [Community(0, [], accepted), Community(1, [], rejected), Community(2, [], unreadable)]
    llm = FakeLLM(
        [
            '{"answer":"Насос: 4 отказа.","score":6}',
            '{"answer":"Данных для ответа нет.","score":0}',
            "ответ не в JSON",
            "В Севере у насоса 4 отказа.",
        ]
    )

    answer = global_search("Что ломается в Севере?", KnowledgeBase(nx.DiGraph(), communities, {}), llm)

    assert not answer.refused
    assert len(llm.requests) == 4
    assert all(request["json_schema"] == MAP_SCHEMA for request in llm.requests[:3])
    content = llm.requests[-1]["messages"][-1]["content"]
    partials, rows = content.split("Частичные ответы:\n", 1)[1].split("\nПодтверждения:\n", 1)
    useful = json.loads(partials)
    assert len(useful) == 1
    assert useful[0]["evidence_ids"] == [1]
    assert rows == f"1. {accepted}"
    assert rejected not in content
    assert unreadable not in content
    assert "P-9099" not in content and "P-9098" not in content


def test_global_search_cites_full_line_counts_outside_useful_community():
    graph = nx.DiGraph()
    connect(graph, "Model:AR-410", "FAILED", "Component:контроллер", weight=3, source="controller.md#0")
    connect(graph, "Model:AR-420", "FAILED", "Component:компрессор", weight=9, source="compressor.md#0")
    for model in ("AR-410", "AR-420"):
        connect(graph, f"Model:{model}", "BELONGS_TO", "ProductLine:Север", source="north.md#0")
    connect(graph, "Model:BZ-410", "FAILED", "Component:вентилятор", source="unrelated.md#0")
    community = Community(0, ["Component:контроллер"], "Север: контроллер=3, компрессор=9.")
    llm = FakeLLM(['{"answer":"Север: контроллер=3, компрессор=9.","score":8}', "В Севере чаще отказывает компрессор."])

    answer = global_search("Что чаще отказывает в Севере?", KnowledgeBase(graph, [community], {}), llm)

    assert answer.sources == ["compressor.md", "controller.md", "north.md"]
    assert len(llm.requests) == 2
    assert "компрессор=9" in llm.requests[-1]["messages"][-1]["content"]
    assert "unrelated" not in json.dumps(llm.requests, ensure_ascii=False)


def test_global_search_cites_external_replacement_pairs_and_original_parts():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B", source="first_replacement.md#0")
    connect(graph, "Part:P-9070B", "REPLACED_BY", "Part:P-9070C", source="second_replacement.md#0")
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070", source="manual_north.md#0")
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север", source="north.md#0")
    connect(graph, "Model:BZ-610", "HAS_PART", "Part:P-9070B", source="manual_south.md#0")
    connect(graph, "Model:BZ-610", "BELONGS_TO", "ProductLine:Юг", source="south.md#0")
    connect(graph, "Part:P-9080", "REPLACED_BY", "Part:P-9080B", source="unrelated_replacement.md#0")
    connect(graph, "Model:AR-610", "HAS_PART", "Part:P-9080", source="unrelated_manual.md#0")
    community = Community(
        0,
        ["Part:P-9070C"],
        ("Исходная деталь | Север | GS-610 | P-9070\nP-9070 снята; замена P-9070B.\nP-9070B снята; замена P-9070C."),
    )
    llm = FakeLLM(
        [
            '{"answer":"Исходная деталь моделей Севера P-9070 снята с производства.","score":8}',
            "В руководствах Севера указана снятая с производства P-9070.",
        ]
    )

    answer = global_search("Какие детали руководств Севера сняты?", KnowledgeBase(graph, [community], {}), llm)

    assert not answer.refused
    assert answer.sources == [
        "first_replacement.md",
        "manual_north.md",
        "manual_south.md",
        "north.md",
        "second_replacement.md",
        "south.md",
    ]
    assert len(llm.requests) == 2
    assert llm.requests[0]["json_schema"] == MAP_SCHEMA
    assert "unrelated" not in json.dumps(llm.requests, ensure_ascii=False)


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


def cited_response(answer, ids):
    return json.dumps({"answer": answer, "evidence_ids": ids}, ensure_ascii=False)


def test_local_citations_cover_all_returned_facts_with_fewer_duplicate_documents():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070", source="manual_gs.md#0")
    connect(graph, "Part:P-9070", "IS_A", "Component:насос", source="manual_gs.md#0")
    graph.edges["Part:P-9070", "Component:насос"]["sources"].append("unrelated_model.md#0")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B", source="bulletin.md#0")
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9080", source="lighting.md#0")
    kb = KnowledgeBase(graph, [], {"Component": ["насос"]})
    llm = FakeLLM([cited_response("Для GS-610 закажите P-9070B.", [1, 2, 3])])

    answer = local_search("Какая актуальная деталь насоса для GS-610?", kb, llm)

    assert not answer.refused
    assert answer.sources == ["bulletin.md", "manual_gs.md"]
    assert len(answer.facts) == 3
    assert not any("P-9080" in fact for fact in answer.facts)
    assert "Оборудование: GS-610" in llm.requests[0]["messages"][-1]["content"]
    assert llm.requests[0]["json_schema"] is not None


@pytest.mark.parametrize("ids", [[], [0], [-1], [3], [True], ["1"]])
def test_local_rejects_missing_and_invalid_citation_ids(ids):
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    llm = FakeLLM([cited_response("Нужна P-9070B.", ids)] * 2)

    answer = local_search("Чем заменить P-9070?", KnowledgeBase(graph, [], {}), llm)

    assert answer.refused and answer.text == "В базе знаний не найдено достаточно данных для ответа."
    assert len(llm.requests) == 2


def test_local_rejects_a_known_but_uncited_code_then_retries_with_the_same_facts():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070", source="manual.md#0")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B", source="bulletin.md#0")
    llm = FakeLLM(
        [
            cited_response("Нужна P-9070B.", [2]),
            cited_response("Нужна P-9070B.", [1, 2]),
        ]
    )

    answer = local_search("Что заказать для GS-610?", KnowledgeBase(graph, [], {}), llm)

    assert not answer.refused
    assert len(llm.requests) == 2
    assert answer.sources == ["bulletin.md", "manual.md"]
    assert len(answer.facts) == 2


@pytest.mark.parametrize(
    "claim",
    ["P-9070B есть на складе в Казани.", "Остаток: 8 штук.", "Цена P-9070B — 1900 рублей."],
)
def test_local_cannot_claim_live_erp_values_from_document_facts(claim):
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    llm = FakeLLM([cited_response(claim, [1])] * 2)

    answer = local_search("Чем заменить P-9070?", KnowledgeBase(graph, [], {}), llm)

    assert answer.refused
    assert claim not in answer.text
    assert len(llm.requests) == 2


def test_local_can_explain_that_live_price_is_unknown():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    text = "Актуальная деталь P-9070B; цена и наличие на складе неизвестны."
    llm = FakeLLM([cited_response(text, [1])])

    answer = local_search("Какая замена и цена P-9070?", KnowledgeBase(graph, [], {}), llm)

    assert not answer.refused and answer.text == text
    assert len(llm.requests) == 1


def test_global_cites_only_the_selected_replacement_chain_and_model():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "HAS_PART", "Part:P-9070", source="manual.md#0")
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север", source="manual.md#0")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B", source="bulletin.md#0")
    connect(graph, "Part:P-9070B", "SUPPLIED_BY", "Supplier:Поставщик", source="supplier.md#0")
    summary = (
        "Исходная деталь | Север | GS-610 | P-9070\nP-9070 снята; замена P-9070B.\nP-9070B -SUPPLIED_BY-> Поставщик"
    )
    llm = FakeLLM(
        [
            '{"answer":"Исходная P-9070 снята.","score":9}',
            cited_response("В руководстве GS-610 снята P-9070.", [1, 2]),
        ]
    )

    answer = global_search(
        "Какие исходные детали Севера сняты?", KnowledgeBase(graph, [Community(0, list(graph), summary)], {}), llm
    )

    assert not answer.refused
    assert answer.sources == ["bulletin.md", "manual.md"]
    assert answer.facts == summary.splitlines()[:2]
    assert len(llm.requests) == 2


def test_global_comparison_retains_every_report_in_the_cited_count_table():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север", source="manual.md#0")
    connect(graph, "Model:GS-620", "BELONGS_TO", "ProductLine:Север", source="manual.md#1")
    connect(graph, "Model:GS-610", "FAILED", "Component:насос", weight=2, source="report1.md#0")
    graph.edges["Model:GS-610", "Component:насос"]["sources"].append("report2.md#0")
    connect(graph, "Model:GS-620", "FAILED", "Component:датчик", weight=1, source="report3.md#0")
    connect(graph, "Part:P-9070", "SUPPLIED_BY", "Supplier:Поставщик", source="unrelated.md#0")
    summary = "Отказы узлов, Север | датчик: 1; насос: 2\nВсего отказов моделей, Север | GS-610: 2; GS-620: 1"
    llm = FakeLLM(
        [
            '{"answer":"Насос: 2, датчик: 1.","score":9}',
            cited_response("В Севере чаще отказывает насос — 2 отказа.", [1]),
        ]
    )

    answer = global_search(
        "Какой узел чаще отказывает в Севере?", KnowledgeBase(graph, [Community(0, list(graph), summary)], {}), llm
    )

    assert not answer.refused
    assert answer.sources == ["manual.md", "report1.md", "report2.md", "report3.md"]
    assert len(answer.facts) == 1


@pytest.mark.parametrize(
    "final",
    [
        cited_response("Нужна P-9070B.", [2]),
        cited_response("Нужна P-9080.", [1]),
        cited_response("Нужна P-9070B.", []),
    ],
)
def test_global_rejects_invalid_citations_and_codes_outside_cited_evidence(final):
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9080", "IS_A", "Component:насос")
    community = Community(0, list(graph), "P-9070 снята; замена P-9070B.")
    llm = FakeLLM(['{"answer":"Замена P-9070B.","score":9}', final])

    answer = global_search("Чем заменить P-9070?", KnowledgeBase(graph, [community], {}), llm)

    assert answer.refused and not answer.sources
    assert len(llm.requests) == 2


def test_global_rejects_a_cited_relation_that_disagrees_with_the_graph():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "SUPPLIED_BY", "Supplier:Старый")
    connect(graph, "Part:P-9070B", "SUPPLIED_BY", "Supplier:Новый")
    community = Community(0, list(graph), "P-9070B -SUPPLIED_BY-> Старый")
    llm = FakeLLM(
        [
            '{"answer":"Старый поставщик.","score":9}',
            cited_response("Поставщик P-9070B — Старый.", [1]),
        ]
    )

    answer = global_search("Кто поставляет P-9070B?", KnowledgeBase(graph, [community], {}), llm)

    assert answer.refused and not answer.sources
    assert len(llm.requests) == 2


def test_global_rejects_a_count_table_that_does_not_match_its_scope():
    graph = nx.DiGraph()
    connect(graph, "Model:GS-610", "BELONGS_TO", "ProductLine:Север")
    connect(graph, "Model:GS-610", "FAILED", "Component:насос", weight=4)
    summary = "Отказы узлов, Север | насос: 40"
    llm = FakeLLM(['{"answer":"Насос: 40.","score":9}', cited_response("Насос — 40 отказов.", [1])])

    answer = global_search(
        "Что чаще ломается в Севере?", KnowledgeBase(graph, [Community(0, list(graph), summary)], {}), llm
    )

    assert answer.refused and not answer.sources


def test_global_rejects_stock_claims_without_an_erp_tool_even_for_a_known_part():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    summary = "P-9070 снята; замена P-9070B."
    llm = FakeLLM(
        [
            '{"answer":"Актуальная P-9070B.","score":9}',
            cited_response("P-9070B есть на складе в Новосибирске.", [1]),
        ]
    )

    answer = global_search(
        "Есть ли замена P-9070 в Новосибирске?", KnowledgeBase(graph, [Community(0, list(graph), summary)], {}), llm
    )

    assert answer.refused and not answer.sources
    assert len(llm.requests) == 2


def test_global_sends_shared_summary_rows_only_once_and_still_maps_every_community():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    row = "P-9070 снята; замена P-9070B."
    communities = [Community(i, list(graph), row) for i in range(4)]
    partial = '{"answer":"Замена P-9070B.","score":9}'
    llm = FakeLLM([partial] * 4 + [cited_response("Замена P-9070B.", [1])])

    answer = global_search("Чем заменить P-9070?", KnowledgeBase(graph, communities, {}), llm)

    assert not answer.refused and len(llm.requests) == 5
    assert all(request["json_schema"] == MAP_SCHEMA for request in llm.requests[:4])
    content = llm.requests[-1]["messages"][-1]["content"]
    partials, rows = content.split("Частичные ответы:\n", 1)[1].split("\nПодтверждения:\n", 1)
    answers = json.loads(partials)
    assert len(answers) == 4
    assert rows == f"1. {row}"
    assert all(item["evidence_ids"] == [1] for item in answers)


def test_global_cannot_copy_the_supplier_of_an_old_part_to_the_current_part():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "SUPPLIED_BY", "Supplier:Старый", source="old.md#0")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B", source="replacement.md#0")
    connect(graph, "Part:P-9070B", "SUPPLIED_BY", "Supplier:Новый", source="new.md#0")
    summary = "P-9070 -SUPPLIED_BY-> Старый\nP-9070 снята; замена P-9070B.\nP-9070B -SUPPLIED_BY-> Новый"
    llm = FakeLLM(
        [
            '{"answer":"Поставщик актуальной детали P-9070B — Старый.","score":8}',
            cited_response("Поставщик актуальной детали P-9070B — Старый.", [1, 2, 3]),
        ]
    )

    answer = global_search(
        "Кто текущий поставщик замены P-9070?", KnowledgeBase(graph, [Community(0, list(graph), summary)], {}), llm
    )

    assert answer.refused and not answer.sources
    assert len(llm.requests) == 2


def test_global_accepts_the_explicit_supplier_of_the_final_replacement():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "SUPPLIED_BY", "Supplier:Старый", source="old.md#0")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B", source="replacement.md#0")
    connect(graph, "Part:P-9070B", "SUPPLIED_BY", "Supplier:Новый", source="new.md#0")
    summary = "P-9070 -SUPPLIED_BY-> Старый\nP-9070 снята; замена P-9070B.\nP-9070B -SUPPLIED_BY-> Новый"
    llm = FakeLLM(
        [
            '{"answer":"Поставщик актуальной детали P-9070B — Новый.","score":8}',
            cited_response("Поставщик актуальной детали P-9070B — Новый.", [2, 3]),
        ]
    )

    answer = global_search(
        "Кто текущий поставщик замены P-9070?", KnowledgeBase(graph, [Community(0, list(graph), summary)], {}), llm
    )

    assert not answer.refused and answer.sources == ["new.md", "replacement.md"]
    assert len(llm.requests) == 2


def test_local_can_confirm_an_unchanged_supplier_after_a_part_replacement():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "SUPPLIED_BY", "Supplier:Поставщик")
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    connect(graph, "Part:P-9070B", "SUPPLIED_BY", "Supplier:Поставщик")
    llm = FakeLLM([cited_response("Текущий поставщик P-9070B — Поставщик.", [1, 2, 3])])

    answer = local_search("Кто текущий поставщик замены P-9070?", KnowledgeBase(graph, [], {}), llm)

    assert not answer.refused
    assert len(llm.requests) == 1


def test_local_cannot_say_an_order_was_created_without_an_order_tool():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    llm = FakeLLM([cited_response("Заказ создан на P-9070B.", [1])] * 2)

    answer = local_search("Закажи замену P-9070", KnowledgeBase(graph, [], {}), llm)

    assert answer.refused and "Заказ создан" not in answer.text
    assert len(llm.requests) == 2


def test_plain_text_client_uses_the_same_validation_and_conservative_sources():
    from assistant.infra.llm import LLMResponse

    class PlainClient:
        def __init__(self, responses):
            self.responses = iter(responses)

        def chat(self, messages, **kwargs):
            return LLMResponse(next(self.responses))

    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B", source="bulletin.md#0")
    graph.edges["Part:P-9070", "Part:P-9070B"]["sources"].append("manual.md#0")
    kb = KnowledgeBase(graph, [Community(0, list(graph), "P-9070 снята; замена P-9070B.")], {})

    local = local_search("Чем заменить P-9070?", kb, PlainClient(["Нужна P-9070B."]))
    assert not local.refused and local.sources == ["bulletin.md", "manual.md"]
    partial = '{"answer":"Замена P-9070B.","score":9}'
    global_answer = global_search("Чем заменить P-9070?", kb, PlainClient([partial, "P-9070B есть на складе."]))
    assert global_answer.refused and not global_answer.sources
    unsupported = global_search("Чем заменить P-9070?", kb, PlainClient([partial, "Нужна P-9999."]))
    assert unsupported.refused and not unsupported.sources


@pytest.mark.parametrize(
    "claim",
    [
        "На складе в Казани есть P-9070B.",
        "P-9070B в наличии.",
        "P-9070B нет на складе в Казани.",
        "Цена не указана, но P-9070B есть на складе в Казани.",
        "Цена: 2630",
        "Стоимость составляет 2630",
        "Данных о цене нет; заказ создан на P-9070B.",
    ],
)
@pytest.mark.parametrize("mode", ["local", "global"])
def test_search_cannot_confirm_any_stock_status_or_unitless_price_without_erp(claim, mode):
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    community = Community(0, list(graph), "P-9070 снята; замена P-9070B.")
    kb = KnowledgeBase(graph, [community], {})
    final = cited_response(claim, [1])
    if mode == "local":
        answer = local_search("Чем заменить P-9070?", kb, FakeLLM([final, final]))
    else:
        answer = global_search("Чем заменить P-9070?", kb, FakeLLM(['{"answer":"Замена P-9070B.","score":9}', final]))
    assert answer.refused
    assert claim not in answer.text


def test_plain_global_does_not_treat_a_graph_disproved_summary_as_code_evidence():
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B", source="bulletin.md#0")
    community = Community(0, list(graph), "P-9070 снята; замена P-9080.")
    llm = FakeLLM(['{"answer":"Замена P-9080.","score":9}', "Нужна P-9080."])

    answer = global_search("Чем заменить P-9070?", KnowledgeBase(graph, [community], {}), llm)

    assert answer.refused and not answer.sources
    assert len(llm.requests) == 2


@pytest.mark.parametrize("mode", ["local", "global"])
def test_search_can_say_that_stock_information_is_missing(mode):
    graph = nx.DiGraph()
    connect(graph, "Part:P-9070", "REPLACED_BY", "Part:P-9070B")
    community = Community(0, list(graph), "P-9070 снята; замена P-9070B.")
    kb = KnowledgeBase(graph, [community], {})
    text = "Актуальная замена P-9070B; сведений о наличии на складе в Казани нет."
    final = cited_response(text, [1])
    if mode == "local":
        answer = local_search("Какая замена P-9070 и есть ли она в Казани?", kb, FakeLLM([final]))
    else:
        answer = global_search(
            "Какая замена P-9070 и есть ли она в Казани?",
            kb,
            FakeLLM(['{"answer":"Замена P-9070B.","score":9}', final]),
        )
    assert not answer.refused and answer.text == text
