import json

import networkx as nx

from assistant.graphrag.communities import detect_communities, summarize_community
from assistant.infra.schemas import Community
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
