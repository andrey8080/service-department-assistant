import copy
import json

import networkx as nx
import pytest

from assistant.graphrag.communities import summarize_community, summary_records
from assistant.infra.schemas import Community
from assistant.infra.testing import FakeLLM


def connect(graph, start, relation, end, *, weight=1):
    for node in (start, end):
        if node not in graph:
            kind, name = node.split(":", 1)
            graph.add_node(node, type=kind, name=name, sources=["manual.md#0"])
    graph.add_edge(start, end, type=relation, weight=weight, sources=["manual.md#0"])


class ArrangeSummary(FakeLLM):
    def __init__(self, transform=None):
        super().__init__([])
        self.transform = transform or (lambda ids: ids)

    def chat(self, messages, **kwargs):
        records = json.loads(messages[-1]["content"])["compact_records"]
        self._script.append(json.dumps({"record_ids": self.transform([row["id"] for row in records])}))
        return super().chat(messages, **kwargs)


def summary(graph, nodes=None, llm=None):
    llm = llm or ArrangeSummary()
    text = summarize_community(Community(0, sorted(graph) if nodes is None else nodes), graph, llm)
    assert len(llm.requests) == 1
    assert 0 < len(text.split()) <= 150
    return text, summary_records(text, graph), llm


@pytest.fixture
def catalog():
    graph = nx.DiGraph()
    for model, part in (("ZX-710", "P-7100"), ("ZX-720", "P-7200")):
        connect(graph, f"Model:{model}", "BELONGS_TO", "ProductLine:Север")
        connect(graph, f"Model:{model}", "HAS_PART", f"Part:{part}")
        connect(graph, f"Part:{part}", "IS_A", "Component:насос")
        connect(graph, f"Part:{part}", "SUPPLIED_BY", "Supplier:Старый")
    connect(graph, "Part:P-7100", "REPLACED_BY", "Part:P-7100B")
    connect(graph, "Part:P-7100B", "REPLACED_BY", "Part:P-7100C")
    connect(graph, "Part:P-7100C", "SUPPLIED_BY", "Supplier:Новый")
    connect(graph, "Model:ZX-710", "FAILED", "Component:насос", weight=7)
    connect(graph, "Model:ZX-710", "FAILED", "Component:датчик", weight=2)
    connect(graph, "Model:ZX-720", "FAILED", "Component:насос", weight=1)
    return graph


def test_compact_summary_retains_joint_failure_counts_and_completeness(catalog):
    text, records, _ = summary(catalog, llm=ArrangeSummary(lambda ids: list(reversed(ids))))

    counts = {(r["model"], r["component"]): r["count"] for r in records if r["kind"] == "failure"}
    assert counts == {("ZX-710", "насос"): 7, ("ZX-710", "датчик"): 2, ("ZX-720", "насос"): 1}
    coverage = {r["model"]: r for r in records if r["kind"] == "coverage"}
    assert coverage["ZX-710"]["failure_count"] == 2
    assert coverage["ZX-710"]["failure_total"] == 9
    assert coverage["ZX-720"]["failure_total"] == 1
    assert all(r["parts_count"] == 1 for r in coverage.values())
    assert coverage["ZX-710"]["replaced_parts_count"] == 1
    assert coverage["ZX-720"]["replaced_parts_count"] == 0
    assert "7" in text and "2" in text


def test_unchanged_inventory_and_current_suppliers_survive_alongside_replacements(catalog):
    _, records, _ = summary(catalog)

    inventory = {(r["model"], r["component"], r["part"]) for r in records if r["kind"] == "inventory"}
    assert inventory == {("ZX-710", "насос", "P-7100"), ("ZX-720", "насос", "P-7200")}
    assert all(r["complete"] for r in records if r["kind"] == "inventory")
    versions = {r["part"]: r for r in records if r["kind"] == "version"}
    assert versions["P-7100"]["path"] == ["P-7100", "P-7100B", "P-7100C"]
    assert versions["P-7100B"]["path"] == ["P-7100B", "P-7100C"]
    assert versions["P-7200"]["path"] == ["P-7200"]
    assert versions["P-7200"]["current"] == "P-7200"
    suppliers = {(r["part"], r["supplier"]) for r in records if r["kind"] == "supplier"}
    assert ("P-7100C", "Новый") in suppliers
    assert ("P-7100", "Старый") in suppliers
    assert ("P-7200", "Старый") in suppliers


def test_community_boundary_keeps_current_supplier_and_the_whole_replacement_chain(catalog):
    text, records, _ = summary(catalog, ["Part:P-7100B"])

    assert "P-7100>P-7100B>P-7100C" in text
    assert any(r["kind"] == "supplier" and r["part"] == "P-7100C" and r["supplier"] == "Новый"
               for r in records)
    assert not any(r["kind"] == "coverage" for r in records)


def test_union_of_separate_communities_recovers_inventory_and_full_failure_matrix(catalog):
    records = []
    for nodes in (["Model:ZX-710", "Model:ZX-720"], ["Part:P-7100"], ["Part:P-7200"], ["Component:датчик"]):
        _, selected, _ = summary(catalog, nodes)
        records.extend(selected)

    assert {(r["model"], r["component"], r["part"]) for r in records if r["kind"] == "inventory"} == {
        ("ZX-710", "насос", "P-7100"), ("ZX-720", "насос", "P-7200")
    }
    assert {(r["model"], r["component"], r["count"]) for r in records if r["kind"] == "failure"} == {
        ("ZX-710", "насос", 7), ("ZX-710", "датчик", 2), ("ZX-720", "насос", 1)
    }
    line = next(r for r in records if r["kind"] == "line_coverage")
    assert line["models"] == ["ZX-710", "ZX-720"]


def test_inventory_component_group_is_complete_even_when_one_original_part_is_outside_community(catalog):
    connect(catalog, "Model:ZX-710", "HAS_PART", "Part:P-7101")
    connect(catalog, "Part:P-7101", "IS_A", "Component:насос")
    _, records, _ = summary(catalog, ["Part:P-7100"])

    assert {r["part"] for r in records if r["kind"] == "inventory"} == {"P-7100", "P-7101"}
    assert summary_records("Комплектация | Север | ZX-710 | насос=P-7100", catalog) == []


def test_replaced_parts_count_uses_unique_original_parts_and_not_chain_length(catalog):
    connect(catalog, "Model:ZX-710", "HAS_PART", "Part:P-7100B")
    connect(catalog, "Model:ZX-710", "HAS_PART", "Part:P-7100C")
    _, records, _ = summary(catalog)

    coverage = next(row for row in records if row["kind"] == "coverage" and row["model"] == "ZX-710")
    assert coverage["parts_count"] == 3
    assert coverage["replaced_parts_count"] == 2
    assert next(row for row in records if row["kind"] == "coverage" and row["model"] == "ZX-720")[
        "replaced_parts_count"
    ] == 0


def test_legacy_coverage_without_replaced_count_is_not_assumed_to_mean_zero(catalog):
    rows = summary_records("Полнота | Север | ZX-710 | деталей=1; отказов=2; случаев=9", catalog)

    assert len(rows) == 1
    assert rows[0]["parts_count"] == 1
    assert "replaced_parts_count" not in rows[0]


@pytest.mark.parametrize("count", [0, 2, -1])
def test_coverage_rejects_incorrect_replaced_count(catalog, count):
    line = f"Полнота | Север | ZX-710 | деталей=1; отказов=2; случаев=9; снятых={count}"
    assert summary_records(line, catalog) == []


@pytest.mark.parametrize("line", [
    "Отказы | Север | ZX-710 | насос=9",
    "Отказы | Юг | ZX-710 | насос=7",
    "Отказы | Север | ZX-710 | насос=7; насос=1",
    "Полнота | Север | ZX-710 | деталей=1; отказов=1; случаев=9",
    "Модели | Север | ZX-710",
    "Комплектация | Север | ZX-720 | насос=P-7100",
    "Поставщики | Старый | P-7100C",
    "Версии | P-7100",
    "Версии | P-7100>P-7100C",
    "Версии | P-7100>P-7100B",
])
def test_compact_parser_rejects_false_claims_even_when_entities_exist(catalog, line):
    assert summary_records(line, catalog) == []


def test_bad_record_does_not_erase_an_independent_valid_record(catalog):
    rows = summary_records("Отказы | Север | ZX-710 | насос=90\nПоставщики | Новый | P-7100C", catalog)
    assert len(rows) == 1
    assert rows[0]["kind"] == "supplier"


@pytest.mark.parametrize("transform", [
    lambda ids: ids[:-1],
    lambda ids: [*ids, ids[0]],
    lambda ids: [*ids[:-1], 999],
    lambda ids: [True, *ids[1:]],
])
def test_summary_rejects_model_selection_that_loses_or_changes_a_required_group(catalog, transform):
    text, records, _ = summary(catalog, llm=ArrangeSummary(transform))
    assert text == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert records == []


def test_branching_versions_remain_distinct_and_a_cycle_has_no_terminal_version():
    graph = nx.DiGraph()
    connect(graph, "Part:P-7300", "REPLACED_BY", "Part:P-7300B")
    connect(graph, "Part:P-7300", "REPLACED_BY", "Part:P-7300C")
    _, records, _ = summary(graph)

    assert {r["current"] for r in records if r["kind"] == "version" and r["part"] == "P-7300"} == {
        "P-7300B", "P-7300C"
    }
    connect(graph, "Part:P-7300B", "REPLACED_BY", "Part:P-7300")
    connect(graph, "Part:P-7300C", "REPLACED_BY", "Part:P-7300")
    _, records, _ = summary(graph)
    assert not any(r["kind"] == "version" for r in records)
    assert len([r for r in records if r["kind"] == "replacement"]) == 4


def test_over_budget_summary_fails_without_silently_dropping_inventory():
    graph = nx.DiGraph()
    for number in range(30):
        part = f"Part:P-{8000 + number}"
        connect(graph, "Model:ZX-810", "HAS_PART", part)
        connect(graph, part, "IS_A", f"Component:отдельный узел номер {number}")
    text, records, _ = summary(graph)

    assert text == "Сводка не сформирована: ответ модели не прошёл проверку."
    assert records == []


def test_summary_does_not_mutate_graph_and_preserves_error_procedure_facts(catalog):
    connect(catalog, "ErrorCode:E-51", "INDICATES", "Component:насос")
    connect(catalog, "ErrorCode:E-51", "FIXED_BY", "Procedure:SP-951")
    original = copy.deepcopy(catalog)
    text, _, llm = summary(catalog)

    assert "E-51 -INDICATES-> насос" in text
    assert "E-51 -FIXED_BY-> SP-951" in text
    assert nx.utils.graphs_equal(catalog, original)
    assert llm.requests[0]["json_schema"]["properties"]["record_ids"]["uniqueItems"]


def test_isolated_node_still_uses_one_generation_without_inventing_facts():
    graph = nx.DiGraph()
    graph.add_node("Procedure:SP-999", type="Procedure", name="SP-999", sources=[])
    text, records, _ = summary(graph)
    assert text == "В сообществе нет подтверждённых фактов для сводки."
    assert records == []
