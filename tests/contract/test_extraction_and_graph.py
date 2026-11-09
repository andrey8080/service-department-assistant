"""Контракт FR-1 и FR-2: извлечение из чанка и сборка графа."""

import json

from assistant.graphrag.extraction import extract_chunk
from assistant.graphrag.graph_builder import build_graph
from assistant.infra.schemas import Chunk, Entity, ExtractionResult, Relation
from assistant.infra.testing import FakeLLM

BULLETIN = Chunk(
    "bulletin_SB-2025-03.md#0",
    "bulletin_SB-2025-03.md",
    "# Сервисный бюллетень SB-2025-03\n\nДеталь P-1042 (пусковое реле) снята с производства. "
    "Вместо неё устанавливается деталь P-1042B.\n\nПоставщик детали P-1042B – «ХолодКомплект». "
    "Замена выполняется по регламенту SP-104.",
)
REPORT = Chunk(
    "report_R-2025-003.md#0",
    "report_R-2025-003.md",
    "# Отчёт о выезде R-2025-003\n\nОборудование: витрина ar 210. На панели горел код Е-17.\n\n"
    "Диагностика показала отказ узла: пусковое реле.",
)


def reply(entities=(), relations=()) -> str:
    return json.dumps(
        {
            "entities": [{"type": t, "name": n} for t, n in entities],
            "relations": [
                {"subject_type": st, "subject": s, "type": r, "object_type": ot, "object": o}
                for st, s, r, ot, o in relations
            ],
        },
        ensure_ascii=False,
    )


GOOD = reply(
    entities=[
        ("Part", "P-1042"),
        ("Part", "p-1042b"),
        ("Component", "Пусковое реле"),
        ("Supplier", "ООО «ХолодКомплект»"),
    ],
    relations=[
        ("Part", "P-1042", "REPLACED_BY", "Part", "P-1042B"),
        ("Part", "P-1042", "IS_A", "Component", "пусковое реле"),
        ("Part", "P-1042B", "SUPPLIED_BY", "Supplier", "ХолодКомплект"),
    ],
)


def triples(result: ExtractionResult) -> set:
    return {(r.subject.node_id, r.type, r.object.node_id) for r in result.relations}


def test_valid_answer_becomes_canonical_entities_and_relations():
    llm = FakeLLM([GOOD])
    result = extract_chunk(BULLETIN, llm)
    assert (result.chunk_id, result.doc_id) == (BULLETIN.chunk_id, BULLETIN.doc_id)
    assert {e.node_id for e in result.entities} >= {
        "Part:P-1042",
        "Part:P-1042B",
        "Component:пусковое реле",
        "Supplier:ХолодКомплект",
    }
    assert triples(result) == {
        ("Part:P-1042", "REPLACED_BY", "Part:P-1042B"),
        ("Part:P-1042", "IS_A", "Component:пусковое реле"),
        ("Part:P-1042B", "SUPPLIED_BY", "Supplier:ХолодКомплект"),
    }
    assert result.dropped == 0


def test_model_is_asked_once_with_the_extraction_schema_and_the_chunk_text():
    llm = FakeLLM([GOOD])
    extract_chunk(BULLETIN, llm)
    assert len(llm.requests) == 1 and llm.requests[0]["json_schema"] is not None
    assert "P-1042B" in " ".join(str(m["content"]) for m in llm.requests[0]["messages"])


def test_code_that_is_not_in_the_chunk_is_dropped_with_its_relations():
    invented = reply(
        entities=[("Part", "P-1042"), ("Procedure", "SP-1-04"), ("Part", "P-7777")],
        relations=[
            ("Part", "P-1042", "REPLACED_BY", "Part", "P-7777"),
            ("Part", "P-1042", "IS_A", "Component", "пусковое реле"),
        ],
    )
    result = extract_chunk(BULLETIN, FakeLLM([invented]))
    assert "Part:P-7777" not in {e.node_id for e in result.entities}
    assert triples(result) == {("Part:P-1042", "IS_A", "Component:пусковое реле")}
    assert result.dropped >= 3  # два выдуманных кода и связь с выдуманным концом


def test_name_outside_the_vocabulary_is_dropped():
    answer = reply(
        entities=[("Component", "конденсатор"), ("Component", "пусковое реле")],
        relations=[("Part", "P-1042", "IS_A", "Component", "конденсатор")],
    )
    result = extract_chunk(BULLETIN, FakeLLM([answer]))
    assert "Component:конденсатор" not in {e.node_id for e in result.entities}
    assert triples(result) == set() and result.dropped >= 2


def test_sloppy_codes_in_the_chunk_still_confirm_canonical_entities():
    answer = reply(
        entities=[("Model", "AR-210"), ("ErrorCode", "E-17")],
        relations=[("Model", "AR-210", "FAILED", "Component", "пусковое реле")],
    )
    result = extract_chunk(REPORT, FakeLLM([answer]))
    assert triples(result) == {("Model:AR-210", "FAILED", "Component:пусковое реле")}
    assert {"Model:AR-210", "ErrorCode:E-17"} <= {e.node_id for e in result.entities}


def test_broken_answer_is_retried_once():
    llm = FakeLLM(["это не JSON", GOOD])
    result = extract_chunk(BULLETIN, llm)
    assert len(llm.requests) == 2 and len(result.relations) == 3


def test_two_broken_answers_give_an_empty_result_without_an_exception():
    for broken in (
        "это не JSON",
        "[1, 2, 3]",
        '{"entities": "нет", "relations": null}',
        '{"entities": [{"type": "Part"}]}',
    ):
        llm = FakeLLM([broken, broken])
        result = extract_chunk(BULLETIN, llm)
        assert len(llm.requests) == 2, broken
        assert (result.chunk_id, result.entities, result.relations) == (BULLETIN.chunk_id, [], []), broken


OLD, NEW = Entity("Part", "P-1042"), Entity("Part", "P-1042B")
RELAY, MODEL = Entity("Component", "пусковое реле"), Entity("Model", "AR-210")


def result_of(chunk_id: str, relations: list[Relation]) -> ExtractionResult:
    entities = list({end.node_id: end for r in relations for end in (r.subject, r.object)}.values())
    return ExtractionResult(chunk_id, chunk_id.split("#")[0], entities, relations)


def test_same_entity_from_different_chunks_is_one_node_with_all_sources():
    graph = build_graph(
        [
            result_of("manual_AR-210.md#0", [Relation(MODEL, "HAS_PART", OLD)]),
            result_of("bulletin_SB-2025-03.md#0", [Relation(OLD, "REPLACED_BY", NEW)]),
        ]
    )
    assert set(graph.nodes) == {"Model:AR-210", "Part:P-1042", "Part:P-1042B"}
    node = graph.nodes["Part:P-1042"]
    assert (node["type"], node["name"]) == ("Part", "P-1042")
    assert sorted(node["sources"]) == ["bulletin_SB-2025-03.md#0", "manual_AR-210.md#0"]


def test_edge_weight_counts_distinct_chunks():
    relation = Relation(OLD, "IS_A", RELAY)
    graph = build_graph(
        [result_of("a.md#0", [relation, relation]), result_of("b.md#0", [relation]), result_of("c.md#0", [relation])]
    )
    edge = graph.edges["Part:P-1042", "Component:пусковое реле"]
    assert edge["type"] == "IS_A" and edge["weight"] == 3 and sorted(edge["sources"]) == ["a.md#0", "b.md#0", "c.md#0"]


def test_relation_that_breaks_the_schema_is_dropped():
    graph = build_graph([result_of("a.md#0", [Relation(MODEL, "REPLACED_BY", NEW), Relation(OLD, "LIKES", NEW)])])
    assert graph.number_of_edges() == 0


def test_opposite_replacements_keep_the_heavier_one():
    forward, backward = Relation(OLD, "REPLACED_BY", NEW), Relation(NEW, "REPLACED_BY", OLD)
    graph = build_graph(
        [result_of("a.md#0", [forward]), result_of("b.md#0", [forward]), result_of("c.md#0", [backward])]
    )
    assert graph.has_edge("Part:P-1042", "Part:P-1042B") and not graph.has_edge("Part:P-1042B", "Part:P-1042")
    assert graph.graph["conflicts"] == 1


def test_opposite_replacements_of_equal_weight_are_both_dropped():
    graph = build_graph(
        [
            result_of("a.md#0", [Relation(OLD, "REPLACED_BY", NEW)]),
            result_of("b.md#0", [Relation(NEW, "REPLACED_BY", OLD)]),
        ]
    )
    assert graph.number_of_edges() == 0 and graph.graph["conflicts"] == 1


def test_graph_without_conflicts_reports_zero():
    assert build_graph([result_of("a.md#0", [Relation(OLD, "REPLACED_BY", NEW)])]).graph["conflicts"] == 0
