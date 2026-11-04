import json

import pytest

from assistant.graphrag.extraction import extract_chunk
from assistant.graphrag.graph_builder import build_graph
from assistant.infra.schemas import Chunk, Entity, ExtractionResult, Relation
from assistant.infra.testing import FakeLLM


def test_repeated_spellings_produce_one_entity_and_one_relation():
    chunk = Chunk("a.md#0", "a.md", "Деталь Р1042 заменена на p 1042b.")
    relation = {
        "subject_type": "Part",
        "subject": "P-1042",
        "type": "REPLACED_BY",
        "object_type": "Part",
        "object": "P-1042B",
    }
    response = json.dumps(
        {
            "entities": [{"type": "Part", "name": name} for name in ("Р1042", "P-1042", "p 1042b")],
            "relations": [relation, {**relation, "subject": "р 1042", "object": "р1042в"}],
        }
    )
    result = extract_chunk(chunk, FakeLLM([response]))
    assert result.entities == [Entity("Part", "P-1042"), Entity("Part", "P-1042B")]
    assert len(result.relations) == 1
    assert result.dropped == 0


@pytest.mark.parametrize(
    "invalid",
    [
        {"entities": [None], "relations": []},
        {"entities": [{"type": "Part", "name": 1042}], "relations": []},
        {"entities": [{"type": "Unknown", "name": "P-1042"}], "relations": []},
        {
            "entities": [],
            "relations": [
                {
                    "subject_type": "Part",
                    "subject": "P-1042",
                    "type": "LIKES",
                    "object_type": "Part",
                    "object": "P-1042B",
                }
            ],
        },
    ],
)
def test_invalid_schema_gets_one_retry(invalid):
    chunk = Chunk("a.md#0", "a.md", "Деталь P-1042")
    valid = '{"entities":[{"type":"Part","name":"P-1042"}],"relations":[]}'
    llm = FakeLLM([json.dumps(invalid), valid])
    assert extract_chunk(chunk, llm).entities == [Entity("Part", "P-1042")]
    assert len(llm.requests) == 2


def test_dropped_counts_entities_and_relations_separately():
    chunk = Chunk("a.md#0", "a.md", "Деталь P-1042")
    response = json.dumps(
        {
            "entities": [
                {"type": "Part", "name": "P-1042"},
                {"type": "Model", "name": "P-1042"},
                {"type": "Part", "name": "P-7777"},
            ],
            "relations": [
                {
                    "subject_type": "Part",
                    "subject": "P-1042",
                    "type": "REPLACED_BY",
                    "object_type": "Part",
                    "object": "P-7777",
                }
            ],
        }
    )
    result = extract_chunk(chunk, FakeLLM([response]))
    assert result.entities == [Entity("Part", "P-1042")]
    assert result.relations == []
    assert result.dropped == 3


def test_invalid_relation_direction_is_filtered():
    chunk = Chunk("a.md#0", "a.md", "В модели AR-210 установлена деталь P-1042.")
    response = json.dumps(
        {
            "entities": [],
            "relations": [
                {
                    "subject_type": "Part",
                    "subject": "P-1042",
                    "type": "HAS_PART",
                    "object_type": "Model",
                    "object": "AR-210",
                }
            ],
        }
    )
    result = extract_chunk(chunk, FakeLLM([response]))
    assert result.relations == []
    assert result.dropped == 1


def test_repeated_chunk_does_not_increase_edge_weight():
    old, new = Entity("Part", "P-1042"), Entity("Part", "P-1042B")
    relation = Relation(old, "REPLACED_BY", new)
    results = [ExtractionResult(f"a.md#{i}", "a.md", [old, new], [relation, relation]) for i in range(2)]
    graph = build_graph([results[1], results[0], results[0]])
    edge = graph.edges[old.node_id, new.node_id]
    assert edge["weight"] == 2
    assert edge["sources"] == ["a.md#0", "a.md#1"]
    assert graph.nodes[old.node_id]["sources"] == edge["sources"]


def test_conflicts_are_counted_once_regardless_of_input_order():
    first, second, third = [Entity("Part", name) for name in ("P-2054", "P-2054B", "P-2054C")]
    forward = Relation(first, "REPLACED_BY", second)
    backward = Relation(second, "REPLACED_BY", first)
    onward = Relation(second, "REPLACED_BY", third)
    back = Relation(third, "REPLACED_BY", second)
    results = [
        ExtractionResult("a.md#0", "a.md", [first, second, third], [forward, onward]),
        ExtractionResult("b.md#0", "b.md", [first, second, third], [forward, back]),
        ExtractionResult("c.md#0", "c.md", [first, second], [backward]),
    ]
    for ordered in (results, results[::-1]):
        graph = build_graph(ordered)
        assert set(graph.edges) == {(first.node_id, second.node_id)}
        assert set(graph.nodes) == {first.node_id, second.node_id, third.node_id}
        assert graph.graph["conflicts"] == 2


def test_empty_graph_and_entities_without_relations_are_preserved():
    empty = build_graph([])
    assert len(empty) == 0
    assert empty.graph["conflicts"] == 0
    part = Entity("Part", "P-1042")
    graph = build_graph([ExtractionResult("a.md#0", "a.md", [part])])
    assert graph.nodes[part.node_id] == {"type": "Part", "name": "P-1042", "sources": ["a.md#0"]}
    assert graph.number_of_edges() == 0
