import json

import pytest

from assistant.graphrag.extraction import extract_chunk
from assistant.infra.schemas import Chunk, Entity
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
