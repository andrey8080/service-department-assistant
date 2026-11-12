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
    result = extract_chunk(chunk, FakeLLM([response, response]))
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
    llm = FakeLLM([response, response])
    result = extract_chunk(chunk, llm)
    assert result.relations == []
    assert result.dropped == 1
    assert len(llm.requests) == 2


ASSEMBLY = Chunk("manual.md#0", "manual.md", "# Витрина AR-210\n## Состав узлов\nПусковое реле — деталь P-1042.")


def assembly_response(model_type="Model", include_part=True):
    relations = [
        {
            "subject_type": "Part",
            "subject": "P-1042",
            "type": "IS_A",
            "object_type": "Component",
            "object": "пусковое реле",
        }
    ]
    if include_part:
        relations.append(
            {
                "subject_type": model_type,
                "subject": "AR-210",
                "type": "HAS_PART",
                "object_type": "Part",
                "object": "P-1042",
            }
        )
    return json.dumps(
        {"entities": [{"type": model_type, "name": "AR-210"}], "relations": relations}, ensure_ascii=False
    )


@pytest.mark.parametrize("bad", [assembly_response("ProductLine"), assembly_response(include_part=False)])
def test_wrong_model_type_and_missing_installed_part_get_one_repair(bad):
    llm = FakeLLM([bad, assembly_response()])

    result = extract_chunk(ASSEMBLY, llm)
    graph = build_graph([result])

    assert graph.has_edge("Model:AR-210", "Part:P-1042")
    assert graph.has_edge("Part:P-1042", "Component:пусковое реле")
    assert "ProductLine:AR-210" not in graph
    assert len(llm.requests) == 2 and result.dropped == 0


@pytest.mark.parametrize("repair", ["не JSON", '{"entities": [], "relations": []}'])
def test_broken_repair_keeps_valid_facts_from_first_response(repair):
    llm = FakeLLM([assembly_response("ProductLine", include_part=False), repair])

    result = extract_chunk(ASSEMBLY, llm)

    assert len(result.relations) == 1
    assert result.relations[0].type == "IS_A"
    assert result.dropped == 1 and len(llm.requests) == 2


def test_second_semantic_error_is_filtered_without_a_third_call():
    response = assembly_response("ProductLine")
    llm = FakeLLM([response, response])

    result = extract_chunk(ASSEMBLY, llm)

    assert [relation.type for relation in result.relations] == ["IS_A"]
    assert result.dropped == 2 and len(llm.requests) == 2


def test_repair_does_not_authorize_a_code_missing_from_the_document():
    invented = assembly_response().replace("P-1042", "P-7777")
    llm = FakeLLM([assembly_response(include_part=False), invented])

    result = extract_chunk(ASSEMBLY, llm)

    assert result.relations == [Relation(Entity("Part", "P-1042"), "IS_A", Entity("Component", "пусковое реле"))]
    assert all(entity.name != "P-7777" for entity in result.entities)
    assert result.dropped == 2 and len(llm.requests) == 2


def test_model_name_with_equipment_prefix_keeps_failure_relation():
    chunk = Chunk("report.md#0", "report.md", "Витрина АR-220: отказ узла пусковое реле.")
    response = json.dumps(
        {
            "entities": [],
            "relations": [
                {
                    "subject_type": "Model",
                    "subject": "витрина АR-220",
                    "type": "FAILED",
                    "object_type": "Component",
                    "object": "пусковое реле",
                }
            ],
        },
        ensure_ascii=False,
    )
    llm = FakeLLM([response])

    result = extract_chunk(chunk, llm)

    assert result.relations[0].subject == Entity("Model", "AR-220")
    assert result.dropped == 0 and len(llm.requests) == 1


@pytest.mark.parametrize("supplied_count", [0, 1])
def test_missing_supplier_is_checked_for_each_part(supplied_count):
    chunk = Chunk(
        "manual.md#0",
        "manual.md",
        ASSEMBLY.text + ", поставщик «ХолодКомплект».\nДатчик температуры — P-1018, поставщик «ТермоПрибор».",
    )
    complete = json.loads(assembly_response())
    complete["relations"].extend(
        [
            {
                "subject_type": "Model",
                "subject": "AR-210",
                "type": "HAS_PART",
                "object_type": "Part",
                "object": "P-1018",
            },
            {
                "subject_type": "Part",
                "subject": "P-1018",
                "type": "IS_A",
                "object_type": "Component",
                "object": "датчик температуры",
            },
        ]
    )
    suppliers = [
        {"subject_type": "Part", "subject": part, "type": "SUPPLIED_BY", "object_type": "Supplier", "object": supplier}
        for part, supplier in [("P-1042", "ХолодКомплект"), ("P-1018", "ТермоПрибор")]
    ]
    partial = {**complete, "relations": complete["relations"] + suppliers[:supplied_count]}
    complete["relations"] += suppliers
    llm = FakeLLM([json.dumps(partial, ensure_ascii=False), json.dumps(complete, ensure_ascii=False)])

    result = extract_chunk(chunk, llm)
    graph = build_graph([result])

    assert graph.has_edge("Part:P-1042", "Supplier:ХолодКомплект")
    assert graph.has_edge("Part:P-1018", "Supplier:ТермоПрибор")
    assert graph.has_edge("Model:AR-210", "Part:P-1042")
    assert graph.has_edge("Model:AR-210", "Part:P-1018")
    assert result.dropped == 0 and len(llm.requests) == 2


def test_supplier_is_not_requested_when_document_does_not_name_one():
    llm = FakeLLM([assembly_response()])

    result = extract_chunk(ASSEMBLY, llm)

    assert not any(relation.type == "SUPPLIED_BY" for relation in result.relations)
    assert len(llm.requests) == 1


def document_link_response(kind, start, end, reversed_types=False):
    types = {"BELONGS_TO": ("Model", "ProductLine"), "FIXED_BY": ("ErrorCode", "Procedure")}
    start_type, end_type = types[kind]
    if reversed_types:
        start_type, end_type = end_type, start_type
        start, end = end, start
    return json.dumps(
        {
            "entities": [],
            "relations": [
                {
                    "subject_type": start_type,
                    "subject": start,
                    "type": kind,
                    "object_type": end_type,
                    "object": end,
                }
            ],
        },
        ensure_ascii=False,
    )


MEMBERSHIP = Chunk("equipment.md#0", "equipment.md", "Витрина ar 650 входит в линейку «Арктика».")
REGULATION = Chunk("repair.md#0", "repair.md", "Регламент sp 250 применяется при ошибке Е 35.")


@pytest.mark.parametrize(
    "chunk,kind,start,end",
    [
        (MEMBERSHIP, "BELONGS_TO", "AR-650", "Арктика"),
        (REGULATION, "FIXED_BY", "E-35", "SP-250"),
    ],
)
def test_missing_explicit_document_link_gets_one_targeted_retry(chunk, kind, start, end):
    llm = FakeLLM(['{"entities": [], "relations": []}', document_link_response(kind, start, end)])

    result = extract_chunk(chunk, llm)

    assert [(r.subject.name, r.type, r.object.name) for r in result.relations] == [(start, kind, end)]
    assert len(llm.requests) == 2
    feedback = llm.requests[1]["messages"][1]["content"]
    assert chunk.text in feedback and f"{start} -{kind}-> {end}" in feedback
    expected = json.loads(document_link_response(kind, start, end))["relations"][0]
    assert json.dumps(expected, ensure_ascii=False) in feedback


@pytest.mark.parametrize("repaired", [True, False])
def test_model_self_link_is_repaired_by_the_model_or_dropped_after_two_calls(repaired):
    chunk = Chunk("equipment.md#0", "equipment.md", "Холодильник GS-640 входит в линейку «Гольфстрим».")
    wrong = json.loads(document_link_response("BELONGS_TO", "GS-640", "GS-640"))
    wrong["relations"][0]["object_type"] = "Model"
    wrong = json.dumps(wrong)
    correct = document_link_response("BELONGS_TO", "GS-640", "Гольфстрим")
    llm = FakeLLM([wrong, correct if repaired else wrong])

    result = extract_chunk(chunk, llm)

    feedback = llm.requests[1]["messages"][1]["content"]
    assert '"object_type": "ProductLine"' in feedback and '"object": "Гольфстрим"' in feedback
    assert len(llm.requests) == 2
    if repaired:
        assert result.relations == [
            Relation(Entity("Model", "GS-640"), "BELONGS_TO", Entity("ProductLine", "Гольфстрим"))
        ]
        assert result.dropped == 0
    else:
        assert not result.relations and result.dropped == 1


@pytest.mark.parametrize("wrong_line", ["Бриз", "Гольфстрим"])
def test_membership_with_another_vocabulary_line_is_retried_and_rejected(wrong_line):
    wrong = document_link_response("BELONGS_TO", "AR-650", wrong_line)
    llm = FakeLLM([wrong, wrong])

    result = extract_chunk(MEMBERSHIP, llm)

    assert not result.relations and result.dropped == 1
    assert len(llm.requests) == 2


def test_correct_membership_survives_a_repair_that_omits_it():
    chunk = Chunk("equipment.md#0", "equipment.md", MEMBERSHIP.text + "\nРегламент SP-250 применяется при ошибке E-35.")
    llm = FakeLLM(
        [
            document_link_response("BELONGS_TO", "AR-650", "Арктика"),
            document_link_response("FIXED_BY", "E-35", "SP-250"),
        ]
    )

    result = extract_chunk(chunk, llm)

    assert {(r.subject.name, r.type, r.object.name) for r in result.relations} == {
        ("AR-650", "BELONGS_TO", "Арктика"),
        ("E-35", "FIXED_BY", "SP-250"),
    }
    assert result.dropped == 0 and len(llm.requests) == 2


def test_reversed_regulation_relation_is_normalized_only_when_the_text_confirms_it():
    llm = FakeLLM([document_link_response("FIXED_BY", "E-35", "SP-250", reversed_types=True)])

    result = extract_chunk(REGULATION, llm)

    assert result.relations == [Relation(Entity("ErrorCode", "E-35"), "FIXED_BY", Entity("Procedure", "SP-250"))]
    assert result.dropped == 0 and len(llm.requests) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Регламент SP-250 не применяется при ошибке E-35.",
        "В документе упомянуты SP-250 и E-35 без описания связи.",
        "Регламент SP-250 применяется при ошибке E-35 или E-36.",
    ],
)
def test_unsupported_regulation_direction_is_filtered(text):
    response = document_link_response("FIXED_BY", "E-35", "SP-250", reversed_types=True)
    llm = FakeLLM([response, response])

    result = extract_chunk(Chunk("repair.md#0", "repair.md", text), llm)

    assert not result.relations and result.dropped == 1 and len(llm.requests) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Витрина AR-650 не входит в линейку «Арктика».",
        "Витрина AR-650 не   входит в линейку «Арктика».",
        "В руководстве упомянуты AR-650 и линейка «Арктика».",
        "Витрины AR-650 и AR-660 входят в линейку «Арктика».",
        "Витрина AR-650 входит в линейку «Арктика» или «Бриз».",
        "Витрина AR-650 входит в линейку «Арктика». Витрина AR-650 входит в линейку «Бриз».",
    ],
)
def test_no_completeness_retry_for_ambiguous_or_negative_membership(text):
    llm = FakeLLM(['{"entities": [], "relations": []}'])

    result = extract_chunk(Chunk("equipment.md#0", "equipment.md", text), llm)

    assert not result.relations and len(llm.requests) == 1


def test_an_explicit_link_is_never_created_without_a_model_relation():
    llm = FakeLLM(['{"entities": [], "relations": []}', '{"entities": [], "relations": []}'])

    result = extract_chunk(MEMBERSHIP, llm)

    assert not result.entities and not result.relations and len(llm.requests) == 2


def test_complete_document_link_needs_one_call():
    llm = FakeLLM([document_link_response("BELONGS_TO", "AR-650", "Арктика")])

    result = extract_chunk(MEMBERSHIP, llm)

    assert len(result.relations) == 1 and result.dropped == 0 and len(llm.requests) == 1


def replacement_response(old, new):
    return json.dumps(
        {
            "entities": [],
            "relations": [
                {"subject_type": "Part", "subject": old, "type": "REPLACED_BY", "object_type": "Part", "object": new},
                {
                    "subject_type": "Part",
                    "subject": "P-4321",
                    "type": "SUPPLIED_BY",
                    "object_type": "Supplier",
                    "object": "ТермоПрибор",
                },
            ],
        },
        ensure_ascii=False,
    )


REPLACEMENT = Chunk(
    "notice.md#0",
    "notice.md",
    "Деталь P-9876C снята с производства. Вместо неё устанавливается деталь P-4321. "
    "Поставщик P-4321 — ТермоПрибор.",
)


def test_reversed_replacement_is_normalized_from_the_original_document():
    wrong = replacement_response("P-4321", "P-9876C")
    llm = FakeLLM([wrong])

    result = extract_chunk(REPLACEMENT, llm)

    assert [(r.subject.name, r.object.name) for r in result.relations if r.type == "REPLACED_BY"] == [
        ("P-9876C", "P-4321")
    ]
    assert len(llm.requests) == 1 and result.dropped == 0
    assert REPLACEMENT.text == llm.requests[0]["messages"][1]["content"]


def test_direction_normalization_preserves_other_facts():
    llm = FakeLLM([replacement_response("P-4321", "P-9876C")])

    result = extract_chunk(REPLACEMENT, llm)

    assert [(r.subject.name, r.object.name) for r in result.relations if r.type == "SUPPLIED_BY"] == [
        ("P-4321", "ТермоПрибор")
    ]
    assert result.dropped == 0 and len(llm.requests) == 1


def test_correct_direction_is_kept_without_an_extra_call():
    llm = FakeLLM([replacement_response("P-9876C", "P-4321")])

    result = extract_chunk(REPLACEMENT, llm)

    assert result.relations[0] == Relation(Entity("Part", "P-9876C"), "REPLACED_BY", Entity("Part", "P-4321"))
    assert len(llm.requests) == 1 and result.dropped == 0


def test_replacement_direction_uses_normalized_codes_and_not_the_suffix():
    chunk = Chunk("notice.md#0", "notice.md", REPLACEMENT.text.replace("P-9876C", "р 9876с"))
    llm = FakeLLM([replacement_response("P-4321", "P-9876C")])

    result = extract_chunk(chunk, llm)

    assert result.relations[0].subject.name == "P-9876C"
    assert result.relations[0].object.name == "P-4321"
    assert len(llm.requests) == 1


def test_two_directions_from_the_model_become_one_source_confirmed_relation():
    reply = json.loads(replacement_response("P-4321", "P-9876C"))
    reply["relations"] += json.loads(replacement_response("P-9876C", "P-4321"))["relations"]
    result = extract_chunk(REPLACEMENT, FakeLLM([json.dumps(reply, ensure_ascii=False)]))

    replacements = [r for r in result.relations if r.type == "REPLACED_BY"]
    assert replacements == [Relation(Entity("Part", "P-9876C"), "REPLACED_BY", Entity("Part", "P-4321"))]
    assert len(result.relations) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Деталь P-9876C не снята с производства. Вместо неё на складе оказалась P-4321.",
        "Деталь P-9876C снята с производства. Вместо неё не устанавливается P-4321.",
        "Деталь P-9876C снята с производства. Вместо неё устанавливается P-4321 или P-9876D.",
        "Детали P-9876C и P-4321 сняты с производства. Вместо них устанавливается P-9876D.",
        "Деталь P-9876C снята с производства. Вместо неё устанавливается P-4321. "
        "Деталь P-4321 снята с производства. Вместо неё устанавливается P-9876C.",
    ],
)
def test_direction_check_does_not_resolve_ambiguous_or_contradictory_text(text):
    llm = FakeLLM([replacement_response("P-4321", "P-9876C")])

    result = extract_chunk(Chunk("notice.md#0", "notice.md", text), llm)

    assert result.relations[0].type == "REPLACED_BY"
    assert len(llm.requests) == 1


def test_direction_check_never_adds_a_relation_missing_from_the_model_response():
    llm = FakeLLM(['{"entities": [], "relations": []}'])

    result = extract_chunk(REPLACEMENT, llm)

    assert not result.relations and not result.entities
    assert len(llm.requests) == 1


def test_a_part_cannot_replace_itself():
    llm = FakeLLM([replacement_response("P-9876C", "P-9876C")])

    result = extract_chunk(REPLACEMENT, llm)

    assert [r.type for r in result.relations] == ["SUPPLIED_BY"]
    assert result.dropped == 1 and len(llm.requests) == 1
    assert build_graph([result]).graph["conflicts"] == 0


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


def test_explicit_supplier_direction_is_normalized_without_losing_other_relations():
    text = "Деталь P-4321 заменена на P-9876C. Поставщик детали P-9876C – «ТермоПрибор»."
    reply = replacement_response("P-4321", "P-9876C")
    data = json.loads(reply)
    data["relations"][-1] = {
        "subject_type": "Supplier", "subject": "ТермоПрибор", "type": "SUPPLIED_BY",
        "object_type": "Part", "object": "P-9876C",
    }
    llm = FakeLLM([json.dumps(data, ensure_ascii=False)])

    result = extract_chunk(Chunk("notice.md#0", "notice.md", text), llm)

    assert Relation(Entity("Part", "P-9876C"), "SUPPLIED_BY", Entity("Supplier", "ТермоПрибор")) in result.relations
    assert any(relation.type == "REPLACED_BY" for relation in result.relations)
    assert len(llm.requests) == 1


@pytest.mark.parametrize("suffix", ["не «ТермоПрибор»", "«ТермоПрибор» или «ВентПром»"])
def test_uncertain_supplier_text_does_not_reverse_the_relation(suffix):
    row = {"subject_type": "Supplier", "subject": "ТермоПрибор", "type": "SUPPLIED_BY",
           "object_type": "Part", "object": "P-9876C"}
    data = json.dumps({"entities": [], "relations": [row]}, ensure_ascii=False)
    text = f"Поставщик детали P-9876C – {suffix}."
    llm = FakeLLM([data, '{"entities": [], "relations": []}'])

    result = extract_chunk(Chunk("notice.md#0", "notice.md", text), llm)

    assert not result.relations
