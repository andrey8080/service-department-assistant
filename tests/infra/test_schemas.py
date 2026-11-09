from assistant.infra.schemas import ENTITY_TYPES, EXTRACTION_SCHEMA, RELATION_SIGNATURES, Entity, Relation


def test_schema_has_seven_entity_types_and_eight_relations():
    assert len(ENTITY_TYPES) == 7 and len(RELATION_SIGNATURES) == 8
    assert RELATION_SIGNATURES["REPLACED_BY"] == ("Part", "Part")


def test_every_pair_of_types_has_at_most_one_relation():
    assert len(set(RELATION_SIGNATURES.values())) == len(RELATION_SIGNATURES)


def test_node_id():
    assert Entity("Part", "P-1042").node_id == "Part:P-1042"


def test_relation_matches_schema_only_with_right_types_at_both_ends():
    old, new, model = Entity("Part", "P-1042"), Entity("Part", "P-1042B"), Entity("Model", "AR-210")
    assert Relation(old, "REPLACED_BY", new).matches_schema()
    assert not Relation(model, "REPLACED_BY", new).matches_schema()
    assert not Relation(old, "LIKES", new).matches_schema()


def test_extraction_schema_lists_types_as_enums():
    entity = EXTRACTION_SCHEMA["properties"]["entities"]["items"]["properties"]
    relation = EXTRACTION_SCHEMA["properties"]["relations"]["items"]["properties"]
    assert entity["type"]["enum"] == list(ENTITY_TYPES)
    assert relation["type"]["enum"] == list(RELATION_SIGNATURES)
    assert relation["subject_type"]["enum"] == list(ENTITY_TYPES)


def test_map_schema_asks_for_a_partial_answer_and_a_score():
    from assistant.infra.schemas import MAP_SCHEMA

    assert MAP_SCHEMA["required"] == ["answer", "score"]
    assert MAP_SCHEMA["properties"]["score"] == {"type": "integer", "minimum": 0, "maximum": 10}
