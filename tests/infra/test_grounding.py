import json

import pytest

from assistant.infra.grounding import canonical_name, code_type, find_codes, find_names, load_vocabulary, normalize_code

VOCABULARY = {
    "ProductLine": ["Арктика"],
    "Component": ["пусковое реле", "реле давления"],
    "Supplier": ["ХолодКомплект"],
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AR-210", "AR-210"),
        ("ar-210", "AR-210"),
        ("AR210", "AR-210"),
        ("AR 210", "AR-210"),
        ("АR-210", "AR-210"),  # первая буква кириллическая
        ("е-17", "E-17"),  # кириллическая «е»
        ("Р-1042В", "P-1042B"),  # кириллические «Р» и «В»
        ("sp-104", "SP-104"),
    ],
)
def test_normalize_code_accepts_sloppy_spelling(raw, expected):
    assert normalize_code(raw) == expected


@pytest.mark.parametrize("raw", ["SB-2025", "R-2025", "AR-21", "P-104", "E-170", "AR-210B", "за 25", "насос"])
def test_normalize_code_rejects_non_codes(raw):
    assert normalize_code(raw) is None


def test_find_codes_returns_canonical_unique_codes_in_order():
    text = "Оборудование: витрина ar 210. Код е-17, затем снова AR-210. Заменили «P-1042B»."
    assert find_codes(text) == ["AR-210", "E-17", "P-1042B"]


def test_find_codes_ignores_document_numbers_and_plain_numbers():
    assert find_codes("Бюллетень SB-2025-03, отчёт R-2025-014, режим вышел за 25 минут в 2025 году.") == []


def test_code_type():
    assert [code_type(c) for c in ("AR-210", "P-1042B", "E-17", "SP-104")] == [
        "Model",
        "Part",
        "ErrorCode",
        "Procedure",
    ]


def test_load_vocabulary_reads_the_shipped_file(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps(VOCABULARY, ensure_ascii=False), encoding="utf-8")
    assert load_vocabulary(path) == VOCABULARY


def test_find_names_is_case_insensitive_and_keeps_canonical_spelling():
    found = find_names("Кто сейчас поставляет узел «Пусковое реле» для линейки арктика?", VOCABULARY)
    assert found == [("ProductLine", "Арктика"), ("Component", "пусковое реле")]


def test_canonical_name_for_codes_checks_the_type():
    assert canonical_name("Part", "р-1042в", VOCABULARY) == "P-1042B"
    assert canonical_name("Model", "P-1042", VOCABULARY) is None
    assert canonical_name("ErrorCode", "ошибка", VOCABULARY) is None


def test_canonical_name_for_names_uses_the_vocabulary():
    assert canonical_name("Supplier", "ООО «ХолодКомплект»", VOCABULARY) == "ХолодКомплект"
    assert canonical_name("Component", "Пусковое реле", VOCABULARY) == "пусковое реле"
    assert canonical_name("Component", "конденсатор", VOCABULARY) is None
    assert canonical_name("Тип", "пусковое реле", VOCABULARY) is None
