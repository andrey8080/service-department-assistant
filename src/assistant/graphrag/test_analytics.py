from copy import deepcopy

import pytest

from assistant.graphrag.analytics import overview
from assistant.infra.grounding import find_codes


@pytest.fixture
def vocabulary():
    return {
        "ProductLine": ["Плато", "Долина"],
        "Component": ["насос", "датчик", "контроллер", "вентилятор", "компрессор"],
        "Supplier": ["Партнёр", "Старая фабрика", "Новая фабрика"],
    }


@pytest.fixture
def records():
    lines = {"Плато": ["AR-701", "AR-702"], "Долина": ["BZ-801"]}
    failures = {
        "AR-701": {"насос": 3, "датчик": 2, "контроллер": 1},
        "AR-702": {"насос": 2, "датчик": 1, "контроллер": 2},
        "BZ-801": {"вентилятор": 1, "компрессор": 1},
    }
    inventory = {
        "AR-701": {"насос": "P-7001", "датчик": "P-7002", "контроллер": "P-7003"},
        "AR-702": {"насос": "P-7001", "датчик": "P-7102", "контроллер": "P-7003"},
        "BZ-801": {"насос": "P-8001", "вентилятор": "P-8002", "компрессор": "P-8003"},
    }
    rows = []
    for line, models in lines.items():
        rows.append({"kind": "line_coverage", "product_line": line, "models": models, "edges": set()})
        for model in models:
            membership = (f"Model:{model}", f"ProductLine:{line}")
            rows.append({
                "kind": "coverage", "model": model, "product_lines": [line],
                "failure_count": len(failures[model]), "failure_total": sum(failures[model].values()),
                "parts_count": len(inventory[model]), "edges": {membership},
            })
            for component, count in failures[model].items():
                rows.append({
                    "kind": "failure", "model": model, "component": component, "count": count,
                    "product_lines": [line], "edges": {(f"Model:{model}", f"Component:{component}")},
                })
            for component, part in inventory[model].items():
                rows.append({
                    "kind": "inventory", "model": model, "component": component, "part": part,
                    "product_lines": [line], "complete": True,
                    "edges": {membership, (f"Model:{model}", f"Part:{part}"),
                              (f"Part:{part}", f"Component:{component}")},
                })
    for parts in inventory.values():
        for part in parts.values():
            if not any(row["kind"] == "version" and row["part"] == part for row in rows):
                path = [part, "P-8001B", "P-8001C"] if part == "P-8001" else [part]
                rows.append({
                    "kind": "version", "part": part, "current": path[-1], "path": path,
                    "edges": {(f"Part:{a}", f"Part:{b}") for a, b in zip(path, path[1:], strict=False)},
                })
    for part, supplier in (("P-7001", "Партнёр"), ("P-8001", "Старая фабрика"),
                           ("P-8001C", "Новая фабрика")):
        rows.append({
            "kind": "supplier", "part": part, "supplier": supplier,
            "edges": {(f"Part:{part}", f"Supplier:{supplier}")},
        })
    return rows


def model_row(result, model):
    return next(row for row in result.rows if row.startswith(model + " —"))


def test_joint_distribution_keeps_each_model_component_and_total(records, vocabulary):
    result = overview("Сравни AR-701 и AR-702 по отказам узлов. Укажи число случаев и общий итог.",
                      records, vocabulary)

    assert result.complete
    first, second = model_row(result, "AR-701"), model_row(result, "AR-702")
    assert all(value in first for value in ("насос: 3", "датчик: 2", "контроллер: 1", "всего: 6"))
    assert all(value in second for value in ("насос: 2", "датчик: 1", "контроллер: 2", "всего: 5"))
    assert "Общее число отказов: 11." in result.rows
    assert not any("BZ-801" in row for row in result.rows)
    assert ("Model:AR-701", "Component:насос") in result.edges


def test_component_distribution_does_not_use_total_failures_of_model(records, vocabulary):
    result = overview("По линейке Плато распиши отказы датчик по моделям.", records, vocabulary)

    assert result.complete
    assert "датчик: 2" in model_row(result, "AR-701")
    assert "всего: 2" in model_row(result, "AR-701")
    assert "датчик: 1" in model_row(result, "AR-702")
    assert all("насос" not in row and "контроллер" not in row for row in result.rows)


def test_minimum_returns_all_tied_components_with_actual_failure_models(records, vocabulary):
    result = overview("Какие узлы реже всего отказывали по всем отчётам? Назови модели.",
                      records, vocabulary)

    assert result.complete
    assert len(result.rows) == 2
    assert {row.split(":", 1)[0] for row in result.rows} == {"вентилятор", "компрессор"}
    assert all("1 отказ" in row and "BZ-801" in row for row in result.rows)
    assert all("AR-701" not in row and "AR-702" not in row for row in result.rows)


def test_maximum_can_compare_models_within_one_component(records, vocabulary):
    result = overview("У какой модели Плато больше всего отказов датчик?", records, vocabulary)

    assert result.complete
    assert result.rows == ["AR-701: 2 отказа."]


@pytest.mark.parametrize("wording", ["Топ-2", "Первые два"])
def test_ranking_keeps_every_candidate_tied_at_the_cutoff(records, vocabulary, wording):
    result = overview(f"{wording} самых частых отказов узлов по всем отчётам.", records, vocabulary)

    assert result.complete
    assert result.rows == ["насос: 5 отказов.", "датчик: 3 отказа.", "контроллер: 3 отказа."]


def test_ranking_accepts_a_number_expressed_as_a_noun(records, vocabulary):
    result = overview("Покажи тройку самых частых отказов узлов по всем отчётам.", records, vocabulary)

    assert result.complete
    assert result.rows == ["насос: 5 отказов.", "датчик: 3 отказа.", "контроллер: 3 отказа."]


def test_ranking_distinguishes_three_items_from_three_distinct_places(records, vocabulary):
    items = overview("Топ-3 самых частых отказов узлов по всем отчётам.", records, vocabulary)
    places = overview("Топ-3 места по частоте отказов узлов по всем отчётам.", records, vocabulary)

    assert items.complete and places.complete
    assert len(items.rows) == 3
    assert len(places.rows) == 5
    assert set(items.rows) < set(places.rows)


@pytest.mark.parametrize("question", [
    "Сравни AR-701 и AR-702 по отказам узлов, укажи общее число случаев.",
    "Топ-2 самых частых отказов узлов по всем отчётам.",
    "Сравни узлы Плато и Долина: какие есть только у каждой линейки?",
    "Для каждой модели Плато и Долина укажи узел насос, актуальный артикул и нынешний поставщик.",
])
def test_repeated_records_do_not_change_counts_catalog_or_provenance(records, vocabulary, question):
    original = overview(question, records, vocabulary)
    repeated = overview(question, records + deepcopy(records), vocabulary)

    assert original.complete and repeated.complete
    assert repeated.rows == original.rows
    assert repeated.edges == original.edges


@pytest.mark.parametrize("missing", ["failure", "coverage", "model", "line_coverage"])
def test_missing_candidate_or_scope_cannot_produce_a_global_winner(records, vocabulary, missing):
    remaining = [
        row for row in records
        if not (
            (missing == "failure" and row["kind"] == "failure" and row["component"] == "компрессор")
            or (missing == "coverage" and row["kind"] == "coverage" and row["model"] == "BZ-801")
            or (missing == "model" and row.get("model") == "BZ-801")
            or (missing == "line_coverage" and row["kind"] == "line_coverage"
                and row["product_line"] == "Долина")
        )
    ]
    result = overview("Какие узлы реже всего отказывали по всем отчётам?", remaining, vocabulary)

    assert not result.complete
    assert result.reason
    assert result.rows == []


def test_conflicting_failure_counts_are_rejected_instead_of_added(records, vocabulary):
    conflict = deepcopy(next(row for row in records if row["kind"] == "failure"))
    conflict["count"] += 1
    result = overview("Сравни AR-701 и AR-702 по отказам узлов.", records + [conflict], vocabulary)

    assert not result.complete and result.rows == []


def test_set_difference_uses_component_names_not_part_numbers(records, vocabulary):
    result = overview("Сравни узлы Плато и Долина: какие есть только у каждой линейки?",
                      records, vocabulary)

    assert result.complete
    assert set(result.rows) == {"Только Долина: вентилятор, компрессор.",
                                "Только Плато: датчик, контроллер."}
    assert all("насос" not in row and "P-" not in row for row in result.rows)


def test_set_difference_requires_complete_inventory_of_every_model(records, vocabulary):
    remaining = [row for row in records if not (
        row["kind"] == "inventory" and row["model"] == "AR-702" and row["component"] == "датчик"
    )]
    result = overview("Сравни узлы Плато и Долина: какие есть только у каждой линейки?",
                      remaining, vocabulary)

    assert not result.complete and result.rows == []


def test_catalog_keeps_unchanged_parts_and_complete_replacements_with_new_supplier(records, vocabulary):
    result = overview("Для каждой модели Плато и Долина укажи узел насос, исходный артикул, "
                      "актуальную замену и нынешний поставщик.", records, vocabulary)

    assert result.complete
    assert len(result.rows) == 3
    for model in ("AR-701", "AR-702"):
        row = model_row(result, model)
        assert find_codes(row) == [model, "P-7001"]
        assert "без замены" in row and "Партнёр" in row
    changed = model_row(result, "BZ-801")
    assert find_codes(changed) == ["BZ-801", "P-8001", "P-8001B", "P-8001C"]
    assert "Новая фабрика" in changed and "Старая фабрика" not in changed
    assert {("Part:P-8001", "Part:P-8001B"), ("Part:P-8001B", "Part:P-8001C"),
            ("Part:P-8001C", "Supplier:Новая фабрика")} <= result.edges


@pytest.mark.parametrize("missing", ["inventory", "version", "supplier"])
def test_catalog_does_not_invent_a_missing_part_chain_end_or_supplier(records, vocabulary, missing):
    remaining = [row for row in records if not (
        row["kind"] == missing and row.get("part") == ("P-8001C" if missing == "supplier" else "P-8001")
    )]
    result = overview("Для каждой модели Плато и Долина укажи насос, актуальный артикул "
                      "и нынешний поставщик.", remaining, vocabulary)

    assert not result.complete and result.rows == []


def test_unknown_question_is_left_to_normal_search(records, vocabulary):
    assert overview("По какому регламенту проверяют E-42?", records, vocabulary) is None


@pytest.mark.parametrize("question", [
    "Сравни число отказов насоса в линейке Плато и назови нынешних поставщиков.",
    "Сравни сбои AR-701 и AR-702 и перечисли артикулы деталей.",
    "По отчётам линейки Долина назови коды деталей насоса.",
    "Для всей линейки Плато укажи отказы и номера запчастей.",
    "Сравни отказы Плато и Долины и покажи текущие детали насоса.",
    "По выездам линейки Долина нужны отказы и цепочка замен насоса.",
    "Сравни число отказов Плато и Долины и перечисли снятые детали.",
    "По отказам Плато укажи номер в исходном руководстве для насоса.",
    "Сравни число отказов Плато и Долины и перечисли детали каждой модели.",
    "Сравни число отказов Плато и Долины и перечисли запчасти насоса.",
    "Сравни число отказов Плато и Долины и укажи каталожные номера.",
])
def test_mixed_failure_and_catalog_fields_are_left_to_normal_search(records, vocabulary, question):
    assert overview(question, records, vocabulary) is None


@pytest.mark.parametrize("question", [
    "Сравни число отказов AR-701 и AR-702 по узлам.",
    "Какие узлы Плато реже всего отказывали? Укажи номер модели.",
    "Сравни сбои узла датчик по моделям Плато.",
    "Какая модель Долины чаще встречается в отчётах о сбоях?",
    "Сравни число отказов деталей AR-701 и AR-702.",
])
def test_failure_questions_without_catalog_fields_keep_analytics(records, vocabulary, question):
    result = overview(question, records, vocabulary)
    assert result is not None and result.complete and result.rows


def test_overview_does_not_modify_verified_records(records, vocabulary):
    before = deepcopy(records)
    overview("Топ-2 самых частых отказов узлов по всем отчётам.", records, vocabulary)

    assert records == before


@pytest.mark.parametrize("question, expected", [
    ("Топ-2 моделей по числу отказов по всем отчётам.", ["AR-701: 6 отказов.", "AR-702: 5 отказов."]),
    ("У какой линейки больше всего отказов по всей базе?", ["Плато: 11 отказов."]),
    ("Какие узлы чаще отказывают: насос или датчик? Сравни по всем отчётам.", ["насос: 5 отказов."]),
    ("Сравни состав узлов линеек Плато и Долина. Какие узлы общие?", ["Общие узлы: насос."]),
])
def test_overview_preserves_the_requested_dimension(records, vocabulary, question, expected):
    result = overview(question, records, vocabulary)
    assert result.complete and result.rows == expected


@pytest.mark.parametrize("question", [
    "У какой модели линейки «Равнина» больше всего отказов?",
    "Сколько отказов узла «осушитель» в линейке Плато?",
])
def test_unknown_explicit_scope_is_not_replaced_by_the_whole_corpus(records, vocabulary, question):
    result = overview(question, records, vocabulary)
    assert not result.complete and result.rows == []


def test_catalog_can_show_both_suppliers_or_only_current_codes(records, vocabulary):
    both = overview("Для каждой модели Долины укажи исходного поставщика насоса, "
                    "актуальный артикул и нынешнего поставщика.", records, vocabulary)
    assert both.complete
    assert "Старая фабрика" in both.rows[0] and "Новая фабрика" in both.rows[0]
    current = overview("Для всей линейки Долина перечисли только текущие артикулы насоса, "
                       "исходные номера не нужны.", records, vocabulary)
    assert current.complete and find_codes(current.rows[0]) == ["BZ-801", "P-8001C"]


def test_overview_does_not_expand_a_specific_part_to_the_whole_line(records, vocabulary):
    assert overview("Кто нынешний поставщик P-8001C в линейке Долина?", records, vocabulary) is None


def test_current_catalog_keeps_historical_numbers_out_of_the_main_rows(records, vocabulary):
    result = overview("Составь таблицу насоса для всей линейки Долина: модель, актуальный артикул, "
                      "нынешний поставщик. Исторические номера можно пояснить отдельно, "
                      "в основной таблице нужны текущие.", records, vocabulary)
    assert result.complete
    assert find_codes(result.rows[0]) == ["BZ-801", "P-8001C"]
    assert "Новая фабрика" in result.rows[0]


def test_component_distribution_cites_only_the_requested_failure_relations(records, vocabulary):
    result = overview("По линейке Плато распиши отказы датчика по моделям.", records, vocabulary)
    assert result.complete
    assert {edge for edge in result.edges if edge[1].startswith("Component:")} == {
        ("Model:AR-701", "Component:датчик"), ("Model:AR-702", "Component:датчик"),
    }


@pytest.mark.parametrize("missing", [None, "unchanged", "replaced", "version"])
def test_removed_list_checks_the_coverage_of_removed_parts(records, vocabulary, missing):
    for row in records:
        if row["kind"] == "coverage":
            row["replaced_parts_count"] = int(row["model"] == "BZ-801")
    selected = [row for row in records if not (
        (missing == "unchanged" and row["kind"] == "inventory" and row["part"] in {"P-8002", "P-8003"})
        or (missing == "replaced" and row["kind"] == "inventory" and row["part"] == "P-8001")
        or (missing == "version" and row["kind"] == "version" and row["part"] == "P-8001")
    )]
    result = overview("Какие исходные детали линейки Долина сняты с производства?", selected, vocabulary)
    assert result.complete == (missing in {None, "unchanged"})
    if result.complete:
        assert find_codes("\n".join(result.rows)) == ["P-8001"]


def test_old_summary_does_not_guess_how_many_original_parts_were_replaced(records, vocabulary):
    selected = [row for row in records if not (
        row["kind"] == "inventory" and row["part"] in {"P-8002", "P-8003"}
    )]
    result = overview("Какие детали линейки Долина сняты с производства?", selected, vocabulary)
    assert not result.complete
