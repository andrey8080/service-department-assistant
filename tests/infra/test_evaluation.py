from assistant.infra.evaluation import Outcome, evaluate, evaluate_extraction, fabricated_facts, has_fact
from assistant.infra.schemas import Entity, ExtractionResult, Relation, ToolStep


def question(kind="multihop", facts=(("P-1042B",),), **extra) -> dict:
    base = {
        "id": "dev-01",
        "type": kind,
        "question": "Какую деталь заказать при ошибке E-17 на AR-210?",
        "key_facts": [list(f) for f in facts],
        "source_groups": [],
        "expected_calls": [],
        "expected_pending": None,
        "must_refuse": kind == "refusal",
    }
    return {**base, **extra}


def test_code_fact_is_matched_as_a_whole_code_in_any_spelling():
    assert has_fact("Закажите p-1042b.", ["P-1042B"])
    assert not has_fact("Закажите P-1042B.", ["P-1042"])  # старая деталь не засчитывается по вхождению в новую
    assert not has_fact("Закажите P-1042.", ["P-1042B"])


def test_number_fact_is_a_standalone_number_and_ignores_thousand_spaces():
    assert has_fact("Цена 2 630 руб., на складе 3 шт.", ["2630"])
    assert has_fact("Цена 2 630 руб., на складе 3 шт.", ["3"])
    assert not has_fact("Деталь P-1042B для AR-310 стоит 2630.", ["3"])
    assert not has_fact("Цена 12630 руб.", ["2630"])


def test_name_fact_accepts_any_listed_spelling_case_insensitively():
    assert has_fact("Чаще всего отказ Пускового реле.", ["пусковое реле", "пускового реле"])
    assert not has_fact("Чаще всего отказывает реле давления.", ["пусковое реле", "пускового реле"])


def test_fabricated_codes_are_those_absent_from_corpus_and_question():
    known = {"P-1042", "P-1042B", "AR-210", "E-17"}
    assert fabricated_facts("Закажите P-1043 для AR-210.", "Что заказать на AR-210?", known, "") == ["P-1043"]
    assert fabricated_facts("Модель AR-999 в документах не найдена.", "Что заказать на AR-999?", known, "") == []


def test_fabricated_numbers_are_prices_and_stock_not_returned_by_tools():
    tool_text = '{"price_rub": 2630, "qty": 3}'
    assert fabricated_facts("Цена 2 630 руб., есть 3 шт.", "?", set(), tool_text) == []
    assert fabricated_facts("Цена 4850 руб., есть 3 шт.", "?", set(), tool_text) == ["4850 руб"]
    assert fabricated_facts("Примерно 3000 ₽.", "?", set(), "") == ["3000 ₽"]
    assert fabricated_facts("Норма времени 40 минут.", "?", set(), "") == []


def test_answer_is_correct_when_all_facts_are_present():
    result = evaluate(
        [question()], lambda q: Outcome("Закажите P-1042B.", sources=["bulletin_SB-2025-03.md"]), {"P-1042B"}
    )
    assert result["rows"][0]["correct"] and result["summary"]["accuracy"] == 1.0
    assert result["summary"]["by_type"] == {"multihop": {"questions": 1, "accuracy": 1.0}}


def test_refusal_instead_of_an_answer_is_wrong_and_an_answer_instead_of_refusal_is_wrong():
    refused = Outcome("В базе знаний не найдено.", refused=True)
    assert not evaluate([question()], lambda q: refused, set())["rows"][0]["correct"]
    refusal = question("refusal", facts=(), question="Сколько стоит деталь P-9999?")
    assert evaluate([refusal], lambda q: refused, set())["rows"][0]["correct"]
    invented = Outcome("Деталь P-9999 стоит 1500 руб.")
    row = evaluate([refusal], lambda q: invented, set())["rows"][0]
    assert not row["correct"] and row["fabricated"] == ["1500 руб"]


def test_tool_question_needs_the_expected_calls_with_exact_arguments():
    q = question(
        "tool",
        facts=(("P-1042B",), ("3",)),
        expected_calls=[{"tool": "check_stock", "args": {"part_id": "P-1042B", "warehouse": "Казань"}}],
    )
    good = Outcome(
        "P-1042B, на складе 3 шт.",
        steps=[ToolStep("check_stock", {"part_id": "P-1042B", "warehouse": "Казань"}, {"qty": 3})],
    )
    wrong = Outcome(
        "P-1042B, на складе 3 шт.",
        steps=[ToolStep("check_stock", {"part_id": "P-1042", "warehouse": "Казань"}, {"qty": 3})],
    )
    assert evaluate([q], lambda _: good, {"P-1042B"})["rows"][0]["correct"]
    assert not evaluate([q], lambda _: wrong, {"P-1042B", "P-1042"})["rows"][0]["correct"]


def test_order_question_needs_a_pending_action_and_no_created_order():
    pending = {
        "tool": "create_service_order",
        "args": {"part_id": "P-2054C", "warehouse": "Москва", "qty": 2, "model": "BZ-120"},
    }
    q = question("tool", facts=(("P-2054C",),), expected_pending=pending)
    waits = Outcome("Готов заказать P-2054C, подтвердите.", needs_confirmation=True, pending_action=pending)
    done = Outcome(
        "Заказал P-2054C.", steps=[ToolStep("create_service_order", pending["args"], {"order_id": "SO-000001"})]
    )
    assert evaluate([q], lambda _: waits, {"P-2054C"})["rows"][0]["correct"]
    assert not evaluate([q], lambda _: done, {"P-2054C"})["rows"][0]["correct"]


def test_source_recall_counts_groups_hit_by_the_answer_sources():
    q = question(source_groups=[["manual_AR-210.md", "manual_AR-220.md"], ["bulletin_SB-2025-03.md"]])
    result = evaluate([q], lambda _: Outcome("P-1042B", sources=["manual_AR-220.md"]), {"P-1042B"})
    assert result["rows"][0]["source_recall"] == 0.5 and result["summary"]["source_recall"] == 0.5


def test_summary_reports_fabrication_rejected_calls_and_cost():
    steps = [
        ToolStep("check_stock", {"warehouse": "Екатеринбург"}, error="неизвестный склад"),
        ToolStep("get_price", {"part_id": "P-1042B"}, {"price_rub": 2630}),
    ]

    class Stats:
        calls, prompt_tokens, completion_tokens = 0, 0, 0

    stats = Stats()

    def run(q):
        stats.calls += 3
        stats.prompt_tokens += 100
        stats.completion_tokens += 20
        return Outcome("P-1042B стоит 9999 руб.", steps=steps)

    summary = evaluate([question(), question()], run, {"P-1042B"}, stats)["summary"]
    assert summary["fabricated_rate"] == 1.0 and summary["rejected_calls_rate"] == 0.5
    assert summary["llm_calls_per_question"] == 3.0 and summary["tokens_per_question"] == 120.0


def test_extraction_scores_do_not_punish_optional_relations():
    part, new, relay = Entity("Part", "P-1042"), Entity("Part", "P-1042B"), Entity("Component", "пусковое реле")
    gold = {
        "bulletin.md": {
            "entities": ["Part:P-1042", "Part:P-1042B", "Component:пусковое реле"],
            "relations": [
                ["Part:P-1042", "REPLACED_BY", "Part:P-1042B"],
                ["Part:P-1042", "IS_A", "Component:пусковое реле"],
            ],
            "optional_relations": [["Part:P-1042B", "IS_A", "Component:пусковое реле"]],
        }
    }
    found = ExtractionResult(
        "bulletin.md#0",
        "bulletin.md",
        [part, new, relay],
        [
            Relation(part, "REPLACED_BY", new),  # обязательная
            Relation(new, "IS_A", relay),  # необязательная, но верная
            Relation(new, "REPLACED_BY", part),  # перевёрнутая, ошибка
        ],
        dropped=2,
    )
    scores = evaluate_extraction(gold, [found])
    assert scores["relations"] == {"precision": round(2 / 3, 3), "recall": 0.5, "found": 3, "gold": 2}
    assert scores["entities"] == {"precision": 1.0, "recall": 1.0, "found": 3, "gold": 3}
    assert scores["dropped"] == 2 and scores["documents"] == 1
