"""Оценка ответов и извлечения. Всё считается сравнением строк."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from assistant.infra.grounding import code_type, find_codes, normalize_code
from assistant.infra.schemas import ExtractionResult, ToolStep

_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[   ]\d{3})+|\d+)(?![\w-])"
_WITH_UNIT = re.compile(rf"({_NUMBER})\s*(руб|₽|шт)")


@dataclass
class Outcome:
    """Ответ ассистента в любом режиме, приведённый к одному виду."""

    text: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False
    steps: list[ToolStep] = field(default_factory=list)
    needs_confirmation: bool = False
    pending_action: dict | None = None


def _digits(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def has_fact(answer: str, spellings: list[str]) -> bool:
    """Есть ли в ответе ключевой факт в одном из допустимых написаний.

    Код сравнивается целиком и в каноническом виде: P-1042 не засчитывается внутри P-1042B.
    Число должно стоять отдельно, пробелы между разрядами не мешают. Название ищется без учёта регистра.
    """
    codes = set(find_codes(answer))
    numbers = {_digits(match) for match in re.findall(_NUMBER, answer)}
    for spelling in spellings:
        code = normalize_code(spelling)
        if code:
            found = code in codes
        elif spelling.isdigit():
            found = spelling in numbers
        else:
            found = spelling.lower() in answer.lower()
        if found:
            return True
    return False


def fabricated_facts(answer: str, question: str, known_codes: set[str], tool_text: str) -> list[str]:
    """Выдуманные факты в ответе.

    Код выдуман, если его нет ни в корпусе, ни в вопросе. Цена или остаток выдуманы,
    если такого числа нет в результатах инструментов этого диалога.
    """
    asked = set(find_codes(question))
    invented = [code for code in find_codes(answer) if code not in known_codes and code not in asked]
    tool_numbers = set(re.findall(r"\d+", tool_text))
    for raw, unit in _WITH_UNIT.findall(answer):
        if _digits(raw) not in tool_numbers:
            invented.append(f"{_digits(raw)} {unit}")
    return invented


def _calls_ok(question: dict, outcome: Outcome) -> bool:
    done = [(step.tool, step.args) for step in outcome.steps if step.error is None]
    if any((call["tool"], call["args"]) not in done for call in question["expected_calls"]):
        return False
    pending = question["expected_pending"]
    if pending is None:
        return True
    created = any(tool == "create_service_order" for tool, _ in done)
    return outcome.needs_confirmation and outcome.pending_action == pending and not created


def _is_correct(question: dict, outcome: Outcome, fabricated: list[str]) -> bool:
    if question["must_refuse"]:
        asked = set(find_codes(question["question"]))
        extra_parts = [c for c in find_codes(outcome.text) if code_type(c) == "Part" and c not in asked]
        return outcome.refused and not extra_parts and not _WITH_UNIT.search(outcome.text)
    facts_ok = all(has_fact(outcome.text, spellings) for spellings in question["key_facts"])
    return facts_ok and not outcome.refused and _calls_ok(question, outcome)


def _share(part: float, whole: float) -> float:
    return round(part / whole, 3) if whole else 0.0


def evaluate(questions: list[dict], run: Callable[[dict], Outcome], known_codes: set[str], stats=None) -> dict:
    """Прогоняет вопросы через `run` и считает метрики.

    stats – счётчики LLMClient. По их приросту считается стоимость одного вопроса.
    """
    rows = []
    for question in questions:
        before = (stats.calls, stats.prompt_tokens + stats.completion_tokens) if stats else (0, 0)
        outcome = run(question)
        after = (stats.calls, stats.prompt_tokens + stats.completion_tokens) if stats else (0, 0)
        tool_text = json.dumps([step.result for step in outcome.steps], ensure_ascii=False)
        fabricated = fabricated_facts(outcome.text, question["question"], known_codes, tool_text)
        groups = question["source_groups"]
        hit = sum(1 for group in groups if set(group) & set(outcome.sources))
        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "answer": outcome.text,
                "correct": _is_correct(question, outcome, fabricated),
                "refused": outcome.refused,
                "fabricated": fabricated,
                "sources": outcome.sources,
                "source_recall": round(hit / len(groups), 3) if groups else None,
                "steps": [step.__dict__ for step in outcome.steps],
                "rejected_steps": sum(1 for step in outcome.steps if step.error is not None),
                "llm_calls": after[0] - before[0],
                "tokens": after[1] - before[1],
            }
        )
    by_type = {}
    for kind in dict.fromkeys(row["type"] for row in rows):
        selected = [row for row in rows if row["type"] == kind]
        by_type[kind] = {
            "questions": len(selected),
            "accuracy": _share(sum(row["correct"] for row in selected), len(selected)),
        }
    recalls = [row["source_recall"] for row in rows if row["source_recall"] is not None]
    total_steps = sum(len(row["steps"]) for row in rows)
    summary = {
        "questions": len(rows),
        "accuracy": _share(sum(row["correct"] for row in rows), len(rows)),
        "by_type": by_type,
        "fabricated_rate": _share(sum(1 for row in rows if row["fabricated"]), len(rows)),
        "source_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "rejected_calls_rate": _share(sum(row["rejected_steps"] for row in rows), total_steps),
        "llm_calls_per_question": _share(sum(row["llm_calls"] for row in rows), len(rows)),
        "tokens_per_question": _share(sum(row["tokens"] for row in rows), len(rows)),
    }
    return {"summary": summary, "rows": rows}


def _score(found: set, required: set, allowed: set) -> dict:
    return {
        "precision": _share(len(found & allowed), len(found)) if found else (1.0 if not required else 0.0),
        "recall": _share(len(found & required), len(required)) if required else 1.0,
        "found": len(found),
        "gold": len(required),
    }


def evaluate_extraction(gold: dict, results: list[ExtractionResult]) -> dict:
    """Precision и Recall извлечения на документах с эталоном.

    Recall считается по обязательным связям. Связь из `optional_relations` верна,
    поэтому Precision она не снижает, хотя находить её не обязательно.
    """
    found_rel, found_ent, need_rel, need_ent, ok_rel, ok_ent = set(), set(), set(), set(), set(), set()
    documents, dropped = set(), 0
    for result in results:
        if result.doc_id not in gold:
            continue
        documents.add(result.doc_id)
        dropped += result.dropped
        found_ent |= {(result.doc_id, entity.node_id) for entity in result.entities}
        found_rel |= {(result.doc_id, r.subject.node_id, r.type, r.object.node_id) for r in result.relations}
    for doc_id in documents:
        entry = gold[doc_id]
        required = {(doc_id, *triple) for triple in entry["relations"]}
        optional = {(doc_id, *triple) for triple in entry["optional_relations"]}
        need_rel |= required
        ok_rel |= required | optional
        need_ent |= {(doc_id, name) for name in entry["entities"]}
        ok_ent |= {(doc_id, name) for name in entry["entities"]} | {
            (doc_id, end) for _, s, _, o in optional for end in (s, o)
        }
    return {
        "relations": _score(found_rel, need_rel, ok_rel),
        "entities": _score(found_ent, need_ent, ok_ent),
        "dropped": dropped,
        "documents": len(documents),
    }
