"""Проверка кодов и чисел по вопросу и результатам инструментов."""

import json
import re
from decimal import Decimal

from assistant.infra.config import vocabulary
from assistant.infra.grounding import code_type, find_codes

_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?(?![\w-])"
_PRICE = re.compile(rf"({_NUMBER})\s*(?:руб\w*|₽|р\.)", re.IGNORECASE)
_STOCK = re.compile(rf"({_NUMBER})\s*(?:шт\w*|единиц\w*)", re.IGNORECASE)
_NAMED_PRICE = re.compile(rf"(?:цен\w*|стоимост\w*|стоит)\s*[:=]?\s*({_NUMBER})", re.IGNORECASE)
_NAMED_STOCK = re.compile(rf"(?:остаток|в наличии|на складе)\s*[:=]?\s*({_NUMBER})", re.IGNORECASE)
_ORDER_ID = re.compile(r"\bSO-\d+\b", re.IGNORECASE)
_SUPPLIER = re.compile(r"поставщик\w*|кто\s+поставля\w*", re.IGNORECASE)
_ERP_LOOKUP = re.compile(
    r"провер\w*|уточн\w*|узна\w*|выясн\w*|определ\w*|получ\w*|запрос\w*|най[тд]\w*|"
    r"хран\w*|содерж\w*|покаж\w*|верн\w*|возвращ\w*",
    re.IGNORECASE,
)
_ORDER_COUNT = re.compile(
    r"(?:количеств\w*|закаж\w*|заказ(?:ать)?(?:\s+на)?)\s*[:=]?\s*$", re.IGNORECASE
)
_OTHER_UNIT = re.compile(
    r"\s*(?:руб\w*|₽|р\.|%|час\w*|минут\w*|секунд\w*|дн\w*|день\b|дней\b|сут\w*|:\d)",
    re.IGNORECASE,
)
_WORD_COUNTS = {
    "одну": 1,
    "один": 1,
    "одна": 1,
    "две": 2,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}


def _number(value) -> Decimal:
    return Decimal(re.sub(r"[ \u00a0\u202f]", "", str(value)).replace(",", "."))


def requested_quantities(question: str) -> set[int]:
    """Числа рядом с количеством/единицами или глаголом заказа; цифры кодов не подходят."""
    found = set()
    for match in re.finditer(_NUMBER, question):
        value = _number(match.group())
        if value <= 0 or value != value.to_integral_value():
            continue
        units = re.match(r"\s*(?:шт\w*|единиц\w*)", question[match.end() :], re.IGNORECASE)
        order = _ORDER_COUNT.search(question[: match.start()])
        if units or (order and not _OTHER_UNIT.match(question[match.end() :])):
            found.add(int(value))
    for word, count in _WORD_COUNTS.items():
        for match in re.finditer(rf"\b{word}\b", question, re.IGNORECASE):
            suffix = question[match.end() :]
            units = re.match(r"\s*(?:шт\w*|единиц\w*)", suffix, re.IGNORECASE)
            order = _ORDER_COUNT.search(question[: match.start()])
            if units or (order and not _OTHER_UNIT.match(suffix)):
                found.add(count)
    return found


def allowed_codes(question: str, results: list[tuple[str, dict]]) -> set[str]:
    evidence = question + json.dumps([result for _, result in results], ensure_ascii=False)
    return set(find_codes(evidence))


def _knowledge_facts(results):
    facts = set()
    for tool, result in results:
        if tool not in {"search_knowledge", "summarize_topic"}:
            continue
        for fact in result.get("facts", []):
            match = re.fullmatch(r"(.+?) -([A-Z_]+)-> (.+)", fact) if isinstance(fact, str) else None
            if match:
                facts.add(match.groups())
    return facts


def supplier_names_for_answer(question: str, results: list[tuple[str, dict]], text: str = "") -> set[str]:
    """Допустимые имена поставщиков по области вопроса и проверенным версиям деталей.

    Факты графа остаются доказательствами и при отказе поиска сформулировать ответ.
    Эта проверка не выбирает ответ: недостающие связи должна искать модель.
    """
    facts = _knowledge_facts(results)
    supplied = {(part, name) for part, relation, name in facts if relation == "SUPPLIED_BY"}
    replacements = {(old, new) for old, relation, new in facts if relation == "REPLACED_BY"}
    codes = find_codes(question)
    parts = {code for code in codes if code_type(code) == "Part"}
    explicit_parts = bool(parts)
    models = {code for code in codes if code_type(code) == "Model"}
    errors = {code for code in codes if code_type(code) == "ErrorCode"}
    components = {node for _, relation, node in facts if relation == "IS_A" and node.lower() in question.lower()}
    components.update(node for error, relation, node in facts if relation == "INDICATES" and error in errors)
    if not parts:
        model_parts = {part for model, relation, part in facts if relation == "HAS_PART" and model in models}
        component_parts = {part for part, relation, node in facts if relation == "IS_A" and node in components}
        parts = model_parts & component_parts if model_parts and components else model_parts or component_parts
    if not parts:
        # Некоторые инструменты/сводки дают только артикул и его связи, без HAS_PART.
        parts = {part for part, _ in supplied} | {part for pair in replacements for part in pair}
        parts -= {new for _, new in replacements}
    original = any(word in question.lower() for word in ("руководств", "исходн", "изначальн", "штатн"))
    current = any(word in question.lower() for word in (
        "актуальн", "текущ", "нынешн", "последн", "теперь", "сейчас", "замен",
    ))
    if current or (not original and not explicit_parts):
        reached, pending = set(), list(parts)
        while pending:
            part = pending.pop()
            if part not in reached:
                reached.add(part)
                pending.extend(new for old, new in replacements if old == part)
        terminal = reached - {old for old, _ in replacements}
        parts = parts | terminal if original else terminal
    names = {name for part, name in supplied if part in parts}
    candidates = set(vocabulary()["Supplier"]) | {name for _, name in supplied}
    for tool, result in results:
        if result.get("refused") or not result.get("answer"):
            continue
        # Сводка может быть самостоятельным источником; локальный текст не отменяет
        # уже известную цепочку замен или явные связи поставщиков.
        if (tool == "summarize_topic" and not result.get("facts")) or (
            tool == "search_knowledge" and not supplied and not replacements
        ):
            names.update(name for name in candidates if name.lower() in result["answer"].lower())
    return names


_WAREHOUSE_NAMES = {
    "Москва": r"москв(?:а|е|у|ы|ой|ою)",
    "Казань": r"казан(?:ь|и|ью)",
    "Санкт-Петербург": r"(?:санкт[- ]петербург(?:а|е|у|ом)?|спб|питер(?:а|е|у|ом)?)",
    "Новосибирск": r"новосибирск(?:а|е|у|ом)?",
}


def warehouse_matches_question(question: str, warehouse: str) -> bool:
    """Не даёт подменить явно названный склад, сохраняя свободный перебор складов."""
    question = question.lower().replace("ё", "е").replace("–", "-").replace("—", "-")

    def mentioned(value):
        return {name for name, pattern in _WAREHOUSE_NAMES.items() if re.search(rf"\b{pattern}\b", value)}

    requested = set()
    explicit = False
    for match in re.finditer(r"\bсклад(?:е|а|у|ом)?\b\s*[:«\"']*\s*(?:в\s+)?([а-я-]+)", question):
        location = match[1]
        if location in {
            "есть", "нет", "для", "по", "с", "со", "и", "или", "нужен", "доступен", "деталь", "детали", "остаток",
        }:
            continue
        explicit = True
        if location == "санкт":
            second = re.match(r"\s+петербург(?:а|е|у|ом)?\b", question[match.end(1):])
            if second:
                location += second[0]
        requested.update(mentioned(location))
    if explicit:
        return warehouse in requested
    for name, pattern in _WAREHOUSE_NAMES.items():
        for match in re.finditer(rf"\b{pattern}\b", question):
            if not re.search(r"\b(?:не|кроме|без)\s+(?:(?:в|на|со?|из)\s+)?$", question[:match.start()]):
                requested.add(name)
    return not requested or warehouse in requested


def unsupported_erp_claims(text: str) -> bool:
    """Замечает утверждения о поиске поставщика в ERP, которого нет среди её возможностей."""
    for sentence in re.split(r"[.!?;\n]+", text):
        for clause in sentence.split(","):
            supplier = _SUPPLIER.search(clause) or (
                _SUPPLIER.search(sentence)
                and re.search(r"\b(?:его|эту информацию|эти сведения)\b", clause, re.IGNORECASE)
            )
            if not supplier or not re.search(r"\bERP\b", clause, re.IGNORECASE):
                continue
            if re.search(r"\b(?:нельзя|невозможно)\b", clause, re.IGNORECASE):
                continue
            for action in _ERP_LOOKUP.finditer(clause):
                if not re.search(
                    r"\bне\s+(?:(?:может|могут|позволяет|умеет|будет|нужно|надо|следует|стоит)\s+)?$",
                    clause[: action.start()],
                    re.IGNORECASE,
                ):
                    return True
    return False


def unsupported_facts(text: str, question: str, results: list[tuple[str, dict]]) -> list[str]:
    """Проверяет коды, версии деталей и числа; тип цены отделён от количества."""
    wrong = [code for code in find_codes(text) if code not in allowed_codes(question, results)]
    if any(word in question.lower() for word in ("поставщик", "поставля")):
        supplied_names = {name for _, relation, name in _knowledge_facts(results) if relation == "SUPPLIED_BY"}
        candidates = set(vocabulary()["Supplier"]) | supplied_names
        supported = supplier_names_for_answer(question, results, text)
        wrong.extend(name for name in sorted(candidates) if name.lower() in text.lower() and name not in supported)
    if any(word in question.lower() for word in ("поставщик", "поставля")) and any(
        word in question.lower() for word in ("актуальн", "текущ", "нынешн", "последн", "теперь", "сейчас")
    ):
        replacements = {}
        for _, result in results:
            for fact in result.get("facts", []):
                if not isinstance(fact, str):
                    continue
                match = re.fullmatch(r"(P-\d{4}[A-Z]?) -REPLACED_BY-> (P-\d{4}[A-Z]?)", fact)
                if match:
                    replacements[match[1]] = match[2]
        mentioned = set(find_codes(text))
        for part in mentioned & replacements.keys():
            current, visited = part, set()
            while current in replacements and current not in visited:
                visited.add(current)
                current = replacements[current]
            if current not in mentioned or current in replacements:
                wrong.append(part)
    prices = {_number(value) for value in _PRICE.findall(question)}
    quantities = {_number(value) for value in requested_quantities(question)}
    order_ids = set(_ORDER_ID.findall(question))
    for name, result in results:
        if name == "get_price" and isinstance(result.get("price_rub"), (int, float)):
            prices.add(_number(result["price_rub"]))
        if name in ("check_stock", "create_service_order") and type(result.get("qty")) is int:
            quantities.add(_number(result["qty"]))
        if name == "create_service_order" and result.get("order_id"):
            order_ids.add(result["order_id"])
    for pattern, allowed in (
        (_PRICE, prices),
        (_NAMED_PRICE, prices),
        (_STOCK, quantities),
        (_NAMED_STOCK, quantities),
    ):
        wrong.extend(value for value in pattern.findall(text) if _number(value) not in allowed)
    wrong.extend(value for value in _ORDER_ID.findall(text) if value not in order_ids)
    return list(dict.fromkeys(wrong))
