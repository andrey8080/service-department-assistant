"""Проверка кодов и чисел по вопросу и результатам инструментов."""

import json
import re
from decimal import Decimal

from assistant.infra.grounding import find_codes

_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?(?![\w-])"
_PRICE = re.compile(rf"({_NUMBER})\s*(?:руб\w*|₽|р\.)", re.IGNORECASE)
_STOCK = re.compile(rf"({_NUMBER})\s*(?:шт\w*|единиц\w*)", re.IGNORECASE)
_NAMED_PRICE = re.compile(rf"(?:цен\w*|стоимост\w*|стоит)\s*[:=]?\s*({_NUMBER})", re.IGNORECASE)
_NAMED_STOCK = re.compile(rf"(?:остаток|в наличии|на складе)\s*[:=]?\s*({_NUMBER})", re.IGNORECASE)
_ORDER_ID = re.compile(r"\bSO-\d+\b", re.IGNORECASE)
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


def unsupported_facts(text: str, question: str, results: list[tuple[str, dict]]) -> list[str]:
    """Проверяет коды и числовые утверждения; тип цены отделён от количества."""
    wrong = [code for code in find_codes(text) if code not in allowed_codes(question, results)]
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
