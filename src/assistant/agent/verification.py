"""Проверка номера детали и количества перед вызовом инструмента."""

import json
import re

from assistant.infra.grounding import find_codes

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


def requested_quantities(question: str) -> set[int]:
    """Числа рядом с количеством/единицами или глаголом заказа; цифры кодов не подходят."""
    found = set()
    number = r"(?<![\w-])([1-9]\d*)(?![\w-])"
    patterns = [
        rf"{number}\s*(?:шт\w*|единиц\w*)",
        rf"(?:количеств\w*|закаж\w*|заказ(?:ать)?)\s*[:=]?\s*{number}",
    ]
    for pattern in patterns:
        found.update(int(value) for value in re.findall(pattern, question, re.IGNORECASE))
    for word, count in _WORD_COUNTS.items():
        if re.search(rf"\b{word}\s+(?:шт\w*|единиц\w*)", question, re.IGNORECASE):
            found.add(count)
        if re.search(rf"\bзакаж\w*\s+{word}\b", question, re.IGNORECASE):
            found.add(count)
    return found


def allowed_codes(question: str, results: list[tuple[str, dict]]) -> set[str]:
    evidence = question + json.dumps([result for _, result in results], ensure_ascii=False)
    return set(find_codes(evidence))
