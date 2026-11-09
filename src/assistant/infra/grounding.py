"""Привязка к фактам: коды и названия, которые можно проверить без модели.

Код – это модель оборудования (AR-210), деталь (P-1042B), ошибка (E-17) или регламент (SP-104).
Название – линейка, узел или поставщик из закрытого словаря data/vocabulary.json.
"""

import json
import re
from pathlib import Path

# Кириллические буквы, которые в небрежной записи подменяют латинские.
LOOKALIKES = str.maketrans("АВЕКМНОРСТХ", "ABEKMHOPCTX")

# Префикс кода -> (число цифр, тип сущности).
KINDS = {
    "AR": (3, "Model"),
    "BZ": (3, "Model"),
    "GS": (3, "Model"),
    "P": (4, "Part"),
    "E": (2, "ErrorCode"),
    "SP": (3, "Procedure"),
}
CODE_TYPES = {"Model", "Part", "ErrorCode", "Procedure"}

_LETTER = "A-Za-zА-Яа-яЁё"
_CODE = re.compile(rf"(?<![0-9{_LETTER}])([{_LETTER}]{{1,2}})[- ]?(\d{{2,4}})([{_LETTER}]?)(?![0-9{_LETTER}])")


def _canonical(letters: str, digits: str, suffix: str) -> str | None:
    letters = letters.upper().translate(LOOKALIKES)
    suffix = suffix.upper().translate(LOOKALIKES)
    spec = KINDS.get(letters)
    if spec is None or len(digits) != spec[0]:
        return None
    if suffix and (spec[1] != "Part" or not suffix.isascii()):
        return None
    return f"{letters}-{digits}{suffix}"


def normalize_code(raw: str) -> str | None:
    """Приводит запись кода к каноническому виду. Возвращает None, если это не код."""
    match = _CODE.fullmatch(raw.strip())
    return _canonical(*match.groups()) if match else None


def find_codes(text: str) -> list[str]:
    """Все коды из текста в каноническом виде, без повторов, в порядке появления."""
    found: list[str] = []
    for match in _CODE.finditer(text):
        code = _canonical(*match.groups())
        if code and code not in found:
            found.append(code)
    return found


def code_type(code: str) -> str:
    """Тип сущности по каноническому коду: Model, Part, ErrorCode или Procedure."""
    return KINDS[code.split("-")[0]][1]


def load_vocabulary(path: Path) -> dict[str, list[str]]:
    """Словарь названий: {"ProductLine": [...], "Component": [...], "Supplier": [...]}."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_names(text: str, vocabulary: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Названия из словаря, которые встречаются в тексте: список пар (тип, название).

    Сравнение без учёта регистра. Название ищется в том виде, как оно записано в словаре,
    то есть в именительном падеже.
    """
    lowered = text.lower()
    return [(kind, name) for kind, names in vocabulary.items() for name in names if name.lower() in lowered]


def canonical_name(entity_type: str, raw: str, vocabulary: dict[str, list[str]]) -> str | None:
    """Каноническое имя сущности или None, если такой сущности быть не может.

    Для кодов проверяет вид кода и его тип. Для названий ищет словарное название внутри строки,
    поэтому «ООО «ХолодКомплект»» превращается в «ХолодКомплект».
    """
    if entity_type in CODE_TYPES:
        code = normalize_code(raw)
        return code if code and code_type(code) == entity_type else None
    lowered = raw.lower()
    for name in vocabulary.get(entity_type, []):
        if name.lower() in lowered:
            return name
    return None
