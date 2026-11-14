"""Подсчёты и сравнения по проверенным записям сводок."""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from assistant.infra.grounding import code_type, find_codes, find_names


@dataclass
class Overview:
    rows: list[str] = field(default_factory=list)
    edges: set[tuple[str, str]] = field(default_factory=set)
    complete: bool = True
    reason: str = ""


def _incomplete(reason):
    return Overview(complete=False, reason=reason)


def _failure_count(number):
    ending = number % 100
    form = "отказов" if 11 <= ending <= 14 else (
        "отказ" if ending % 10 == 1 else "отказа" if 2 <= ending % 10 <= 4 else "отказов"
    )
    return f"{number} {form}"


def _scope(question, vocabulary):
    names = set(find_names(question, vocabulary))
    for kind in ("ProductLine", "Component"):
        for name in vocabulary.get(kind, []):
            words = []
            for word in name.lower().split():
                if word.endswith(("ое", "ый", "ий")):
                    pattern = re.escape(word[:-2]) + (
                        r"(?:ое|ый|ий|ого|ому|ым|ом|ые|ых|ыми|ая|ую|ой|его|ему|им|ем|ие|их)"
                    )
                elif word.endswith(("а", "я", "ы")) and len(word) > 3:
                    pattern = re.escape(word[:-1]) + r"(?:а|я|ы|и|е|у|ю|ой|ою|ей|ям|ями|ях|ам|ами|ах)"
                elif word[-1:] not in "аеёиоуыэюяь" and len(word) > 3:
                    pattern = re.escape(word) + r"(?:а|у|ом|е|ы|и|ов|ам|ами|ах)?"
                elif word.endswith("ь") and len(word) > 3:
                    pattern = re.escape(word[:-1]) + r"(?:ь|я|ю|ем|е|и|ей|ям|ями|ях)"
                else:
                    pattern = re.escape(word)
                words.append(pattern)
            if re.search(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", question, re.I):
                names.add((kind, name))
    return (
        {name for kind, name in names if kind == "ProductLine"},
        {code for code in find_codes(question) if code_type(code) == "Model"},
        {name for kind, name in names if kind == "Component"},
    )


def _models(records, lines, requested):
    known = defaultdict(set)
    for row in records:
        if row["kind"] == "line_coverage":
            known[row["product_line"]].update(row["models"])
    if lines and not lines <= known.keys():
        return None
    covered = set().union(*(known[line] for line in lines or known)) if known else set()
    if requested:
        return requested if not lines or requested <= covered else None
    return covered or None


def _unique(records, kind, keys, value):
    selected = {}
    for row in records:
        if row["kind"] != kind:
            continue
        key = tuple(row[name] for name in keys)
        if key in selected and selected[key][value] != row[value]:
            return None
        if key in selected:
            row = {**row, "edges": row["edges"] | selected[key]["edges"]}
        selected[key] = row
    return selected


def _failures(question, records, lines, models, components):
    selected = _models(records, lines, models)
    if not selected:
        return _incomplete("Не подтверждён полный список моделей выбранной области.")
    counts = _unique(records, "failure", ("model", "component"), "count")
    coverage = _unique(records, "coverage", ("model",), "failure_total")
    if counts is None or coverage is None:
        return _incomplete("Противоречивые записи об отказах.")
    for model in selected:
        row = coverage.get((model,))
        failures = [r for r in counts.values() if r["model"] == model]
        if row is None or len(failures) != row["failure_count"] or sum(
            r["count"] for r in failures
        ) != row["failure_total"]:
            return _incomplete("В сводках пропущена часть распределения отказов.")
    candidates = [row for row in counts.values() if row["model"] in selected]
    text = question.lower()
    if components:
        candidates = [row for row in candidates if row["component"] in components]
    proof = {edge for row in candidates for edge in row["edges"]}
    proof.update(edge for row in records if row["kind"] == "line_coverage"
                 and (not lines or row["product_line"] in lines) for edge in row["edges"]
                 if edge[0].removeprefix("Model:") in selected)
    rare = bool(re.search(r"реже|редк|наимен|меньше\s+всего|миним", text))
    common = bool(re.search(r"чаще|частот|част[ыао]\w*|наибольш|больше\s+всего|максим", text))
    rank = re.search(r"(?:топ[ -]?|первые?\s+)(\d+|три|два|пять)", text)
    if rank is None:
        rank = re.search(r"(\d+|три|два|пять)\s+(?:самых|наиболее|лидирующ)", text)
    if rank is None:
        rank = re.search(r"(тройк\w*|двойк\w*|пят[её]рк\w*)", text)
    ranked = rare or common or rank is not None
    if ranked:
        by_model = bool(re.search(r"(?:какая|какой|каких|какие|у\s+какой)\s+модел|модел[ьи]\s+с\s+", text))
        by_line = bool(re.search(r"(?:какая|какой|каких|какие|у\s+какой)\s+линейк", text))
        if rank:
            suffix = text[rank.end():]
            by_model = by_model or bool(re.match(r"\s+(?:самых\s+\w+\s+)?модел", suffix))
            by_line = by_line or bool(re.match(r"\s+(?:самых\s+\w+\s+)?линейк", suffix))
        dimension = "product_line" if by_line else "model" if by_model else "component"
        groups = defaultdict(int)
        members = defaultdict(set)
        for row in candidates:
            keys = ([name for name in row["product_lines"] if not lines or name in lines]
                    if by_line else [row[dimension]])
            for key in keys:
                groups[key] += row["count"]
                members[key].add(row["model"])
        if not groups:
            return _incomplete("Нет зарегистрированных отказов для сравнения.")
        ordered = sorted(groups, key=lambda key: (groups[key] if rare else -groups[key], key))
        limit = 1
        if rank:
            word = rank[1]
            limit = int(word) if word.isdigit() else next(
                (number for prefix, number in (("три", 3), ("тройк", 3), ("два", 2), ("двойк", 2),
                                              ("пять", 5), ("пятёрк", 5), ("пятерк", 5))
                 if word.startswith(prefix)), 1,
            )
        distinct_places = rank and (
            re.match(r"\s+(?:различн\w*\s+)?мест", text[rank.end():])
            or re.search(r"различн\w*\s+(?:мест|значен)", text)
        )
        if distinct_places:
            boundary = sorted(set(groups.values()), reverse=not rare)[min(limit, len(set(groups.values()))) - 1]
        else:
            boundary = groups[ordered[min(limit, len(ordered)) - 1]]
        winners = [key for key in ordered if groups[key] <= boundary] if rare else [
            key for key in ordered if groups[key] >= boundary
        ]
        rows = [f"{key}: {_failure_count(groups[key])}" + (
            f"; модели: {', '.join(sorted(members[key]))}" if dimension == "component" and rare else ""
        ) + "." for key in winners]
        return Overview(rows, proof)
    if re.search(r"отч[её]т|выезд", text) and not models and not components:
        totals = defaultdict(int)
        for model in selected:
            row = coverage[(model,)]
            for line in row["product_lines"]:
                if not lines or line in lines:
                    totals[line] += row["failure_total"]
        rows = [f"{line}: {total} отчётов." for line, total in sorted(totals.items())]
        if re.search(r"всего|общее|сумм", text):
            rows.append(f"Общее число отказов: {sum(row['count'] for row in candidates)}.")
        return Overview(rows, proof)
    rows = []
    for model in sorted(selected):
        items = sorted((row for row in candidates if row["model"] == model), key=lambda row: row["component"])
        details = "; ".join(f"{row['component']}: {row['count']}" for row in items)
        total = sum(row["count"] for row in items)
        rows.append(f"{model} — {details or 'зарегистрированных отказов нет'}; всего: {total}.")
    if len(selected) > 1 and (not components or re.search(r"всего|общее|сумм", text)):
        rows.append(f"Общее число отказов: {sum(row['count'] for row in candidates)}.")
    if components and len(selected) > 1 and re.search(r"одинаков|больше|меньше", text):
        totals = {model: sum(row["count"] for row in candidates if row["model"] == model) for model in selected}
        if len(set(totals.values())) == 1:
            rows.append(f"Во всех выбранных моделях одинаковое число случаев: {next(iter(totals.values()))}.")
        else:
            maximum = max(totals.values())
            winners = ", ".join(sorted(model for model, total in totals.items() if total == maximum))
            rows.append(f"Больше всего случаев: {winners} — {maximum}.")
    return Overview(rows, proof)


def _inventory(records, selected, components, removed=False, versions=None):
    rows = {}
    for row in records:
        if row["kind"] == "inventory" and row["model"] in selected:
            rows[(row["model"], row["component"], row["part"])] = row
    coverage = _unique(records, "coverage", ("model",), "parts_count")
    if coverage is None:
        return None
    for model in selected:
        items = [row for row in rows.values() if row["model"] == model]
        expected = coverage.get((model,), {})
        complete = len({row["part"] for row in items}) == expected.get("parts_count")
        if removed and not components and "replaced_parts_count" in expected:
            replaced = {
                row["part"] for row in items
                if (row["part"],) in versions and versions[(row["part"],)]["current"] != row["part"]
            }
            complete = len(replaced) == expected["replaced_parts_count"]
        elif removed and any(
            (row["part"],) not in versions for row in items if not components or row["component"] in components
        ):
            return None
        if not complete and not (components and all(
            any(row["component"] == component and row.get("complete", False) for row in items)
            for component in components
        )):
            return None
    return [row for row in rows.values() if (not components or row["component"] in components) and (
        not removed or (row["part"],) in versions and versions[(row["part"],)]["current"] != row["part"]
    )]


def _catalog(question, records, lines, models, components):
    selected = _models(records, lines, models)
    if not selected:
        return _incomplete("Не подтверждён список моделей каталога.")
    text = question.lower()
    removed = bool(re.search(r"снят\w*\s+с\s+производств|не\s+выпуска", text))
    versions = _unique(records, "version", ("part",), "current")
    if versions is None:
        return _incomplete("Неоднозначная цепочка замен.")
    inventory = _inventory(records, selected, components, removed, versions)
    if inventory is None:
        return _incomplete("Неполная комплектация выбранных моделей.")
    difference = len(lines) == 2 and bool(re.search(r"только|различ|отлич|общи[ех]|общее|одинаков", text))
    if difference:
        sets = {line: set() for line in lines}
        for row in inventory:
            for line in row["product_lines"]:
                if line in sets:
                    sets[line].add(row["component"])
        a, b = sorted(lines)
        proof = {edge for row in inventory for edge in row["edges"]}
        if re.search(r"общи[ех]|общее|одинаков", text) and not re.search(r"только|отлич|различ", text):
            return Overview([f"Общие узлы: {', '.join(sorted(sets[a] & sets[b])) or 'нет'}."], proof)
        return Overview([
            f"Только {a}: {', '.join(sorted(sets[a] - sets[b])) or 'нет'}.",
            f"Только {b}: {', '.join(sorted(sets[b] - sets[a])) or 'нет'}.",
        ], proof)
    current = removed or bool(re.search(r"актуальн|текущ|нынешн|сейчас|замен|конечн", text))
    supplier = bool(re.search(r"поставщик|поставля", text))
    both_suppliers = supplier and current and bool(re.search(
        r"(?:исходн|изначальн)\w*\s+поставщик|поставщик\w*\s+(?:исходн|изначальн)", text,
    ))
    current_only = not re.search(r"исходн|изначальн|руководств|цепочк", text) or bool(re.search(
        r"исходн\w*\s+(?:номера|артикулы)\s+не\s+нуж|только\s+(?:текущ|актуальн)", text,
    ))
    suppliers = _unique(records, "supplier", ("part",), "supplier")
    if versions is None or suppliers is None:
        return _incomplete("Неоднозначная цепочка замен или поставщик.")
    proof, rows, removed_parts = set(), [], set()
    for row in sorted(inventory, key=lambda row: (row["model"], row["component"], row["part"])):
        part = row["part"]
        version = versions.get((part,))
        if current and version is None:
            return _incomplete("В сводке нет подтверждения конца цепочки замен.")
        terminal = version["current"] if current else part
        if removed and terminal == part:
            continue
        vendor = suppliers.get((terminal,))
        if supplier and vendor is None:
            return _incomplete("Не подтверждён поставщик запрошенной версии детали.")
        original_vendor = suppliers.get((part,)) if both_suppliers else None
        if both_suppliers and original_vendor is None:
            return _incomplete("Не подтверждён поставщик исходной детали.")
        proof.update(row["edges"])
        if current:
            proof.update(version["edges"])
        if supplier:
            proof.update(vendor["edges"])
        if both_suppliers:
            proof.update(original_vendor["edges"])
        if removed:
            removed_parts.add(part)
            continue
        detail = part
        if current:
            chain = version.get("path", [part, terminal] if part != terminal else [part])
            detail = terminal if current_only else " → ".join(chain) + (" (без замены)" if terminal == part else "")
        if supplier:
            detail += f"; поставщик {vendor['supplier']}"
        if both_suppliers:
            detail += f"; исходный поставщик {part}: {original_vendor['supplier']}"
        rows.append(f"{row['model']} — {row['component']}: {detail}.")
    if removed:
        proof.update(edge for row in records if row["kind"] == "coverage" and row["model"] in selected
                     for edge in row["edges"] if edge[1].startswith(("Part:", "ProductLine:")))
        rows = [f"Снятые исходные детали: {', '.join(sorted(removed_parts)) or 'не обнаружены'}. "]
    return Overview(rows, proof)


def overview(question: str, records: list[dict], vocabulary: dict[str, list[str]]) -> Overview | None:
    """Возвращает проверяемые строки для reduce; неизвестный тип вопроса оставляет обычному поиску."""
    lines, models, components = _scope(question, vocabulary)
    text = question.lower()
    explicit = re.findall(r'(линейк\w*|узел|узла|узлу|узлом|узлы|узлов)\s*[«"]([^»"]+)[»"]', question, re.I)
    explicit += [("линейка", name) for name in re.findall(r'линейк\w*\s+([А-ЯЁ][а-яё]+)', question)]
    for kind, name in explicit:
        recognized = _scope(name, vocabulary)
        if not recognized[0 if kind.lower().startswith("линейк") else 2]:
            return _incomplete("Явно указанная линейка или узел отсутствуют в словаре.")
    if any(code_type(code) in {"Part", "ErrorCode", "Procedure"} for code in find_codes(question)):
        return None
    if re.search(r"цен[аыуе]|стоим|стоит|остат|склад|достав|гаранти|прошив|серийн|телефон|контакт", text):
        return None
    if re.search(r"отказ|сбо[йяеивё]|отч[её]т|выезд", text):
        if re.search(
            r"артикул|поставщик|поставля|комплектац|замен|снят|"
            r"(?:номер|код)\w*\s+(?:детал|запчаст)|"
            r"(?:исходн|актуальн|текущ|нынешн)\w*\s+(?:детал|номер|код)|"
            r"номер\w*\s+в\s+(?:исходн\w*\s+)?руководств|"
            r"перечисл\w*\s+(?:детал|запчаст)|каталожн\w*\s+номер",
            text,
        ):
            # Распределение отказов не покрывает отдельно запрошенные поля каталога.
            return None
        if re.search(r"почему|причин|время|длитель|месяц|процент|вероятност|ни\s+разу|без\s+отказ", text):
            return None
        if not lines and not models:
            lines = set(vocabulary.get("ProductLine", []))
        return _failures(question, records, lines, models, components)
    catalog = components or re.search(r"уз(?:ел|лы|лов|ла)|компонент|детал|артикул|поставщик|поставля|комплектац", text)
    if catalog and (lines or len(models) > 1):
        return _catalog(question, records, lines, models, components)
    return None
