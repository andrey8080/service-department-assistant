from pathlib import Path

from assistant.infra.chunking import chunk_document, load_chunks

MANUAL = """# Руководство: витрина AR-210

Витрина AR-210 входит в линейку «Арктика».

## Состав узлов

В AR-210 установлены следующие узлы.

- пусковое реле – деталь P-1042

## Коды ошибок

- E-17 – нет пускового тока
"""


def test_short_document_is_one_chunk():
    chunks = chunk_document("report_R-2025-001.md", "# Отчёт о выезде\n\nОборудование: витрина AR-210.\n")
    assert [c.chunk_id for c in chunks] == ["report_R-2025-001.md#0"]
    assert chunks[0].doc_id == "report_R-2025-001.md" and "AR-210" in chunks[0].text


def test_long_document_is_split_by_sections_and_every_chunk_keeps_the_title():
    chunks = chunk_document("manual_AR-210.md", MANUAL, max_chars=120)
    assert [c.chunk_id for c in chunks] == ["manual_AR-210.md#0", "manual_AR-210.md#1", "manual_AR-210.md#2"]
    assert all(c.text.startswith("# Руководство: витрина AR-210") for c in chunks)
    assert "P-1042" in chunks[1].text and "E-17" in chunks[2].text


def test_sections_are_packed_together_while_they_fit():
    assert len(chunk_document("manual_AR-210.md", MANUAL, max_chars=5000)) == 1


def test_no_text_is_lost():
    chunks = chunk_document("manual_AR-210.md", MANUAL, max_chars=120)
    joined = "\n".join(c.text for c in chunks)
    for line in MANUAL.splitlines():
        assert line in joined


def test_load_chunks_reads_documents_in_name_order(tmp_path: Path):
    (tmp_path / "b.md").write_text("# Б\n\nтекст\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# А\n\nтекст\n", encoding="utf-8")
    assert [c.chunk_id for c in load_chunks(tmp_path)] == ["a.md#0", "b.md#0"]
