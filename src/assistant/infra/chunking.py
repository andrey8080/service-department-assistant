"""Разбиение документов корпуса на чанки."""

from pathlib import Path

from assistant.infra.schemas import Chunk

MAX_CHARS = 1000


def chunk_document(doc_id: str, text: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Делит документ по заголовкам второго уровня и склеивает соседние разделы, пока они помещаются.

    Каждый чанк начинается с заголовка документа: без него из раздела «Состав узлов» не понять,
    о какой модели речь.
    """
    lines = text.strip().splitlines()
    title = lines[0] if lines and lines[0].startswith("# ") else ""
    sections: list[str] = []
    current: list[str] = []
    for line in lines[1:] if title else lines:
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    packed: list[str] = []
    for section in sections:
        if packed and len(packed[-1]) + len(section) + 2 <= max_chars - len(title):
            packed[-1] = f"{packed[-1]}\n\n{section}"
        else:
            packed.append(section)
    bodies = packed or [""]
    return [Chunk(f"{doc_id}#{i}", doc_id, f"{title}\n\n{body}".strip()) for i, body in enumerate(bodies)]


def load_chunks(corpus_dir: Path, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Все чанки корпуса. Документы читаются в алфавитном порядке имён файлов."""
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        chunks += chunk_document(path.name, path.read_text(encoding="utf-8"), max_chars)
    return chunks
