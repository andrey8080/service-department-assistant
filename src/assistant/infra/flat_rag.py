"""Плоский RAG: точка отсчёта, с которой сравнивается граф."""

from assistant.infra.schemas import Answer, Chunk
from assistant.infra.vector_index import VectorIndex

SYSTEM = (
    "Ты ассистент сервисной службы компании «ПолярТех». Отвечай на вопрос только по фрагментам документов ниже. "
    "Коды моделей, деталей, ошибок и регламентов пиши точно как в тексте. "
    "Если ответа во фрагментах нет, напиши: «в базе знаний не найдено»."
)


class FlatRag:
    def __init__(self, chunks: list[Chunk], llm) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._llm = llm
        self._index = VectorIndex(list(self._chunks), llm.embed([chunk.text for chunk in chunks]))

    def answer(self, question: str, top_k: int = 4) -> Answer:
        found = [self._chunks[chunk_id] for chunk_id in self._index.search(self._llm.embed([question])[0], top_k)]
        context = "\n\n".join(f"[{chunk.doc_id}]\n{chunk.text}" for chunk in found)
        reply = self._llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{context}\n\nВопрос: {question}"}]
        )
        sources = list(dict.fromkeys(chunk.doc_id for chunk in found))
        return Answer(reply.content, sources=sources, refused="не найдено" in reply.content.lower())
