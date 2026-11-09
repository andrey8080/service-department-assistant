"""Подставная модель для тестов: отвечает по заранее заданному сценарию и не требует сети."""

import hashlib

from assistant.infra.llm import LLMResponse, LLMStats


class FakeLLM:
    """Заменяет LLMClient в тестах.

    script – ответы в порядке вызовов `chat`. Строка превращается в LLMResponse с этим текстом.
    Все запросы сохраняются в `requests`, по ним тест проверяет, что именно спросили у модели.
    """

    def __init__(self, script: list[LLMResponse | str]) -> None:
        self._script = list(script)
        self.requests: list[dict] = []
        self.stats = LLMStats()

    def chat(self, messages, *, tools=None, json_schema=None, max_tokens: int = 1024) -> LLMResponse:
        self.requests.append({"messages": list(messages), "tools": tools, "json_schema": json_schema})
        assert self._script, f"сценарий FakeLLM закончился на вызове номер {len(self.requests)}"
        item = self._script.pop(0)
        self.stats.calls += 1
        return LLMResponse(item) if isinstance(item, str) else item

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[byte / 255 for byte in hashlib.sha256(text.encode("utf-8")).digest()[:16]] for text in texts]
