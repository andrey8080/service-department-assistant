"""Настройки из переменных окружения. Значения по умолчанию подходят для запуска без контейнеров."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from assistant.infra.erp_client import ErpClient
from assistant.infra.grounding import load_vocabulary
from assistant.infra.llm import LLMClient


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    embed_model: str = "bge-m3"
    llm_api_key: str = "ollama"
    erp_url: str = "http://localhost:8100"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        defaults = cls()
        return cls(
            llm_base_url=env.get("LLM_BASE_URL", defaults.llm_base_url),
            llm_model=env.get("LLM_MODEL", defaults.llm_model),
            embed_model=env.get("EMBED_MODEL", defaults.embed_model),
            llm_api_key=env.get("LLM_API_KEY", defaults.llm_api_key),
            erp_url=env.get("ERP_URL", defaults.erp_url),
            data_dir=Path(env.get("DATA_DIR", defaults.data_dir)),
            artifacts_dir=Path(env.get("ARTIFACTS_DIR", defaults.artifacts_dir)),
        )


@lru_cache
def vocabulary() -> dict[str, list[str]]:
    """Словарь названий из data/vocabulary.json. Нужен там, где нет доступа к KnowledgeBase."""
    return load_vocabulary(Settings.from_env().data_dir / "vocabulary.json")


def make_llm(settings: Settings) -> LLMClient:
    return LLMClient(
        settings.llm_base_url,
        settings.llm_model,
        settings.embed_model,
        api_key=settings.llm_api_key,
        cache_dir=settings.artifacts_dir / "llm_cache",
    )


def make_erp(settings: Settings) -> ErpClient:
    return ErpClient(settings.erp_url)
