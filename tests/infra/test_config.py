from pathlib import Path

from assistant.infra.config import Settings


def test_defaults_point_to_the_local_services():
    settings = Settings.from_env({})
    assert settings.llm_model == "qwen3:4b-instruct-2507-q4_K_M" and settings.embed_model == "bge-m3"
    assert settings.llm_base_url == "http://localhost:11434/v1" and settings.erp_url == "http://localhost:8100"
    assert settings.data_dir == Path("data") and settings.artifacts_dir == Path("artifacts")


def test_environment_overrides_defaults():
    settings = Settings.from_env(
        {"LLM_MODEL": "t-tech/T-lite-it-2.1:q4_K_M", "ERP_URL": "http://erp:8100", "DATA_DIR": "/srv/data"}
    )
    assert settings.llm_model == "t-tech/T-lite-it-2.1:q4_K_M" and settings.erp_url == "http://erp:8100"
    assert settings.data_dir == Path("/srv/data")


def test_vocabulary_is_read_from_the_data_directory(tmp_path, monkeypatch):
    from assistant.infra import config

    (tmp_path / "vocabulary.json").write_text('{"Component": ["пусковое реле"]}', encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.vocabulary.cache_clear()
    assert config.vocabulary() == {"Component": ["пусковое реле"]}
    config.vocabulary.cache_clear()
