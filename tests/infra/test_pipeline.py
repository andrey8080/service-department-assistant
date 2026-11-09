import json

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from assistant.infra import api, cli, indexing
from assistant.infra.config import Settings
from assistant.infra.flat_rag import FlatRag
from assistant.infra.schemas import Answer, Chunk, Community, Entity, ExtractionResult, KnowledgeBase, Relation
from assistant.infra.testing import FakeLLM


@pytest.fixture()
def settings(tmp_path) -> Settings:
    corpus = tmp_path / "data" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "manual_AR-210.md").write_text(
        "# Руководство AR-210\n\nпусковое реле – деталь P-1042\n", encoding="utf-8"
    )
    (corpus / "bulletin_SB-2025-03.md").write_text("# Бюллетень\n\nP-1042 заменена на P-1042B\n", encoding="utf-8")
    (tmp_path / "data" / "vocabulary.json").write_text(
        json.dumps({"Component": ["пусковое реле"]}, ensure_ascii=False), encoding="utf-8"
    )
    question = {
        "id": "dev-01",
        "type": "local",
        "question": "Чем заменена деталь P-1042?",
        "key_facts": [["P-1042B"]],
        "source_groups": [["bulletin_SB-2025-03.md"]],
        "expected_calls": [],
        "expected_pending": None,
        "must_refuse": False,
    }
    (tmp_path / "data" / "questions_dev.jsonl").write_text(
        json.dumps(question, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (tmp_path / "data" / "gold_dev.json").write_text(
        json.dumps(
            {
                "bulletin_SB-2025-03.md": {
                    "entities": ["Part:P-1042", "Part:P-1042B"],
                    "relations": [["Part:P-1042", "REPLACED_BY", "Part:P-1042B"]],
                    "optional_relations": [],
                }
            }
        ),
        encoding="utf-8",
    )
    return Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")


def fake_extract(chunk: Chunk, llm) -> ExtractionResult:
    old, new = Entity("Part", "P-1042"), Entity("Part", "P-1042B")
    if "заменена" in chunk.text:
        return ExtractionResult(
            chunk.chunk_id, chunk.doc_id, [old, new], [Relation(old, "REPLACED_BY", new)], dropped=1
        )
    return ExtractionResult(chunk.chunk_id, chunk.doc_id, [old], [])


def fake_build_graph(results) -> nx.DiGraph:
    graph = nx.DiGraph(conflicts=0)
    for result in results:
        for entity in result.entities:
            graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[result.chunk_id])
        for relation in result.relations:
            graph.add_edge(
                relation.subject.node_id,
                relation.object.node_id,
                type=relation.type,
                weight=1,
                sources=[result.chunk_id],
            )
    return graph


@pytest.fixture()
def student_code(monkeypatch):
    from assistant.graphrag import communities, extraction, graph_builder

    monkeypatch.setattr(extraction, "extract_chunk", fake_extract)
    monkeypatch.setattr(graph_builder, "build_graph", fake_build_graph)
    monkeypatch.setattr(communities, "detect_communities", lambda graph: [Community(0, sorted(graph.nodes))])
    monkeypatch.setattr(communities, "summarize_community", lambda community, graph, llm: "сводка")


def test_flat_rag_answers_from_the_nearest_chunks_and_reports_their_documents():
    chunks = [Chunk("a.md#0", "a.md", "пусковое реле"), Chunk("b.md#0", "b.md", "компрессор")]
    llm = FakeLLM(["Деталь P-1042."])
    answer = FlatRag(chunks, llm).answer("пусковое реле", top_k=1)
    assert answer == Answer("Деталь P-1042.", sources=["a.md"])
    assert (
        "пусковое реле" in llm.requests[0]["messages"][-1]["content"]
        and "компрессор" not in llm.requests[0]["messages"][-1]["content"]
    )


def test_flat_rag_marks_an_honest_refusal():
    llm = FakeLLM(["В базе знаний не найдено."])
    assert FlatRag([Chunk("a.md#0", "a.md", "текст")], llm).answer("вопрос").refused


def test_build_index_runs_student_code_and_saves_artifacts(settings, student_code):
    kb = indexing.build_index(settings, FakeLLM([]), log=lambda line: None)
    assert set(kb.graph.nodes) == {"Part:P-1042", "Part:P-1042B"} and kb.communities[0].summary == "сводка"
    stats = json.loads((settings.artifacts_dir / "index_stats.json").read_text(encoding="utf-8"))
    assert (
        stats["chunks"] == 2
        and stats["nodes"] == 2
        and stats["edges"] == 1
        and stats["dropped"] == 1
        and stats["conflicts"] == 0
    )
    loaded = indexing.load_kb(settings)
    assert loaded.graph.edges["Part:P-1042", "Part:P-1042B"] == {
        "type": "REPLACED_BY",
        "weight": 1,
        "sources": ["bulletin_SB-2025-03.md#0"],
    }
    assert loaded.communities == kb.communities and loaded.vocabulary == {"Component": ["пусковое реле"]}


def test_load_kb_without_index_explains_what_to_run(settings):
    with pytest.raises(FileNotFoundError, match="make index"):
        indexing.load_kb(settings)


def test_eval_command_writes_the_report_for_the_baseline(settings, capsys):
    context = cli.Context(settings, FakeLLM(["Деталь заменена на P-1042B."]), erp=None)
    cli.main(["eval", "--mode", "baseline"], context)
    report = json.loads((settings.artifacts_dir / "eval_baseline.json").read_text(encoding="utf-8"))
    assert report["summary"]["accuracy"] == 1.0 and report["summary"]["mode"] == "baseline"
    assert "baseline" in capsys.readouterr().out


def test_eval_extraction_command_scores_student_extraction(settings, student_code):
    cli.main(["eval-extraction"], cli.Context(settings, FakeLLM([]), erp=None))
    report = json.loads((settings.artifacts_dir / "eval_extraction.json").read_text(encoding="utf-8"))
    assert report["relations"]["recall"] == 1.0 and report["documents"] == 1


def test_ask_command_prints_the_answer_and_sources(settings, capsys):
    cli.main(
        ["ask", "Чем заменена P-1042?", "--mode", "baseline"], cli.Context(settings, FakeLLM(["На P-1042B."]), erp=None)
    )
    out = capsys.readouterr().out
    assert "На P-1042B." in out and "bulletin_SB-2025-03.md" in out


def test_local_mode_uses_student_search(settings, student_code, monkeypatch):
    from assistant.graphrag import search

    monkeypatch.setattr(
        search, "local_search", lambda question, kb, llm: Answer("P-1042B", sources=["bulletin_SB-2025-03.md"])
    )
    context = cli.Context(settings, FakeLLM([]), erp=None)
    cli.main(["index"], context)
    cli.main(["eval", "--mode", "local"], context)
    report = json.loads((settings.artifacts_dir / "eval_local.json").read_text(encoding="utf-8"))
    assert report["summary"]["accuracy"] == 1.0 and report["summary"]["source_recall"] == 1.0


def test_http_api_answers_through_the_same_runner(settings, monkeypatch):
    monkeypatch.setattr(api, "make_context", lambda: cli.Context(settings, FakeLLM(["На P-1042B."]), erp=None))
    client = TestClient(api.create_app())
    assert client.get("/health").json() == {"status": "ok"}
    reply = client.post("/ask", json={"question": "Чем заменена P-1042?", "mode": "baseline"}).json()
    assert reply["answer"] == "На P-1042B." and reply["sources"] and reply["refused"] is False
    assert client.post("/ask", json={"question": "?", "mode": "telepathy"}).status_code == 422


def test_http_api_explains_a_missing_index_and_unfinished_code(settings, monkeypatch):
    monkeypatch.setattr(api, "make_context", lambda: cli.Context(settings, FakeLLM([]), erp=None))
    client = TestClient(api.create_app())
    missing = client.post("/ask", json={"question": "Чем заменена P-1042?", "mode": "local"})
    assert missing.status_code == 409 and "make index" in missing.json()["detail"]
    indexing.save_kb(KnowledgeBase(nx.DiGraph(), [], {}), settings)
    from assistant.graphrag import search

    def unfinished_search(question, kb, llm):
        raise NotImplementedError("Реализуйте local_search")

    monkeypatch.setattr(search, "local_search", unfinished_search)
    unfinished = client.post("/ask", json={"question": "Чем заменена P-1042?", "mode": "local"})
    assert unfinished.status_code == 501 and "local_search" in unfinished.json()["detail"]


def test_command_line_reports_a_missing_index_without_a_traceback(settings):
    with pytest.raises(SystemExit, match="make index"):
        cli.main(["eval", "--mode", "local"], cli.Context(settings, FakeLLM([]), erp=None))
