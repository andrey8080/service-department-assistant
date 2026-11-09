"""Контракт FR-3, FR-4, FR-5 и FR-10: сообщества, сводки и два режима поиска."""

import networkx as nx

from assistant.graphrag.communities import detect_communities, summarize_community
from assistant.graphrag.search import global_search, local_search
from assistant.infra.schemas import Answer, Community
from assistant.infra.testing import FakeLLM


def said_to_model(llm: FakeLLM, index: int = -1) -> str:
    return " ".join(str(message["content"]) for message in llm.requests[index]["messages"])


def two_triangles() -> nx.DiGraph:
    graph = nx.DiGraph()
    for group in (["Part:P-1001", "Part:P-1002", "Part:P-1003"], ["Part:P-2001", "Part:P-2002", "Part:P-2003"]):
        for node in group:
            graph.add_node(node, type="Part", name=node.split(":")[1], sources=[])
        for a, b in zip(group, group[1:] + group[:1], strict=True):
            graph.add_edge(a, b, type="REPLACED_BY", weight=1, sources=[])
    return graph


def test_every_node_is_in_exactly_one_community(graph):
    communities = detect_communities(graph)
    members = [node for community in communities for node in community.nodes]
    assert sorted(members) == sorted(graph.nodes)
    assert [community.community_id for community in communities] == list(range(len(communities)))


def test_disconnected_groups_are_separate_communities():
    communities = detect_communities(two_triangles())
    assert sorted(sorted(community.nodes) for community in communities) == [
        ["Part:P-1001", "Part:P-1002", "Part:P-1003"],
        ["Part:P-2001", "Part:P-2002", "Part:P-2003"],
    ]


def test_partition_is_reproducible(graph):
    first = [sorted(community.nodes) for community in detect_communities(graph)]
    second = [sorted(community.nodes) for community in detect_communities(graph)]
    assert first == second


def test_summary_is_written_by_one_call_and_is_at_most_150_words(graph):
    community = Community(0, sorted(graph.nodes))
    llm = FakeLLM([" ".join(["слово"] * 400)])
    summary = summarize_community(community, graph, llm)
    assert len(llm.requests) == 1 and 0 < len(summary.split()) <= 150


def test_numbers_for_the_summary_are_counted_by_code_and_given_to_the_model(graph):
    llm = FakeLLM(["Чаще всего отказывает пусковое реле."])
    summarize_community(Community(0, sorted(graph.nodes)), graph, llm)
    prompt = said_to_model(llm)
    assert "37" in prompt and "пусковое реле" in prompt  # вес связи FAILED из графа


def test_local_search_gives_the_model_only_facts_on_paths_between_anchors(kb):
    llm = FakeLLM(["Закажите P-1042B: P-1042 снята с производства."])
    answer = local_search("Какую деталь заказать при ошибке E-17 на AR-210?", kb, llm)
    prompt = said_to_model(llm)
    assert len(llm.requests) == 1
    assert "P-1042" in prompt and "P-1042B" in prompt  # путь и цепочка замен
    assert "P-1142" not in prompt  # реле другой модели на пути не лежит
    assert isinstance(answer, Answer) and answer.text == "Закажите P-1042B: P-1042 снята с производства."
    assert not answer.refused and "bulletin_SB-2025-03.md" in answer.sources
    assert all("#" not in source for source in answer.sources)  # документы, а не чанки


def test_local_search_with_one_anchor_reads_its_neighbourhood(kb):
    llm = FakeLLM(["Поставщик актуальной детали P-1042B – ХолодКомплект."])
    answer = local_search("Кто поставляет замену для детали P-1042?", kb, llm)
    prompt = said_to_model(llm)
    assert "P-1042B" in prompt and "ХолодКомплект" in prompt  # глубина две связи
    assert not answer.refused


def test_local_search_finds_anchors_written_sloppily_and_by_vocabulary_name(kb):
    llm = FakeLLM(["Деталь P-1042 заменена на P-1042B."])
    local_search("какая деталь стоит как «пусковое реле» в ar 210?", kb, llm)
    prompt = said_to_model(llm)
    assert "P-1042" in prompt and "P-1142" not in prompt


def test_local_search_refuses_without_calling_the_model_when_a_code_is_unknown(kb):
    for question in ("Что означает ошибка E-99 на AR-210?", "Какую деталь заказать при ошибке E-17 на AR-999?"):
        llm = FakeLLM([])
        answer = local_search(question, kb, llm)
        assert answer.refused and llm.requests == [], question


def test_local_search_refuses_when_the_question_has_no_anchors(kb):
    llm = FakeLLM([])
    assert local_search("Как дела у сервисной службы?", kb, llm).refused and llm.requests == []


def test_global_search_asks_every_summary_then_merges_useful_answers(kb):
    llm = FakeLLM(
        [
            '{"answer": "Чаще всего отказывает пусковое реле.", "score": 8}',
            '{"answer": "БЕСПОЛЕЗНЫЙ-ЧАСТИЧНЫЙ-ОТВЕТ", "score": 0}',
            "В линейке «Арктика» чаще всего отказывает пусковое реле.",
        ]
    )
    answer = global_search("Какой узел чаще всего отказывает в линейке «Арктика»?", kb, llm)
    assert len(llm.requests) == 3
    assert all(request["json_schema"] is not None for request in llm.requests[:2])
    assert "Сводка первого сообщества." in said_to_model(llm, 0) and "Сводка второго сообщества." in said_to_model(
        llm, 1
    )
    merged = said_to_model(llm, 2)
    assert "Чаще всего отказывает пусковое реле." in merged and "БЕСПОЛЕЗНЫЙ-ЧАСТИЧНЫЙ-ОТВЕТ" not in merged
    assert answer.text == "В линейке «Арктика» чаще всего отказывает пусковое реле." and not answer.refused


def test_global_search_refuses_when_no_summary_is_useful(kb):
    llm = FakeLLM(['{"answer": "нет", "score": 0}', '{"answer": "нет", "score": 0}'])
    answer = global_search("Какая погода в Казани?", kb, llm)
    assert answer.refused and len(llm.requests) == 2


def test_global_search_survives_a_broken_partial_answer(kb):
    llm = FakeLLM(["это не JSON", '{"answer": "Отказывает пусковое реле.", "score": 5}', "Отказывает пусковое реле."])
    answer = global_search("Какой узел отказывает чаще всего?", kb, llm)
    assert answer.text == "Отказывает пусковое реле." and len(llm.requests) == 3
