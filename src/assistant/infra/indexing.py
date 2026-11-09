"""Сборка индекса: чанки -> извлечение -> граф -> сообщества -> сводки. Вызывает ваши функции."""

import json
from collections.abc import Callable

import networkx as nx

from assistant.infra.chunking import load_chunks
from assistant.infra.config import Settings
from assistant.infra.grounding import load_vocabulary
from assistant.infra.schemas import Community, KnowledgeBase


def build_index(settings: Settings, llm, log: Callable[[str], None] = print) -> KnowledgeBase:
    from assistant.graphrag import communities as communities_module
    from assistant.graphrag import extraction, graph_builder

    chunks = load_chunks(settings.data_dir / "corpus")
    results = []
    for number, chunk in enumerate(chunks, start=1):
        results.append(extraction.extract_chunk(chunk, llm))
        log(f"извлечение {number}/{len(chunks)}: {chunk.chunk_id}")
    graph = graph_builder.build_graph(results)
    communities = communities_module.detect_communities(graph)
    for community in communities:
        community.summary = communities_module.summarize_community(community, graph, llm)
        log(f"сводка сообщества {community.community_id}: узлов {len(community.nodes)}")
    kb = KnowledgeBase(graph, communities, load_vocabulary(settings.data_dir / "vocabulary.json"))
    save_kb(kb, settings)
    stats = {
        "chunks": len(chunks),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "communities": len(communities),
        "dropped": sum(result.dropped for result in results),
        "conflicts": graph.graph.get("conflicts", 0),
        "llm": llm.stats.as_dict(),
    }
    _dump(settings.artifacts_dir / "index_stats.json", stats)
    log(f"индекс готов: {json.dumps(stats, ensure_ascii=False)}")
    return kb


def save_kb(kb: KnowledgeBase, settings: Settings) -> None:
    graph = {
        "graph": dict(kb.graph.graph),
        "nodes": [{"id": node, **attrs} for node, attrs in kb.graph.nodes(data=True)],
        "edges": [{"source": source, "target": target, **attrs} for source, target, attrs in kb.graph.edges(data=True)],
    }
    _dump(settings.artifacts_dir / "graph.json", graph)
    _dump(settings.artifacts_dir / "communities.json", [community.__dict__ for community in kb.communities])


def load_kb(settings: Settings) -> KnowledgeBase:
    graph_path = settings.artifacts_dir / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"нет файла {graph_path}: сначала постройте индекс командой make index")
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    graph = nx.DiGraph(**data["graph"])
    for node in data["nodes"]:
        graph.add_node(node.pop("id"), **node)
    for edge in data["edges"]:
        graph.add_edge(edge.pop("source"), edge.pop("target"), **edge)
    raw = json.loads((settings.artifacts_dir / "communities.json").read_text(encoding="utf-8"))
    return KnowledgeBase(
        graph, [Community(**item) for item in raw], load_vocabulary(settings.data_dir / "vocabulary.json")
    )


def _dump(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=sorted) + "\n", encoding="utf-8")
