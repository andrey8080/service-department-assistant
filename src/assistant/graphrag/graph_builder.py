"""Объединение извлечённых фактов в граф знаний."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет канонические сущности, считает разные чанки и разрешает встречные замены."""
    graph = nx.DiGraph(conflicts=0)

    def add_entity(entity: Entity, chunk_id: str) -> None:
        if entity.node_id not in graph:
            graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
        sources = graph.nodes[entity.node_id]["sources"]
        if chunk_id not in sources:
            sources.append(chunk_id)

    for result in results:
        for entity in result.entities:
            add_entity(entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            start, end = relation.subject.node_id, relation.object.node_id
            add_entity(relation.subject, result.chunk_id)
            add_entity(relation.object, result.chunk_id)
            if not graph.has_edge(start, end):
                graph.add_edge(start, end, type=relation.type, weight=0, sources=[])
            edge = graph.edges[start, end]
            if result.chunk_id not in edge["sources"]:
                edge["sources"].append(result.chunk_id)
            edge["weight"] = len(edge["sources"])

    checked = set()
    for start, end, edge in list(graph.edges(data=True)):
        pair = frozenset((start, end))
        if pair in checked or edge["type"] != "REPLACED_BY" or not graph.has_edge(end, start):
            continue
        reverse = graph.edges[end, start]
        if reverse["type"] != "REPLACED_BY":
            continue
        checked.add(pair)
        graph.graph["conflicts"] += 1
        forward_weight, backward_weight = edge["weight"], reverse["weight"]
        if forward_weight <= backward_weight:
            graph.remove_edge(start, end)
        if backward_weight <= forward_weight and graph.has_edge(end, start):
            graph.remove_edge(end, start)

    for _, attrs in graph.nodes(data=True):
        attrs["sources"].sort()
    for _, _, attrs in graph.edges(data=True):
        attrs["sources"].sort()
    return graph
