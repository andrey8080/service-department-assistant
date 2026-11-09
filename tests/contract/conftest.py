"""Общие заготовки контрактных тестов: маленький граф знаний и подставная ERP."""

import networkx as nx
import pytest

from assistant.infra.erp_client import ErpNotFound, ErpRejected
from assistant.infra.schemas import Community, KnowledgeBase

VOCABULARY = {
    "ProductLine": ["Арктика"],
    "Component": ["пусковое реле", "датчик температуры"],
    "Supplier": ["ХолодКомплект"],
}

# (начало, тип, конец, вес, чанки-источники)
EDGES = [
    ("Model:AR-210", "BELONGS_TO", "ProductLine:Арктика", 1, ["manual_AR-210.md#0"]),
    ("Model:AR-310", "BELONGS_TO", "ProductLine:Арктика", 1, ["manual_AR-310.md#0"]),
    ("ErrorCode:E-17", "INDICATES", "Component:пусковое реле", 2, ["manual_AR-210.md#1", "manual_AR-310.md#1"]),
    ("ErrorCode:E-17", "FIXED_BY", "Procedure:SP-104", 1, ["procedure_SP-104.md#0"]),
    ("Model:AR-210", "HAS_PART", "Part:P-1042", 1, ["manual_AR-210.md#0"]),
    ("Part:P-1042", "IS_A", "Component:пусковое реле", 1, ["manual_AR-210.md#0"]),
    ("Model:AR-310", "HAS_PART", "Part:P-1142", 1, ["manual_AR-310.md#0"]),
    ("Part:P-1142", "IS_A", "Component:пусковое реле", 1, ["manual_AR-310.md#0"]),
    ("Part:P-1042", "REPLACED_BY", "Part:P-1042B", 1, ["bulletin_SB-2025-03.md#0"]),
    ("Part:P-1042B", "SUPPLIED_BY", "Supplier:ХолодКомплект", 1, ["bulletin_SB-2025-03.md#0"]),
    ("Model:AR-210", "FAILED", "Component:пусковое реле", 37, ["report_R-2025-001.md#0"]),
    ("Model:AR-310", "FAILED", "Component:пусковое реле", 2, ["report_R-2025-002.md#0"]),
]


def make_graph() -> nx.DiGraph:
    graph = nx.DiGraph(conflicts=0)
    for source, kind, target, weight, chunks in EDGES:
        for node in (source, target):
            if node not in graph:
                node_type, name = node.split(":", 1)
                graph.add_node(node, type=node_type, name=name, sources=[])
            graph.nodes[node]["sources"] = sorted(set(graph.nodes[node]["sources"]) | set(chunks))
        graph.add_edge(source, target, type=kind, weight=weight, sources=list(chunks))
    return graph


@pytest.fixture()
def graph() -> nx.DiGraph:
    return make_graph()


@pytest.fixture()
def kb(graph) -> KnowledgeBase:
    nodes = sorted(graph.nodes)
    communities = [
        Community(0, nodes[: len(nodes) // 2], "Сводка первого сообщества."),
        Community(1, nodes[len(nodes) // 2 :], "Сводка второго сообщества."),
    ]
    return KnowledgeBase(graph, communities, VOCABULARY)


class FakeErp:
    """ERP без сети: два склада, одна актуальная деталь и одна снятая с производства."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def warehouses(self) -> list[str]:
        return ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

    def get_price(self, part_id: str) -> dict:
        self.calls.append(("get_price", part_id))
        if part_id not in ("P-1042", "P-1042B"):
            raise ErpNotFound(f"деталь {part_id} не найдена")
        return {"part_id": part_id, "name": "пусковое реле", "price_rub": 2630, "discontinued": part_id == "P-1042"}

    def get_stock(self, part_id: str, warehouse: str) -> dict:
        self.calls.append(("get_stock", part_id, warehouse))
        if part_id not in ("P-1042", "P-1042B"):
            raise ErpNotFound(f"деталь {part_id} не найдена")
        qty = 3 if warehouse == "Казань" else 0
        return {"part_id": part_id, "warehouse": warehouse, "qty": qty, "discontinued": part_id == "P-1042"}

    def create_order(self, part_id: str, warehouse: str, qty: int, model: str) -> dict:
        self.calls.append(("create_order", part_id, warehouse, qty, model))
        if part_id == "P-1042":
            raise ErpRejected("деталь P-1042 снята с производства, заказ невозможен")
        return {
            "order_id": "SO-000001",
            "part_id": part_id,
            "warehouse": warehouse,
            "qty": qty,
            "model": model,
            "status": "created",
        }


@pytest.fixture()
def erp() -> FakeErp:
    return FakeErp()
