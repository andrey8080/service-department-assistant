import json

import httpx
import pytest
from fastapi.testclient import TestClient

from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected, ErpUnavailable
from erp.app import create_app

SEED = {
    "warehouses": ["Москва", "Казань"],
    "parts": [
        {"part_id": "P-1042", "name": "пусковое реле", "price_rub": 3460, "discontinued": True},
        {"part_id": "P-1042B", "name": "пусковое реле", "price_rub": 2630, "discontinued": False},
    ],
    "stock": [
        {"part_id": "P-1042", "warehouse": "Москва", "qty": 1},
        {"part_id": "P-1042", "warehouse": "Казань", "qty": 0},
        {"part_id": "P-1042B", "warehouse": "Москва", "qty": 0},
        {"part_id": "P-1042B", "warehouse": "Казань", "qty": 3},
    ],
}


@pytest.fixture()
def seed_path(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(SEED, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def erp(seed_path) -> ErpClient:
    return ErpClient("http://testserver", http=TestClient(create_app(seed_path, fail_rate=0.0)))


def test_value_received(erp):
    assert erp.get_price("P-1042B") == {
        "part_id": "P-1042B",
        "name": "пусковое реле",
        "price_rub": 2630,
        "discontinued": False,
    }
    assert erp.get_stock("P-1042B", "Казань")["qty"] == 3
    assert erp.get_stock("P-1042B", "Москва")["qty"] == 0
    assert erp.warehouses() == ["Москва", "Казань"]


def test_no_data_is_a_separate_outcome(erp):
    with pytest.raises(ErpNotFound, match="P-9999"):
        erp.get_price("P-9999")
    with pytest.raises(ErpNotFound):
        erp.get_stock("P-9999", "Казань")


def test_rejected_requests(erp):
    with pytest.raises(ErpRejected, match="Екатеринбург"):
        erp.get_stock("P-1042B", "Екатеринбург")
    with pytest.raises(ErpRejected, match="снята с производства"):
        erp.create_order("P-1042", "Казань", 1, "AR-210")


def test_order_is_created(erp):
    order = erp.create_order("P-1042B", "Казань", 2, "AR-210")
    assert order["order_id"] == "SO-000001" and order["qty"] == 2


def test_unavailable_service_is_a_separate_outcome(seed_path):
    broken = ErpClient("http://testserver", http=TestClient(create_app(seed_path, fail_rate=1.0)))
    with pytest.raises(ErpUnavailable):
        broken.get_price("P-1042B")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет соединения")

    offline = ErpClient("http://erp", http=httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://erp"))
    with pytest.raises(ErpUnavailable):
        offline.get_stock("P-1042B", "Казань")
