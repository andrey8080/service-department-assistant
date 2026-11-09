import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
def seed_path(tmp_path: Path) -> Path:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(SEED, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def client(seed_path: Path) -> TestClient:
    return TestClient(create_app(seed_path, fail_rate=0.0))


def test_price_and_discontinued_flag(client):
    assert client.get("/parts/P-1042B").json() == {
        "part_id": "P-1042B",
        "name": "пусковое реле",
        "price_rub": 2630,
        "discontinued": False,
    }
    assert client.get("/parts/P-1042").json()["discontinued"] is True


def test_unknown_part_is_404_not_a_zero_price(client):
    response = client.get("/parts/P-9999")
    assert response.status_code == 404 and "P-9999" in response.json()["detail"]


def test_stock_distinguishes_zero_from_unknown(client):
    assert client.get("/stock/P-1042B", params={"warehouse": "Казань"}).json()["qty"] == 3
    zero = client.get("/stock/P-1042B", params={"warehouse": "Москва"})
    assert zero.status_code == 200 and zero.json()["qty"] == 0
    assert client.get("/stock/P-9999", params={"warehouse": "Казань"}).status_code == 404


def test_unknown_warehouse_is_rejected_with_the_allowed_list(client):
    response = client.get("/stock/P-1042B", params={"warehouse": "Екатеринбург"})
    assert response.status_code == 422 and "Москва, Казань" in response.json()["detail"]


def test_order_is_created_and_can_be_read_back(client):
    created = client.post("/orders", json={"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"})
    assert created.status_code == 201
    assert created.json() == {
        "order_id": "SO-000001",
        "part_id": "P-1042B",
        "warehouse": "Казань",
        "qty": 2,
        "model": "AR-210",
        "status": "created",
    }
    assert client.get("/orders/SO-000001").json() == created.json()
    assert client.get("/orders/SO-000002").status_code == 404


def test_discontinued_part_cannot_be_ordered(client):
    response = client.post("/orders", json={"part_id": "P-1042", "warehouse": "Казань", "qty": 1, "model": "AR-210"})
    assert response.status_code == 409 and "снята с производства" in response.json()["detail"]


@pytest.mark.parametrize("qty", [0, -1])
def test_order_quantity_must_be_positive(client, qty):
    response = client.post("/orders", json={"part_id": "P-1042B", "warehouse": "Казань", "qty": qty, "model": "AR-210"})
    assert response.status_code == 422


def test_fail_rate_one_breaks_everything_except_health(seed_path):
    broken = TestClient(create_app(seed_path, fail_rate=1.0))
    assert broken.get("/health").status_code == 200
    assert broken.get("/parts/P-1042B").status_code == 503


def test_fail_rate_is_reproducible(seed_path):
    def statuses() -> list[int]:
        flaky = TestClient(create_app(seed_path, fail_rate=0.5, fail_seed=1))
        return [flaky.get("/parts/P-1042B").status_code for _ in range(20)]

    first = statuses()
    assert first == statuses() and {200, 503} == set(first)


def test_generated_seed_loads_if_present():
    seed = Path(__file__).resolve().parents[1] / "seed.json"
    if not seed.exists():
        pytest.skip("seed.json ещё не сгенерирован")
    client = TestClient(create_app(seed, fail_rate=0.0))
    assert client.get("/warehouses").json() == ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
    assert client.get("/stock/P-1042B", params={"warehouse": "Казань"}).json() == {
        "part_id": "P-1042B",
        "warehouse": "Казань",
        "qty": 3,
        "discontinued": False,
    }


def test_orders_get_consecutive_numbers(client):
    body = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 1, "model": "AR-210"}
    assert [client.post("/orders", json=body).json()["order_id"] for _ in range(3)] == [
        "SO-000001",
        "SO-000002",
        "SO-000003",
    ]


def test_fail_rate_is_read_from_the_environment(seed_path, monkeypatch):
    monkeypatch.setenv("ERP_FAIL_RATE", "1")
    assert TestClient(create_app(seed_path)).get("/parts/P-1042B").status_code == 503
    monkeypatch.setenv("ERP_FAIL_RATE", "0")
    assert TestClient(create_app(seed_path)).get("/parts/P-1042B").status_code == 200
