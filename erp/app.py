"""Учебная учётная система «ПолярТех»: цены, остатки и заказы деталей.

Данные загружаются из seed.json в SQLite в памяти, поэтому заказы живут до перезапуска сервиса.
Переменная окружения ERP_FAIL_RATE задаёт долю запросов, на которые сервис отвечает кодом 503:
так проверяются повторы на стороне ассистента.
"""

import json
import os
import random
import sqlite3
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

SEED_PATH = Path(__file__).with_name("seed.json")


class OrderIn(BaseModel):
    part_id: str
    warehouse: str
    qty: int = Field(ge=1)
    model: str


def create_app(seed_path: Path = SEED_PATH, fail_rate: float | None = None, fail_seed: int = 0) -> FastAPI:
    seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    warehouses: list[str] = seed["warehouses"]
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE parts (part_id TEXT PRIMARY KEY, name TEXT, price_rub INTEGER, discontinued INTEGER);
        CREATE TABLE stock (part_id TEXT, warehouse TEXT, qty INTEGER, PRIMARY KEY (part_id, warehouse));
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT, part_id TEXT, warehouse TEXT, qty INTEGER, model TEXT
        );
        """
    )
    db.executemany("INSERT INTO parts VALUES (:part_id, :name, :price_rub, :discontinued)", seed["parts"])
    db.executemany("INSERT INTO stock VALUES (:part_id, :warehouse, :qty)", seed["stock"])
    lock = threading.Lock()
    rate = float(os.environ.get("ERP_FAIL_RATE", "0")) if fail_rate is None else fail_rate
    rng = random.Random(fail_seed)
    app = FastAPI(title="ПолярТех ERP")

    @app.middleware("http")
    async def unstable(request: Request, call_next):
        if request.url.path != "/health" and rng.random() < rate:
            return JSONResponse({"detail": "ERP временно недоступна"}, status_code=503)
        return await call_next(request)

    def part_row(part_id: str) -> sqlite3.Row:
        with lock:
            row = db.execute("SELECT * FROM parts WHERE part_id = ?", (part_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"деталь {part_id} не найдена")
        return row

    def check_warehouse(warehouse: str) -> None:
        if warehouse not in warehouses:
            raise HTTPException(422, f"неизвестный склад {warehouse}, допустимые: {', '.join(warehouses)}")

    def order_view(row: sqlite3.Row) -> dict:
        return {
            "order_id": f"SO-{row['id']:06d}",
            "part_id": row["part_id"],
            "warehouse": row["warehouse"],
            "qty": row["qty"],
            "model": row["model"],
            "status": "created",
        }

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/warehouses")
    def list_warehouses() -> list[str]:
        return warehouses

    @app.get("/parts/{part_id}")
    def get_part(part_id: str) -> dict:
        row = part_row(part_id)
        return {
            "part_id": row["part_id"],
            "name": row["name"],
            "price_rub": row["price_rub"],
            "discontinued": bool(row["discontinued"]),
        }

    @app.get("/stock/{part_id}")
    def get_stock(part_id: str, warehouse: str) -> dict:
        part = part_row(part_id)
        check_warehouse(warehouse)
        with lock:
            row = db.execute(
                "SELECT qty FROM stock WHERE part_id = ? AND warehouse = ?", (part_id, warehouse)
            ).fetchone()
        return {
            "part_id": part_id,
            "warehouse": warehouse,
            "qty": row["qty"],
            "discontinued": bool(part["discontinued"]),
        }

    @app.post("/orders", status_code=201)
    def create_order(order: OrderIn) -> dict:
        part = part_row(order.part_id)
        check_warehouse(order.warehouse)
        if part["discontinued"]:
            raise HTTPException(409, f"деталь {order.part_id} снята с производства, заказ невозможен")
        with lock:
            cursor = db.execute(
                "INSERT INTO orders (part_id, warehouse, qty, model) VALUES (?, ?, ?, ?)",
                (order.part_id, order.warehouse, order.qty, order.model),
            )
            row = db.execute("SELECT * FROM orders WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return order_view(row)

    @app.get("/orders/{order_id}")
    def get_order(order_id: str) -> dict:
        number = order_id.removeprefix("SO-")
        with lock:
            row = (
                db.execute("SELECT * FROM orders WHERE id = ?", (int(number),)).fetchone() if number.isdigit() else None
            )
        if row is None:
            raise HTTPException(404, f"заказ {order_id} не найден")
        return order_view(row)

    return app


app = create_app() if SEED_PATH.exists() else None
