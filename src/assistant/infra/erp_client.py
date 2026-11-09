"""Клиент учётной системы. У любого запроса три исхода, и их нельзя путать.

- значение получено: метод возвращает словарь
- данных нет: исключение ErpNotFound, такой детали или заказа в ERP не существует
- сервис недоступен: исключение ErpUnavailable, запрос можно повторить

Отдельный случай – ErpRejected: ERP поняла запрос и отказала (неизвестный склад,
деталь снята с производства). Повторять такой запрос бессмысленно.
"""

import httpx


class ErpNotFound(LookupError):
    pass


class ErpUnavailable(ConnectionError):
    pass


class ErpRejected(ValueError):
    pass


class ErpClient:
    def __init__(self, base_url: str, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=10)

    def warehouses(self) -> list[str]:
        return self._request("GET", "/warehouses")

    def get_price(self, part_id: str) -> dict:
        """{"part_id", "name", "price_rub", "discontinued"}"""
        return self._request("GET", f"/parts/{part_id}")

    def get_stock(self, part_id: str, warehouse: str) -> dict:
        """{"part_id", "warehouse", "qty", "discontinued"}"""
        return self._request("GET", f"/stock/{part_id}", params={"warehouse": warehouse})

    def create_order(self, part_id: str, warehouse: str, qty: int, model: str) -> dict:
        """{"order_id", "part_id", "warehouse", "qty", "model", "status"}"""
        body = {"part_id": part_id, "warehouse": warehouse, "qty": qty, "model": model}
        return self._request("POST", "/orders", json=body)

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ErpUnavailable(f"ERP недоступна: {exc}") from exc
        if response.status_code >= 500:
            raise ErpUnavailable(f"ERP недоступна: код {response.status_code}")
        if response.status_code == 404:
            raise ErpNotFound(_detail(response))
        if response.status_code >= 400:
            raise ErpRejected(_detail(response))
        return response.json()


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text
    return detail if isinstance(detail, str) else str(detail)
