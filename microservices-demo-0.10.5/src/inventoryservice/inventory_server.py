import json
import os
import threading
import time
import uuid
from concurrent import futures
from pathlib import Path

import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

import demo_pb2
import demo_pb2_grpc
from logger import get_json_logger


logger = get_json_logger("inventoryservice-server")


class InventoryStore:
    def __init__(self, inventory_path, reservations_path):
        self.inventory_path = Path(inventory_path)
        self.reservations_path = Path(reservations_path)
        self.lock = threading.Lock()
        self.inventory = self._load_inventory()
        self.reservations = self._load_reservations()

    def _load_inventory(self):
        with self.inventory_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        inventory = {}
        for product_id, value in raw.items():
            if isinstance(value, int):
                inventory[product_id] = {"quantity": value, "reserved": 0}
            else:
                inventory[product_id] = {
                    "quantity": int(value.get("quantity", 0)),
                    "reserved": int(value.get("reserved", 0)),
                }
        return inventory

    def _load_reservations(self):
        if not self.reservations_path.exists():
            return {}
        with self.reservations_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def get_stock(self, product_id):
        item = self.inventory.get(product_id)
        if item is None:
            return 0
        return max(0, int(item["quantity"]))

    def check_stock(self, items):
        statuses = []
        available = True
        for item in items:
            current = self.get_stock(item.product_id)
            item_available = current >= item.quantity
            available = available and item_available
            statuses.append({
                "product_id": item.product_id,
                "requested_quantity": item.quantity,
                "available_quantity": current,
                "available": item_available,
            })
        return available, statuses

    def reserve_stock(self, user_id, items):
        with self.lock:
            available, statuses = self.check_stock(items)
            if not available:
                return False, "", "insufficient stock", statuses

            for item in items:
                entry = self.inventory[item.product_id]
                entry["quantity"] -= item.quantity
                entry["reserved"] += item.quantity

            reservation_id = "resv_" + uuid.uuid4().hex
            now = int(time.time())
            self.reservations[reservation_id] = {
                "user_id": user_id,
                "status": "RESERVED",
                "items": [
                    {"product_id": item.product_id, "quantity": item.quantity}
                    for item in items
                ],
                "created_at": now,
                "updated_at": now,
            }
            return True, reservation_id, "stock reserved", statuses

    def release_stock(self, reservation_id):
        with self.lock:
            reservation = self.reservations.get(reservation_id)
            if reservation is None:
                return False, "reservation not found"

            if reservation.get("status") == "RELEASED":
                return True, "reservation already released"

            if reservation.get("status") != "RESERVED":
                return False, "reservation cannot be released"

            for item in reservation.get("items", []):
                product_id = item["product_id"]
                quantity = int(item["quantity"])
                entry = self.inventory.setdefault(product_id, {"quantity": 0, "reserved": 0})
                entry["quantity"] += quantity
                entry["reserved"] = max(0, entry["reserved"] - quantity)

            reservation["status"] = "RELEASED"
            reservation["updated_at"] = int(time.time())
            return True, "stock released"


def _set_invalid_argument(context, message):
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(message)


def _validate_items(items, context):
    if not items:
        _set_invalid_argument(context, "items must not be empty")
        return False
    for item in items:
        if not item.product_id:
            _set_invalid_argument(context, "product_id is required")
            return False
        if item.quantity <= 0:
            _set_invalid_argument(context, "quantity must be greater than zero")
            return False
    return True


class InventoryService(demo_pb2_grpc.InventoryServiceServicer):
    def __init__(self, store):
        self.store = store

    def GetStock(self, request, context):
        if not request.product_id:
            _set_invalid_argument(context, "product_id is required")
            return demo_pb2.GetStockResponse()

        quantity = self.store.get_stock(request.product_id)
        logger.info("GetStock product_id=%s quantity=%s", request.product_id, quantity)
        return demo_pb2.GetStockResponse(
            product_id=request.product_id,
            quantity=quantity,
        )

    def CheckStock(self, request, context):
        if not _validate_items(request.items, context):
            return demo_pb2.CheckStockResponse()

        available, statuses = self.store.check_stock(request.items)
        logger.info("CheckStock available=%s item_count=%s", available, len(request.items))
        return demo_pb2.CheckStockResponse(
            available=available,
            item_statuses=[
                demo_pb2.StockItemStatus(
                    product_id=status["product_id"],
                    requested_quantity=status["requested_quantity"],
                    available_quantity=status["available_quantity"],
                    available=status["available"],
                )
                for status in statuses
            ],
        )

    def ReserveStock(self, request, context):
        if not request.user_id:
            _set_invalid_argument(context, "user_id is required")
            return demo_pb2.ReserveStockResponse(success=False, message="user_id is required")
        if not _validate_items(request.items, context):
            return demo_pb2.ReserveStockResponse(success=False, message="invalid items")

        success, reservation_id, message, statuses = self.store.reserve_stock(
            request.user_id,
            request.items,
        )
        logger.info(
            "ReserveStock user_id=%s success=%s reservation_id=%s item_count=%s",
            request.user_id,
            success,
            reservation_id,
            len(request.items),
        )
        return demo_pb2.ReserveStockResponse(
            success=success,
            reservation_id=reservation_id,
            message=message,
        )

    def ReleaseStock(self, request, context):
        if not request.reservation_id:
            _set_invalid_argument(context, "reservation_id is required")
            return demo_pb2.ReleaseStockResponse(success=False, message="reservation_id is required")

        success, message = self.store.release_stock(request.reservation_id)
        logger.info(
            "ReleaseStock reservation_id=%s success=%s",
            request.reservation_id,
            success,
        )
        return demo_pb2.ReleaseStockResponse(success=success, message=message)

    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

    def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.UNIMPLEMENTED)


def serve():
    port = os.environ.get("PORT", "50052")
    listen_addr = os.environ.get("LISTEN_ADDR", "[::]")
    inventory_path = os.environ.get("INVENTORY_DATA_FILE", "data/inventory.json")
    reservations_path = os.environ.get("RESERVATIONS_DATA_FILE", "data/reservations.json")

    store = InventoryStore(inventory_path, reservations_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = InventoryService(store)
    demo_pb2_grpc.add_InventoryServiceServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(service, server)

    logger.info("inventoryservice loaded product_count=%s", len(store.inventory))
    logger.info("listening on %s:%s", listen_addr, port)
    server.add_insecure_port(listen_addr + ":" + port)
    server.start()
    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
