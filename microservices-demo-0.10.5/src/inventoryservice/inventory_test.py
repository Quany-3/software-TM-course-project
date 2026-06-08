import tempfile
import unittest
from pathlib import Path

import demo_pb2
from inventory_server import InventoryStore


class InventoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.inventory_path = Path(self.temp_dir.name) / "inventory.json"
        self.reservations_path = Path(self.temp_dir.name) / "reservations.json"
        self.inventory_path.write_text(
            '{"product-a": {"quantity": 3, "reserved": 0}}',
            encoding="utf-8",
        )
        self.reservations_path.write_text("{}", encoding="utf-8")
        self.store = InventoryStore(self.inventory_path, self.reservations_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reserve_stock_decrements_available_quantity(self):
        success, reservation_id, message, _ = self.store.reserve_stock(
            "user-1",
            [demo_pb2.CartItem(product_id="product-a", quantity=2)],
        )

        self.assertTrue(success)
        self.assertTrue(reservation_id.startswith("resv_"))
        self.assertEqual("stock reserved", message)
        self.assertEqual(1, self.store.get_stock("product-a"))

    def test_reserve_stock_fails_without_partial_update(self):
        success, reservation_id, message, _ = self.store.reserve_stock(
            "user-1",
            [demo_pb2.CartItem(product_id="product-a", quantity=4)],
        )

        self.assertFalse(success)
        self.assertEqual("", reservation_id)
        self.assertEqual("insufficient stock", message)
        self.assertEqual(3, self.store.get_stock("product-a"))

    def test_release_stock_is_idempotent(self):
        success, reservation_id, _, _ = self.store.reserve_stock(
            "user-1",
            [demo_pb2.CartItem(product_id="product-a", quantity=2)],
        )
        self.assertTrue(success)

        success, _ = self.store.release_stock(reservation_id)
        self.assertTrue(success)
        self.assertEqual(3, self.store.get_stock("product-a"))

        success, _ = self.store.release_stock(reservation_id)
        self.assertTrue(success)
        self.assertEqual(3, self.store.get_stock("product-a"))


if __name__ == "__main__":
    unittest.main()
