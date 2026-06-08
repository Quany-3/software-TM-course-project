import tempfile
import unittest
from pathlib import Path

import demo_pb2
from coupon_server import CouponService, CouponStore, money_to_nanos


class FakeContext:
    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class CouponServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.coupons_path = Path(self.temp_dir.name) / "coupons.json"
        self.coupons_path.write_text(
            """
            {
              "SAVE10": {
                "type": "fixed",
                "value": 10.0,
                "min_subtotal": 50.0,
                "enabled": true,
                "expires_at": null
              },
              "OFF20": {
                "type": "percentage",
                "value": 0.2,
                "min_subtotal": 100.0,
                "enabled": true,
                "expires_at": null
              },
              "FREESHIP": {
                "type": "free_shipping",
                "value": 0.0,
                "min_subtotal": 0.0,
                "enabled": true,
                "expires_at": null
              }
            }
            """,
            encoding="utf-8",
        )
        self.service = CouponService(CouponStore(self.coupons_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_apply_fixed_coupon(self):
        response = self.service.ApplyCoupon(
            demo_pb2.ApplyCouponRequest(
                coupon_code="SAVE10",
                user_id="user-1",
                subtotal=demo_pb2.Money(currency_code="USD", units=60, nanos=0),
                shipping_cost=demo_pb2.Money(currency_code="USD", units=5, nanos=0),
            ),
            FakeContext(),
        )

        self.assertTrue(response.success)
        self.assertEqual(10_000_000_000, money_to_nanos(response.discount))
        self.assertEqual(55_000_000_000, money_to_nanos(response.final_total))

    def test_apply_percentage_coupon(self):
        response = self.service.ApplyCoupon(
            demo_pb2.ApplyCouponRequest(
                coupon_code="OFF20",
                user_id="user-1",
                subtotal=demo_pb2.Money(currency_code="USD", units=100, nanos=0),
                shipping_cost=demo_pb2.Money(currency_code="USD", units=5, nanos=0),
            ),
            FakeContext(),
        )

        self.assertTrue(response.success)
        self.assertEqual(20_000_000_000, money_to_nanos(response.discount))
        self.assertEqual(85_000_000_000, money_to_nanos(response.final_total))

    def test_apply_free_shipping_coupon(self):
        response = self.service.ApplyCoupon(
            demo_pb2.ApplyCouponRequest(
                coupon_code="FREESHIP",
                user_id="user-1",
                subtotal=demo_pb2.Money(currency_code="USD", units=20, nanos=0),
                shipping_cost=demo_pb2.Money(currency_code="USD", units=5, nanos=0),
            ),
            FakeContext(),
        )

        self.assertTrue(response.success)
        self.assertEqual(5_000_000_000, money_to_nanos(response.discount))
        self.assertEqual(20_000_000_000, money_to_nanos(response.final_total))


if __name__ == "__main__":
    unittest.main()
