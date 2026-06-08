import json
import os
import time
from concurrent import futures
from datetime import datetime, timezone
from pathlib import Path

import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

import demo_pb2
import demo_pb2_grpc
from logger import get_json_logger


logger = get_json_logger("couponservice-server")
NANOS_PER_UNIT = 1_000_000_000


def money_to_nanos(money):
    return money.units * NANOS_PER_UNIT + money.nanos


def nanos_to_money(amount_nanos, currency_code):
    units = int(amount_nanos // NANOS_PER_UNIT)
    nanos = int(amount_nanos % NANOS_PER_UNIT)
    return demo_pb2.Money(currency_code=currency_code, units=units, nanos=nanos)


def decimal_units_to_nanos(value):
    return int(round(float(value) * NANOS_PER_UNIT))


class CouponStore:
    def __init__(self, coupons_path):
        self.coupons_path = Path(coupons_path)
        self.coupons = self._load_coupons()

    def _load_coupons(self):
        with self.coupons_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def normalize_code(self, coupon_code):
        return coupon_code.strip().upper()

    def get_coupon(self, coupon_code):
        return self.coupons.get(self.normalize_code(coupon_code))

    def validate_coupon(self, coupon_code):
        code = self.normalize_code(coupon_code)
        if not code:
            return False, "coupon_code is required", None

        coupon = self.coupons.get(code)
        if coupon is None:
            return False, "coupon not found", None

        if not coupon.get("enabled", True):
            return False, "coupon is disabled", coupon

        expires_at = coupon.get("expires_at")
        if expires_at:
            expires_at_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) >= expires_at_dt:
                return False, "coupon is expired", coupon

        return True, "coupon is valid", coupon


def _set_invalid_argument(context, message):
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(message)


def _validate_money_pair(subtotal, shipping_cost, context):
    if not subtotal.currency_code:
        _set_invalid_argument(context, "subtotal currency_code is required")
        return False
    if subtotal.currency_code != shipping_cost.currency_code:
        _set_invalid_argument(context, "subtotal and shipping_cost currencies must match")
        return False
    if money_to_nanos(subtotal) < 0 or money_to_nanos(shipping_cost) < 0:
        _set_invalid_argument(context, "money values must not be negative")
        return False
    return True


class CouponService(demo_pb2_grpc.CouponServiceServicer):
    def __init__(self, store):
        self.store = store

    def ValidateCoupon(self, request, context):
        valid, message, coupon = self.store.validate_coupon(request.coupon_code)
        discount_type = coupon.get("type", "") if coupon else ""
        discount_value = float(coupon.get("value", 0.0)) if coupon else 0.0
        logger.info(
            "ValidateCoupon coupon_code=%s user_id=%s valid=%s",
            self.store.normalize_code(request.coupon_code),
            request.user_id,
            valid,
        )
        return demo_pb2.ValidateCouponResponse(
            valid=valid,
            message=message,
            discount_type=discount_type,
            discount_value=discount_value,
        )

    def ApplyCoupon(self, request, context):
        if not _validate_money_pair(request.subtotal, request.shipping_cost, context):
            return demo_pb2.ApplyCouponResponse(success=False, message="invalid money")

        currency = request.subtotal.currency_code
        subtotal_nanos = money_to_nanos(request.subtotal)
        shipping_nanos = money_to_nanos(request.shipping_cost)
        total_nanos = subtotal_nanos + shipping_nanos

        if not request.coupon_code.strip():
            return demo_pb2.ApplyCouponResponse(
                success=True,
                discount=nanos_to_money(0, currency),
                final_total=nanos_to_money(total_nanos, currency),
                message="no coupon applied",
            )

        valid, message, coupon = self.store.validate_coupon(request.coupon_code)
        if not valid:
            logger.info("ApplyCoupon coupon_code=%s success=false message=%s", request.coupon_code, message)
            return demo_pb2.ApplyCouponResponse(
                success=False,
                discount=nanos_to_money(0, currency),
                final_total=nanos_to_money(total_nanos, currency),
                message=message,
            )

        min_subtotal_nanos = decimal_units_to_nanos(coupon.get("min_subtotal", 0.0))
        if subtotal_nanos < min_subtotal_nanos:
            message = "subtotal does not meet coupon minimum"
            logger.info("ApplyCoupon coupon_code=%s success=false message=%s", request.coupon_code, message)
            return demo_pb2.ApplyCouponResponse(
                success=False,
                discount=nanos_to_money(0, currency),
                final_total=nanos_to_money(total_nanos, currency),
                message=message,
            )

        discount_nanos = self._calculate_discount(coupon, subtotal_nanos, shipping_nanos)
        discount_nanos = min(discount_nanos, total_nanos)
        final_total_nanos = total_nanos - discount_nanos

        logger.info(
            "ApplyCoupon coupon_code=%s user_id=%s success=true discount_nanos=%s",
            self.store.normalize_code(request.coupon_code),
            request.user_id,
            discount_nanos,
        )
        return demo_pb2.ApplyCouponResponse(
            success=True,
            discount=nanos_to_money(discount_nanos, currency),
            final_total=nanos_to_money(final_total_nanos, currency),
            message="coupon applied",
        )

    def _calculate_discount(self, coupon, subtotal_nanos, shipping_nanos):
        coupon_type = coupon.get("type")
        value = float(coupon.get("value", 0.0))
        if coupon_type == "fixed":
            return decimal_units_to_nanos(value)
        if coupon_type == "percentage":
            return int(round(subtotal_nanos * value))
        if coupon_type == "free_shipping":
            return shipping_nanos
        return 0

    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

    def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.UNIMPLEMENTED)


def serve():
    port = os.environ.get("PORT", "50053")
    listen_addr = os.environ.get("LISTEN_ADDR", "[::]")
    coupons_path = os.environ.get("COUPON_DATA_FILE", "data/coupons.json")

    store = CouponStore(coupons_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = CouponService(store)
    demo_pb2_grpc.add_CouponServiceServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(service, server)

    logger.info("couponservice loaded coupon_count=%s", len(store.coupons))
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
