import os

import grpc

import demo_pb2
import demo_pb2_grpc


def main():
    addr = os.environ.get("COUPON_SERVICE_ADDR", "localhost:50053")
    channel = grpc.insecure_channel(addr)
    stub = demo_pb2_grpc.CouponServiceStub(channel)

    coupon_code = os.environ.get("COUPON_CODE", "SAVE10")
    response = stub.ApplyCoupon(
        demo_pb2.ApplyCouponRequest(
            coupon_code=coupon_code,
            user_id="demo-user",
            subtotal=demo_pb2.Money(currency_code="USD", units=120, nanos=0),
            shipping_cost=demo_pb2.Money(currency_code="USD", units=5, nanos=0),
        )
    )
    print(
        "coupon success={} discount={}.{:09d} final_total={}.{:09d} message={}".format(
            response.success,
            response.discount.units,
            response.discount.nanos,
            response.final_total.units,
            response.final_total.nanos,
            response.message,
        )
    )


if __name__ == "__main__":
    main()
