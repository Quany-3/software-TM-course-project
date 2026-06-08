import os

import grpc

import demo_pb2
import demo_pb2_grpc


def main():
    addr = os.environ.get("INVENTORY_SERVICE_ADDR", "localhost:50052")
    channel = grpc.insecure_channel(addr)
    stub = demo_pb2_grpc.InventoryServiceStub(channel)

    product_id = os.environ.get("PRODUCT_ID", "OLJCESPC7Z")
    response = stub.GetStock(demo_pb2.GetStockRequest(product_id=product_id))
    print(f"stock product_id={response.product_id} quantity={response.quantity}")

    check = stub.CheckStock(
        demo_pb2.CheckStockRequest(
            items=[demo_pb2.CartItem(product_id=product_id, quantity=1)]
        )
    )
    print(f"check available={check.available}")


if __name__ == "__main__":
    main()
