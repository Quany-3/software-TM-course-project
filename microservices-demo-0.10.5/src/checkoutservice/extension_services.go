package main

import (
	"context"
	"fmt"
	"sync"

	pb "github.com/GoogleCloudPlatform/microservices-demo/src/checkoutservice/genproto"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/dynamicpb"
)

const (
	inventoryReserveStockMethod = "/hipstershop.InventoryService/ReserveStock"
	inventoryReleaseStockMethod = "/hipstershop.InventoryService/ReleaseStock"
	couponApplyCouponMethod     = "/hipstershop.CouponService/ApplyCoupon"
)

var (
	extensionDescOnce sync.Once
	extensionDescs    *extensionDescriptors
	extensionDescErr  error
)

type extensionDescriptors struct {
	cartItem             protoreflect.MessageDescriptor
	money                protoreflect.MessageDescriptor
	reserveStockRequest  protoreflect.MessageDescriptor
	reserveStockResponse protoreflect.MessageDescriptor
	releaseStockRequest  protoreflect.MessageDescriptor
	releaseStockResponse protoreflect.MessageDescriptor
	applyCouponRequest   protoreflect.MessageDescriptor
	applyCouponResponse  protoreflect.MessageDescriptor
}

func getExtensionDescriptors() (*extensionDescriptors, error) {
	extensionDescOnce.Do(func() {
		file, err := protodesc.NewFile(&descriptorpb.FileDescriptorProto{
			Name:    protoString("checkout_extensions.proto"),
			Package: protoString("hipstershop"),
			Syntax:  protoString("proto3"),
			MessageType: []*descriptorpb.DescriptorProto{
				message("CartItem",
					field("product_id", 1, descriptorpb.FieldDescriptorProto_TYPE_STRING),
					field("quantity", 2, descriptorpb.FieldDescriptorProto_TYPE_INT32)),
				message("Money",
					field("currency_code", 1, descriptorpb.FieldDescriptorProto_TYPE_STRING),
					field("units", 2, descriptorpb.FieldDescriptorProto_TYPE_INT64),
					field("nanos", 3, descriptorpb.FieldDescriptorProto_TYPE_INT32)),
				message("ReserveStockRequest",
					field("user_id", 1, descriptorpb.FieldDescriptorProto_TYPE_STRING),
					repeatedMessageField("items", 2, ".hipstershop.CartItem")),
				message("ReserveStockResponse",
					field("success", 1, descriptorpb.FieldDescriptorProto_TYPE_BOOL),
					field("reservation_id", 2, descriptorpb.FieldDescriptorProto_TYPE_STRING),
					field("message", 3, descriptorpb.FieldDescriptorProto_TYPE_STRING)),
				message("ReleaseStockRequest",
					field("reservation_id", 1, descriptorpb.FieldDescriptorProto_TYPE_STRING)),
				message("ReleaseStockResponse",
					field("success", 1, descriptorpb.FieldDescriptorProto_TYPE_BOOL),
					field("message", 2, descriptorpb.FieldDescriptorProto_TYPE_STRING)),
				message("ApplyCouponRequest",
					field("coupon_code", 1, descriptorpb.FieldDescriptorProto_TYPE_STRING),
					field("user_id", 2, descriptorpb.FieldDescriptorProto_TYPE_STRING),
					messageField("subtotal", 3, ".hipstershop.Money"),
					messageField("shipping_cost", 4, ".hipstershop.Money")),
				message("ApplyCouponResponse",
					field("success", 1, descriptorpb.FieldDescriptorProto_TYPE_BOOL),
					messageField("discount", 2, ".hipstershop.Money"),
					messageField("final_total", 3, ".hipstershop.Money"),
					field("message", 4, descriptorpb.FieldDescriptorProto_TYPE_STRING)),
			},
		}, nil)
		if err != nil {
			extensionDescErr = err
			return
		}

		messages := file.Messages()
		extensionDescs = &extensionDescriptors{
			cartItem:             messages.ByName("CartItem"),
			money:                messages.ByName("Money"),
			reserveStockRequest:  messages.ByName("ReserveStockRequest"),
			reserveStockResponse: messages.ByName("ReserveStockResponse"),
			releaseStockRequest:  messages.ByName("ReleaseStockRequest"),
			releaseStockResponse: messages.ByName("ReleaseStockResponse"),
			applyCouponRequest:   messages.ByName("ApplyCouponRequest"),
			applyCouponResponse:  messages.ByName("ApplyCouponResponse"),
		}
	})
	return extensionDescs, extensionDescErr
}

func protoString(v string) *string { return &v }
func protoInt32(v int32) *int32    { return &v }

func message(name string, fields ...*descriptorpb.FieldDescriptorProto) *descriptorpb.DescriptorProto {
	return &descriptorpb.DescriptorProto{
		Name:  protoString(name),
		Field: fields,
	}
}

func field(name string, number int32, fieldType descriptorpb.FieldDescriptorProto_Type) *descriptorpb.FieldDescriptorProto {
	return &descriptorpb.FieldDescriptorProto{
		Name:   protoString(name),
		Number: protoInt32(number),
		Label:  descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
		Type:   fieldType.Enum(),
	}
}

func messageField(name string, number int32, typeName string) *descriptorpb.FieldDescriptorProto {
	f := field(name, number, descriptorpb.FieldDescriptorProto_TYPE_MESSAGE)
	f.TypeName = protoString(typeName)
	return f
}

func repeatedMessageField(name string, number int32, typeName string) *descriptorpb.FieldDescriptorProto {
	f := messageField(name, number, typeName)
	f.Label = descriptorpb.FieldDescriptorProto_LABEL_REPEATED.Enum()
	return f
}

func setString(msg *dynamicpb.Message, name string, value string) {
	msg.Set(msg.Descriptor().Fields().ByName(protoreflect.Name(name)), protoreflect.ValueOfString(value))
}

func setInt32(msg *dynamicpb.Message, name string, value int32) {
	msg.Set(msg.Descriptor().Fields().ByName(protoreflect.Name(name)), protoreflect.ValueOfInt32(value))
}

func setInt64(msg *dynamicpb.Message, name string, value int64) {
	msg.Set(msg.Descriptor().Fields().ByName(protoreflect.Name(name)), protoreflect.ValueOfInt64(value))
}

func getString(msg *dynamicpb.Message, name string) string {
	return msg.Get(msg.Descriptor().Fields().ByName(protoreflect.Name(name))).String()
}

func getBool(msg *dynamicpb.Message, name string) bool {
	return msg.Get(msg.Descriptor().Fields().ByName(protoreflect.Name(name))).Bool()
}

func moneyToDynamic(desc protoreflect.MessageDescriptor, money *pb.Money) *dynamicpb.Message {
	msg := dynamicpb.NewMessage(desc)
	if money == nil {
		return msg
	}
	setString(msg, "currency_code", money.GetCurrencyCode())
	setInt64(msg, "units", money.GetUnits())
	setInt32(msg, "nanos", money.GetNanos())
	return msg
}

func dynamicToMoney(msg protoreflect.Message) *pb.Money {
	fields := msg.Descriptor().Fields()
	return &pb.Money{
		CurrencyCode: msg.Get(fields.ByName("currency_code")).String(),
		Units:        msg.Get(fields.ByName("units")).Int(),
		Nanos:        int32(msg.Get(fields.ByName("nanos")).Int()),
	}
}

func cartItemToDynamic(desc protoreflect.MessageDescriptor, item *pb.CartItem) *dynamicpb.Message {
	msg := dynamicpb.NewMessage(desc)
	setString(msg, "product_id", item.GetProductId())
	setInt32(msg, "quantity", item.GetQuantity())
	return msg
}

func (cs *checkoutService) reserveStock(ctx context.Context, userID string, items []*pb.CartItem) (string, error) {
	descs, err := getExtensionDescriptors()
	if err != nil {
		return "", err
	}

	req := dynamicpb.NewMessage(descs.reserveStockRequest)
	setString(req, "user_id", userID)
	itemsField := req.Descriptor().Fields().ByName("items")
	itemsList := req.Mutable(itemsField).List()
	for _, item := range items {
		itemsList.Append(protoreflect.ValueOfMessage(cartItemToDynamic(descs.cartItem, item)))
	}

	resp := dynamicpb.NewMessage(descs.reserveStockResponse)
	if err := cs.inventorySvcConn.Invoke(ctx, inventoryReserveStockMethod, req, resp, grpc.StaticMethod()); err != nil {
		return "", err
	}
	if !getBool(resp, "success") {
		return "", fmt.Errorf("%s", getString(resp, "message"))
	}
	return getString(resp, "reservation_id"), nil
}

func (cs *checkoutService) releaseStock(ctx context.Context, reservationID string) error {
	if reservationID == "" {
		return nil
	}
	descs, err := getExtensionDescriptors()
	if err != nil {
		return err
	}

	req := dynamicpb.NewMessage(descs.releaseStockRequest)
	setString(req, "reservation_id", reservationID)
	resp := dynamicpb.NewMessage(descs.releaseStockResponse)
	if err := cs.inventorySvcConn.Invoke(ctx, inventoryReleaseStockMethod, req, resp, grpc.StaticMethod()); err != nil {
		return err
	}
	if !getBool(resp, "success") {
		return fmt.Errorf("%s", getString(resp, "message"))
	}
	return nil
}

func (cs *checkoutService) applyCoupon(ctx context.Context, couponCode, userID string, subtotal, shippingCost *pb.Money) (*pb.Money, *pb.Money, error) {
	descs, err := getExtensionDescriptors()
	if err != nil {
		return nil, nil, err
	}

	req := dynamicpb.NewMessage(descs.applyCouponRequest)
	setString(req, "coupon_code", couponCode)
	setString(req, "user_id", userID)
	req.Set(req.Descriptor().Fields().ByName("subtotal"), protoreflect.ValueOfMessage(moneyToDynamic(descs.money, subtotal)))
	req.Set(req.Descriptor().Fields().ByName("shipping_cost"), protoreflect.ValueOfMessage(moneyToDynamic(descs.money, shippingCost)))

	resp := dynamicpb.NewMessage(descs.applyCouponResponse)
	if err := cs.couponSvcConn.Invoke(ctx, couponApplyCouponMethod, req, resp, grpc.StaticMethod()); err != nil {
		return nil, nil, err
	}
	if !getBool(resp, "success") {
		return nil, nil, fmt.Errorf("%s", getString(resp, "message"))
	}

	fields := resp.Descriptor().Fields()
	discount := dynamicToMoney(resp.Get(fields.ByName("discount")).Message())
	finalTotal := dynamicToMoney(resp.Get(fields.ByName("final_total")).Message())
	return discount, finalTotal, nil
}
