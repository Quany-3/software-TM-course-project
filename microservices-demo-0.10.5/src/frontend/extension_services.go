package main

import (
	"context"
	"fmt"
	"sync"

	pb "github.com/GoogleCloudPlatform/microservices-demo/src/frontend/genproto"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/dynamicpb"
)

const (
	inventoryGetStockMethod = "/hipstershop.InventoryService/GetStock"
	couponApplyCouponMethod = "/hipstershop.CouponService/ApplyCoupon"
)

var (
	frontendExtensionDescOnce sync.Once
	frontendExtensionDescs    *frontendExtensionDescriptors
	frontendExtensionDescErr  error
)

type frontendExtensionDescriptors struct {
	money               protoreflect.MessageDescriptor
	getStockRequest     protoreflect.MessageDescriptor
	getStockResponse    protoreflect.MessageDescriptor
	applyCouponRequest  protoreflect.MessageDescriptor
	applyCouponResponse protoreflect.MessageDescriptor
}

func getFrontendExtensionDescriptors() (*frontendExtensionDescriptors, error) {
	frontendExtensionDescOnce.Do(func() {
		file, err := protodesc.NewFile(&descriptorpb.FileDescriptorProto{
			Name:    protoString("frontend_extensions.proto"),
			Package: protoString("hipstershop"),
			Syntax:  protoString("proto3"),
			MessageType: []*descriptorpb.DescriptorProto{
				message("Money",
					field("currency_code", 1, descriptorpb.FieldDescriptorProto_TYPE_STRING),
					field("units", 2, descriptorpb.FieldDescriptorProto_TYPE_INT64),
					field("nanos", 3, descriptorpb.FieldDescriptorProto_TYPE_INT32)),
				message("GetStockRequest",
					field("product_id", 1, descriptorpb.FieldDescriptorProto_TYPE_STRING)),
				message("GetStockResponse",
					field("product_id", 1, descriptorpb.FieldDescriptorProto_TYPE_STRING),
					field("quantity", 2, descriptorpb.FieldDescriptorProto_TYPE_INT32)),
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
			frontendExtensionDescErr = err
			return
		}

		messages := file.Messages()
		frontendExtensionDescs = &frontendExtensionDescriptors{
			money:               messages.ByName("Money"),
			getStockRequest:     messages.ByName("GetStockRequest"),
			getStockResponse:    messages.ByName("GetStockResponse"),
			applyCouponRequest:  messages.ByName("ApplyCouponRequest"),
			applyCouponResponse: messages.ByName("ApplyCouponResponse"),
		}
	})
	return frontendExtensionDescs, frontendExtensionDescErr
}

func protoString(v string) *string { return &v }
func protoInt32(v int32) *int32    { return &v }

func message(name string, fields ...*descriptorpb.FieldDescriptorProto) *descriptorpb.DescriptorProto {
	return &descriptorpb.DescriptorProto{Name: protoString(name), Field: fields}
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

func setString(msg *dynamicpb.Message, name string, value string) {
	msg.Set(msg.Descriptor().Fields().ByName(protoreflect.Name(name)), protoreflect.ValueOfString(value))
}

func setInt64(msg *dynamicpb.Message, name string, value int64) {
	msg.Set(msg.Descriptor().Fields().ByName(protoreflect.Name(name)), protoreflect.ValueOfInt64(value))
}

func setInt32(msg *dynamicpb.Message, name string, value int32) {
	msg.Set(msg.Descriptor().Fields().ByName(protoreflect.Name(name)), protoreflect.ValueOfInt32(value))
}

func getString(msg *dynamicpb.Message, name string) string {
	return msg.Get(msg.Descriptor().Fields().ByName(protoreflect.Name(name))).String()
}

func getBool(msg *dynamicpb.Message, name string) bool {
	return msg.Get(msg.Descriptor().Fields().ByName(protoreflect.Name(name))).Bool()
}

func getInt32(msg *dynamicpb.Message, name string) int32 {
	return int32(msg.Get(msg.Descriptor().Fields().ByName(protoreflect.Name(name))).Int())
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

func (fe *frontendServer) getStock(ctx context.Context, productID string) (int32, error) {
	descs, err := getFrontendExtensionDescriptors()
	if err != nil {
		return 0, err
	}

	req := dynamicpb.NewMessage(descs.getStockRequest)
	setString(req, "product_id", productID)
	resp := dynamicpb.NewMessage(descs.getStockResponse)
	if err := fe.inventorySvcConn.Invoke(ctx, inventoryGetStockMethod, req, resp, grpc.StaticMethod()); err != nil {
		return 0, err
	}
	return getInt32(resp, "quantity"), nil
}

func (fe *frontendServer) applyCoupon(ctx context.Context, couponCode, userID string, subtotal, shippingCost *pb.Money) (*pb.Money, *pb.Money, string, error) {
	descs, err := getFrontendExtensionDescriptors()
	if err != nil {
		return nil, nil, "", err
	}

	req := dynamicpb.NewMessage(descs.applyCouponRequest)
	setString(req, "coupon_code", couponCode)
	setString(req, "user_id", userID)
	req.Set(req.Descriptor().Fields().ByName("subtotal"), protoreflect.ValueOfMessage(moneyToDynamic(descs.money, subtotal)))
	req.Set(req.Descriptor().Fields().ByName("shipping_cost"), protoreflect.ValueOfMessage(moneyToDynamic(descs.money, shippingCost)))

	resp := dynamicpb.NewMessage(descs.applyCouponResponse)
	if err := fe.couponSvcConn.Invoke(ctx, couponApplyCouponMethod, req, resp, grpc.StaticMethod()); err != nil {
		return nil, nil, "", err
	}

	message := getString(resp, "message")
	if !getBool(resp, "success") {
		return nil, nil, message, fmt.Errorf("%s", message)
	}

	fields := resp.Descriptor().Fields()
	discount := dynamicToMoney(resp.Get(fields.ByName("discount")).Message())
	finalTotal := dynamicToMoney(resp.Get(fields.ByName("final_total")).Message())
	return discount, finalTotal, message, nil
}
