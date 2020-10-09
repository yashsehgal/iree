// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: hal.constant_pool @pool0
hal.constant_pool @pool0 {
  // CHECK-NEXT: hal.constant_pool.value @cst0 = dense<0.{{.+}}> : tensor<2x3xf32>
  hal.constant_pool.value @cst0 = dense<0.0> : tensor<2x3xf32>
  // CHECK-NEXT: hal.constant_pool.value @cst1 = dense<1.{{.+}}> : tensor<3x2xf32>
  hal.constant_pool.value @cst1 = dense<1.0> : tensor<3x2xf32>
}
// CHECK: func @func()
func @func() -> (tensor<2x3xf32>, tensor<3x2xf32>) {
  // CHECK-NEXT: = hal.constant_pool.load @pool0::@cst0 : tensor<2x3xf32>
  %cst0 = hal.constant_pool.load @pool0::@cst0 : tensor<2x3xf32>
  // CHECK-NEXT: = hal.constant_pool.load @pool0::@cst1 : tensor<3x2xf32>
  %cst1 = hal.constant_pool.load @pool0::@cst1 : tensor<3x2xf32>
  return %cst0, %cst1 : tensor<2x3xf32>, tensor<3x2xf32>
}

// -----

// CHECK-LABEL: hal.constant_pool @pool0
hal.constant_pool @pool0 {
  // CHECK-NEXT: hal.constant_pool.span @cst0 : tensor<2x3xf32> = @storage0[#hal.byte_range<0, 1024>]
  hal.constant_pool.span @cst0 : tensor<2x3xf32> = @storage0[#hal.byte_range<0, 1024>]
  // CHECK-NEXT: hal.constant_pool.span @cst1 : tensor<3x2xf32> = @storage0[#hal.byte_range<1024, 1024>]
  hal.constant_pool.span @cst1 : tensor<3x2xf32> = @storage0[#hal.byte_range<1024, 1024>]
}
// CHECK: hal.variable @storage0 = dense<0> : vector<2048xi8>
hal.variable @storage0 = dense<0> : vector<2048xi8>

// -----

hal.constant_pool @pool0 {
  hal.constant_pool.span @cst0 : tensor<2x3xf32> = @storage0[#hal.byte_range<0, 1024>]
  hal.constant_pool.span @cst1 : tensor<3x2xf32> = @storage0[#hal.byte_range<1024, 1024>]
}
// CHECK: hal.variable @storage = dense<0> : vector<2048xi8>
hal.variable @storage0 = dense<0> : vector<2048xi8>
// CHECK: hal.variable @storage0_buffer0 : !hal.buffer
hal.variable @storage0_buffer0 : !hal.buffer
// CHECK-NEXT: func @func()
func @func() {
  // CHECK-NEXT: = hal.constant.subspan @storage0::@cst0 : !hal.constant
  %cst0 = hal.constant.subspan @storage0::@cst0 : !hal.constant
  // CHECK-NEXT: = hal.constant.subspan @storage0::@cst1 : !hal.constant
  %cst1 = hal.constant.subspan @storage0::@cst1 : !hal.constant

  // CHECK-NEXT: = hal.constant.buffer %{{.+}} : !hal.buffer
  %buffer0 = hal.constant.buffer %cst0 : !hal.buffer
  // CHECK-NEXT: = hal.constant.buffer.offset %{{.+}} : index
  %offset0 = hal.constant.buffer.offset %cst0 : index
  // CHECK-NEXT: = hal.constant.buffer.byte_length %{{.+}} : index
  %length0 = hal.constant.buffer.byte_length %cst0 : index

  return
}
