// RUN: iree-opt -split-input-file -iree-convert-to-hal -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: hal.variable @var_i32 mutable : !hal.buffer
flow.variable @var_i32 mutable : tensor<i32>
func @fn() {
  // CHECK: %[[V:.+]] = hal.variable.load @var_i32 : !hal.buffer
  %0 = flow.variable.load @var_i32 : tensor<i32>
  // CHECK-NEXT: hal.variable.store %[[V]], @var_i32 : !hal.buffer
  flow.variable.store %0, @var_i32 : tensor<i32>
  return
}

// -----

// CHECK-LABEL: hal.variable @var_i1 mutable : !hal.buffer
flow.variable @var_i1 mutable : tensor<i1>
func @fn() {
  // CHECK: %[[V:.+]] = hal.variable.load @var_i1 : !hal.buffer
  %0 = flow.variable.load @var_i1 : tensor<i1>
  // CHECK-NEXT: hal.variable.store %[[V]], @var_i1 : !hal.buffer
  flow.variable.store %0, @var_i1 : tensor<i1>
  return
}

// -----

// CHECK-LABEL: hal.variable @var_indirect mutable : !hal.buffer
flow.variable @var_indirect mutable : tensor<i32>
func @fn() {
  // CHECK: %[[ADDR:.+]] = hal.variable.address @var_indirect
  %0 = flow.variable.address @var_indirect : !iree.ptr<tensor<i32>>
  // CHECK-NEXT: %[[VALUE:.+]] = hal.variable.load.indirect %[[ADDR]]
  %1 = flow.variable.load.indirect %0 : !iree.ptr<tensor<i32>> -> tensor<i32>
  // CHECK-NEXT: hal.variable.store.indirect %[[VALUE]], %[[ADDR]]
  flow.variable.store.indirect %1, %0 : tensor<i32> -> !iree.ptr<tensor<i32>>
  return
}

// -----
// Checks that an initializer function is generated, used and operates on
// a hal.buffer (versus tensor).
// CHECK-LABEL: func @__var_with_tensor_initializer_initializer() -> !hal.buffer
// CHECK: hal.variable @var_with_tensor_initializer
// CHECK-SAME: init(@__var_with_tensor_initializer_initializer)
// CHECK-SAME: : !hal.buffer
flow.variable @var_with_tensor_initializer mutable dense<0.000000e+00> : tensor<f32>
func @fn() {
  %0 = flow.variable.load @var_with_tensor_initializer : tensor<f32>
  flow.variable.store %0, @var_with_tensor_initializer : tensor<f32>
  return
}

// -----
// TODO(b/145839814): It should not be possible to produce a name collision
// expected-error @+3 {{redefinition of symbol named '__var_with_initializer_initializer'}}
// expected-note @+1 {{see existing symbol definition here}}
func @__var_with_initializer_initializer() -> ()
flow.variable @var_with_initializer mutable dense<0.000000e+00> : tensor<f32>
func @fn() {
  %0 = flow.variable.load @var_with_initializer : tensor<f32>
  flow.variable.store %0, @var_with_initializer : tensor<f32>
  return
}


// DO NOT SUBMIT
hal.constant_pool @pool attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  hal.constant_pool.span @cst0 : tensor<4xf32> = @_storage[#hal.byte_range<0, 16>]
  hal.constant_pool.span @cst1 : tensor<3xi8> = @_storage_0[#hal.byte_range<0, 3>]
  hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32>
  hal.constant_pool.splat @cst3 = dense<1234567890> : tensor<8xi32>
  hal.constant_storage @_storage = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
  hal.constant_storage @_storage_0 = dense<[6, 7, 8, 0]> : vector<4xi8>
}

// CHECK: DO NOT SUBMIT
func @fn1() -> (tensor<4xf32>, tensor<3xi8>, tensor<1xf32>, tensor<8xi32>) {
  %cst0 = hal.constant_pool.load @pool::@cst0 : tensor<4xf32>
  %cst1 = hal.constant_pool.load @pool::@cst1 : tensor<3xi8>
  %cst2 = hal.constant_pool.load @pool::@cst2 : tensor<1xf32>
  %cst3 = hal.constant_pool.load @pool::@cst3 : tensor<8xi32>
  return %cst0, %cst1, %cst2, %cst3 : tensor<4xf32>, tensor<3xi8>, tensor<1xf32>, tensor<8xi32>
}
