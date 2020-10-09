// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

hal.constant_pool @storage = dense<[1,2,3]> : vector<3xi8> {
  hal.constant_pool.span @cst0, data = #hal.byte_range<0, 1024>, buffer = @storage_buffer0[#hal.byte_range<128, 1024>]
}
hal.variable @storage_buffer0 : !hal.buffer

// CHECK-LABEL: func @constant_buffer_access()
func @constant_buffer_access() -> (!hal.buffer, index, index) {
  %cst0 = hal.constant_pool.lookup @storage::@cst0 : !hal.constant
  // CHECK-DAG: %[[BUFFER:.+]] = hal.variable.load @storage_buffer0
  %buffer = hal.constant.buffer %cst0 : !hal.buffer
  // CHECK-DAG: %[[OFFSET:.+]] = constant 128 : index
  %offset = hal.constant.buffer.offset %cst0 : index
  // CHECK-DAG: %[[LENGTH:.+]] = constant 1024 : index
  %length = hal.constant.buffer.byte_length %cst0 : index
  // CHECK-NEXT: return %[[BUFFER]], %[[OFFSET]], %[[LENGTH]]
  return %buffer, %offset, %length : !hal.buffer, index, index
}
