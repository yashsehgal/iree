// RUN: iree-opt -split-input-file -iree-vm-hoist-inlined-rodata %s | IreeFileCheck %s

vm.module @module {
  // CHECK: vm.rodata @_inline_const dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK-LABEL: @fn
  vm.func @fn() {
    // CHECK-NEXT: = vm.const.ref.rodata @_inline_const : !vm.ref<!iree.byte_buffer>
    %0 = vm.rodata.inline : !vm.ref<!iree.byte_buffer> = dense<[1, 2, 3]> : tensor<3xi32>
    vm.return
  }
}
