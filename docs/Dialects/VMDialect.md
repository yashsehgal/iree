---
layout: default
permalink: Dialects/VMDialect
parent: Dialect Definitions
title: "'vm' Dialect"
---

<!-- Autogenerated by mlir-tblgen; don't manually edit -->
# 'vm' Dialect
{: .no_toc }


A dialect representing operations against an abstract virtual machine.


The virtual machine ops are designed to be either serialized to a bytecode
representation that can be interpreted at runtime or lowered further to
static representations such as LLVM IR, C, etc. The idea is that the types
and operations performed are generally just encoding resource ownership
rules and control flow that can be represented in many different ways by
target runtimes. For example, it should be possible to lower the VM dialect
to SPIR-V and run the VM entirely within a persistent Vulkan kernel.

With this scalable runtime approach we make some limiting assumptions to
keep the required implementations simple. As we assume all real math is
happening within dispatch regions the only math we provide is scalar
operations used for offset and shape calculations. This also enables simple
flow control such as fixed-range loops.

Besides integer values the only other storage type is a variant reference
modeling an abstract iree::ref_ptr. This allows automated reference counting
to be relied upon by other dialects built on top of the VM dialect and
avoids the need for more verbose manual reference counting logic (that may
be difficult or impossible to manage given the coroutine-like nature of the
VM). Lowering targets can insert the reference counting as needed.

1. TOC
{:toc}

## Operation definition

### `vm.add.i32` (IREE::VM::AddI32Op)

integer add operation

Syntax:

```
operation ::= `vm.add.i32` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.and.i32` (IREE::VM::AndI32Op)

integer binary and operation

Syntax:

```
operation ::= `vm.and.i32` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.br` (IREE::VM::BranchOp)

unconditional branch operation

Syntax:

```
operation ::= `vm.br` $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
```


 Represents an unconditional branch operation that branches to a target block
 with the given set of arguments.

 ```
 ^bb0(...):
   vm.br ^bb1(%a)
 ^bb1(%blockArg1):
   ...
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`destOperands` | 32-bit signless integer or 32-bit signless integer or ref

#### Successors:

| Successor | Description |
| :-------: | ----------- |
`dest` | any successor

### `vm.break` (IREE::VM::BreakOp)

unconditional debug break operation

Syntax:

```
operation ::= `vm.break` $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
```


Breaks into the attached debugger or asks for attaching a debugger. After
resuming (or if a debugger is not attached) execution will continue at the
target block.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`destOperands` | 32-bit signless integer or 32-bit signless integer or ref

#### Successors:

| Successor | Description |
| :-------: | ----------- |
`dest` | any successor

### `vm.call` (IREE::VM::CallOp)

call operation

Syntax:

```
operation ::= `vm.call` $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
```


Calls an internal VM function with the given arguments.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`callee` | FlatSymbolRefAttr | symbol reference attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operands` | 32-bit signless integer or 32-bit signless integer or ref

#### Results:

| Result | Description |
| :----: | ----------- |
`results` | 32-bit signless integer or 32-bit signless integer or ref

### `vm.call.variadic` (IREE::VM::CallVariadicOp)

call operation with variadic arguments

Calls an internal VM function with the given arguments. One or more of the
arguments may be variadic, encoded as segmented sized operand lists.

Variadic arguments must be specified with a total count in the segment_sizes
attribute.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`callee` | FlatSymbolRefAttr | symbol reference attribute
`segment_sizes` | DenseIntElementsAttr | 16-bit signless integer elements attribute
`segment_types` | ArrayAttr | type array attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operands` | 32-bit signless integer or 32-bit signless integer or ref

#### Results:

| Result | Description |
| :----: | ----------- |
`results` | 32-bit signless integer or 32-bit signless integer or ref

### `vm.cmp.eq.i32` (IREE::VM::CmpEQI32Op)

integer equality comparison operation

Syntax:

```
operation ::= `vm.cmp.eq.i32` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.eq.ref` (IREE::VM::CmpEQRefOp)

ref_ptr equality comparison operation

Syntax:

```
operation ::= `vm.cmp.eq.ref` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | ref
`rhs` | ref

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.gte.i32.s` (IREE::VM::CmpGTEI32SOp)

signed integer greater-than-or-equal comparison operation

Syntax:

```
operation ::= `vm.cmp.gte.i32.s` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.gte.i32.u` (IREE::VM::CmpGTEI32UOp)

unsigned integer greater-than-or-equal comparison operation

Syntax:

```
operation ::= `vm.cmp.gte.i32.u` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.gt.i32.s` (IREE::VM::CmpGTI32SOp)

signed integer greater-than comparison operation

Syntax:

```
operation ::= `vm.cmp.gt.i32.s` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.gt.i32.u` (IREE::VM::CmpGTI32UOp)

unsigned integer greater-than comparison operation

Syntax:

```
operation ::= `vm.cmp.gt.i32.u` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.lte.i32.s` (IREE::VM::CmpLTEI32SOp)

signed integer less-than-or-equal comparison operation

Syntax:

```
operation ::= `vm.cmp.lte.i32.s` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.lte.i32.u` (IREE::VM::CmpLTEI32UOp)

unsigned integer less-than-or-equal comparison operation

Syntax:

```
operation ::= `vm.cmp.lte.i32.u` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.lt.i32.s` (IREE::VM::CmpLTI32SOp)

signed integer less-than comparison operation

Syntax:

```
operation ::= `vm.cmp.lt.i32.s` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.lt.i32.u` (IREE::VM::CmpLTI32UOp)

unsigned integer less-than comparison operation

Syntax:

```
operation ::= `vm.cmp.lt.i32.u` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.ne.i32` (IREE::VM::CmpNEI32Op)

integer inequality comparison operation

Syntax:

```
operation ::= `vm.cmp.ne.i32` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.ne.ref` (IREE::VM::CmpNERefOp)

ref_ptr inequality comparison operation

Syntax:

```
operation ::= `vm.cmp.ne.ref` operands attr-dict `:` type($lhs)
```


Compares two operands with the specified predicate.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | ref
`rhs` | ref

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cmp.nz.ref` (IREE::VM::CmpNZRefOp)

ref_ptr non-zero comparison operation

Syntax:

```
operation ::= `vm.cmp.nz.ref` $operand attr-dict `:` type($operand)
```


Compares the given ref_ptr operand for a non-zero/null value.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | ref

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.cond_br` (IREE::VM::CondBranchOp)

conditional branch operation

Syntax:

```
operation ::= `vm.cond_br` $condition `,`
              $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
              $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
              attr-dict
```


 Represents a conditional branch operation that branches to one of the two
 target blocks with the given set of arguments.

 ```
 ^bb0(...):
   vm.cond_br %condition, ^bb1(%a), ^bb2(%b)
 ^bb1(%blockArg1):
   ...
 ^bb2(%blockArg2):
   ...
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`condition` | 32-bit signless integer
`trueDestOperands` | 32-bit signless integer or 32-bit signless integer or ref
`falseDestOperands` | 32-bit signless integer or 32-bit signless integer or ref

#### Successors:

| Successor | Description |
| :-------: | ----------- |
`trueDest` | any successor
`falseDest` | any successor

### `vm.cond_break` (IREE::VM::CondBreakOp)

conditional debug break operation

Syntax:

```
operation ::= `vm.cond_break` $condition `,` $dest (`(` $destOperands^ `:` type($destOperands) `)`)?
              attr-dict
```


Breaks into the attached debugger or asks for attaching a debugger if the
provided condition is true. After resuming (or if a debugger is not
attached) execution will continue at the target block.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`condition` | 32-bit signless integer
`destOperands` | 32-bit signless integer or 32-bit signless integer or ref

#### Successors:

| Successor | Description |
| :-------: | ----------- |
`dest` | any successor

### `vm.const.i32` (IREE::VM::ConstI32Op)

32-bit integer constant operation

Defines a constant value that is treated as a scalar literal at runtime.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`value` | Attribute | anonymous_386

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.const.i32.zero` (IREE::VM::ConstI32ZeroOp)

32-bit integer constant zero operation

Syntax:

```
operation ::= `vm.const.i32.zero` `:` type($result) attr-dict
```


Defines a constant zero 32-bit integer.

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.const.ref.rodata` (IREE::VM::ConstRefRodataOp)

constant rodata access operation

Syntax:

```
operation ::= `vm.const.ref.rodata` $rodata attr-dict `:` type($value)
```


Returns a reference to a read-only buffer.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`rodata` | FlatSymbolRefAttr | symbol reference attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`value` | ref<byte_buffer>

### `vm.const.ref.zero` (IREE::VM::ConstRefZeroOp)

null ref_ptr constant operation

Syntax:

```
operation ::= `vm.const.ref.zero` `:` type($result) attr-dict
```


Defines a constant null ref_ptr that can be used in comparisons and
initialization.

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ref

### `vm.div.i32.s` (IREE::VM::DivI32SOp)

signed integer division operation

Syntax:

```
operation ::= `vm.div.i32.s` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.div.i32.u` (IREE::VM::DivI32UOp)

unsigned integer division operation

Syntax:

```
operation ::= `vm.div.i32.u` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.export` (IREE::VM::ExportOp)

exports a function from the module

Specifies an exported function with an externally-visible alias. Multiple
exports can reference the same internal functions.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`function_ref` | FlatSymbolRefAttr | flat symbol reference attribute
`export_name` | StringAttr | string attribute
`ordinal` | IntegerAttr | ordinal value

### `vm.ext.i16.i32.s` (IREE::VM::ExtI16I32SOp)

integer sign extend 16 bits to 32 bits

Syntax:

```
operation ::= `vm.ext.i16.i32.s` $operand attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.ext.i8.i32.s` (IREE::VM::ExtI8I32SOp)

integer sign extend 8 bits to 32 bits

Syntax:

```
operation ::= `vm.ext.i8.i32.s` $operand attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.fail` (IREE::VM::FailOp)

raises a global failure

Syntax:

```
operation ::= `vm.fail` $status (`,` $message^)? attr-dict
```


Signals a runtime failure that causes the entire active invocation - and
possibly *all* in-flight and pending invocations - to fail with the given
status. The status will be propagated back via the available runtime error
handling mechanisms such as semaphores or synchronous invocation results.

As the IREE execution model is deeply pipelined it's possible that failures
have a latency between when they are emitted and when the application can
observe the failure. It's also possible that other work that is in-flight
or pending when the failure occurs will complete.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`message` | StringAttr | string attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`status` | 32-bit signless integer

### `vm.func` (IREE::VM::FuncOp)

function defined with VM control flow ops

Represents a function containing VM ops and those of compatible dialects.
All flow control is performed by VM ops.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`ordinal` | IntegerAttr | ordinal value
`noinline` | UnitAttr | unit attribute

### `vm.global.address` (IREE::VM::GlobalAddressOp)

returns an address reference to a global

Syntax:

```
operation ::= `vm.global.address` $global attr-dict `:` type($result)
```


Returns an indirect address reference to the given global. During export the
address will be converted to the natural format of the global table (for
example, ordinals for refs and byte offsets for primitive types).

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`global` | FlatSymbolRefAttr | symbol reference attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ptr

### `vm.global.i32` (IREE::VM::GlobalI32Op)

32-bit integer global declaration

Defines a global value that is treated as a scalar literal at runtime.
Initialized to zero unless a custom initializer function is specified.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`sym_name` | StringAttr | string attribute
`type` | TypeAttr | any type attribute
`is_mutable` | UnitAttr | unit attribute
`initializer` | FlatSymbolRefAttr | flat symbol reference attribute
`initial_value` | Attribute | anonymous_389
`ordinal` | IntegerAttr | ordinal value

### `vm.global.load.i32` (IREE::VM::GlobalLoadI32Op)

global 32-bit integer load operation

Syntax:

```
operation ::= `vm.global.load.i32` $global attr-dict `:` type($value)
```


Loads the value of a global containing a 32-bit integer.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`global` | FlatSymbolRefAttr | symbol reference attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`value` | 32-bit signless integer

### `vm.global.load.indirect.i32` (IREE::VM::GlobalLoadIndirectI32Op)

global 32-bit integer load operation

Syntax:

```
operation ::= `vm.global.load.indirect.i32` $global attr-dict `:` type($global) `->` type($value)
```


Loads the value of a global containing a 32-bit integer.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`global` | 32-bit signless integer or ptr<32-bit signless integer>

#### Results:

| Result | Description |
| :----: | ----------- |
`value` | 32-bit signless integer

### `vm.global.load.indirect.ref` (IREE::VM::GlobalLoadIndirectRefOp)

global ref_ptr<T> load operation

Syntax:

```
operation ::= `vm.global.load.indirect.ref` $global attr-dict `:` type($global) `->` type($value)
```


Loads the value of a global containing a ref_ptr of the given type.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`global` | 32-bit signless integer or ptr<ref>

#### Results:

| Result | Description |
| :----: | ----------- |
`value` | ref

### `vm.global.load.ref` (IREE::VM::GlobalLoadRefOp)

global ref_ptr<T> load operation

Syntax:

```
operation ::= `vm.global.load.ref` $global attr-dict `:` type($value)
```


Loads the value of a global containing a ref_ptr of the given type.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`global` | FlatSymbolRefAttr | symbol reference attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`value` | ref

### `vm.global.ref` (IREE::VM::GlobalRefOp)

ref_ptr<T> global declaration

Defines a global value that is a ref_ptr of a specific type. The global will
retain the ref object for the lifetime of the context or until the value is
replaced with a store or reset.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`sym_name` | StringAttr | string attribute
`type` | TypeAttr | any type attribute
`is_mutable` | UnitAttr | unit attribute
`initializer` | FlatSymbolRefAttr | flat symbol reference attribute
`initial_value` | UnitAttr | unit attribute
`ordinal` | IntegerAttr | ordinal value

### `vm.global.store.i32` (IREE::VM::GlobalStoreI32Op)

global 32-bit integer store operation

Syntax:

```
operation ::= `vm.global.store.i32` $value `,` $global attr-dict `:` type($value)
```


Stores the 32-bit integer value to a global.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`global` | FlatSymbolRefAttr | symbol reference attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | 32-bit signless integer

### `vm.global.store.indirect.i32` (IREE::VM::GlobalStoreIndirectI32Op)

global 32-bit integer store operation

Syntax:

```
operation ::= `vm.global.store.indirect.i32` $value `,` $global attr-dict `:` type($value) `->` type($global)
```


Stores the 32-bit integer value to a global.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | 32-bit signless integer
`global` | 32-bit signless integer or ptr<32-bit signless integer>

### `vm.global.store.indirect.ref` (IREE::VM::GlobalStoreIndirectRefOp)

global ref_ptr<T> stores operation

Syntax:

```
operation ::= `vm.global.store.indirect.ref` $value `,` $global attr-dict `:` type($value) `->` type($global)
```


Stores a ref_ptr<T> to a global, retaining it until the global is reset.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | ref
`global` | 32-bit signless integer or ptr<ref>

### `vm.global.store.ref` (IREE::VM::GlobalStoreRefOp)

global ref_ptr<T> stores operation

Syntax:

```
operation ::= `vm.global.store.ref` $value `,` $global attr-dict `:` type($value)
```


Stores a ref_ptr<T> to a global, retaining it until the global is reset.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`global` | FlatSymbolRefAttr | symbol reference attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | ref

### `vm.import` (IREE::VM::ImportOp)

imports a function from an external module

Specifies a function that should be imported from either the runtime or
an external VM module.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`ordinal` | IntegerAttr | ordinal value

### `vm.module` (IREE::VM::ModuleOp)

module containing VM functions and variables

Top-level container for VM functions.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`sym_name` | StringAttr | string attribute
`ordinal_counts` | DictionaryAttr | dictionary of named attribute values

### `vm.module_terminator` (IREE::VM::ModuleTerminatorOp)

terminator pseudo-op for the module op

Syntax:

```
operation ::= `vm.module_terminator` attr-dict
```



### `vm.mul.i32` (IREE::VM::MulI32Op)

integer multiplication operation

Syntax:

```
operation ::= `vm.mul.i32` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.not.i32` (IREE::VM::NotI32Op)

integer binary not operation

Syntax:

```
operation ::= `vm.not.i32` $operand attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.or.i32` (IREE::VM::OrI32Op)

integer binary or operation

Syntax:

```
operation ::= `vm.or.i32` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.print` (IREE::VM::PrintOp)

message printing operation

Syntax:

```
operation ::= `vm.print` $message `(` operands `)` attr-dict `:` type(operands)
```


Prints the given string message and zero or more values.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`message` | StringAttr | string attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operands` | 32-bit signless integer or 32-bit signless integer or ref

### `vm.rem.i32.s` (IREE::VM::RemI32SOp)

signed integer division remainder operation

Syntax:

```
operation ::= `vm.rem.i32.s` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.rem.i32.u` (IREE::VM::RemI32UOp)

unsigned integer division remainder operation

Syntax:

```
operation ::= `vm.rem.i32.u` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.return` (IREE::VM::ReturnOp)

return operation

Syntax:

```
operation ::= `vm.return` attr-dict ($operands^ `:` type($operands))?
```


Represents a return operation within a function.

```
vm.func @foo(%0, %1) : (i32, f8) {
  vm.return %0, %1 : i32, f8
}
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operands` | 32-bit signless integer or 32-bit signless integer or ref

### `vm.rodata` (IREE::VM::RodataOp)

read-only data definition operation

Defines a blob of read-only constant data that can be represented as a
ref_ptr. This can be used to store arbitrary data within modules such as
large constant buffers and other file contents.

Note that the data is reference counted as a way to track its usage once the
value leaves the module. For example, returning rodata from an exported
function must keep the data (possibly backed by mmap) valid for its entire
lifetime.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`sym_name` | StringAttr | string attribute
`value` | ElementsAttr | constant vector/tensor attribute
`ordinal` | IntegerAttr | ordinal value

### `vm.select.i32` (IREE::VM::SelectI32Op)

integer select operation

Syntax:

```
operation ::= `vm.select.i32` operands attr-dict `:` type($result)
```


Chooses one value based on a binary condition supplied as its first operand.
If the value of the condition is true the `true_value` operand is chosen,
otherwise the `false_value` operand is chosen. The true and false values
must have the same types. For example, the maximum operation is obtained by
combining "select" with "cmpi" as follows:

```
%2 = vm.cmp.gt.i32.s %0, %1 : i32
%3 = vm.select.i32 %2, %0, %1 : i32
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`condition` | 32-bit signless integer
`true_value` | 32-bit signless integer
`false_value` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.select.ref` (IREE::VM::SelectRefOp)

ref_ptr select operation

Syntax:

```
operation ::= `vm.select.ref` operands attr-dict `:` type($result)
```


Chooses one value based on a binary condition supplied as its first operand.
If the value of the condition is true the `true_value` operand is chosen,
otherwise the `false_value` operand is chosen.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`condition` | 32-bit signless integer
`true_value` | ref
`false_value` | ref

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ref

### `vm.shl.i32` (IREE::VM::ShlI32Op)

integer shift left operation

Syntax:

```
operation ::= `vm.shl.i32` $operand `,` $amount attr-dict `:` type($operand)
```


Shifts the operand in a direction by the number of bits specified.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`amount` | IntegerAttr | 8-bit signless integer attribute within the range [0, 32] inclusive

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.shr.i32.s` (IREE::VM::ShrI32SOp)

signed integer (arithmetic) shift right operation

Syntax:

```
operation ::= `vm.shr.i32.s` $operand `,` $amount attr-dict `:` type($operand)
```


Shifts the operand in a direction by the number of bits specified.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`amount` | IntegerAttr | 8-bit signless integer attribute within the range [0, 32] inclusive

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.shr.i32.u` (IREE::VM::ShrI32UOp)

unsigned integer (logical) shift right operation

Syntax:

```
operation ::= `vm.shr.i32.u` $operand `,` $amount attr-dict `:` type($operand)
```


Shifts the operand in a direction by the number of bits specified.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`amount` | IntegerAttr | 8-bit signless integer attribute within the range [0, 32] inclusive

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.sub.i32` (IREE::VM::SubI32Op)

integer subtract operation

Syntax:

```
operation ::= `vm.sub.i32` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.switch.i32` (IREE::VM::SwitchI32Op)

integer switch operation

Syntax:

```
operation ::= `vm.switch.i32` $index `[` $values `]` `else` $default_value attr-dict `:` type($result)
```


Returns the value with the given `index` in `values` or `default_value` if
the index is out of bounds.

```mlir
// Switch %arg0 to cases of %c100/%c200/%c300 if arg0==0, ==1, ==2.
// If %arg0 is out of range (<0 or >2) then default to %c5.
%0 = vm.switch.i32 %index[%c100, %c200, %c300] else %c5 : i32
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`index` | 32-bit signless integer
`default_value` | 32-bit signless integer
`values` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.switch.ref` (IREE::VM::SwitchRefOp)

ref_ptr switch operation

Returns the value with the given `index` in `values` or `default_value` if
the index is out of bounds.

```mlir
// Switch %arg0 to cases of %r0/%r1/%r2 if arg0==0, ==1, ==2.
// If %arg0 is out of range (<0 or >2) then default to %null.
%0 = vm.switch.ref %index[%r0, %r1, %r2] else %null : vm.ref<!foo>
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`index` | 32-bit signless integer
`default_value` | ref
`values` | ref

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ref

### `vm.trace` (IREE::VM::TraceOp)

trace value(s) operation

Syntax:

```
operation ::= `vm.trace` $event_name `(` operands `)` attr-dict `:` type(operands)
```


Traces one or more values at the time the operation is executed.
These values will be encoded into the active trace depending on the active
trace verbosity setting.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`event_name` | StringAttr | string attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operands` | 32-bit signless integer or 32-bit signless integer or ref

### `vm.trunc.i16` (IREE::VM::TruncI16Op)

integer truncate to 16 bits

Syntax:

```
operation ::= `vm.trunc.i16` $operand attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.trunc.i8` (IREE::VM::TruncI8Op)

integer truncate to 8 bits

Syntax:

```
operation ::= `vm.trunc.i8` $operand attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.xor.i32` (IREE::VM::XorI32Op)

integer binary exclusive-or operation

Syntax:

```
operation ::= `vm.xor.i32` operands attr-dict `:` type($result)
```



#### Operands:

| Operand | Description |
| :-----: | ----------- |
`lhs` | 32-bit signless integer
`rhs` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | 32-bit signless integer

### `vm.yield` (IREE::VM::YieldOp)

unconditional fiber yield operation

Syntax:

```
operation ::= `vm.yield` attr-dict
```


Yields the fiber for some (likely short) amount of time. This can be used to
perform cooperative scheduling and ensure fair (enough) execution.