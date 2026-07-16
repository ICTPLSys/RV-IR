# RAIR Dialect Operations Reference

RAIR is the **RISC-V Accelerator Interface Representation** dialect in RV-IR.
It is implemented as an MLIR dialect with the textual namespace `rair` and the
C++ namespace `rair`.

Source files:

```text
include/torch-mlir/Dialect/RISCV/IR/RISCVDialect.td
include/torch-mlir/Dialect/RISCV/IR/RISCVOps.td
lib/Dialect/RISCV/IR/RISCVDialect.cpp
```

---

## Design Position

RAIR sits between high-level compute IR and target-specific backend IR. Its
central contribution is making accelerator interface semantics compiler-visible:

```text
Torch / Linalg / Tensor IR
  -> RAIR (accelerator semantics visible here)
  -> Affine / Async / LLVM / CIM backends
```

RAIR exposes:

- **Resource management**: accelerator context acquisition and release
- **Data management**: explicit memory-space transfers and buffer allocation
- **Compute operations**: matrix multiply, convolution, pooling, element-wise, reduction
- **Structured execution regions**: kernel and launch boundaries for fusion/scheduling
- **Synchronization**: async tokens, await, wait-all, barrier
- **Target attributes**: accelerator, tile_size, dataflow, fallback

---

## Dialect Types

### `!rair.async.token`

Represents completion of an asynchronous RAIR operation.

```mlir
%token = rair.wait_all async [%dep0, %dep1]
```

### `!rair.event`

Represents an event-style synchronization value used during lowering.

### `!rair.context`

Represents an accelerator execution context. The context models resource
ownership of a RoCC-style accelerator and must be explicitly acquired and
released.

```mlir
%ctx = rair.acquire {accelerator = "gemmini"} : !rair.context
// ... use accelerator ...
rair.release %ctx : !rair.context
```

---

## Memory Spaces

| Name | Value | Intended Use |
| --- | ---: | --- |
| `LMEM` | 0 | local accelerator memory |
| `CIMC0` | 1 | CIM compute/memory region |
| `SPAD3` | 2 | scratchpad bank 3 |
| `SPAD2` | 3 | scratchpad bank 2 |
| `SPAD1` | 4 | scratchpad bank 1 |
| `SPAD0` | 5 | scratchpad bank 0 |
| `GMEM` | 6 | global memory (host DRAM) |

---

## Interfaces

### `RAIR_AsyncOpInterface`

Async dependency tokens + optional async token result.
Used by: `rair.launch`, `rair.herd`, `rair.load`, `rair.store`, `rair.wait_all`.

### `RAIR_HierarchyInterface`

Structured region ops with iteration space.
Used by: `rair.launch`, `rair.herd`.

### `RAIR_MemcpyInterface`

DMA-like ops with src/dst memrefs, offsets, sizes, strides.
Used by: `rair.load`, `rair.store`.

---

## Optional Accelerator Attributes

Most compute ops support the following optional attributes:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `accelerator` | `StrAttr` | Target accelerator identifier (e.g., `"gemmini"`, `"cim_npu"`) |
| `tile_size` | `DenseI64ArrayAttr` | Target-aware tiling parameters (e.g., `array<i64: 16, 16, 16>`) |
| `dataflow` | `StrAttr` | Accelerator dataflow strategy (e.g., `"weight_stationary"`, `"output_stationary"`) |
| `fallback` | `FlatSymbolRefAttr` | CPU fallback function for correctness-preserving offload |

---

## Resource Management Operations

### `rair.acquire`

Acquires an accelerator execution context. The context represents resource
ownership of a RoCC-style accelerator.

```mlir
// Acquire context for the default accelerator
%ctx = rair.acquire {accelerator = "default"} : !rair.context

// Acquire context for a specific accelerator
%ctx_gem = rair.acquire {accelerator = "gemmini"} : !rair.context
```

### `rair.release`

Releases a previously acquired accelerator context, freeing associated resources.

```mlir
rair.release %ctx : !rair.context
```

### `rair.await`

Blocks until all pending accelerator operations associated with the given context
have completed. Makes synchronization an explicit, analyzable program event.

```mlir
rair.await %ctx : !rair.context
```

---

## Data Management Operations

### `rair.alloc`

Allocates a memref in a selected memory space (via memref type's memory space).

```mlir
%buf = rair.alloc : memref<1024xi32, 6>
%spad = rair.alloc : memref<128x128xf32, 5>
```

### `rair.dealloc`

Deallocates a memref.

```mlir
rair.dealloc %buf : memref<1024xi32, 6>
```

### `rair.alloc_buffer`

Allocates a buffer in accelerator-visible memory, associated with an accelerator
context. Enables buffer lifetime analysis and memory reuse across kernels.

```mlir
%buf_sp = rair.alloc_buffer %ctx {memory_space = "scratchpad"}
          : memref<128x128xf32>

%buf_acc = rair.alloc_buffer %ctx {memory_space = "accumulator"}
           : memref<64x64xf32>
```

### `rair.dealloc_buffer`

Deallocates a buffer previously allocated in accelerator-visible memory.

```mlir
rair.dealloc_buffer %ctx, %buf_sp : memref<128x128xf32>
```

### `rair.transfer`

Explicitly transfers data between memory spaces. Making transfers explicit
enables redundant transfer elimination, transfer/compute overlap, and memory
lifetime analysis.

```mlir
// Transfer host data to scratchpad
rair.transfer %host_A to %spad_A
  {src_memory_space = "host", dst_memory_space = "scratchpad"}
  : memref<128x128xf32>, memref<128x128xf32>

// Transfer result back from accelerator to host
rair.transfer %accel_out to %host_out : memref<64x64xf32>, memref<64x64xf32>
```

### `rair.load`

DMA-like asynchronous load/copy operation with offset/size/stride support.

```mlir
%token = rair.load async [%dep]
  (%dst[][][][], %src[][][][])
  : (memref<128x128xf32>, memref<128x128xf32>)
```

### `rair.store`

DMA-like asynchronous store/copy operation (same structure as `rair.load`).

```mlir
%token = rair.store async [%dep]
  (%dst[%d_off][%d_sz][%d_str], %src[%s_off][%s_sz][%s_str])
  : (memref<64x64xf32>, memref<64x64xf32>)
```

---

## Structured Execution Region Operations

### `rair.launch`

Models a structured accelerator launch region. Supports async dependencies,
index iteration space, captured kernel operands, and optional fallback.

```mlir
%token = rair.launch async [%dep] (%i) in (%sz=%c4)
  args(%a = %hostA : memref<128x128xf32>)
  attributes {accelerator = "gemmini", fallback = @cpu_fallback} {
  rair.matmul ins(%a, %b : memref<128x128xf32>, memref<128x128xf32>)
               outs(%c : memref<128x128xf32>)
  rair.launch_terminator
}
```

### `rair.launch_terminator`

Terminator for `rair.launch` bodies.

```mlir
rair.launch_terminator
```

### `rair.herd`

Models a structured group of accelerator workers or tiles with 2-D placement.

```mlir
%token = rair.herd async (%x, %y) in (%sx=%c2, %sy=%c2)
  args(%buf = %data : memref<256x256xf32>) {
  // per-tile computation
  rair.herd_terminator
}
```

### `rair.herd_terminator`

Terminator for `rair.herd` bodies.

```mlir
rair.herd_terminator
```

### `rair.kernel`

Defines a structured kernel region for accelerator execution. Encapsulates one
or more compute operations that should execute as a unit. Provides the boundary
for fusion, scheduling, and accelerator-side optimization. Supports `fallback`
for correctness-preserving offload selection.

```mlir
rair.kernel attributes {accelerator = "gemmini",
                        fallback = @cpu_matmul,
                        tile_size = array<i64: 16, 16, 16>,
                        dataflow = "weight_stationary"} {
  rair.matmul ins(%a, %b : memref<128x128xf32>, memref<128x128xf32>)
               outs(%c : memref<128x128xf32>)
  rair.kernel_terminator
}
```

### `rair.kernel_terminator`

Terminator for `rair.kernel` bodies.

```mlir
rair.kernel_terminator
```

---

## Synchronization Operations

### `rair.wait_all`

Waits for all async dependency tokens. May itself produce an async token.

```mlir
// Fire-and-forget wait
rair.wait_all [%tok_a, %tok_b]

// Produce a new token after all deps complete
%done = rair.wait_all async [%tok_a, %tok_b]
```

### `rair.barrier`

Tensor-level barrier. Blocks until the value of the associated future is present.

```mlir
%result = rair.barrier %future_val : tensor<4x4xf32> -> tensor<4x4xf32>
```

### `rair.noc`

Placeholder for on-chip network communication.

```mlir
%result = rair.noc %data : tensor<16xf32> -> tensor<16xf32>
```

---

## Compute Operations: Matrix and Tensor

### `rair.matmul`

In-place 2-D matrix multiplication: `output = lhs * rhs`.

```mlir
// Basic usage
rair.matmul ins(%A, %B : memref<16x32xf32>, memref<32x64xf32>)
             outs(%C : memref<16x64xf32>)

// With accelerator attributes
rair.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
             outs(%C : memref<128x128xf32>)
             {accelerator = "gemmini",
              tile_size = array<i64: 16, 16, 16>,
              dataflow = "weight_stationary"}
```

### `rair.batch_matmul`

In-place batched matrix multiplication: `output[b] = lhs[b] * rhs[b]`.

```mlir
rair.batch_matmul ins(%lhs, %rhs : memref<4x128x256xf32>, memref<4x256x512xf32>)
                   outs(%out : memref<4x128x512xf32>)
                   {accelerator = "gemmini"}
```

### `rair.matvec`

Matrix-vector multiplication (GEMV): `y = A * x`.

```mlir
%y = "rair.matvec"(%A, %x) {accelerator = "default"}
     : (tensor<128x64xf32>, tensor<64xf32>) -> tensor<128xf32>
```

### `rair.conv2d`

2-D convolution on tensor operands.

```mlir
%output = "rair.conv2d"(%input, %kernel)
           {accelerator = "gemmini",
            tile_size = array<i64: 8, 8>,
            dataflow = "output_stationary"}
           : (tensor<32x32xf32>, tensor<3x3xf32>) -> tensor<30x30xf32>
```

### `rair.conv_2d_nchw_fchw`

NCHW-layout 2-D convolution on memref operands with stride/dilation.

```mlir
rair.conv_2d_nchw_fchw
  {dilations = dense<[1, 1]> : vector<2xi64>,
   strides   = dense<[1, 1]> : vector<2xi64>,
   accelerator = "gemmini"}
  ins(%input, %filter : memref<1x3x224x224xf32>, memref<64x3x7x7xf32>)
  outs(%output : memref<1x64x218x218xf32>)
```

### `rair.pooling_nchw_max`

NCHW max pooling.

```mlir
rair.pooling_nchw_max
  {dilations = dense<[1, 1]> : vector<2xi64>,
   strides   = dense<[2, 2]> : vector<2xi64>,
   accelerator = "gemmini"}
  ins(%input, %kernel : memref<1x64x112x112xf32>, memref<3x3xf32>)
  outs(%output : memref<1x64x56x56xf32>)
```

### `rair.pooling_nchw_sum`

NCHW sum pooling (average pooling without division).

```mlir
rair.pooling_nchw_sum
  {dilations = dense<[1, 1]> : vector<2xi64>,
   strides   = dense<[2, 2]> : vector<2xi64>}
  ins(%input, %kernel : memref<1x64x112x112xf32>, memref<3x3xf32>)
  outs(%output : memref<1x64x56x56xf32>)
```

### `rair.transpose`

In-place tensor transpose with arbitrary permutation.

```mlir
rair.transpose ins(%src : memref<64x32xf32>)
               outs(%dst : memref<32x64xf32>)
               {permutation = array<i64: 1, 0>, accelerator = "default"}
```

### `rair.reshape`

Tensor reshape (same number of elements, different shape).

```mlir
%dest = "rair.reshape"(%src, %shape)
        : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
```

### `rair.reduce`

Reduction over one or more dimensions. Supports `kind = "sum" | "max" | "min"`.

```mlir
// Sum reduction over dimension 0
%res = "rair.reduce"(%input) {kind = "sum", dim = [0], accelerator = "default"}
       : (tensor<3x2x4xf64>) -> tensor<1x2x4xf64>

// Max reduction over dimensions 0 and 1
%res2 = "rair.reduce"(%input) {kind = "max", dim = [0, 1]}
        : (tensor<3x2x4xf64>) -> tensor<1x1x4xf64>
```

---

## Compute Operations: Element-wise Arithmetic

### `rair.add`

Element-wise addition (integer or float).

```mlir
%result = "rair.add"(%a, %b) {accelerator = "default"}
          : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
```

### `rair.sub`

Element-wise subtraction.

```mlir
%result = "rair.sub"(%a, %b)
          : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
```

### `rair.mul`

Element-wise multiplication.

```mlir
%result = "rair.mul"(%a, %b) {accelerator = "cim_npu"}
          : (tensor<8x8xf16>, tensor<8x8xf16>) -> tensor<8x8xf16>
```

### `rair.div`

Element-wise division.

```mlir
%result = "rair.div"(%a, %b)
          : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
```

### `rair.max`

Element-wise maximum.

```mlir
%result = "rair.max"(%a, %b)
          : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
```

### `rair.min`

Element-wise minimum.

```mlir
%result = "rair.min"(%a, %b)
          : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
```

### `rair.negf`

Floating-point negation.

```mlir
%neg = "rair.negf"(%x) {accelerator = "default"}
       : (tensor<4x2xf32>) -> tensor<4x2xf32>
```

---

## Compute Operations: Integer Bitwise

### `rair.andi`

Integer bitwise AND.

```mlir
// Scalar
%a = rair.andi %b, %c : i64

// Tensor element-wise
%x = rair.andi %y, %z : tensor<4x8xi8>
```

### `rair.xori`

Integer bitwise XOR.

```mlir
%a = rair.xori %b, %c : i64
%f = rair.xori %g, %h : tensor<4xi32>
```

### `rair.ori`

Integer bitwise OR.

```mlir
%a = rair.ori %b, %c : i64
%f = rair.ori %g, %h : tensor<4xi32>
```

---

## Compute Operations: Comparison

### `rair.cmpi`

Integer comparison. Predicates: `eq`, `ne`, `slt`, `sle`, `sgt`, `sge`, `ult`,
`ule`, `ugt`, `uge`.

```mlir
// Scalar comparison
%0 = "rair.cmpi"(%lhs, %rhs) {predicate = "slt"} : (i32, i32) -> i1

// Tensor element-wise comparison
%1 = "rair.cmpi"(%a, %b) {predicate = "eq"}
     : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
```

### `rair.cmpf`

Floating-point comparison.

```mlir
%0 = "rair.cmpf"(%lhs, %rhs) {predicate = "olt"} : (f32, f32) -> i1

%1 = "rair.cmpf"(%a, %b) {predicate = "oeq"}
     : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
```

---

## Function-Like and Utility Operations

### `rair.func`

RAIR function-like operation implementing `FunctionOpInterface`.

```mlir
rair.func @main() {
  %0 = rair.constant dense<5.5> : tensor<f64>
  rair.print %0 : tensor<f64>
  rair.return
}
```

### `rair.return`

Return terminator for `rair.func`.

```mlir
rair.func @add(%a: tensor<2xf64>, %b: tensor<2xf64>) -> tensor<2xf64> {
  %c = "rair.add"(%a, %b) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  rair.return %c : tensor<2xf64>
}
```

### `rair.constant`

Constant operation that turns a literal into an SSA value.

```mlir
// Dense tensor constant
%0 = "rair.constant"()
     {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>}
     : () -> tensor<2x2xf64>

// Scalar constant
%1 = "rair.constant"() {value = dense<42> : tensor<i64>}
     : () -> tensor<i64>
```

### `rair.print`

Debug print operation for tensors or memrefs.

```mlir
rair.print %tensor : tensor<2x3xf64>
rair.print %memref : memref<4x4xf32>
```

### `rair.world`

Debug operation that prints "RAIR, World!".

```mlir
rair.world
```

---

## Linalg to RAIR Lowering

Pass flag:

```bash
--convert-linalg-to-rair
```

Source: `lib/Conversion/LinalgToRISCV/LinalgToRISCV.cpp`

### Operation Mappings

| Linalg Source | RAIR Target |
| --- | --- |
| `linalg.matmul` | `rair.matmul` |
| `linalg.batch_matmul` | `rair.batch_matmul` |
| `linalg.matvec` | `rair.matvec` |
| `linalg.reduce` | `rair.reduce` |
| `linalg.conv2d` | `rair.conv2d` |
| `linalg.conv_2d_nchw_fchw` | `rair.conv_2d_nchw_fchw` |
| `linalg.transpose` | `rair.transpose` |
| `linalg.add` | `rair.add` |
| `linalg.sub` | `rair.sub` |
| `linalg.mul` | `rair.mul` |
| `linalg.div` | `rair.div` |
| `linalg.negf` | `rair.negf` |
| `linalg.max` | `rair.max` |
| `linalg.min` | `rair.min` |
| `linalg.pooling_nchw_max` | `rair.pooling_nchw_max` |
| `linalg.pooling_nchw_sum` | `rair.pooling_nchw_sum` |

### Context Management

After converting individual operations, the pass automatically inserts accelerator
context management for functions containing RAIR compute ops:

```mlir
// Before (Linalg):
func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>,
                  %C: memref<128x128xf32>) {
  linalg.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
                 outs(%C : memref<128x128xf32>)
  return
}

// After (RAIR):
func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>,
                  %C: memref<128x128xf32>) {
  %ctx = rair.acquire {accelerator = "default"} : !rair.context
  rair.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
               outs(%C : memref<128x128xf32>)
  rair.release %ctx : !rair.context
  return
}
```

---

## RAIR Downstream Lowering

### RAIR to Affine (CPU path)

```bash
--convert-rair-to-affine
```

Lowers compute ops to affine loop nests, launch/herd to nested loops, async to
`async.execute`/`async.await`.

### RAIR to LLVM

```bash
--convert-rair-to-llvm
```

Lowers debug ops (`rair.print`, `rair.world`) plus standard MLIR dialects to
LLVM IR.

### RAIR to CIM (accelerator path)

```bash
--convert-rair-to-cim
```

Lowers compute ops to CIM hardware intrinsic calls (`llvm.riscv.vv.v.drv`,
`llvm.riscv.conv.drv`, etc.).

---

## End-to-End Example

Full pipeline from Linalg to executable LLVM IR (CPU path):

```bash
torch-mlir-opt input.mlir \
  --convert-linalg-to-rair \
  --convert-rair-to-affine \
  --convert-rair-to-llvm
```

Full pipeline to CIM hardware (accelerator path):

```bash
torch-mlir-opt input.mlir \
  --convert-linalg-to-rair \
  --convert-rair-to-cim \
  --convert-cim-to-llvm
```

---

## Complete Operation Summary

| # | Operation | Category | Description |
|---|-----------|----------|-------------|
| 1 | `rair.acquire` | Resource | Acquire accelerator context |
| 2 | `rair.release` | Resource | Release accelerator context |
| 3 | `rair.await` | Resource | Wait for accelerator completion |
| 4 | `rair.alloc` | Memory | Allocate memref |
| 5 | `rair.dealloc` | Memory | Deallocate memref |
| 6 | `rair.alloc_buffer` | Memory | Allocate accelerator-visible buffer |
| 7 | `rair.dealloc_buffer` | Memory | Deallocate accelerator-visible buffer |
| 8 | `rair.transfer` | Memory | Transfer between memory spaces |
| 9 | `rair.load` | DMA | Async DMA load |
| 10 | `rair.store` | DMA | Async DMA store |
| 11 | `rair.launch` | Region | Structured accelerator launch |
| 12 | `rair.launch_terminator` | Region | Terminator for launch |
| 13 | `rair.herd` | Region | Structured worker group |
| 14 | `rair.herd_terminator` | Region | Terminator for herd |
| 15 | `rair.kernel` | Region | Structured kernel region |
| 16 | `rair.kernel_terminator` | Region | Terminator for kernel |
| 17 | `rair.wait_all` | Sync | Wait for multiple async tokens |
| 18 | `rair.barrier` | Sync | Tensor-level barrier |
| 19 | `rair.noc` | Sync | On-chip network communication |
| 20 | `rair.matmul` | Compute | Matrix multiplication |
| 21 | `rair.batch_matmul` | Compute | Batched matrix multiplication |
| 22 | `rair.matvec` | Compute | Matrix-vector multiplication |
| 23 | `rair.conv2d` | Compute | 2-D convolution (tensor) |
| 24 | `rair.conv_2d_nchw_fchw` | Compute | NCHW convolution (memref) |
| 25 | `rair.pooling_nchw_max` | Compute | NCHW max pooling |
| 26 | `rair.pooling_nchw_sum` | Compute | NCHW sum pooling |
| 27 | `rair.transpose` | Compute | Tensor transpose |
| 28 | `rair.reshape` | Compute | Tensor reshape |
| 29 | `rair.reduce` | Compute | Tensor reduction |
| 30 | `rair.add` | Arithmetic | Element-wise addition |
| 31 | `rair.sub` | Arithmetic | Element-wise subtraction |
| 32 | `rair.mul` | Arithmetic | Element-wise multiplication |
| 33 | `rair.div` | Arithmetic | Element-wise division |
| 34 | `rair.max` | Arithmetic | Element-wise maximum |
| 35 | `rair.min` | Arithmetic | Element-wise minimum |
| 36 | `rair.negf` | Arithmetic | Float negation |
| 37 | `rair.andi` | Bitwise | Integer AND |
| 38 | `rair.xori` | Bitwise | Integer XOR |
| 39 | `rair.ori` | Bitwise | Integer OR |
| 40 | `rair.cmpi` | Comparison | Integer comparison |
| 41 | `rair.cmpf` | Comparison | Float comparison |
| 42 | `rair.func` | Function | RAIR function definition |
| 43 | `rair.return` | Function | RAIR function return |
| 44 | `rair.constant` | Utility | Constant value |
| 45 | `rair.print` | Utility | Debug print |
| 46 | `rair.world` | Utility | Print "RAIR, World!" |
