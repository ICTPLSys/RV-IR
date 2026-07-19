# RAIR Dialect Operations Reference

RAIR is the **Resource-Adaptive Intermediate Representation** dialect in RV-IR.
It is a target-neutral resource-effect layer with the textual namespace `rair`
and the C++ namespace `rair`; RISC-V/Gemmini is the first target instance.

Source files:

```text
include/torch-mlir/Dialect/RAIR/IR/RAIRDialect.td
include/torch-mlir/Dialect/RAIR/IR/RAIROps.td
lib/Dialect/RAIR/IR/RAIRDialect.cpp
```

---

## Design Position

RAIR sits between high-level compute IR and target-specific backend IR. Its
central contribution is making accelerator interface semantics compiler-visible:

```text
Torch / Linalg / Tensor IR
  -> RAIR (accelerator semantics visible here)
  -> RAIR Core/Plan-aware target lowering
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

### `!rair.event<@target>`

Represents a target-aware RAIR Plan completion event.

### `!rair.region`

Represents an analyzable buffer slice created by `rair.view`.

### `!rair.lease`

Represents the linear lifetime capability returned by `rair.reserve`.

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

Core v0.1 uses only the target-neutral spellings:

| Name | Value | Intended Use |
| --- | ---: | --- |
| `HOST` | 7 | host-visible memory |
| `DEVICE` | 8 | device-global memory |
| `SPAD` | 9 | explicitly managed scratchpad |
| `ACC` | 10 | accumulator/partial-sum memory |
| `UNKNOWN` | 11 | conservative import only; rejected by Core |

The following values are pre-Core compatibility spellings and are rejected by
the new Core op verifiers:

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

## Core v0.1 Operations

| Operation | Contract |
| --- | --- |
| `rair.target` | module-level static target symbol |
| `rair.scope` | single-block Core reference trace and lifetime boundary |
| `rair.view` | in-bounds static zero-copy region over a typed memref |
| `rair.reserve` | static device/spad/acc buffer plus one linear lease |
| `rair.release_lease` | matching lease/buffer release after every use |
| `rair.move` | equal-count, equal-element-type region copy with read/write effects |
| `rair.compute` | one static memref `linalg.matmul` with read/read/readwrite effects |

`rair.compute` retains Linalg as the compute semantic anchor. Core does not
introduce a second matmul definition. `rair.release_lease` is intentionally
distinct from the pre-Core context operation `rair.release` during migration.

The implementation rejects dynamic shapes, out-of-bounds or negative-stride
views, legacy memory-space spellings, unmatched or multiply-used leases,
use-after-release, unequal move element counts, implicit element conversion,
and non-matmul compute payloads.

### `--rair-materialize-static-matmul`

This independent module pass converts eligible static buffer-semantics
`linalg.matmul` operations into Core. It does not call or change legacy
`--convert-linalg-to-rair`.

```text
external A/B/C views
  -> reserve A/B in SPAD and C in ACC
  -> move A, B, and initial C into local storage
  -> rair.compute { cloned linalg.matmul }
  -> move C back
  -> release C, B, A leases
```

The initial C move is semantically required because Linalg matmul accumulates
into a ReadWrite output. The pass validates all candidates before rewriting,
creates or reuses `rair.target @rair_default`, strips external layout from
contiguous local buffers, and is idempotent. v0.1 accepts only rank-2 static
memrefs in `host`/`device` space and gives each matmul its own scope; target
selection, tiling, capacity checks, Plan IR, and backend lowering are deferred.

### `--rair-infer-effects`

This module-level pass reports Core effects without modifying IR. Within each
`rair.scope`, views and actions receive deterministic textual-order IDs.

| Action | Normalized footprint |
| --- | --- |
| `rair.reserve` | lease/buffer allocation |
| `rair.move` | `Read(src), Write(dst)` |
| `rair.compute` | `Read(lhs), Read(rhs), ReadWrite(output)` |
| `rair.release_lease` | matching lease/buffer free |

The pass consumes the operations' generated `MemoryEffectOpInterface` rather
than duplicating the ODS effect declaration. Region relations are
`disjoint`, `overlap`, or `may_overlap`. Same-base static unit-stride rectangles
are decided from their bounds; different fresh reservations and different
known physical spaces are disjoint; unresolved external aliasing and ambiguous
strided intersections remain may-overlap.

For two actions in reference-trace order, Read/Read is independent. Any RAW,
WAR, or WAW pair on Overlap/MayOverlap regions creates a directed conflict
edge. Lease edges separately encode allocate-before-use, use-before-free, and
allocate-before-free. The report lists all raw edges and every no-edge action
pair under `independent`; it does not perform transitive reduction or add Plan
IR.

```text
actions:
  a3 move effects=[read(r0), write(r3)]
  a4 move effects=[read(r1), write(r4)]
  a5 move effects=[read(r2), write(r5)]
  a6 compute effects=[read(r3), read(r4), readwrite(r5)]
  a7 move effects=[read(r5), write(r2)]
edges:
  a3 -> a6 memory conflict=RAW ... relation=overlap
  a4 -> a6 memory conflict=RAW ... relation=overlap
  a5 -> a6 memory conflict=RAW+WAW ... relation=overlap
  a6 -> a7 memory conflict=RAW ... relation=overlap
independent:
  a3 || a4
```

Same-space function arguments conservatively may alias, so additional
WAR/MayOverlap edges can appear until an explicit no-alias contract exists.

#### Reusable typed graph API

The pass is a thin consumer of
`rair::RAIRStaticEffectGraph::build(rair::ScopeOp)`, declared in
`RAIRStaticEffectGraph.h`. The immutable graph uses typed enums/structs rather
than report strings:

| API value | Meaning |
| --- | --- |
| `StaticOverlapKind` | Disjoint, Overlap, or MayOverlap |
| `StaticAccessKind` | Read, Write, or ReadWrite |
| `StaticActionKind` | Reserve, Move, Compute, or ReleaseLease |
| `StaticMemoryConflict` | RAW/WAR/WAW bitset |
| `StaticLifetimeConstraint` | allocate-before-use, use-before-free, or allocate-before-free |
| `StaticEffectEdge` | ordered action pair with one or more typed reasons |

IDs index stable textual-order arrays. `getRegionId`, `getActionId`,
`getOverlap`, and `hasEdge` support consumers without exposing internal maps.
The graph is valid only until its source scope is mutated. Report spelling and
ordering live in `PrintEffectReport.cpp`; correctness inference lives only in
`RAIRStaticEffectGraph.cpp`.

---

## Plan v0.1 Operations

| Operation / attribute | Contract |
| --- | --- |
| `rair.plan @target` | single-block target-bound task DAG, structurally nested last in its source Scope |
| `rair.task` | one source action descriptor with dependency events and one completion event |
| `rair.plan_terminator` | implicit structural terminator; no completion semantics |
| `#rair.task_kind<...>` | `reserve`, `move`, `compute`, or `release_lease` |

A hand-written Plan uses function-like dependency/result types and remains
inside its source Core scope:

```mlir
rair.scope @gemmini {
  // ... retained Core reference trace ...
  rair.plan @gemmini {
    %reserve = rair.task () {
      kind = #rair.task_kind<reserve>, source_action = 0 : i64
    } : () -> !rair.event<@gemmini>
    %move = rair.task (%reserve) {
      kind = #rair.task_kind<move>, source_action = 1 : i64
    } : (!rair.event<@gemmini>) -> !rair.event<@gemmini>
  }
}
```

Verifier invariants are:

- every Scope has at most one Plan and it follows all Core operations;
- the Plan is directly nested in its source Scope;
- the Plan target resolves to `rair.target` and matches its Scope target;
- the body contains only direct `rair.task` operations and its terminator;
- `source_action` IDs are non-negative, unique, and dense `[0, N)` IDs matching
  the stable action-numbering domain of the source effect graph;
- each result and dependency is `!rair.event` for the Plan target;
- every dependency is produced by a textually earlier task in the same Plan;
- one task cannot list the same dependency twice.

SSA provides the unique event producer, and earlier-producer/single-block rules
provide acyclicity. Event fan-out is legal. Tasks are not `Pure`, so
`--canonicalize --cse` preserves the schedule even when the final event is not
yet consumed.

### `--rair-materialize-plan`

This module pass directly consumes `RAIRStaticEffectGraph` and materializes one
associated Plan per Core Scope:

- action `aN` becomes task `source_action = N` with the corresponding typed
  `TaskKind`;
- every raw edge `aI -> aJ` becomes an event dependency from task `I` to task
  `J`;
- dependency operands are emitted in source-action order;
- Core views/actions/payloads remain in place as the semantic reference trace;
- graph data is copied into a detached specification before Scope mutation;
- all scopes are analyzed before mutation, avoiding partial module updates on
  graph-build or stale-Plan failure.

Running the pass again validates and reuses an existing Plan. Validation checks
task count, kind-to-source-action correspondence, and the dependency set for
every task. Operand order does not affect set equality. A mismatch is diagnosed
as a stale Plan rather than overwritten.

The pass emits all raw graph edges; it does not perform transitive reduction,
resource-capacity or queue scheduling, expose a host completion result, define
a runtime ABI, or invoke a backend.

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
%buf = rair.alloc : memref<1024xi32, #rair.space<gmem>>
%spad = rair.alloc : memref<128x128xf32, #rair.space<spad0>>
```

### `rair.dealloc`

Deallocates a memref.

```mlir
rair.dealloc %buf : memref<1024xi32, #rair.space<gmem>>
```

### `rair.alloc_buffer`

Allocates a buffer in accelerator-visible memory, associated with an accelerator
context. Enables buffer lifetime analysis and memory reuse across kernels.

```mlir
%buf_sp = rair.alloc_buffer %ctx {memory_space = #rair.space<spad0>}
          : memref<128x128xf32, #rair.space<spad0>>

%buf_acc = rair.alloc_buffer %ctx {memory_space = #rair.space<cimc0>}
           : memref<64x64xf32, #rair.space<cimc0>>
```

### `rair.dealloc_buffer`

Deallocates a buffer previously allocated in accelerator-visible memory.

```mlir
rair.dealloc_buffer %ctx, %buf_sp
  : memref<128x128xf32, #rair.space<spad0>>
```

### `rair.transfer`

Explicitly transfers data between memory spaces. Making transfers explicit
enables redundant transfer elimination, transfer/compute overlap, and memory
lifetime analysis.

```mlir
// Transfer global-memory data to scratchpad
rair.transfer %gmem_A to %spad_A
  {src_memory_space = #rair.space<gmem>,
   dst_memory_space = #rair.space<spad0>}
  : memref<128x128xf32, #rair.space<gmem>>,
    memref<128x128xf32, #rair.space<spad0>>

// Transfer result back to global memory
rair.transfer %accel_out to %gmem_out
  {src_memory_space = #rair.space<cimc0>,
   dst_memory_space = #rair.space<gmem>}
  : memref<64x64xf32, #rair.space<cimc0>>,
    memref<64x64xf32, #rair.space<gmem>>
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

Source: `lib/Conversion/LinalgToRAIR/LinalgToRAIR.cpp`

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

### Linear Lifetime Verification

Pass flag:

```bash
--rair-verify-lifetimes
```

This Phase 0 verifier freezes the straight-line ownership contract:

- every context produced by `rair.acquire` is released exactly once;
- every buffer produced by `rair.alloc_buffer` is deallocated exactly once;
- allocation and deallocation use the same context SSA value;
- contexts and buffers are not used after release;
- borrowed function arguments are not released by the callee;
- owned resources do not escape through calls, returns, branches, loops,
  nested regions, or derived memref aliases.

The last category currently produces an explicit unsupported diagnostic.
Control-flow-aware, alias-aware, and interprocedural ownership are planned
extensions rather than implicit assumptions in the current verifier.

---

## RAIR Downstream Lowering

The legacy RAIR-to-Affine and RAIR-to-CIM paths have been removed. New target
lowering must consume the RAIR Core/Plan contract instead of bypassing its
effect, dependency, and resource semantics with direct op rewrites.

The old debug-only `--convert-rair-to-llvm` utility has also been removed. It
did not lower RAIR compute or resource semantics and had no RAIR-specific test
coverage. Future target lowering starts from the Core/Plan contract.

---

## Front-End Example

Lower bufferized Linalg into RAIR for inspection and further Core/Plan work:

```bash
torch-mlir-opt input.mlir \
  --convert-linalg-to-rair \
  --rair-verify-lifetimes
```

---

## Pre-Core Compatibility Operation Summary

The operations below remain active only until the existing
`--convert-linalg-to-rair` pipeline is migrated to the Core operations above.

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
