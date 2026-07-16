# RAIR Dialect Design

RAIR, short for RISC-V Accelerator Interface Representation, is the
accelerator-facing MLIR dialect in this project. The dialect textual namespace
is `rair`, and the C++ namespace is also `rair`.

Although parts of the source tree still use `RISCV` in file names and pass
names for historical reasons, the dialect defined in
`include/torch-mlir/Dialect/RISCV/IR/RISCVDialect.td` is named `rair`. Its goal
is not to model the full RISC-V ISA. Instead, RAIR models the compiler-visible
interface between high-level tensor/memref computation and RISC-V attached
accelerators, including accelerator ownership, memory movement, synchronization,
and target-oriented compute operations.

## Design Position

RAIR sits between high-level MLIR compute dialects and lower backend dialects:

```text
Torch / Linalg / MemRef / Tensor
  -> RAIR
  -> Affine / Async / LLVM / CIM / accelerator-specific lowering
```

The design intent is to make accelerator interface semantics explicit in IR.
Without RAIR, these semantics tend to be hidden in runtime helpers, inline
assembly, C macros, or backend-specific intrinsics. RAIR lifts them into MLIR so
passes can inspect and transform them before committing to a concrete backend.

The current implementation already exposes three important pieces of that
contract:

- Compute intent: operations such as `rair.matmul`, `rair.batch_matmul`,
  `rair.transpose`, `rair.conv_2d_nchw_fchw`, and `rair.pooling_nchw_max`
  preserve high-level operation meaning after Linalg lowering.
- Resource ownership: functions containing RAIR compute ops get explicit
  `rair.acquire` and `rair.release` operations.
- Memory movement: matmul-like operations are rewritten to use local accelerator
  buffers via `rair.alloc_buffer`, `rair.transfer`, and
  `rair.dealloc_buffer`.

This means RAIR is best understood as an accelerator interface dialect, not as a
thin rename of Linalg operations.

## Source Layout

The core dialect implementation is spread across the following files:

- `include/torch-mlir/Dialect/RISCV/IR/RISCVDialect.td`: dialect definition,
  type constraints, async/hierarchy/memcpy interfaces, dialect-specific types,
  and memory-space enum.
- `include/torch-mlir/Dialect/RISCV/IR/RISCVOps.td`: RAIR operation
  definitions and assembly formats.
- `include/torch-mlir/Dialect/RISCV/IR/RISCVDialect.h`: hand-written type
  declarations and async dependency parser/printer helpers.
- `lib/Dialect/RISCV/IR/RISCVDialect.cpp`: dialect initialization, custom type
  parsing/printing, builders, hierarchy parsing/printing helpers, and constant
  materialization.
- `lib/Conversion/LinalgToRISCV/LinalgToRISCV.cpp`: upper lowering from Linalg
  and MemRef into RAIR.
- `lib/Conversion/RISCVToAffine/LowerToAffine.cpp`: CPU-style lowering from
  RAIR compute and utility ops to Affine/Arith/SCF/Async.
- `lib/Conversion/RISCVToCIM/RISCVToCIM.cpp`: accelerator-style lowering from
  selected RAIR ops to CIM/RISC-V intrinsic calls.
- `lib/Conversion/RISCVToLLVM/LowerToLLVM.cpp`: utility lowering for
  `rair.print` and `rair.world` to LLVM/printf.

The existing operation reference in `docs/rair_ops_design.md` is a companion
document. This document focuses on the dialect-level design rather than listing
every operation in reference style.

## Core Abstractions

### Accelerator Context

`!rair.context` represents ownership of an accelerator execution context.
The current upper lowering inserts:

```mlir
%ctx = rair.acquire {accelerator = "default"} : !rair.context
...
rair.release %ctx : !rair.context
```

The context is the anchor for accelerator resource lifetime. Buffer allocation
ops such as `rair.alloc_buffer` are tied to a context so later passes can reason
about memory ownership and reuse.

### Explicit Memory Movement

RAIR models data movement explicitly instead of treating it as an implicit
backend side effect. The main operations are:

- `rair.alloc` / `rair.dealloc` for general memref allocation and deallocation.
- `rair.alloc_buffer` / `rair.dealloc_buffer` for accelerator-visible buffers
  associated with `!rair.context`.
- `rair.transfer` for memory-space-to-memory-space copies.
- `rair.load` / `rair.store` for DMA-like async transfer operations with
  offset, size, and stride operands.

The Linalg-to-RAIR pass currently applies this most concretely to matmul and
batch matmul. It allocates local `LMEM` buffers, transfers operands from `GMEM`
to `LMEM`, rewrites the compute op to consume local buffers, transfers the
result back to `GMEM`, and deallocates the local buffers.

### Structured Execution

RAIR provides region operations for structured accelerator execution:

- `rair.launch` models an accelerator launch region.
- `rair.herd` models a structured group of accelerator workers or tiles.
- `rair.kernel` models a higher-level kernel region for fusion, scheduling, and
  target-aware optimization.

`rair.launch` and `rair.herd` implement a shared hierarchy interface. They own a
single region, expose iteration-space ids and sizes as block arguments, and
support captured kernel operands. These operations are designed to give later
passes a stable boundary for scheduling, fusion, and async dependency handling.

### Async and Synchronization

`!rair.async.token` represents completion of an async operation. Operations that
implement `RAIR_AsyncOpInterface` consume zero or more dependency tokens and may
produce a token.

The current async-aware operations are:

- `rair.launch`
- `rair.herd`
- `rair.load`
- `rair.store`
- `rair.wait_all`

RAIR also has `rair.await` for context-level waiting and `rair.barrier` for a
tensor-level synchronization point.

## Dialect Types and Attributes

RAIR defines three dialect-specific types:

- `!rair.async.token`: dependency token for async scheduling.
- `!rair.event`: event-style synchronization value used by some lowering paths.
- `!rair.context`: accelerator execution context.

The dialect also defines a `MemorySpace` enum:

- `LMEM = 0`
- `CIMC0 = 1`
- `SPAD3 = 2`
- `SPAD2 = 3`
- `SPAD1 = 4`
- `SPAD0 = 5`
- `GMEM = 6`

In the current operation definitions, memory space is sometimes represented by
string attributes such as `"GMEM"` and `"LMEM"` rather than this enum. This is a
practical implementation detail to be aware of when writing or extending passes.
Long term, using a single representation would make verification and lowering
cleaner.

Many compute operations accept optional target attributes:

- `accelerator`: string identifier such as `"default"` or `"gemmini"`.
- `tile_size`: dense i64 array carrying target-aware tiling parameters.
- `dataflow`: string strategy such as weight-stationary or output-stationary.
- `fallback`: symbol reference for CPU fallback paths on region operations.

These attributes are the intended carrier for legality, scheduling, and fallback
decisions, even though the current conversion pass mostly emits empty optional
attributes or `accelerator = "default"`.

## Operation Families

### Resource Operations

`rair.acquire`, `rair.release`, and `rair.await` model accelerator resource
ownership and context-level synchronization. They make the lifetime of an
accelerator session explicit in the IR.

### Memory Operations

`rair.alloc`, `rair.dealloc`, `rair.alloc_buffer`, `rair.dealloc_buffer`, and
`rair.transfer` model allocation and movement across host and accelerator memory
spaces.

`rair.load` and `rair.store` are DMA-style async operations. They implement both
the async interface and memcpy interface, so passes can inspect dependency
tokens and source/destination views uniformly.

### Structured Region Operations

`rair.launch`, `rair.herd`, and `rair.kernel` are region operations. They are
intended to describe accelerator execution boundaries rather than scalar
operations. Their terminators are `rair.launch_terminator`,
`rair.herd_terminator`, and `rair.kernel_terminator`.

### Compute Operations

The main accelerator-relevant compute operations are:

- Matrix operations: `rair.matmul`, `rair.batch_matmul`, `rair.matvec`.
- Convolution and pooling: `rair.conv2d`, `rair.conv_2d_nchw_fchw`,
  `rair.pooling_nchw_max`, `rair.pooling_nchw_sum`.
- Shape/layout operations: `rair.transpose`, `rair.reshape`.
- Reductions: `rair.reduce`.
- Elementwise arithmetic: `rair.add`, `rair.sub`, `rair.mul`, `rair.div`,
  `rair.max`, `rair.min`, `rair.negf`.
- Bitwise and comparisons: `rair.andi`, `rair.xori`, `rair.ori`, `rair.cmpi`,
  `rair.cmpf`.

There is an important split in operand style:

- Several accelerator-oriented ops are in-place memref operations with explicit
  `ins(...)` and `outs(...)`, for example `rair.matmul`,
  `rair.batch_matmul`, `rair.transpose`, NCHW pooling, and NCHW/FCHW conv.
- Some older tensor-style ops return tensor values, for example `rair.matvec`,
  `rair.conv2d`, `rair.reduce`, and elementwise arithmetic.

This mixed style reflects the current evolution of the dialect. The lowerings
already support both forms in selected places, but future work should decide
whether accelerator compute should consistently prefer buffer-style side-effect
semantics.

### Function and Utility Operations

`rair.func` and `rair.return` provide a RAIR function-like container using
`FunctionOpInterface`. The project also keeps debug and utility operations such
as `rair.constant`, `rair.print`, and `rair.world`.

## Upper Lowering Contract

The main upper lowering pass is:

```bash
--convert-linalg-to-rair
```

It currently performs direct named-op conversions for supported Linalg
operations:

- `linalg.matmul` -> `rair.matmul`
- `linalg.batch_matmul` -> `rair.batch_matmul`
- `linalg.matvec` -> `rair.matvec`
- `linalg.reduce` -> `rair.reduce`
- `linalg.conv_2d` -> `rair.conv2d`
- `linalg.conv_2d_nchw_fchw` -> `rair.conv_2d_nchw_fchw`
- `linalg.transpose` -> `rair.transpose`
- `linalg.pooling_nchw_max` -> `rair.pooling_nchw_max`
- `linalg.pooling_nchw_sum` -> `rair.pooling_nchw_sum`
- `linalg.add/sub/mul/div/max/min/negf` -> corresponding RAIR arithmetic ops
- `memref.alloc` -> `rair.alloc`
- `memref.copy` -> `rair.transfer`

After conversion, the pass walks functions containing RAIR compute operations
and inserts accelerator context management. For matmul and batch matmul, it also
materializes local memory staging:

```text
acquire context
allocate LMEM buffers
transfer GMEM -> LMEM
run RAIR compute on LMEM buffers
transfer LMEM -> GMEM
deallocate LMEM buffers
release context
```

This post-processing is important because it turns RAIR into a representation of
accelerator interface behavior, not only accelerator compute behavior.

## Downstream Lowering Paths

RAIR has multiple downstream paths:

```bash
--convert-rair-to-affine
--convert-rair-to-cim
--convert-rair-to-llvm
```

The Affine path lowers many RAIR compute operations into loops and standard MLIR
arithmetic/memory operations. It is useful as a CPU-style executable path and as
a correctness-oriented lowering.

The CIM path lowers selected accelerator operations to intrinsic-like calls such
as `llvm.riscv.vv.v.drv`, `llvm.riscv.conv.drv`, `llvm.riscv.trans.drv`,
`llvm.riscv.load`, `llvm.riscv.store`, and `llvm.riscv.sync`. This path is the
accelerator-oriented backend direction.

The LLVM path currently focuses on utility operations such as `rair.print` and
`rair.world`, lowering them to `printf`-style LLVM calls. It is not the full
RAIR compute lowering path by itself.

## Example Design Flow

A matmul coming from bufferized Linalg is transformed from:

```mlir
linalg.matmul ins(%lhs, %rhs : memref<16x32xf32>, memref<32x64xf32>)
              outs(%out : memref<16x64xf32>)
```

into a RAIR form that contains:

```mlir
%ctx = rair.acquire {accelerator = "default"} : !rair.context
%lhs_lmem = rair.alloc_buffer %ctx {memory_space = "LMEM"} : memref<16x32xf32>
%rhs_lmem = rair.alloc_buffer %ctx {memory_space = "LMEM"} : memref<32x64xf32>
%out_lmem = rair.alloc_buffer %ctx {memory_space = "LMEM"} : memref<16x64xf32>

rair.transfer %lhs to %lhs_lmem
  {src_memory_space = "GMEM", dst_memory_space = "LMEM"}
  : memref<16x32xf32>, memref<16x32xf32>
rair.transfer %rhs to %rhs_lmem
  {src_memory_space = "GMEM", dst_memory_space = "LMEM"}
  : memref<32x64xf32>, memref<32x64xf32>

rair.matmul ins(%lhs_lmem, %rhs_lmem : memref<16x32xf32>, memref<32x64xf32>)
             outs(%out_lmem : memref<16x64xf32>)

rair.transfer %out_lmem to %out
  {src_memory_space = "LMEM", dst_memory_space = "GMEM"}
  : memref<16x64xf32>, memref<16x64xf32>

rair.release %ctx : !rair.context
```

This example shows the key RAIR design choice: the IR preserves the high-level
matmul operation while also making memory placement and movement visible.

## Current Design Limitations

The implementation is functional but still evolving. The most important design
limitations are:

- Naming is inconsistent: source paths and some comments still say `RISCV`, but
  the dialect namespace is `rair`.
- Tensor-style and memref-style compute operations coexist. This is workable,
  but it complicates verification and lowering contracts.
- Optional target attributes are defined on many ops, but the upper lowering
  does not yet consistently fill legality, capability, tile, or fallback data.
- `MemorySpace` exists as an enum, while some operations use string attributes
  for memory spaces.
- `rair.launch`, `rair.herd`, and `rair.kernel` provide useful structure, but
  the current Linalg lowering mostly emits individual compute ops rather than
  wrapping accepted offload regions in launch/kernel boundaries.
- Verification is light. Many shape, rank, layout, type, memory-space, and
  attribute invariants are implied by lowering patterns rather than enforced by
  op verifiers.
- Some high-level model patterns remain outside RAIR lowering, including
  arbitrary `linalg.generic`, `linalg.fill`, broadcast/map patterns, bias-add
  generics, and activation patterns.

## Recommended Evolution

The next design steps should strengthen RAIR as an accelerator interface layer:

1. Add verifiers for major compute ops, especially memref rank and shape
   compatibility for matmul, batch matmul, transpose, conv, pooling, and
   transfer.
2. Normalize memory-space representation so allocation, transfer, and lowering
   all use either the `MemorySpace` enum or a clearly documented string
   convention.
3. Make upper lowering fill target attributes consistently:
   `accelerator`, `tile_size`, `dataflow`, and possibly `fallback`.
4. Introduce a capability/profitability analysis before materializing RAIR so
   offload decisions are explicit and explainable.
5. Use `rair.kernel` or `rair.launch` around accepted offload regions, not only
   around individual operations, to support fusion and shared-buffer reuse.
6. Extend pattern recognition for model-level regions such as matmul+bias,
   matmul+bias+relu, conv+relu, and pooling pipelines.
7. Add tests that check not only operation replacement but also context
   lifetime, memory movement placement, and backend-specific lowering.

## Summary

RAIR is designed as the compiler-visible boundary for RISC-V accelerator
offload. Its central value is that it keeps high-level compute operations
recognizable while exposing accelerator interface concerns that are normally
hidden: context ownership, memory placement, transfer scheduling,
synchronization, target attributes, and backend lowering choices.

The current implementation already supports a meaningful subset of this design:
Linalg named ops lower to RAIR, matmul-like operations get explicit local memory
staging, and RAIR can lower either to affine loops or to accelerator-oriented
CIM/RISC-V intrinsic calls. The main opportunity now is to make the offload
decision layer more explicit through verifiers, target capability metadata,
structured launch/kernel regions, and richer pattern recognition.
