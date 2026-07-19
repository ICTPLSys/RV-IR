# RAIR Dialect Design

RAIR, short for Resource-Adaptive Intermediate Representation, is the
target-neutral resource-effect MLIR dialect in this project. The dialect
textual namespace is `rair`, and the C++ namespace is also `rair`. RISC-V and
Gemmini are target instances, not part of the IR's semantic name.

The implementation now consistently uses RAIR in its active source paths,
generated artifacts, CMake targets, tests, and pass headers. The dialect's goal
is not to model the full RISC-V ISA. Instead, RAIR models the compiler-visible
interface between high-level tensor/memref computation and RISC-V attached
accelerators, including accelerator ownership, memory movement,
synchronization, and target-oriented compute operations.

## Design Position

RAIR sits between high-level MLIR compute dialects and lower backend dialects:

```text
Torch / Linalg / MemRef / Tensor
  -> RAIR
  -> RAIR Core/Plan-aware target lowering
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

- `include/torch-mlir/Dialect/RAIR/IR/RAIRDialect.td`: dialect definition,
  type constraints, async/hierarchy/memcpy interfaces, dialect-specific types,
  and memory-space enum.
- `include/torch-mlir/Dialect/RAIR/IR/RAIROps.td`: RAIR operation
  definitions and assembly formats.
- `include/torch-mlir/Dialect/RAIR/IR/RAIRDialect.h`: generated type/attribute
  declarations and async dependency parser/printer helpers.
- `lib/Dialect/RAIR/IR/RAIRDialect.cpp`: dialect initialization, generated type
  registration, builders, hierarchy parsing/printing helpers, and constant
  materialization.
- `lib/Conversion/LinalgToRAIR/LinalgToRAIR.cpp`: upper lowering from Linalg
  and MemRef into RAIR.
- `lib/Dialect/RAIR/Transforms/VerifyLifetimes.cpp`: straight-line context and
  accelerator-buffer ownership verification.
- `lib/Dialect/RAIR/Transforms/PrintEffectReport.cpp`: deterministic formatting
  consumer for the static effect report.
- `lib/Dialect/RAIR/Transforms/MaterializePlan.cpp`: transactional Core-to-Plan
  producer and stale-Plan validator.
- `include/torch-mlir/Dialect/RAIR/Transforms/RAIRStaticEffectGraph.h` and
  `lib/Dialect/RAIR/Transforms/RAIRStaticEffectGraph.cpp`: reusable typed Core
  region/effect/lifetime correctness graph.

The obsolete RAIR-to-Affine and RAIR-to-CIM conversion paths have been
removed. New execution lowering should consume the RAIR Core/Plan contract
instead of extending those legacy paths.

The existing operation reference in `docs/rair_ops_design.md` is a companion
document. This document focuses on the dialect-level design rather than listing
every operation in reference style.

## Core Abstractions

### Core v0.1 Scaffold

RAIR is now defined as the **Resource-Adaptive Intermediate Representation**.
The first Core scaffold is deliberately target-neutral and supports one
single-block, static-shape matmul trace without adding a backend:

- `rair.target @name` is a module-level static capability/resource symbol;
- `rair.scope @target` is the reference-trace and lifetime boundary;
- `rair.view` creates a zero-copy `!rair.region` from static offsets, sizes,
  and strides and checks bounds;
- `rair.reserve` returns a statically shaped memref and one linear
  `!rair.lease`;
- `rair.release_lease` consumes the matching lease and buffer after every
  buffer and derived-region use;
- `rair.move` copies equal element counts between two typed regions and has a
  read footprint on the source plus a write footprint on the destination;
- `rair.compute` wraps exactly one memref `linalg.matmul`, preserving Linalg as
  the tensor semantic anchor while exposing read/read/readwrite region effects.

The lease terminator is named `rair.release_lease` because the active pre-Core
lowering still uses `rair.release` for `!rair.context`. No compatibility alias
is added. A later legacy-to-Core migration can remove the old context op and
settle the final spelling without overloading one textual operation.

Core memrefs use `#rair.space<host>`, `device`, `spad`, or `acc`. The old
`gmem`, `lmem`, numbered scratchpad, and CIM spellings remain parseable only for
the existing Linalg-to-RAIR compatibility path; Core verifiers reject them.
`unknown` is reserved for conservative import and is also rejected by Core.

This scaffold is not produced by legacy `--convert-linalg-to-rair`.
`--rair-materialize-static-matmul` is the independent opt-in Core producer;
keeping the two pipelines separate prevents the compatibility lowering from
silently defining the new IR contract.

### Static Matmul Core Materialization

`--rair-materialize-static-matmul` replaces every eligible buffer-semantics
`linalg.matmul` with one self-contained `rair.scope`. Eligibility is checked for
all candidates before any mutation: lhs, rhs, and output must be rank-2 static
memrefs in target-neutral `host` or `device` space. Dynamic shapes, tensor
results, missing typed spaces, and pre-Core spaces are rejected.

The pass creates `rair.target @rair_default {kind = "generic"}` when absent and
reuses an existing `rair.target @rair_default` without changing its kind or
capacity attributes. A non-target symbol with that name is an error. Running
the pass again is a no-op because matmuls already nested in `rair.compute` are
not candidates.

For each matmul, materialization emits:

1. three whole-buffer external views;
2. two SPAD reservations and one ACC reservation;
3. three local whole-buffer views;
4. lhs, rhs, and initial-output moves into local storage;
5. one `rair.compute` containing a clone of the original Linalg payload on the
   local buffers;
6. output writeback and ACC/SPAD leases released in reverse order.

The initial-output move is required: memref `linalg.matmul` reads and
accumulates into its output, so a fresh ACC buffer cannot be assumed zeroed.
Local buffers preserve shape and element type but use contiguous identity
layout. The pass does not tile, choose a target, check capacity, create Plan
tasks/events, or invoke a backend.

### Core v0.1 Static Effect Analysis

`--rair-infer-effects` is a module-level, analysis-only report pass over the
hand-written Core trace. Module scope makes multi-function output deterministic;
each `rair.scope` still forms an independent analysis boundary. The pass does
not add attributes, rewrite operations, materialize Plan tasks/events, select a
schedule, or call a backend, and it explicitly preserves all analyses.

Inference is implemented by the reusable
`rair::RAIRStaticEffectGraph::build(scope)` API. The graph is an immutable
snapshot whose op handles remain valid until the scope is mutated. It exposes
typed, scope-local arrays with textual-order IDs:

- `ViewOp` regions and `StaticRegionRelation` pairs;
- `StaticActionKind` actions and normalized `StaticFootprint` entries;
- `StaticMemoryConflict` bitsets, including combined RAW/WAR/WAW reasons;
- `StaticLifetimeConstraint` reasons;
- aggregated, sorted `StaticEffectEdge` constraints;
- region/action lookup, pair-overlap, and direct-edge queries.

The builder returns `FailureOr<RAIRStaticEffectGraph>` rather than printing or
signalling diagnostics itself. It expects verified Core v0.1 IR and fails if an
action has no memory-effect interface, an effect region is not a known direct
view, or a reserve cannot be matched to its release. This keeps policy and
diagnostics in consumers. `PrintEffectReport.cpp` contains formatting only;
future Core-to-Plan construction must consume the same typed graph rather than
copying overlap/effect rules or parsing the textual report.

The report assigns scope-local stable identifiers in textual order: `rN` for
`rair.view` descriptors and `aN` for direct Core actions. It reads action
effects from the generated `MemoryEffectOpInterface`, then normalizes them to:

- `rair.reserve`: allocation/lifetime start;
- `rair.move`: source Read plus destination Write;
- `rair.compute`: lhs Read, rhs Read, output ReadWrite;
- `rair.release_lease`: free/lifetime end.

Region pairs use a conservative three-state relation:

- `overlap`: the same descriptor, or same-base unit-stride rectangles whose
  static bounds intersect;
- `disjoint`: a separating static dimension, different fresh `rair.reserve`
  buffers, or different known target-neutral memory spaces;
- `may_overlap`: aliasing is not disproved, including different same-space
  function arguments and ambiguous non-unit-stride intersections.

For actions `aI` before `aJ`, an overlapping or may-overlapping pair produces a
memory edge exactly when at least one footprint writes. Reasons are classified
as RAW, WAR, or WAW; Read/Read never creates an edge. Each lease also contributes
the unreduced raw constraints `reserve -> use`, `use -> release`, and
`reserve -> release`. Capacity/queue constraints and transitive reduction are
deliberately deferred to Plan construction.

The reference matmul report proves the lhs/rhs moves are independent, both feed
the compute, the initial-output move feeds the ReadWrite accumulator, and the
compute feeds output writeback. Because same-space host memref arguments have
no no-alias contract, it can also conservatively report input-read to
output-write WAR/MayOverlap edges. Those are sound constraints, not false
target serialization.

### Plan v0.1 and Core-to-Plan Materialization

Plan v0.1 freezes the target-aware task-DAG contract independently from target
execution:

- `rair.plan @target` is the last non-terminator operation in its source
  `rair.scope`, giving every Plan one structural Core association without a
  positional ID or pseudo-symbol;
- the Plan target must resolve and equal the containing Scope target;
- `rair.task` is a descriptor for one source Core action, identified by a
  required `source_action` i64 and a typed `#rair.task_kind` attribute whose
  value is `reserve`, `move`, `compute`, or `release_lease`;
- every task produces exactly one `!rair.event<@target>` and accepts zero or
  more dependency events;
- source action IDs are unique, non-negative, and dense in `[0, task-count)`,
  preserving the textual-order action numbering of `RAIRStaticEffectGraph`;
- every dependency must be the event of an earlier task in the same Plan and
  must target the containing Plan's target; duplicate entries are rejected;
- task result events must also target the containing Plan's target.

SSA gives each event exactly one producer. Requiring all dependency producers
to be earlier direct tasks in the same single block makes the represented graph
a DAG while still allowing one completion event to fan out to multiple later
tasks. Events from function arguments, another Plan, or another target cannot
be injected into the graph.

Tasks deliberately have conservative side effects: they are execution-plan
descriptors, not pure metadata, so generic canonicalization or dead-code
elimination may not erase an otherwise unused task. The implicit
`rair.plan_terminator` has no completion/event semantics.

`--rair-materialize-plan` builds `RAIRStaticEffectGraph` for every Scope and
creates exactly one task per action. Task kind and `source_action` preserve the
typed action and its textual-order ID. For each raw graph edge `aI -> aJ`, the
event from task `aI` becomes a dependency of task `aJ`; dependencies are emitted
in source-action order. The Core operations remain unchanged and inspectable.
The graph is converted to a detached task specification before the Scope is
mutated, honoring the graph snapshot lifetime contract.

The pass is transactional across the module: all graphs are built and all
existing Plans are validated before any missing Plan is added. An existing Plan
is accepted only if its task count, typed kinds, and dependency sets exactly
match the current raw graph. This makes repeated execution idempotent and turns
stale Plan data into a diagnostic instead of silently trusting it. Dependency
comparison is set-based; textual operand order is not semantically significant.

The produced Plan deliberately preserves all raw correctness edges, including
transitively redundant memory/lifetime constraints. It does not yet perform
transitive reduction, add capacity/queue edges, choose an optimized schedule,
expose a host completion event, or define runtime/backend execution.

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

RAIR defines five dialect-specific types:

- `!rair.region`: analyzable static buffer slice used by Core effects.
- `!rair.lease`: linear capability returned by `rair.reserve`.
- `!rair.event<@target>`: target-aware Plan completion event.
- `!rair.context`: pre-Core accelerator execution context.
- `!rair.async.token`: pre-Core async dependency token.

The target-neutral Core memory spaces are:

- `HOST = 7`
- `DEVICE = 8`
- `SPAD = 9`
- `ACC = 10`
- `UNKNOWN = 11` (import only; rejected by Core verification)

Values 0-6 retain the pre-Core `LMEM`, `CIMC0`, `SPAD0..3`, and `GMEM`
spellings until the current Linalg lowering is migrated.

- `LMEM = 0`
- `CIMC0 = 1`
- `SPAD3 = 2`
- `SPAD2 = 3`
- `SPAD1 = 4`
- `SPAD0 = 5`
- `GMEM = 6`

The canonical textual representation is the typed attribute
`#rair.space<name>`, for example `#rair.space<host>` and
`#rair.space<spad>`. `rair.alloc_buffer` and `rair.transfer` use this attribute
for operation metadata. When a memref type declares an explicit memory space,
the verifier requires it to agree with the corresponding operation attribute.
An omitted memref memory space remains legal, which lets upper lowering retain
existing function signatures while transfer attributes make the boundary
placement explicit.

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

## Linear Resource Lifetime Contract

The Phase 0 lifetime verifier is available as:

```bash
--rair-verify-lifetimes
```

A typical upper pipeline is therefore:

```bash
torch-mlir-opt input.mlir \
  --convert-linalg-to-rair \
  --rair-verify-lifetimes
```

The pass distinguishes owned resources from borrowed function arguments:

- A context returned by `rair.acquire` is owned and must have exactly one
  `rair.release` in the same straight-line function block.
- A buffer returned by `rair.alloc_buffer` is owned and must have exactly one
  `rair.dealloc_buffer` in that block.
- The deallocation context must be the same SSA value used for allocation.
- Context and buffer uses after their release are rejected.
- Releasing a borrowed context argument or deallocating a borrowed buffer
  argument is rejected. A borrowed context may still own a locally allocated
  buffer, provided that buffer is deallocated with the same context.

This first verifier is intentionally conservative. Owned resources may not
cross calls, returns, CFG branches, loops, nested regions, or derived memref
aliases. Each such case receives an explicit unsupported diagnostic rather than
being silently accepted. Control flow that does not capture or contain RAIR
resources remains legal. A future dataflow analysis can extend the contract to
path-sensitive and interprocedural ownership without weakening this baseline.

## Downstream Lowering

The legacy RAIR-to-Affine and RAIR-to-CIM paths have been removed. Future
target lowering must consume the RAIR Core/Plan contract so that scheduling,
resource demand, and effect ordering are not bypassed by direct op rewrites.

The old debug-only `--convert-rair-to-llvm` utility has also been removed. It
lowered only `rair.print` and `rair.world`, had no RAIR-specific regression
coverage, and could be mistaken for an execution backend. Debug I/O should be
handled independently from the future Core/Plan target lowering.

## Example Design Flow

A matmul coming from bufferized Linalg is transformed from:

```mlir
linalg.matmul ins(%lhs, %rhs : memref<16x32xf32>, memref<32x64xf32>)
              outs(%out : memref<16x64xf32>)
```

into a RAIR form that contains:

```mlir
%ctx = rair.acquire {accelerator = "default"} : !rair.context
%lhs_lmem = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>} : memref<16x32xf32, #rair.space<lmem>>
%rhs_lmem = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>} : memref<32x64xf32, #rair.space<lmem>>
%out_lmem = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>} : memref<16x64xf32, #rair.space<lmem>>

rair.transfer %lhs to %lhs_lmem
  {src_memory_space = #rair.space<gmem>, dst_memory_space = #rair.space<lmem>}
  : memref<16x32xf32>, memref<16x32xf32, #rair.space<lmem>>
rair.transfer %rhs to %rhs_lmem
  {src_memory_space = #rair.space<gmem>, dst_memory_space = #rair.space<lmem>}
  : memref<32x64xf32>, memref<32x64xf32, #rair.space<lmem>>

rair.matmul ins(%lhs_lmem, %rhs_lmem
  : memref<16x32xf32, #rair.space<lmem>>,
    memref<32x64xf32, #rair.space<lmem>>)
  outs(%out_lmem : memref<16x64xf32, #rair.space<lmem>>)

rair.transfer %out_lmem to %out
  {src_memory_space = #rair.space<lmem>, dst_memory_space = #rair.space<gmem>}
  : memref<16x64xf32, #rair.space<lmem>>, memref<16x64xf32>

rair.dealloc_buffer %ctx, %lhs_lmem
  : memref<16x32xf32, #rair.space<lmem>>
rair.dealloc_buffer %ctx, %rhs_lmem
  : memref<32x64xf32, #rair.space<lmem>>
rair.dealloc_buffer %ctx, %out_lmem
  : memref<16x64xf32, #rair.space<lmem>>

rair.release %ctx : !rair.context
```

This example shows the key RAIR design choice: the IR preserves the high-level
matmul operation while also making memory placement and movement visible.

## Current Design Limitations

The implementation is functional but still evolving. The most important design
limitations are:

- Tensor-style and memref-style compute operations coexist. This is workable,
  but it complicates verification and lowering contracts.
- Optional target attributes are defined on many ops, but the upper lowering
  does not yet consistently fill legality, capability, tile, or fallback data.
- Function arguments and existing global buffers may still have an unspecified
  memref memory space; explicit operation attributes carry their intended
  placement until signature conversion is introduced.
- `rair.launch`, `rair.herd`, and `rair.kernel` provide useful structure, but
  the current Linalg lowering mostly emits individual compute ops rather than
  wrapping accepted offload regions in launch/kernel boundaries.
- Verification covers the main compute shape contracts, explicit memory-space
  consistency, and straight-line owned-resource lifetimes. Layout,
  bank-capacity, byte-size, coherence, path-sensitive lifetime, alias, and
  interprocedural invariants are not yet fully enforced.
- Some high-level model patterns remain outside RAIR lowering, including
  arbitrary `linalg.generic`, `linalg.fill`, broadcast/map patterns, bias-add
  generics, and activation patterns.

## Recommended Evolution

The next design steps should strengthen RAIR as an accelerator interface layer:

1. Extend linear lifetime verification to path-sensitive CFG, nested-region,
   alias-aware, and interprocedural ownership analysis.
2. Introduce function-signature conversion where explicit global-memory types
   are required, without making existing boundary types invalid prematurely.
3. Make upper lowering fill target attributes consistently:
   `accelerator`, `tile_size`, `dataflow`, and possibly `fallback`.
4. Introduce a capability/profitability analysis before materializing RAIR so
   offload decisions are explicit and explainable.
5. Use `rair.kernel` or `rair.launch` around accepted offload regions, not only
   around individual operations, to support fusion and shared-buffer reuse.
6. Extend pattern recognition for model-level regions such as matmul+bias,
   matmul+bias+relu, conv+relu, and pooling pipelines.
7. Extend tests from the current straight-line lifetime and memory placement
   coverage to path-sensitive Core/Plan and target-lowering contracts.

## Summary

RAIR is designed as the compiler-visible boundary for RISC-V accelerator
offload. Its central value is that it keeps high-level compute operations
recognizable while exposing accelerator interface concerns that are normally
hidden: context ownership, memory placement, transfer scheduling,
synchronization, target attributes, and backend lowering choices.

The current implementation already supports a meaningful subset of this design:
Linalg named ops lower to RAIR, matmul-like operations get explicit typed local
memory staging, and straight-line context/buffer ownership can be verified
before downstream work. The main opportunity now is to make the offload
decision and RAIR Core/Plan layers more explicit through target capability
metadata, structured launch/kernel regions, control-flow-aware ownership, and
richer pattern recognition.
