# AICompiler StoryLine

## Core Thesis

RoCC-style AI accelerator interfaces hide the semantics a compiler needs behind runtime calls, intrinsics, helper libraries, and inline assembly. RAIR lifts those semantics into compiler-visible IR constructs so that accelerator offloading can be checked, optimized, and specialized before final backend lowering.

The work should be framed as an abstraction-boundary paper:

> RAIR defines a compiler IR layer for tightly coupled RISC-V AI accelerators, where accelerator contexts, memory spaces, data movement, synchronization, launch boundaries, target capabilities, and CPU fallback are first-class program concepts rather than opaque host-side effects.

This is stronger than saying “we support multiple RISC-V accelerator backends.” Multi-backend support is important evidence, but the research claim is that RoCC-style accelerator interaction itself deserves an analyzable IR representation.

## Problem Statement

RISC-V edge and embedded platforms increasingly combine a scalar control processor with one or more domain accelerators. In RoCC-style systems, accelerator invocation is tightly coupled with host execution: software must acquire accelerator resources, move data into accelerator-visible memories, issue custom commands, synchronize with the accelerator, and fall back to CPU code when shapes or resources are unsupported.

Today this interaction is usually expressed through low-level programming interfaces:

- C macros and inline assembly for custom instructions;
- accelerator-specific helper libraries;
- device-specific runtime calls;
- manually managed scratchpad, accumulator, DMA, and synchronization conventions.

These interfaces are usable by experts, but they erase the semantic structure needed by a compiler:

- Accelerator resources look like opaque handles rather than analyzable program objects.
- Memory placement and data movement are encoded as side effects.
- Synchronization is hidden inside calls or polling loops.
- Buffer lifetimes cannot be reasoned about across accelerator kernels.
- Offload legality and fallback decisions are scattered across handwritten host code.
- Retargeting requires rewriting target-specific recipes rather than specializing a common representation.

The core gap is therefore not simply missing code generation support. The missing layer is a compiler-visible representation of accelerator interface semantics.

## Key Insight

The key idea is to treat a RoCC-attached accelerator not merely as a backend target and not merely as a library call, but as a structured compiler object with explicit resource, memory, execution, and fallback semantics.

Once accelerator context, memory spaces, transfer, launch, await, and target capability constraints are modeled directly in IR:

- offload legality can be checked before lowering;
- target constraints can guide tiling and placement decisions;
- buffer lifetime and reuse can be optimized across accelerator regions;
- synchronization and transfer/compute overlap become schedulable;
- unsupported or unprofitable regions can remain on a CPU path by construction.

The paper’s strongest claim is that these benefits come from semantic visibility. Individual optimizations such as tiling, fusion, and buffering are not new by themselves; the contribution is making them systematic for RoCC-style accelerators by exposing the right IR boundary.

## Contributions

### 1. RoCC-Style Accelerator Interface IR

RAIR, implemented as a `rocc` dialect, exposes accelerator interface semantics as first-class IR constructs. The dialect should model:

- `rocc.acquire`, `rocc.release`, and `rocc.await` for accelerator resource ownership and synchronization;
- `rocc.alloc_buffer`, `rocc.transfer`, and `rocc.dealloc_buffer` for explicit data movement and memory-space control;
- `rocc.launch` and `rocc.kernel` for structured accelerator execution regions;
- `!rocc.context` and `!rocc.buffer<...>` types for accelerator resources and memory-resident values;
- attributes such as `accelerator`, `memory_space`, `tile_size`, `dataflow`, `capability`, and `fallback`.

This contribution should be written as “we lift accelerator interface semantics into IR,” not “we define a new dialect.”

### 2. Semantics-Enabled Compiler Analyses

The IR should enable a focused set of analyses and transformations that are hard to express over opaque runtime calls:

- legality and profitability based offload selection;
- target-aware tiling under scratchpad, accumulator, banking, and array-shape constraints;
- deterministic buffer lifetime analysis and accelerator memory reuse;
- dependence-aware scheduling, short-region fusion, and transfer/compute overlap.

For the actual paper, it is better to implement and evaluate two or three of these deeply than to list many shallow optimizations. The most convincing combination is likely:

- cost-aware offload selection;
- buffer lifetime and memory reuse;
- fusion/overlap for short operator chains.

### 3. Retargetable Lowering with Correctness-Preserving Fallback

RAIR should lower to target-specific backends through capability specialization. Each backend describes:

- supported operators and shapes;
- memory spaces and capacity constraints;
- tile and dataflow constraints;
- synchronization model;
- transfer cost and setup cost parameters;
- lowering hooks for target commands or runtime calls.

The paper should demonstrate at least two targets or target configurations, such as a Gemmini-like RoCC accelerator plus a second NPU backend or a distinct Gemmini configuration. CPU fallback should be presented as part of correctness-preserving offload selection: if a region is illegal or unprofitable for a target, it remains in the original CPU path.

## Proposed IR Design

### Core Example

```mlir
func.func @example(%A: memref<128x128xf32>, %B: memref<128x128xf32>) {
  %ctx = rocc.acquire {accelerator = "gemmini"} : !rocc.context
  %bufA = rocc.alloc_buffer %ctx, %A {memory_space = "scratchpad"}
    : memref<128x128xf32> -> !rocc.buffer<128x128xf32, "scratchpad">
  %bufB = rocc.alloc_buffer %ctx, %B {memory_space = "scratchpad"}
    : memref<128x128xf32> -> !rocc.buffer<128x128xf32, "scratchpad">

  %out = rocc.launch %ctx @compute_kernel(%bufA, %bufB)
    {fallback = @cpu_compute}
    : (!rocc.buffer<128x128xf32, "scratchpad">,
       !rocc.buffer<128x128xf32, "scratchpad">)
   -> !rocc.buffer<128x128xf32, "accumulator">

  rocc.await %ctx
  rocc.release %ctx
  return
}

rocc.kernel @compute_kernel(
  %A: !rocc.buffer<128x128xf32, "scratchpad">,
  %B: !rocc.buffer<128x128xf32, "scratchpad">
) -> !rocc.buffer<128x128xf32, "accumulator"> {
  %out = rocc.matmul %A, %B
    {tile_size = [16, 16, 16], dataflow = "weight_stationary"}
  rocc.return %out
}
```

### Why These IR Elements Matter

- `rocc.acquire` and `rocc.release` make resource lifetime explicit.
- `rocc.await` turns synchronization into a dependence boundary.
- `rocc.alloc_buffer`, `rocc.transfer`, and `rocc.dealloc_buffer` expose data movement and memory lifetime.
- `!rocc.buffer<..., "scratchpad">` and `!rocc.buffer<..., "accumulator">` make memory placement part of the type contract.
- `rocc.kernel` and `rocc.launch` provide region boundaries for fusion, scheduling, and target lowering.
- `fallback = @cpu_compute` makes the CPU path part of the compilation decision rather than an external recovery mechanism.

## Capability and Legality Model

The paper needs a concrete model for how target-specific knowledge enters the compiler. A compact table or schema would help:

```text
TargetCapability = {
  accelerator: "gemmini",
  ops: [matmul, conv2d, elementwise],
  memory_spaces: [scratchpad, accumulator],
  scratchpad_capacity: S,
  accumulator_capacity: A,
  tile_constraints: Tm % 16 == 0, Tn % 16 == 0,
  dataflows: [weight_stationary, output_stationary],
  sync_model: blocking_await,
  transfer_model: dma_2d,
  setup_cost: C_setup
}
```

This model is important because it turns “retargetability” into something reviewers can inspect. A backend is not just a lowering script; it is a specialization of the same interface semantics under different capabilities.

## Optimization Story

### Cost-Aware Offload Selection

Hot single operators or short fusable patterns are grouped as offload candidates. For a region `R`, the compiler estimates:

$$\Delta T(R)=T_{cpu}(R)-T_{off}(R)$$

with:

$$T_{off}(R) \approx T_{setup}+T_{transfer}+T_{compute}+T_{sync}-T_{overlap}$$

The region is materialized into RAIR only if it is legal and profitable. This separates the work from backend-only approaches, which generally assume the offload decision has already been made.

### Target-Aware Tiling

For matrix multiplication `C[M, N] = A[M, K] x B[K, N]`, tiling can be selected under capacity and alignment constraints:

$$T_mT_k + T_kT_n + T_mT_n \le S$$

$$T_m \bmod A_m = 0,\quad T_n \bmod A_n = 0$$

The exact solver is not the paper’s key novelty. The key point is that the IR keeps memory spaces, dataflow, and target capabilities visible at the level where tiling decisions are still meaningful.

### Buffer Lifetime and Memory Reuse

Because accelerator allocations and kernel boundaries are explicit, the compiler can infer when accelerator buffers are live and when storage can be reused:

- adjacent kernels can reuse scratchpad or accumulator allocation slots;
- intermediate values can remain accelerator-resident across fused regions;
- redundant host-accelerator transfers can be removed;
- memory-space aliasing can be checked explicitly.

### Fusion, Scheduling, and Overlap

Short chains such as `conv2d + relu` or `matmul + bias + activation` are good examples because they expose both launch overhead and intermediate transfer overhead. If dependencies and memory-space constraints allow it, RAIR can fuse such chains inside an accelerator region or overlap transfer with compute:

```mlir
%conv = rocc.conv2d %image, %filter {memory_space = "scratchpad"}
%relu = rocc.elementwise "relu", %conv {memory_space = "accumulator"}
%out = rocc.transfer %relu to host
```

The evaluation should report whether these transformations reduce launch count, transfer volume, and synchronization overhead.

## Related Work Positioning

### 1. TVM, VTA, and End-to-End DNN Compilation

TVM and VTA demonstrate the value of schedule-based tensor compilation and full-stack accelerator integration. Their strength is optimizing tensor programs through scheduling, auto-tuning, and backend code generation. RAIR should not claim to replace this layer. The distinction is that RAIR focuses on the compiler-visible semantics of RoCC-style accelerator invocation: resource ownership, memory spaces, synchronization, and fallback decisions that are often hidden in host-side accelerator interfaces.

### 2. IREE and MLIR Deployment Runtimes

IREE provides an MLIR-based end-to-end deployment stack with portable dispatch and runtime integration across many targets. This validates the broad idea that compiler and runtime boundaries matter. RAIR differs by specializing the abstraction to tightly coupled RISC-V accelerators where scratchpad placement, custom command launch, explicit synchronization, and CPU fallback are central compiler concerns rather than generic device dispatch details.

### 3. MLIR-AIR, MLIR-AIE, and Spatial Accelerator Compilation

MLIR-AIR and MLIR-AIE expose asynchronous execution, data movement, and spatial mapping for NPU fabrics. These works are strong related work because they share the principle that data movement and synchronization should be explicit in IR. RAIR targets a different boundary: the host-coprocessor interface of RoCC-attached accelerators, where the compiler must coordinate CPU code, custom accelerator commands, accelerator-visible memories, and fallback.

### 4. Buddy/Gemmini and Accelerator-Specific Dialects

Buddy/Gemmini is likely the closest comparison. It shows that MLIR-based compilation to a concrete RISC-V accelerator is practical. RAIR must distinguish itself by avoiding a single-target op-wrapper story. The claim should be that RAIR models resource, memory, synchronization, offload, and fallback semantics in a way that can be specialized to multiple RoCC-style targets or configurations.

### 5. RISC-V Custom Backend Work

Recent multi-level backend work for RISC-V custom extensions shows that traditional backend abstractions can be too low-level for custom hardware. RAIR is aligned with that motivation but operates one level earlier. Backend work asks how to lower a chosen kernel efficiently; RAIR asks what should be offloaded, how accelerator interaction should be represented, and how host-accelerator coordination can be optimized before backend lowering.

### 6. Timeloop, MAESTRO, Accelergy, and Mapping Frameworks

Mapping and modeling frameworks explore dataflows, memory hierarchies, and performance/energy tradeoffs for accelerator designs. RAIR can borrow ideas from their cost and mapping models, but its contribution is not hardware design-space exploration. RAIR is a compiler representation that carries accelerator interaction semantics through an executable compilation flow.

### 7. ATLAAS and Automatic Semantic Extraction

ATLAAS-like work asks how tensor-level accelerator semantics can be extracted from RTL or low-level hardware descriptions. RAIR assumes accelerator semantics are available and asks how a compiler should represent and exploit them. These directions are complementary: extracted semantics could eventually populate RAIR capability descriptions.

## Evaluation Strategy

### Research Questions

- **RQ1:** Does RAIR expose accelerator resource, memory, synchronization, and fallback semantics that are opaque in handwritten or runtime-call interfaces?
- **RQ2:** Do RAIR-enabled passes reduce transfer volume, launch/synchronization overhead, or end-to-end latency?
- **RQ3:** Does the same RAIR representation retarget across more than one RoCC-style accelerator or configuration?
- **RQ4:** What overhead does RAIR introduce in compile time, code size, or backend specialization complexity?

### Platforms

The minimum credible platform set is:

- one concrete Gemmini-like RoCC target;
- one second target/configuration, such as a different Gemmini array/memory configuration, a simulator backend, or a custom NPU backend;
- CPU fallback on the host processor.

If a real second accelerator is unavailable, varying Gemmini configuration parameters can still support the retargetability claim, but the paper should be honest and phrase it as cross-configuration rather than cross-accelerator generality.

### Workloads

Use workloads that exercise the claimed semantics:

- isolated `matmul` and `conv2d` for tiling and legality;
- `conv2d + relu` and `matmul + bias + activation` for fusion and intermediate reuse;
- small edge-inference subgraphs for end-to-end offload selection;
- shape-mismatch or capacity-stress cases for fallback behavior.

### Baselines

The strongest baseline set is:

- CPU-only execution;
- handwritten RoCC/Gemmini macro or helper-library implementation;
- direct target-specific lowering without RAIR optimization;
- optional Buddy/Gemmini or TVM/VTA comparison if implementation effort is practical and the comparison is fair.

### Metrics

Report both performance and semantic evidence:

- end-to-end latency;
- kernel latency;
- transfer volume;
- number of host-accelerator transfers;
- launch and synchronization count;
- accelerator buffer reuse rate;
- number of accepted/rejected offload candidates;
- fallback cases by reason;
- compile-time overhead.

### Ablations

Ablations should map directly to contribution claims:

- RAIR with no cost-aware offload selection;
- RAIR with no buffer reuse;
- RAIR with no fusion/overlap;
- direct opaque-call representation;
- single-backend lowering without capability specialization.

## Novelty Statement

The novelty should be stated as:

1. **Compiler-visible accelerator interface semantics.** RAIR elevates accelerator context, typed buffers, memory spaces, transfer, launch, synchronization, and fallback into analyzable IR constructs.
2. **Reusable analyses enabled by explicit semantics.** RAIR enables legality checking, profitability-guided offloading, buffer lifetime analysis, memory reuse, and scheduling/fusion transformations over accelerator interactions.
3. **Whole-program CPU-accelerator coordination.** RAIR handles what to offload, how to coordinate host and accelerator execution, and when to preserve CPU fallback, rather than only lowering already selected kernels.

## Scope and Limitations

The paper should explicitly avoid overclaiming:

- It does not solve fully automatic graph partitioning for arbitrary models and arbitrary accelerators.
- It does not eliminate target-specific backend engineering.
- It does not claim a universal abstraction for all accelerator classes.
- It does not treat the runtime as the primary novelty unless substantial runtime scheduling is implemented and evaluated.
- It is strongest for tightly coupled RISC-V accelerators with explicit memory and synchronization behavior.

These limitations are useful because they make the thesis sharper: RAIR is a compiler abstraction boundary for RoCC-style accelerator interfaces.

## Recommended Paper Angle

The best current angle is:

**RAIR: Compiler-Visible Accelerator Interface Semantics for RoCC-Style RISC-V AI Accelerators.**

The title can be softened for TACO:

**An MLIR Abstraction Boundary for Tightly Coupled RISC-V AI Accelerators.**

## Current Thesis Checklist

A strong submission version should answer yes to these questions:

- Is the main claim about accelerator interface semantics rather than backend coverage?
- Are the contributions limited to IR semantics, semantics-enabled analyses, and retargetable lowering/fallback?
- Is Buddy/Gemmini treated as the closest single-target comparison?
- Are TVM/VTA, IREE, AIR/AIE, Timeloop/MAESTRO, and ATLAAS positioned accurately?
- Does each experiment support one contribution claim?
- Is fallback framed as correctness-preserving offload selection rather than an afterthought?
