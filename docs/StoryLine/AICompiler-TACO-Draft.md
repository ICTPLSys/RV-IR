# TACO Draft for AICompiler

> Note: This draft is written in a TACO-friendly journal style. The abstract avoids citations, displayed mathematics, and first-person language. Bracketed citation keys such as `[CGO25]` and `[AIR]` are placeholders to be replaced by ACM-style numeric citations in the final `acmart` manuscript.

## Candidate Title

RAIR: Compiler-Visible Accelerator Interface Semantics for RoCC-Style RISC-V AI Accelerators

Alternative TACO title:

An MLIR Abstraction Boundary for Tightly Coupled RISC-V AI Accelerators

## Abstract

RISC-V edge platforms increasingly combine general-purpose control cores with tightly coupled AI accelerators attached through custom interfaces such as RoCC. Although these accelerators provide efficient domain-specific execution, their software interfaces often encode accelerator invocation, memory placement, data movement, synchronization, and fallback behavior through handwritten macros, intrinsics, inline assembly, or device-specific runtime calls. This makes the accelerator interface opaque to compiler analyses and forces optimization logic to remain target-specific. This paper presents RAIR, an MLIR-based abstraction boundary for RoCC-style RISC-V AI accelerators. RAIR lifts accelerator contexts, typed accelerator buffers, memory spaces, transfers, launch boundaries, synchronization events, target capabilities, and CPU fallback paths into first-class intermediate representation constructs. Exposing these semantics enables legality checking, cost-aware offload selection, buffer lifetime analysis, accelerator memory reuse, short-region fusion, and transfer/compute overlap before final target-specific lowering. RAIR further supports retargetable backend specialization by separating common accelerator interface semantics from target capabilities such as supported operators, memory capacities, tiling constraints, dataflow choices, and synchronization models. Together, these mechanisms turn tightly coupled accelerator invocation from opaque host-side code into an analyzable compiler representation, enabling systematic CPU-accelerator coordination across RoCC-style systems.

## 1. Introduction

RISC-V-based edge systems are increasingly organized around a heterogeneous execution model in which a scalar host processor cooperates with one or more domain accelerators. In many such systems, accelerators are attached through tightly coupled custom interfaces such as RoCC. This organization gives hardware designers a flexible path to specialized machine-learning, signal-processing, and robotics accelerators, but it shifts substantial complexity into the software stack. A compiler must decide which program regions should run on the accelerator, how data should be placed in accelerator-visible memories, when transfers and synchronization should occur, and when execution must remain on the CPU because accelerator constraints are not satisfied.

Current RoCC-style software stacks typically expose this functionality through low-level programming interfaces. Developers invoke accelerators using C macros, inline assembly, intrinsics, handwritten helper libraries, or device-specific runtime calls. These interfaces are practical for expert programmers and individual accelerator stacks, but they make accelerator interaction opaque to compiler analyses. Accelerator resources appear as ordinary host-side values or hidden runtime state. Scratchpad and accumulator placement are encoded through conventions rather than types. DMA-like transfers and synchronization are represented as side effects. Fallback behavior is implemented manually around accelerator calls. As a result, the compiler cannot easily reason about offload legality, memory-space consistency, buffer lifetime, transfer redundancy, or CPU-accelerator coordination.

This opacity creates a compiler abstraction problem rather than only a backend code generation problem. A backend can lower a selected kernel to custom instructions, but it does not by itself answer which regions should be offloaded, which memory spaces they require, whether the accelerator can legally execute a given shape, how multiple accelerator regions interact through on-chip storage, or when CPU fallback should be preserved. These decisions depend on semantic information that is often lost when accelerator use is expressed as opaque host-side calls.

This paper presents RAIR, an MLIR-based abstraction boundary for RoCC-style RISC-V AI accelerators. RAIR lifts accelerator interface semantics into the intermediate representation: accelerator contexts, typed accelerator buffers, explicit memory spaces, transfers, launch regions, synchronization events, target capability constraints, and CPU fallback paths are represented as first-class IR constructs. The goal is not merely to introduce another target dialect. Rather, RAIR identifies the semantic layer at which tightly coupled accelerator invocation becomes analyzable before being specialized to a concrete backend.

Exposing accelerator interface semantics enables compiler analyses and transformations that are difficult to express over macros or runtime calls. RAIR allows the compiler to check offload legality against target capabilities, estimate profitability using transfer and synchronization costs, analyze accelerator buffer lifetimes, reuse on-chip storage across adjacent regions, fuse short operator chains, and overlap transfer with computation when dependences allow. The same representation can then be specialized to different RoCC-style targets or configurations by supplying backend capability descriptions and lowering hooks. Unsupported or unprofitable regions remain on the CPU path, making fallback part of the compilation flow rather than an external recovery mechanism.

The central claim of this work is that RoCC-style accelerator invocation should be represented as compiler-visible semantics, not as opaque host-side effects and not only as late backend operations. This position distinguishes RAIR from backend-centric RISC-V work that focuses on lowering selected kernels, from single-accelerator dialects tied to one code generation path, and from general deployment runtimes that do not model the fine-grained resource and memory semantics of tightly coupled accelerators.

## 1.1 Contributions

This work makes the following contributions:

1. It introduces RAIR, a RoCC-style accelerator interface IR that models accelerator contexts, typed buffers, memory spaces, data transfers, launch regions, synchronization, target capabilities, and fallback paths as first-class compiler constructs.

2. It develops semantics-enabled compiler analyses for tightly coupled accelerators, including legality and profitability based offload selection, target-aware tiling, buffer lifetime analysis, accelerator memory reuse, short-region fusion, and transfer/compute overlap.

3. It presents a retargetable lowering and fallback flow in which RAIR regions are specialized through backend capability descriptions, while illegal or unprofitable candidates are preserved on a CPU path.

## 2. Related Work

### 2.1 End-to-End DNN Compilation and Accelerator Stacks

Systems such as TVM and VTA demonstrate the value of end-to-end tensor compilation, scheduling, auto-tuning, and accelerator stack integration `[TVM,VTA]`. They provide a strong foundation for mapping tensor computations to diverse targets and have influenced many accelerator software stacks. RAIR is complementary to this line of work. Its focus is not a new tensor scheduling language or auto-tuning system, but the representation of RoCC-style accelerator interface semantics that are often hidden below scheduling abstractions: accelerator resource ownership, explicit accelerator-visible memory spaces, host-accelerator transfers, synchronization, and fallback behavior.

### 2.2 MLIR Deployment Runtimes

IREE and TinyIREE show how MLIR can support portable machine-learning deployment across a wide range of platforms through compiler/runtime integration `[IREE,TINYIREE]`. These systems validate the importance of jointly considering compilation and execution. RAIR targets a narrower but more explicit boundary: tightly coupled RISC-V accelerators where custom command invocation, scratchpad and accumulator placement, synchronization, and CPU fallback must remain visible to compiler analyses. The runtime in RAIR is therefore an execution substrate for compiler-visible decisions rather than the primary abstraction.

### 2.3 Spatial and Asynchronous Accelerator Compilation

MLIR-AIR, MLIR-AIE, and related spatial accelerator compilers expose data movement, hierarchy, and asynchronous execution for NPU fabrics `[AIR,AIE]`. These systems are highly relevant because they support the broader thesis that accelerator performance depends on compiler-visible orchestration rather than only low-level code generation. RAIR differs in its hardware/software boundary. Spatial compiler stacks model execution across accelerator fabrics, while RAIR models the host-coprocessor interface of RoCC-style systems, including accelerator contexts, typed memory spaces, launch boundaries, synchronization, and fallback integration with CPU code.

### 2.4 Accelerator-Specific Dialects and Gemmini Software Stacks

Accelerator-specific dialects and software stacks, including Buddy/Gemmini, show that MLIR-based lowering to concrete RISC-V accelerators is practical `[BUDDY,GEMMINI]`. This is the closest related category for RAIR. The distinction is that RAIR is not intended to be a Gemmini operation wrapper or a single-target lowering path. It defines common interface semantics for RoCC-style accelerators and separates those semantics from backend capability descriptions. The evaluation must therefore demonstrate either multiple accelerator targets or at least multiple target configurations to support the claim that the abstraction is not hard-coded to one accelerator.

### 2.5 RISC-V Compiler Support for Custom Extensions

Prior work on RISC-V compiler support has shown that general-purpose backend abstractions can become semantic bottlenecks for custom architectural features. Recent multi-level backend efforts for RISC-V ISA extensions preserve target structure above traditional backend layers to improve generated code for specialized micro-kernels `[CGO25]`. RAIR is aligned with this motivation but addresses a different question. Backend work primarily asks how a chosen kernel should be lowered. RAIR asks how accelerator use should be represented, analyzed, selected, coordinated with CPU execution, and preserved with fallback before final lowering.

### 2.6 Accelerator Mapping and Modeling Frameworks

Timeloop, MAESTRO, Accelergy, and related mapping frameworks model dataflows, memory hierarchies, and performance or energy tradeoffs for accelerator designs `[TIMELOOP,MAESTRO,ACCELERGY]`. RAIR can use similar ideas in target capability and cost models, but it is not primarily a hardware design-space exploration framework. Its contribution is to carry accelerator interface semantics through a compiler IR so that executable offloading decisions, memory reuse, synchronization, and fallback can be optimized in a program.

### 2.7 Automatic Extraction of Accelerator Semantics

ATLAAS and related work seek to derive tensor-level accelerator semantics from RTL or low-level hardware descriptions and use the recovered semantics to generate software support `[ATLAAS]`. RAIR addresses the next compiler layer. It assumes that accelerator semantics are known or can be exposed through a capability model and asks how those semantics should be represented in IR to enable legality checking, optimization, offloading, retargeting, and fallback. Semantic extraction and RAIR are therefore complementary: extracted semantics could populate RAIR target capability descriptions.

## 3. System Overview and Compiler Design

### 3.1 Design Goals

The compiler is designed around three goals. First, accelerator invocation should preserve the semantic information needed for legality analysis and optimization. Second, target-specific decisions should be made above the final backend layer while memory hierarchy, synchronization, and accelerator region structure are still explicit. Third, accelerator use should remain correct in the presence of unsupported operators, shape mismatches, resource constraints, or poor profitability by integrating CPU fallback into the compilation flow.

### 3.2 Compilation Flow

Figure X will present the end-to-end flow. At a high level, the compiler starts from a high-level MLIR program expressed in tensor or loop-oriented dialects and progressively lowers it toward heterogeneous execution over the CPU and one or more RoCC-attached accelerators.

The flow consists of four stages:

1. **Candidate discovery.** Hot single operators or short fusable regions are identified as potential offload candidates according to operator semantics, shape properties, and accelerator support constraints.

2. **Accelerator-aware materialization.** Legal candidates are rewritten into RAIR, where accelerator context acquisition, memory allocation, explicit transfers, launch boundaries, fallback paths, and synchronization points become first-class operations.

3. **Target-aware optimization.** Once accelerator-visible semantics are explicit, the compiler applies analyses and rewrites such as tiling, fusion, dependence-aware scheduling, memory reuse, and overlap transformations.

4. **Backend specialization and fallback integration.** Optimized RAIR regions are specialized using target capability descriptions and lowered to target-specific code generation paths, while unsupported or unprofitable regions remain on the CPU path.

This organization deliberately separates the decision to offload from the final instruction-level lowering step. The distinction is important because many of the relevant decisions, including legality of memory placement and profitability of synchronization-heavy regions, depend on semantics that are no longer visible once the program has been flattened into backend-oriented code.

### 3.3 Role of the Runtime

The runtime is intentionally lightweight. Its role is to provide concrete execution support for compiler-visible semantics, including accelerator context management, transfer submission, synchronization, and host-side dispatch. The runtime is not the primary source of optimization. Instead, RAIR performs as much reasoning as possible before code generation and leaves the runtime to execute already structured decisions.

### 3.4 Target Capability Model

RAIR uses target capability descriptions to keep common interface semantics separate from backend-specific constraints. A capability description records supported operators, memory spaces, capacity limits, tiling constraints, dataflow choices, synchronization behavior, transfer model, and approximate setup or transfer costs. This information drives legality checks, offload profitability estimation, optimization choices, and final lowering.

A capability model also makes the retargetability claim concrete. A new backend does not require changing the meaning of RAIR operations; it supplies a different specialization of the same accelerator interface semantics.

## 4. RoCC Dialect Design

### 4.1 Abstraction Boundary

RAIR is designed to sit between high-level compute representations and low-level accelerator-specific code generation. It does not replace target-specific backends, nor does it expose only raw custom instructions. Instead, it captures the semantic layer at which accelerator use becomes analyzable: resource ownership, memory placement, transfer intent, synchronization structure, fallback, and region-level accelerator execution.

This design reflects a deliberate abstraction choice. If the dialect is too high-level, it cannot express the operational constraints that determine legality and performance on RoCC-attached accelerators. If it is too low-level, memory and synchronization structure collapse into opaque effects and become difficult to optimize. The proposed dialect therefore aims to preserve exactly those concepts needed for offloading and optimization while still allowing target-specific lowering downstream.

### 4.2 Core Operations

The dialect is organized around five classes of operations and metadata.

**Resource management operations** represent accelerator ownership and execution scope. Operations such as `rocc.acquire`, `rocc.release`, and `rocc.await` make context lifetime and synchronization explicit. This enables the compiler to reason about accelerator resource usage rather than inferring it from host-side library calls.

**Data management operations** represent memory placement and transfer behavior. Operations such as `rocc.alloc_buffer`, `rocc.transfer`, and `rocc.dealloc_buffer` expose where values reside, when data movement occurs, and how long accelerator-side buffers remain live. These properties are central to memory reuse and legality checking.

**Compute operations** represent accelerator-supported kernels or primitive accelerator functions, such as `rocc.matmul`, `rocc.conv2d`, and `rocc.element_wise`. These operations serve as structured accelerator semantics rather than as plain code generation intrinsics.

**Region-based execution operations** such as `rocc.kernel` and `rocc.launch` define boundaries within which fusion, scheduling, and accelerator-side transformation are meaningful. They also provide a natural bridge between high-level offload candidates and target-specific lowering.

**Fallback metadata and operations** connect accelerator materialization to the preserved CPU path. Fallback is part of the compilation contract: if a candidate fails legality or profitability checks, the compiler does not emit an invalid accelerator region.

### 4.3 Types and Attributes

The dialect relies on typed representations to make accelerator-specific invariants explicit. A type such as `!rocc.context` denotes an accelerator execution context, potentially enriched with target identity or capability information. A type such as `!rocc.buffer<..., memory_space>` represents data located in an accelerator-visible memory space. This typing scheme allows the compiler to reject illegal combinations early and to propagate accelerator-specific memory information through transformation passes.

Attributes carry target-aware optimization intent and hardware constraints. Representative examples include:

- `accelerator`, which identifies the target accelerator or accelerator class,
- `memory_space`, which describes placement in accelerator-visible storage hierarchies such as scratchpad or accumulator memory,
- `tile_size`, which conveys target-aware decomposition choices,
- `dataflow`, which expresses accelerator-side scheduling or storage policy assumptions,
- `capability`, which links an operation or region to a target capability description,
- `fallback`, which records the CPU path preserved for correctness.

The purpose of these attributes is not merely annotation. Rather, they form part of the contract between optimization passes and lowering stages, making explicit which decisions remain symbolic and which have already been committed.

### 4.4 Semantics for Analysis and Transformation

The most important property of RAIR is that it makes accelerator effects explicit enough to support transformation. Synchronization becomes a named program event instead of an implicit convention. Memory placement becomes part of the value model instead of a backend detail. Region boundaries make it possible to distinguish between intra-accelerator optimization and host-accelerator coordination. Fallback makes legality and profitability decisions visible in the IR. These design choices make RAIR suitable not only for lowering but also for legality checking, cost modeling, dependence analysis, and optimization across multiple accelerator interaction patterns.

## 5. Optimization Passes

### 5.1 Offload Candidate Discovery

The first optimization stage identifies candidate regions for accelerator execution. In the current design, candidates are intentionally restricted to hot single operators and short fusable patterns, since these offer a favorable balance between optimization opportunity and analysis complexity. Candidate selection uses operator semantics, static legality constraints, and profile information when available. This avoids the need to solve fully general graph partitioning while still exposing the compiler to meaningful offloading decisions.

### 5.2 Cost-Aware Offload Selection

After candidate discovery, the compiler evaluates whether offloading is legal and beneficial. The legality check uses target capabilities such as supported operators, shape restrictions, memory capacity, tile alignment, dataflow support, and synchronization behavior. The profitability model combines setup overhead, estimated transfer latency, expected accelerator compute time, synchronization cost, and overlap opportunities. Shape-dependent utilization effects are particularly important for accelerators whose efficiency depends on tiling compatibility or array occupancy. The objective is therefore not simply to test whether an operator is supported, but to estimate whether a candidate remains profitable once coordination costs are included.

### 5.3 Target-Aware Tiling

For compute-intensive kernels such as matrix multiplication or convolution, the compiler selects tilings under accelerator-specific capacity and alignment constraints. These constraints may reflect scratchpad size, array geometry, banking restrictions, or accelerator dataflow preferences. By making these constraints visible before final lowering, the compiler can choose decompositions that reduce external memory traffic while preserving legality. The proposed design does not depend on a single optimization solver; the implementation may use heuristic search, integer programming, or profile-guided selection depending on the target and evaluation budget.

### 5.4 Buffer Lifetime Analysis and Memory Reuse

Because accelerator allocation and deallocation are explicit in RAIR, the compiler can infer deterministic buffer lifetimes and reuse opportunities. In favorable cases, accelerator buffers can be statically assigned, reused across adjacent kernels, or retained on-chip across short fused regions. This reduces redundant transfers and improves locality. The benefit of this pass depends directly on modeling accelerator memory as a first-class program concept rather than as an implicit side effect of runtime calls.

### 5.5 Dependence Analysis, Scheduling, and Overlap

Explicit memory-space and synchronization semantics make it possible to distinguish dependent and independent accelerator actions. This enables the compiler to reorder operations, overlap transfer with computation, or interleave independent accelerator regions when resource constraints permit. Such scheduling decisions are especially important for RoCC-style accelerators because host-side invocation and synchronization overhead can dominate short kernels if operations are executed naively. The compiler therefore aims to reduce idle time not only through faster kernels but also through better structured coordination.

### 5.6 Region Fusion and Double Buffering

Short operator chains are natural opportunities for region fusion, particularly when intermediate values can remain in accelerator-visible storage. Fusing patterns such as convolution followed by activation can reduce both memory traffic and host-side launch overhead. Similarly, double buffering can be introduced when the target accelerator and memory subsystem support overlap between load and compute phases. These optimizations are not unique in isolation, but the proposed abstraction allows them to be expressed as reusable passes over explicit accelerator semantics rather than as ad hoc accelerator-specific scripts.

## 6. Evaluation Methodology

### 6.1 Research Questions

The evaluation should be organized around the following questions:

- **RQ1:** Does RAIR expose accelerator resource, memory, synchronization, and fallback semantics that are opaque in handwritten interfaces or backend-only lowering?
- **RQ2:** Do the resulting analyses and optimizations improve performance or reduce transfer and synchronization overhead on representative RoCC-attached accelerators?
- **RQ3:** Does the same RAIR representation remain useful across more than one accelerator target or target configuration within the RoCC setting?
- **RQ4:** What is the cost of the abstraction in terms of compilation overhead, code complexity, or remaining target-specific customization?

These questions align the evaluation with the paper's main claims and avoid reducing the experimental section to a pure speedup report.

### 6.2 Claim-to-Experiment Matrix

The experimental section should make the connection between claims and evidence explicit:

| Claim | Experiment | Primary evidence |
| --- | --- | --- |
| RAIR exposes accelerator interface semantics | Compare RAIR IR with handwritten macro/runtime-call paths on representative kernels | explicit contexts, memory spaces, transfers, sync, fallback paths |
| Cost-aware selection avoids invalid or unprofitable offloads | Run legal, illegal, profitable, and unprofitable shape cases | accepted/rejected regions, fallback reasons, end-to-end latency |
| Buffer lifetime analysis reduces movement | Run adjacent kernels and short subgraphs with and without memory reuse | transfer count, transfer volume, buffer reuse rate |
| Fusion/overlap reduces coordination overhead | Run `conv2d + relu` and `matmul + bias + activation` with ablations | launch count, sync count, latency |
| RAIR is not hard-coded to one backend | Lower the same RAIR region to two targets or configurations | shared IR, target capability differences, performance portability |

### 6.3 Baselines

The evaluation should compare against at least three classes of baselines:

1. **CPU-only execution**, which establishes the value of offloading.
2. **Handwritten accelerator interfaces**, such as macro- or intrinsic-based implementations, which test whether the proposed abstraction retains competitiveness against expert-coded paths.
3. **Direct target-specific lowering without RAIR**, which tests whether explicit accelerator semantics above the backend layer provide measurable benefits.
4. **Optional framework baselines**, such as Buddy/Gemmini or TVM/VTA paths, when available and fair to compare.

If a full baseline implementation is unavailable, the paper should state this clearly and compare against the closest practical alternative.

### 6.4 Workloads and Platforms

The most convincing evaluation should include representative workloads composed of both isolated kernels and short operator sequences. Suitable examples include matrix multiplication, convolution, `conv2d + relu`, `matmul + bias + activation`, and selected subgraphs from edge-oriented inference workloads. These workloads should exercise different combinations of compute intensity, transfer cost, synchronization behavior, and fallback behavior.

On the platform side, evaluation should cover at least one concrete RoCC-attached accelerator implementation and one second target or configuration. A strong setup would include a Gemmini-like accelerator and a custom NPU or simulator backend. If a second accelerator is unavailable, varying Gemmini array shape, memory capacity, dataflow support, or supported operator sets can still strengthen the cross-configuration claim, provided the paper describes the limitation clearly.

### 6.5 Metrics

Performance should be reported using end-to-end latency and kernel-level execution time, but the evaluation should also include metrics that directly reflect the paper's abstraction claims. Useful examples include:

- number of offloaded regions selected by the compiler,
- reduction in host-accelerator transfers,
- accelerator buffer reuse rate or lifetime statistics,
- launch and synchronization overhead,
- fallback frequency and rejection reasons,
- sensitivity to shape mismatch or resource constraints,
- compile-time overhead introduced by accelerator-aware analyses.

These measurements help show not only that the system is fast, but also why the proposed abstraction improves optimization opportunities.

### 6.6 Ablation Studies

An ablation study is important for demonstrating that the contribution is the integrated compilation flow rather than any single optimization. Useful ablations include:

- disabling cost-aware offload selection,
- disabling memory reuse or buffer lifetime analysis,
- disabling fusion or overlap transformations,
- replacing typed memory spaces with opaque host-side calls,
- comparing with a direct lowering flow that bypasses RAIR,
- using a single hard-coded backend path instead of capability specialization.

Such experiments would clarify how much of the benefit comes from semantic visibility versus individual target-specific rewrites.

## 7. Limitations and Discussion

The current design makes several scope choices that should be stated explicitly in the paper.

First, the work does not claim to solve fully general partitioning for arbitrary computation graphs and arbitrary accelerator sets. The current flow instead targets hot single operators and short fusable regions, which is a deliberate compromise between tractability and usefulness.

Second, RAIR does not eliminate target-specific backend engineering. Final lowering still depends on accelerator-specific instruction sets, memory systems, and runtime support. The contribution is therefore best understood as a structured abstraction layer above those backends rather than as a replacement for them.

Third, the current paper should avoid overstating runtime novelty. Although the runtime is necessary for execution, the main claim lies in compiler-visible representation and analysis. Unless extensive runtime scheduling mechanisms are implemented and evaluated, the runtime should be framed as an enabling component rather than as a standalone research contribution.

Fourth, the abstraction is strongest for tightly coupled accelerators with explicit memory and synchronization structure, such as RoCC-attached engines. It may transfer only partially to accelerator classes whose programming models hide resource management behind richer device runtimes or whose execution model is fundamentally more dynamic.

These limitations do not weaken the paper if they are stated clearly. On the contrary, they help position the contribution accurately: the value of the work lies in identifying and formalizing the compiler abstraction boundary needed for RoCC-style heterogeneous acceleration.

## 8. Notes for the Final TACO Manuscript

- Replace placeholder citation keys with ACM numeric citations.
- Move the final contribution list into the end of the introduction, as is common in TACO papers.
- Keep the abstract within roughly 150-250 words and do not add citations or displayed equations.
- In the LaTeX submission, use the ACM `acmart` journal template with the `acmsmall` style and review in `manuscript` mode.
- Add a compact related-work comparison table covering TVM/VTA, IREE, AIR/AIE, Buddy/Gemmini, RISC-V backend work, Timeloop/MAESTRO, and ATLAAS.
- Add one figure showing the RAIR flow: high-level MLIR, candidate discovery, RAIR materialization, semantics-enabled optimization, capability specialization, backend lowering, and CPU fallback.
