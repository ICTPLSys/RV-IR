# AICompiler 投稿思路修改

## 推荐主线

> Thesis: RoCC-style AI accelerator interfaces hide resource ownership, memory placement, data movement, synchronization, and fallback decisions behind runtime calls, intrinsics, and inline assembly. RAIR lifts these accelerator interface semantics into compiler-visible IR constructs so that offloading can be analyzed, optimized, and retargeted before final backend lowering.

这篇文章最强的讲法不是“支持多个 RISC-V AI accelerator backend”，而是：

- RISC-V/RoCC AI accelerator 的软件接口正在变成编译器瓶颈；
- 现有接口把 accelerator invocation 伪装成 host-side opaque calls；
- 编译器因此看不见 accelerator context、scratchpad/accumulator memory、DMA、sync、resource lifetime 和 fallback 条件；
- RAIR/`rocc` dialect 提供一个中间层，把这些接口语义显式化；
- 一旦语义进入 IR，legality checking、cost-aware offload selection、buffer lifetime/reuse、fusion、overlap、backend specialization 才能成为系统性的 compiler pass。

## 论文应避免的弱叙事

### 不要主打“多后端统一”

“XuanTie / Gemmini / 自研 NPU 都能 lower”听起来像工程覆盖面，而不是顶会级论文贡献。多后端应该作为证据，证明 RAIR 抽象没有被某一个 accelerator hard-code，而不是作为核心 novelty。

### 不要主打“又一个 MLIR dialect”

Reviewer 会自然联想到 Buddy/Gemmini、MLIR-AIR、MLIR-AIE、IREE、TVM/VTA。论文需要反复强调：RAIR 的贡献不是 dialect 本身，而是 RoCC-style host-accelerator boundary 的 semantic abstraction。

### 不要主打“runtime 框架”

除非后续实现了强 runtime scheduler 并做完整实验，否则 runtime 应该只是执行 IR 决策的支撑层。主贡献应放在 compiler-visible representation and analysis。

## 三个核心贡献

### C1. RoCC-Style Accelerator Interface IR

定义 RAIR/`rocc` dialect，把以下概念变成一等 IR/type/attribute：

- accelerator context and ownership；
- typed accelerator buffers；
- accelerator-visible memory spaces, such as scratchpad and accumulator；
- explicit transfer and synchronization；
- structured launch/kernel regions；
- target capability and legality constraints；
- CPU fallback path metadata。

核心说法：RAIR represents accelerator interaction semantics, not only accelerator operations.

### C2. Semantics-Enabled Compiler Analyses

选择 2-3 个最容易实证、最能体现 IR 价值的 pass，而不是列一堆优化名词：

- legality and profitability based offload selection；
- buffer lifetime analysis and accelerator memory reuse；
- short-region fusion and transfer/compute overlap。

这些优化单独看不一定全新，关键是它们都依赖 RAIR 把 opaque side effects 变成 analyzable IR effects。

### C3. Retargetable Lowering with Correctness-Preserving Fallback

同一 RAIR representation lower 到至少两个 target 或 target configuration：

- Gemmini-like RoCC accelerator；
- 自研 NPU / simulator backend / another Gemmini configuration；
- optional XuanTie vector extension only if its interface can be modeled consistently。

fallback 不是附加功能，而是 offload selection 的 correctness condition：当 shape、memory capacity、operator support、dataflow 或 profitability 不满足条件时，region 保留在 CPU path。

## Related Work 防线

- TVM/VTA: strong in tensor scheduling and end-to-end accelerator stack; RAIR focuses on compiler-visible RoCC invocation, memory, sync, and fallback semantics.
- IREE/TinyIREE: strong in portable ML deployment and runtime dispatch; RAIR targets tightly coupled RISC-V accelerators with explicit resource and memory-space semantics.
- MLIR-AIR/AIE: strong in spatial/asynchronous NPU orchestration; RAIR targets host-coprocessor invocation semantics rather than spatial fabric programming.
- Buddy/Gemmini: closest single-accelerator MLIR stack; RAIR must show cross-target/configuration abstraction and whole-program offload reasoning beyond Gemmini op wrapping.
- RISC-V custom backend work: strong in lowering selected kernels to custom ISA extensions; RAIR decides what should be offloaded and how accelerator interaction is represented before backend lowering.
- Timeloop/MAESTRO/Accelergy: strong in mapping/modeling/design-space exploration; RAIR is a compiler IR and optimization framework, not primarily a hardware mapper.
- ATLAAS/ACT-like semantic extraction: derives accelerator semantics from RTL; RAIR assumes semantics are available and asks how to represent and optimize them in compiler IR.

## 最小可发表系统形态

为了冲 A 类会议或 TACO，系统最好至少做到：

- 一个清晰的 RAIR/`rocc` dialect specification；
- 一个 capability/legality model，描述 target 支持的 ops、memory spaces、tile constraints、sync model；
- 一个 offload materialization pass；
- 一个 cost-aware selection pass；
- 一个 memory reuse 或 fusion/overlap pass；
- 一个 Gemmini-like backend；
- 一个不同 target/configuration backend；
- CPU fallback path；
- 与 handwritten/runtime-call/direct-lowering baseline 的对比实验。

## 推荐标题方向

- **RAIR: Compiler-Visible Accelerator Interface Semantics for RoCC-Style RISC-V AI Accelerators**
- **Lifting RoCC Accelerator Interfaces into Compiler IR**
- **An MLIR Abstraction Boundary for Tightly Coupled RISC-V AI Accelerators**