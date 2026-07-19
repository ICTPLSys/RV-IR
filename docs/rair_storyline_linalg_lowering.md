# RAIR Story Line, Upper Lowering Design, and Motivation Result

This note evaluates the current paper story line and turns it into a concrete
RAIR design plan for the upper compiler path:

```text
Torch / Linalg / Tensor or MemRef IR -> RAIR
```

Lowering below RAIR is intentionally out of scope here.

## Innovation Assessment

The story line is innovative if RAIR is presented and implemented as a
compiler-visible accelerator interface layer, not merely as an operation
wrapper around a single accelerator backend.

The strongest claim is:

> RoCC-style RISC-V AI accelerator interfaces hide resource ownership, memory
> placement, data movement, synchronization, target legality, and fallback
> decisions behind macros, intrinsics, inline assembly, and helper runtimes.
> RAIR lifts these interface semantics into IR so that the compiler can analyze
> and optimize accelerator offloading before final backend lowering.

This is a real abstraction-boundary contribution. It is distinguishable from:

- **TVM/VTA**, which primarily emphasize tensor scheduling and accelerator stack
  integration.
- **IREE**, which focuses on portable dispatch/runtime deployment.
- **MLIR-AIR/AIE**, which model spatial/asynchronous accelerator fabrics.
- **Buddy/Gemmini-like dialects**, which are closer to concrete target lowering.
- **RISC-V custom backend work**, which usually starts after a kernel has
  already been selected for lowering.

However, the story becomes weak if the implementation only does:

```text
linalg.matmul -> rair.matmul -> backend
```

That would look like another target dialect. For the paper-level claim, RAIR
needs to expose at least part of the following upper-level semantics:

- offload candidate boundaries;
- target capability and legality decisions;
- accelerator-visible memory placement;
- explicit transfer/synchronization intent;
- CPU fallback metadata;
- enough structure to enable fusion or memory reuse decisions.

So the novelty is conditional:

| Current Form | Novelty Level |
| --- | --- |
| RAIR as op wrapper only | weak |
| RAIR as Linalg offload materialization with capability/fallback metadata | publishable systems contribution |
| RAIR plus evaluated legality/cost/memory-reuse/fusion passes | strong TACO/A-class systems story |

## Upper Lowering Contract

The RAIR upper lowering should be split into four compiler-visible stages.

### 1. Candidate Discovery

Input IR is Linalg-on-tensors or bufferized Linalg-on-memrefs. The pass should
recognize accelerator-relevant regions:

- single heavy ops: `linalg.matmul`, `linalg.batch_matmul`,
  `linalg.conv_2d_nchw_fchw`, `linalg.pooling_nchw_*`;
- short patterns: `conv2d + relu`, `matmul + bias`, `matmul + bias + relu`;
- target-friendly layout/data movement ops such as `linalg.transpose`;
- rejected or host-only ops such as unsupported `linalg.generic` bodies.

The output of this stage should be a list of candidates with source locations,
operator kinds, shapes, element types, and estimated input/output byte volume.

### 2. Legality and Capability Check

Before materializing RAIR, the compiler should check a target capability model.
For example:

```text
rair.target @gemmini_like {
  ops = [matmul, conv2d_nchw_fchw, pooling_nchw_sum, pooling_nchw_max]
  element_types = [i8, i32, f32]
  memory_spaces = [gmem, spad, accumulator]
  array_shape = [16, 16]
  scratchpad_bytes = 262144
  accumulator_bytes = 65536
  supports_async_transfer = true
}
```

At this layer, the capability model does not need to lower to real hardware.
It only needs to answer:

- is the op supported?
- are the ranks/layouts supported?
- are element types supported?
- can a legal tile fit in accelerator-visible memory?
- should the region stay on the CPU path?

### 3. Profitability Selection

The story line's cost model can be attached to candidate materialization:

```text
T_offload = T_setup + T_transfer + T_compute + T_sync - T_overlap
```

The first implementation can be deliberately simple:

- use static byte counts for transfer estimates;
- use operation counts for compute estimates;
- reject very small shapes because setup/sync dominates;
- record rejection reasons in attributes or an analysis report.

This is enough to support the paper's claim that offload is a compiler decision,
not a backend assumption.

### 4. RAIR Materialization

Once a candidate is legal and profitable, Linalg is rewritten to RAIR. The
current implementation already performs op-level materialization. The paper
story line should evolve this into a richer contract.

Recommended near-term RAIR attributes:

```mlir
rair.matmul
  ins(%lhs, %rhs : memref<16x32xf32>, memref<32x64xf32>)
  outs(%out : memref<16x64xf32>)
  {
    accelerator = "gemmini_like",
    capability = @gemmini_like,
    offload = "accepted",
    tile_size = [16, 16, 16],
    dataflow = "weight_stationary",
    fallback = @cpu_matmul
  }
```

Recommended staged design:

| Stage | IR Form | Purpose |
| --- | --- | --- |
| S0 current | direct `linalg.* -> rair.*` | prove the frontend and op capture path |
| S1 next | RAIR compute ops with target/fallback attributes | expose legality/profitability decisions |
| S2 next | `rair.launch` region around accepted candidates | expose fusion and scheduling boundaries |
| S3 later | `!rair.context`, `!rair.buffer`, `rair.transfer` | full interface-semantics story |

For the current request, S1 and S2 are the most important. They improve the
upper lowering story without depending on RAIR-to-backend work.

## Linalg to RAIR Mapping Plan

### Implemented Today

The current `--convert-linalg-to-rair` pass lowers:

| Linalg Op | RAIR Op |
| --- | --- |
| `linalg.matmul` | `rair.matmul` |
| `linalg.batch_matmul` | `rair.batch_matmul` |
| `linalg.matvec` | `rair.matvec` |
| `linalg.conv_2d` | `rair.conv2d` |
| `linalg.conv_2d_nchw_fchw` | `rair.conv_2d_nchw_fchw` |
| `linalg.pooling_nchw_max` | `rair.pooling_nchw_max` |
| `linalg.pooling_nchw_sum` | `rair.pooling_nchw_sum` |
| `linalg.transpose` | `rair.transpose` |
| `linalg.reduce` | `rair.reduce` |
| `linalg.add/sub/mul/div` | `rair.add/sub/mul/div` |
| `linalg.max/min/negf` | `rair.max/min/negf` |

The `conv_2d_nchw_fchw` and `pooling_nchw_sum` mappings were added while
preparing this note because they appear in the ResNet motivation example.

### Important Gaps

The current pass still leaves many model-level helper computations in Linalg:

- `linalg.fill`;
- `linalg.broadcast`;
- `linalg.map`;
- arbitrary `linalg.generic`;
- activation patterns represented as `cmpf + select`;
- batchnorm/normalization patterns represented as generic math bodies;
- bias add when it appears as `linalg.generic` instead of named `linalg.add`.

For the story line, these gaps are useful because they identify the next
compiler-visible optimization targets:

| Pattern | Suggested RAIR Materialization |
| --- | --- |
| ReLU as `cmpf/select` generic | `rair.relu` or `rair.max` with zero |
| Bias add generic | `rair.add` with broadcast semantics or `rair.bias_add` |
| `matmul + bias` | one RAIR candidate region |
| `conv2d + relu` | one RAIR launch/candidate region |
| `fill` before matmul/pooling | treat as initialization/fallback boundary |
| unsupported generic | keep on CPU path and record rejection reason |

## Motivation Run

The following commands were run from the repository root:

```bash
ninja -C build torch-mlir-opt
build/bin/llvm-lit -sv test/Conversion/LinalgToRAIR
build/bin/torch-mlir-opt projects/pt1/examples/mlir_output/matmul_linalg_memref.mlir --convert-linalg-to-rair -o /tmp/rair_motivation_matmul_after_patch.mlir
build/bin/torch-mlir-opt projects/pt1/examples/mlir_output/mnist_linalg_memref.mlir --convert-linalg-to-rair -o /tmp/rair_motivation_mnist_after_patch.mlir
build/bin/torch-mlir-opt projects/pt1/examples/mlir_output/resnet_simple_memref.mlir --convert-linalg-to-rair -o /tmp/rair_motivation_resnet_simple_after_patch.mlir
build/bin/torch-mlir-opt test/Conversion/LinalgToRAIR/conv2d.mlir --convert-linalg-to-rair -o /tmp/rair_motivation_conv2d.mlir
```

Test result:

```text
test/Conversion/LinalgToRAIR: 13/13 passed
```

Operation counts before and after `--convert-linalg-to-rair`:

| Case | Before | After |
| --- | --- | --- |
| `matmul` | `linalg.fill:1`, `linalg.matmul:1` | `linalg.fill:1`, `rair.matmul:1` |
| `mnist` | `linalg.fill:10`, `linalg.generic:8`, `linalg.index:1`, `linalg.matmul:2`, `linalg.pooling_nchw_max:1`, `linalg.transpose:2`, `linalg.yield:8` | `linalg.fill:10`, `linalg.generic:8`, `linalg.index:1`, `linalg.yield:8`, `rair.matmul:2`, `rair.pooling_nchw_max:1`, `rair.transpose:2` |
| `resnet_simple` | `linalg.broadcast:1`, `linalg.conv_2d_nchw_fchw:1`, `linalg.fill:8`, `linalg.generic:4`, `linalg.map:1`, `linalg.matmul:1`, `linalg.pooling_nchw_sum:1`, `linalg.yield:5` | `linalg.broadcast:1`, `linalg.fill:8`, `linalg.generic:4`, `linalg.map:1`, `linalg.yield:5`, `rair.conv_2d_nchw_fchw:1`, `rair.matmul:1`, `rair.pooling_nchw_sum:1` |
| `conv2d_tensor` | `linalg.conv_2d:1`, `rair.print:1` | `rair.conv2d:1`, `rair.print:1` |

### What This Motivation Shows

The current compiler can already expose important accelerator-relevant compute
ops in RAIR:

- standalone matrix multiplication;
- MNIST-style fully connected layers, pooling, and transposes;
- ResNet-style NCHW/FCHW convolution, pooling, and final matmul;
- tensor-level 2-D convolution.

This supports the first motivation claim: accelerator-relevant operations can
be lifted from generic high-level IR into explicit RAIR operations before
backend lowering.

### What It Does Not Show Yet

This run does not yet prove the full paper claim. It does not measure:

- target legality decisions;
- accepted versus rejected candidates;
- CPU fallback regions;
- transfer volume reduction;
- launch/synchronization count reduction;
- runtime speedup;
- RAIR-to-backend performance.

The next Motivation experiment should therefore add an upper-level report pass:

```text
rair-offload-report:
  candidates: 5
  accepted: 3
  rejected: 2
  rejected_reasons:
    unsupported_generic_body: 1
    unprofitable_small_shape: 1
  estimated_transfer_bytes_before: ...
  estimated_transfer_bytes_after: ...
```

That report would make the paper's motivation much sharper because it would
show not only that RAIR ops exist, but also why explicit semantics help the
compiler make offload decisions.

## Recommended Next Implementation Steps

1. Add capability/offload attributes to RAIR compute ops emitted by
   `--convert-linalg-to-rair`.
2. Add an analysis-only offload report pass for accepted/rejected candidates.
3. Pattern-match `linalg.generic` ReLU and bias-add into RAIR-level ops or RAIR
   candidate regions.
4. Introduce a high-level `rair.launch` or `rair.candidate` wrapper around
   accepted short regions such as `conv2d + relu` and `matmul + bias`.
5. Preserve unsupported regions as CPU fallback and record fallback reasons in
   IR attributes or a pass report.

This sequence stays entirely above RAIR backend lowering, while moving the
implementation closer to the paper's central innovation claim.
