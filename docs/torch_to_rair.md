# Torch Frontend to RAIR IR

This note records the command line flow for importing a PyTorch model through
the torch-mlir frontend and lowering it step by step to RAIR IR.

## Prerequisites

Build `torch-mlir-opt` first:

```bash
ninja -C build torch-mlir-opt
```

Use the Python environment that can import `torch` and `torch_mlir`:

```bash
conda activate torch-mlir
```

All commands below are intended to be run from the RV-IR repository root.

## Quick Path

For a minimal matrix multiplication example:

```bash
python projects/pt1/examples/matmul_to_rair.py
```

This writes:

```text
projects/pt1/examples/mlir_output/matmul_linalg.mlir
projects/pt1/examples/mlir_output/matmul_linalg_memref.mlir
projects/pt1/examples/mlir_output/matmul_rair.mlir
```

Inspect the final RAIR operation:

```bash
rg "rair\\." projects/pt1/examples/mlir_output/matmul_rair.mlir
```

For the built-in MNIST example:

```bash
bash scripts/mnist_torch_linalg_rair_to_cpp.sh --model mnist
```

For the tiny ResNet-style example:

```bash
bash scripts/mnist_torch_linalg_rair_to_cpp.sh --model resnet --spatial 5 --pad 3
```

The script performs:

```text
PyTorch model
  -> torch-mlir linalg-on-tensors
  -> bufferized linalg/memref form
  -> RAIR IR
```

The RAIR output is written under:

```text
projects/pt1/examples/mlir_output/
```

For example:

```text
projects/pt1/examples/mlir_output/mnist_rair_memref.mlir
projects/pt1/examples/mlir_output/resnet_simple_rair_memref.mlir
```

## Manual Step-by-Step Flow

## Matrix Multiplication Example

The standalone matrix multiplication frontend example is:

```text
projects/pt1/examples/matmul_to_rair.py
```

It defines this PyTorch module:

```python
class MatmulModule(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs)
```

Run the default `16x32 @ 32x64` example:

```bash
python projects/pt1/examples/matmul_to_rair.py
```

Use custom matrix sizes:

```bash
python projects/pt1/examples/matmul_to_rair.py --m 4 --k 8 --n 16
```

If your default `python` does not have `torch` and `torch_mlir`, use the
`torch-mlir` conda environment:

```bash
conda activate torch-mlir
python projects/pt1/examples/matmul_to_rair.py
```

Or call its Python directly:

```bash
/home/jianzexin/miniconda3/envs/torch-mlir/bin/python \
  projects/pt1/examples/matmul_to_rair.py
```

The script performs the same steps as the manual flow:

```bash
# 1. Export torch.matmul to linalg-on-tensors:
#    projects/pt1/examples/mlir_output/matmul_linalg.mlir

# 2. Bufferize to linalg/memref:
#    projects/pt1/examples/mlir_output/matmul_linalg_memref.mlir

# 3. Lower linalg.matmul to rair.matmul:
build/bin/torch-mlir-opt \
  projects/pt1/examples/mlir_output/matmul_linalg_memref.mlir \
  --convert-linalg-to-rair \
  -o projects/pt1/examples/mlir_output/matmul_rair.mlir
```

The final file should contain:

```mlir
rair.matmul ins(...)
```

Check it with:

```bash
rg "rair\\.matmul" projects/pt1/examples/mlir_output/matmul_rair.mlir
```

### 1. Import Torch Model to Linalg-on-Tensors

MNIST:

```bash
python projects/pt1/examples/mnistnet_to_linalg.py \
  --emit linalg-tensors \
  --output projects/pt1/examples/mlir_output/mnist_linalg.mlir
```

Tiny ResNet-style model:

```bash
python projects/pt1/examples/resnet_simple_to_linalg.py \
  --emit linalg-tensors \
  --output projects/pt1/examples/mlir_output/resnet_simple_linalg.mlir
```

### 2. Materialize Bufferized Linalg/MemRef IR

The current RAIR lowering patterns expect the memref-oriented linalg form used
by the RISCV/RAIR tests.

MNIST:

```bash
python projects/pt1/examples/mnistnet_to_linalg.py \
  --emit memref-stack-weights \
  --fill-value 1.0 \
  --output projects/pt1/examples/mlir_output/mnist_linalg_memref.mlir
```

Tiny ResNet-style model:

```bash
python projects/pt1/examples/resnet_simple_to_linalg.py \
  --emit memref-stack-weights \
  --fill-value 1.0 \
  --spatial 5 \
  --pad 3 \
  --output projects/pt1/examples/mlir_output/resnet_simple_memref.mlir
```

### 3. Lower Linalg/MemRef IR to RAIR IR

MNIST:

```bash
build/bin/torch-mlir-opt \
  projects/pt1/examples/mlir_output/mnist_linalg_memref.mlir \
  --convert-linalg-to-rair \
  -o projects/pt1/examples/mlir_output/mnist_rair_memref.mlir
```

Tiny ResNet-style model:

```bash
build/bin/torch-mlir-opt \
  projects/pt1/examples/mlir_output/resnet_simple_memref.mlir \
  --convert-linalg-to-rair \
  -o projects/pt1/examples/mlir_output/resnet_simple_rair_memref.mlir
```

Check that the result contains RAIR operations:

```bash
rg "rair\\." projects/pt1/examples/mlir_output/mnist_rair_memref.mlir
```

## Downstream Status

The legacy RAIR-to-Affine and RAIR-to-CIM passes have been removed. Generated
RAIR should currently be treated as an inspectable intermediate result while
the RAIR Core/Plan contract and its target lowering are implemented.

`--convert-rair-to-llvm` remains for utility operations such as `rair.print`
and `rair.world`; it is not a complete compute backend.

## Useful Pass Names

The RAIR-related pass names are:

```text
--convert-linalg-to-rair
--convert-rair-to-llvm
```

Confirm them with:

```bash
build/bin/torch-mlir-opt --help | rg "rair"
```
