# RVV xDSL Microkernel Runner

Generate RVV GEMM microkernels locally, sync them to a RISC-V board, run benchmarks, and fetch results back to your host.

## Quick Start

1. Install local tools
- `git`, `cmake`, `ninja`, a C/C++ compiler, `python3`, `uv`, `ssh`, `rsync`

2. Install remote tools (on the RISC-V board)
- `gcc-14`, `gfortran-14`, `make`, `git`, `ssh`

3. Create the local Python environment
```bash
make install
```

4. Run the pipeline
```bash
bash riscv_compile.sh
```
That’s it. The script will:
- Build `mlir-translate` locally if missing.
- Generate kernels, compile, benchmark, and pull results back.

Example with explicit connection and tuning options:
```bash
./convert_linalg_mlir_to_c.sh tests/riscv_tests/mnist_rair_memref.mlir --wd 16 --type fp
./convert_linalg_mlir_to_c.sh tests/riscv_tests/resnet_rair_memref.mlir --wd 16 --type fp
```

## Outputs

Results are saved to `generated/` :
- `*.cpp` simulator operator
- `*.mlir` emitC



