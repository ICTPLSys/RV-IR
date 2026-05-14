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

4. Edit `compile.sh` (top of file) with your board info
- `RISCV_USER`
- `RISCV_REMOTE_IP`
- `RISCV_SSH_PORT` (optional)
- Optional tuning: `FAMILIES`, `VLEN_BITS`, `KC_PROFILE`, `DATASETS`

5. Run the pipeline
```bash
bash compile.sh
```

Example with explicit connection and tuning options:
```bash
./compile.sh --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 --families "2,2" --vlen 256
```

That’s it. The script will:
- Build `mlir-translate` locally if missing.
- Build OpenBLAS and BLIS on the board if missing.
- Generate kernels, compile, benchmark, and pull results back.

## Outputs

Results are saved to `tests/output/` on your local machine:
- `*.txt` logs
- `*.csv` sweep tables
- `*.png` / `*.pdf` plots

## Common Variations

Use a different MLIR build directory:
```bash
bash compile.sh --mlir-build-dir ../llvm-project
```

Override families or VLEN:
```bash
bash compile.sh --families "4,4" --vlen 256
```

## One-Off Setup Only

Local MLIR/LLVM build:
```bash
bash setup_env.sh --local-mlir-only --mlir-build-dir ../llvm-project
```

Remote OpenBLAS/BLIS build:
```bash
bash setup_env.sh --remote-only --vlen 256 --riscv-workspace xdsl_rvv_microkernel
```
