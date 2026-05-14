# xDSL GEMM Microkernel Development & Benchmarking Guide

This repository contains the tools and scripts to generate RISC-V Vector (RVV) microkernels using xDSL, cross-compile them on a remote RISC-V board, and automatically benchmark them against BLIS and OpenBLAS.

The workflow is fully automated through two main scripts:
1. **`setup_env.sh`**: Installs required dependencies (OpenBLAS, BLIS) on the remote board.
2. **`compile.sh`**: Generates local kernels, pushes them to the board, compiles everything, and runs a comprehensive performance sweep.

---

## 1. Prerequisites

### Local Machine (Development)
- **Python**: 3.8+ with the `xdsl` package installed.
- **LLVM/MLIR tools**: Specifically `mlir-translate` built from your custom xDSL/MLIR tree.
- **System Tools**: `sshpass` (required for default jump-host mode) and `rsync`.

### Remote Machine (RISC-V Board - `jbpi2`)
- **Compiler**: GCC/G++ 14 or higher (required for `riscv64_zvl` and RVV auto-vectorization support).
- **Core Tools**: GNU Make and Git.

> Note: `setup_env.sh` will automatically fetch and compile the following library dependencies on the remote board if they are missing:
> - **OpenBLAS**: Branch `0.3.31`
> - **BLIS**: Tag `2.0`

---

## 2. Connection Methods

The scripts support two ways of connecting to the remote RISC-V board (`jbpi2`):

1. **Jump-Host Mode (Default):**
   - Connects from your local machine → to a frontend `fej` server (using `sshpass` with password `r`) → to the `jbpi2` board.
   - Use this if your local machine cannot reach `jbpi2` directly.
2. **Direct Mode (`--direct`):**
   - Connects directly from your local machine → to the `jbpi2` board.
   - Requires you to be on the correct network (e.g., VPN or physically at the lab) and have SSH keys set up.

---

## 3. Environment Setup (`setup_env.sh`)

Before compiling kernels, you must set up the environment on the remote board. This script checks what is missing and automatically installs it.

### What it does:
1. **Local Check:** Verifies `mlir-translate` is available locally.
2. **OpenBLAS (Remote):** Checks for OpenBLAS `~/.local/openblas_install`. If missing, clones branch `0.3.31` and builds it.
3. **BLIS (Remote):** Checks for BLIS `~/opt/blis`. If missing, clones tag `2.0` and builds it.

### Usage:

```bash
# Default (Jump-host, VLEN=256)
./setup_env.sh

# Specify Vector Length (128 or 256)
./setup_env.sh --vlen 128

# Direct SSH connection method
./setup_env.sh --direct --vlen 128
```

---

## 4. Kernel Generation & Benchmarking (`compile.sh`)

The `compile.sh` script automates the entire loop from generating the C code via xDSL to seeing the peak GFLOPS on the board.

### What it does:
1. **Local Generation:** Runs Python scripts to generate MLIR and translates it to C++ (e.g., `RVV_128_XDSL`) for the requested GEMM dimensions.
2. **Remote Cleanup:** Deletes any stale kernels on the remote board.
3. **Synchronization:** `rsync`s the newly generated C++ kernels and `cnn_models` datasets to the board.
4. **Benchmark Sweep (XDSL):**
   - Injects your microkernels into the `RVV_ukernels_benchmark` framework.
   - Sweeps every sub-tile configuration (e.g. if you request `4,4`, it compiles and runs combinations from `1x1` to `4x4`).
   - Benchmarks against `square_small`, `bert_large`, `gpt2_large`, and `square`.
   - Prints a cleanly formatted table showing peak GFLOPS for each configuration, and extracts the `BEST` performing kernel.
5. **OpenBLAS Baseline:** Runs the same datasets through OpenBLAS for a performance baseline.

### Usage:

```bash
# Run with defaults (Jump-host, VLEN=256, FAMILIES="4,4")
./compile.sh

# Direct connection, VLEN=128
./compile.sh --direct --vlen 128

# Run without the OpenBLAS reference comparisons (faster)
./compile.sh --no-ref

# Specify custom kernel dimensions to generate and sweep
./compile.sh --families "2,2 4,4"
```

### Script Flags Reference:

| Flag | Description | Default |
|------|-------------|---------|
| `--families "X,Y ..."` | Comma-separated max dimensions to generate. | `"4,4"` |
| `--vlen <bits>` | Vector length config (`128` or `256`). | `256` |
| `--mlir-translate <path>` | Path to the `mlir-translate` binary. | (searches PATH) |
| `--no-ref` | Skips OpenBLAS reference benchmark steps. | `false` |
| `--direct` | Uses direct SSH connection instead of jump-host. | `false` |
| `--kc-profile <mode>` | KC generation profile (`full`, `odd`, etc). | `odd` |
| `--verbose` | Enables verbose python output. | `false` |

---

## 5. Understanding the Expected Output

When `compile.sh` finishes compilation, it actively runs the sweep script on the remote board. You will see an output table for each dataset tested (e.g., `square_small`, `bert_large`, etc.).

**Example Output:**
```text
============================================================
 XDSL Kernel Sweep: MR=1..4  NR=1..4
 Dataset: cnn_models/square_small.dat 
 Kernel file: kernels_RVV_4x4_fp32 (merged, contains all sub-kernels)
============================================================
 MR   NR    GFLOPS        Status
------------------------------------------------------------
 1    1     1.24e+00      OK
 1    2     2.31e+00      OK
 ...
 4    4     5.85e+00      OK
------------------------------------------------------------
  BEST FOR square_small: MR=4  NR=4  GFLOPS=5.85e+00
============================================================
```

### What to Expect:
- **Comprehensive Sweeping:** The script iterates through every `(MR, NR)` tile size from `1x1` up to your requested maximum (e.g., `4x4`), linking each dynamically against your **merged** `kernels_RVV_NxN_fp32` file.
- **Status Column (`OK` vs `compile failed`):** 
  - `OK` indicates the microkernel was successfully dispatched and benchmarked.
  - `compile failed` means that specific `(mr, nr)` sub-kernel size was not successfully generated by xDSL (or was omitted). The sweep will gracefully skip it and continue testing the rest.
- **GFLOPS:** The peak floating-point operations per second achieved by that specific microkernel configuration on the given dataset.
- **The BEST Summary:** At the bottom of each table, the script automatically parses the CSV results to extract and highlight the absolute best-performing `MR x NR` configuration for that specific workload.
