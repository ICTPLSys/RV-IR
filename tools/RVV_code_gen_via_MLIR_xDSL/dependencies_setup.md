# MLIR Microkernel Dependencies Compilation Guide

This document outlines the steps to compile the required dependencies for the MLIR microkernel project. The compilation process is split across two architectures: your local x86 host machine, and the remote RISC-V board.

## 1. Local x86 Host (Compiling LLVM / `mlir-translate`)

The core MLIR compiler infrastructure must be built locally. The script requires the `mlir-translate` tool to convert xDSL/Python descriptions into native C/C++ microkernels before they are deployed to the board.

**Requirements:**
- A local x86 machine (e.g., your laptop)
- CMake and a compiler (like GCC or Clang)
- Git

**Compilation Steps:**

Run the following commands locally to fetch and build LLVM:

```bash
# Clone the LLVM repository
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Checkout the specific branch required
git checkout llvmorg-22-init

# Create a build directory
mkdir build && cd build

# Configure the build using CMake and Ninja
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_USE_SPLIT_DWARF=ON \
  -DCMAKE_INSTALL_PREFIX=./install

# Build and verify the MLIR component
cmake --build . --target check-mlir -j6
```

Once built, the `mlir-translate` executable will be located at `llvm-project/build/bin/mlir-translate`.

---

## 2. Remote RISC-V Board

The target execution environment requires mathematical benchmark libraries to be natively compiled on the board itself. These act as the baselines and fallback kernels for evaluating the custom MLIR microkernels.

Both of these steps happen **directly on the RISC-V machine** (e.g., `jbpi2`).

### 2.1 Compiling OpenBLAS

OpenBLAS is used as a highly optimized reference matrix multiplication library. It needs to be explicitly configured to take advantage of the RISC-V Vector Extensions (RVV).

**Compilation Steps:**

Run these commands on the board. Make sure to specify your desired Vector Length (`ZVL256B` for 256-bit or `ZVL128B` for 128-bit).

```bash
# Clone the openblas repository on tag v0.3.31
git clone -b v0.3.31 https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS

# Build OpenBLAS targeting the RISC-V RVV architecture using GCC 14
# Replace ZVL256B with ZVL128B if compiling for a 128-bit VLEN
make TARGET=RISCV64_ZVL256B \
     CC=gcc-14 FC=gfortran-14 \
     BINARY=64 \
     USE_THREAD=1 \
     NUM_THREADS=8

# Install OpenBLAS into a local directory
make PREFIX=$HOME/.local/openblas_install install
```

### 2.2 Compiling BLIS

BLIS is another BLAS-like instantiation framework used structurally by the benchmarks.

**Compilation Steps:**

Run these commands on the board:

```bash
# Clone the BLIS framework repository
git clone --depth 1 -b 2.0 https://github.com/flame/blis.git
cd blis

# Configure BLIS to automatically detect the board's architecture
./configure -p "$HOME/opt/blis" auto

# Build using all available CPU cores securely
make -j$(nproc)

# Install
make install
```
