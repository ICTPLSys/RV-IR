<div align="center">
  <img src="assets/logo.png" width="120"/>
  <h1>RV-IR: RISC-V Oriented IR Extensions<br>Based on MLIR</h1>
  <p>
    <a href="https://mlir.llvm.org/">
      <img src="https://img.shields.io/badge/MLIR-LLVM-1E90FF" alt="MLIR" height="20">
    </a>
    <a href="https://github.com/llvm/torch-mlir">
      <img src="https://img.shields.io/badge/Upstream-torch--mlir-1E90FF" alt="torch-mlir" height="20">
    </a>
    <a href="https://riscv.org/">
      <img src="https://img.shields.io/badge/ISA-RISC--V-1E90FF" alt="RISC-V" height="20">
    </a>
  </p>
  <p>
    <a href="#risc-v-dialect-design"><b>Design</b></a> |
    <a href="#development-guide"><b>Building</b></a>
  </p>
</div>

# RV-IR

> RV-IR is a research-oriented fork of upstream torch-mlir, focusing on
>RISC-V backends and CIM-related IR extensions.
>Internally, RV-IR reuses the torch-mlir https://github.com/llvm/torch-mlir build system, tools, and
>Python bindings.
## RISC-V Dialect Design

RV-IR introduces a custom **RISCV Dialect** to model RISC-V and CIM-oriented execution semantics at the MLIR level.

**Dialect design and operation specifications:**

- https://i02plfgfarl.feishu.cn/wiki/UzuMwmKE7iKmgtkOIp1c4JhonEh

This document describes:

- Core RISCV dialect operations and type system
- Asynchronous and memory-related operations
- Design rationale for mapping high-level ops to RISC-V backends
- Extension points for CIM-specific instructions

## Hardware Architecture

## Hardware Operator Library

RV-IR targets a specific class of RISC-V–based CIM hardware platforms. The corresponding hardware operator library defines the semantic contract between IR-level operations and hardware execution.

**Underlying hardware operator specification:**

<!-- - https://i02plfgfarl.feishu.cn/wiki/ONOqwaxLCidwqPkzfekcnVf0nnb -->

This document covers:

- Supported hardware operators and instruction forms
- Data layout and memory interaction semantics
- Constraints imposed by the underlying CIM architecture
- Guidance for extending the operator set

# Development Guide
> This section documents the minimal build steps required for this project.
> For a complete and up-to-date guide, please refer to:
>
> - https://github.com/llvm/torch-mlir/blob/main/README.md
> - https://github.com/llvm/torch-mlir/blob/main/docs/development.md
## Setting Up Environment
### Clone the Repository
```shell
   git clone https://github.com/ICTPLSys/RV-IR && cd RV-IR
   ```
### Set up the Python environment

1. if you want to switch over multiple versions of Python using conda, you can create a conda environment with Python 3.11.
```shell
conda create -n torch-mlir python=3.11
conda activate torch-mlir
```
2. Install the latest requirements.

    ```shell
    python -m pip install -r requirements.txt -r torchvision-requirements.txt
    ```
## Building
### With CMake
#### Configure for Building
1. **If you haven't already**, [activate the Python environment](#set-up-the-python-environment)
1. Choose command relevant to LLVM setup:
1. Choose command relevant to LLVM setup:
    1. **If you want the more straightforward option**, run the "in-tree" setup:

        ```shell
        cmake -GNinja -Bbuild \
          `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DPython3_FIND_VIRTUALENV=ONLY \
          -DPython_FIND_VIRTUALENV=ONLY \
          -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
          -DLLVM_TARGETS_TO_BUILD=host \
          `# For building LLVM "in-tree"` \
          externals/llvm-project/llvm \
          -DLLVM_ENABLE_PROJECTS=mlir \
          -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
          -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD"
        ```

        - NOTE: uses external/llvm-project/llvm as the main build, so LLVM will be built in addition to torch-mlir and its sub-projects.
    2. **If you want to use a separate build of LLVM from another directory**, run the "out-of-tree" setup:

        ```shell
        cmake -GNinja -Bbuild \
          `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DPython3_FIND_VIRTUALENV=ONLY \
          -DPython_FIND_VIRTUALENV=ONLY \
          -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
          -DLLVM_TARGETS_TO_BUILD=host \
          `# For building LLVM "out-of-tree"` \
          -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir/" \
          -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm/"
        ```

        - Be sure to have built LLVM with `-DLLVM_ENABLE_PROJECTS=mlir`.
        - Be aware that the installed version of LLVM needs in general to match the committed version in `externals/llvm-project`. Using a different version may or may not work.

    - [About MLIR debugging](https://mlir.llvm.org/getting_started/Debugging/)
#### Configure for Building
1. [Configure the build](#configure-for-building) if you haven't already done so.
1. **If you want to...**
    - **...build _everything_** (including LLVM if configured as "in-tree"), run:

      ```shell
      cmake --build build
      ```

    - **...build _just_ torch-mlir** (not all of LLVM), run:

      ```shell
      cmake --build build --target tools/torch-mlir/all
      ```

    - **...run unit tests**, run:

      ```shell
      cmake --build build --target check-torch-mlir
      ```

    - **...run single test**, run:

      ```shell
      ./bin/llvm-lit -v ../test/Conversion/RISCVToLLVM/add.mlir
      ```
    - **...run the MLIR-to-LLVM lowering pipeline test**, run:

      ```shell
      ./bin/torch-mlir-opt --convert-linalg-to-riscv --convert-riscv-to-affine --convert-riscv-to-llvm\ ../test/Conversion/RISCVToLLVM/add.mlir >> add.ll
      ```
    - **...Execute the code with LLVM interpreter**, run:

      ```shell
      lli /path/to/add.ll
      ```


