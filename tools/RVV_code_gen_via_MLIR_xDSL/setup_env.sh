#!/bin/bash
set -e

# ============================================================
# setup_env.sh
# Check and compile dependencies
# Local: MLIR/LLVM (--local-mlir-only)
# Remote RISC-V: BLIS, OpenBLAS (--remote-only)
# ============================================================

REMOTE_WORKSPACE="xdsl_rvv_microkernel"
VLEN_BITS="256"
OPENBLAS_DIR_ENV=""
BLIS_DIR_ENV=""
MLIR_BUILD_DIR="../llvm-project"

LOCAL_MLIR_ONLY=false
REMOTE_ONLY=false

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
require_cmd() {
    local cmd_name="$1"
    local label="$2"
    local cmd_path
    if ! cmd_path="$(command -v "${cmd_name}" 2>/dev/null)"; then
        echo "❌ Missing dependency: ${label} (${cmd_name})"
        echo "   Please install it and re-run this script."
        exit 1
    fi
    echo "✅ ${label} found at ${cmd_path}"
}

# ------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local-mlir-only)  LOCAL_MLIR_ONLY=true ;;
        --remote-only)      REMOTE_ONLY=true ;;
        --mlir-build-dir)   MLIR_BUILD_DIR="$2"; shift ;;
        --vlen)             VLEN_BITS="$2"; shift ;;
        --riscv-workspace)  RISCV_WORKSPACE="$2"; shift ;;
        --openblas-dir)     OPENBLAS_DIR_ENV="$2"; shift ;;
        --blis-dir)         BLIS_DIR_ENV="$2"; shift ;;
                 *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ------------------------------------------------------------
# LOCAL PHASE: MLIR/LLVM Build
# ------------------------------------------------------------
if [ "$LOCAL_MLIR_ONLY" = true ]; then
    echo "============================================================"
    echo "1. Checking Local Dependencies (MLIR/LLVM)"
    echo "============================================================"
    if [ ! -f "${MLIR_BUILD_DIR}/build/bin/mlir-translate" ]; then
        echo "⚠️ mlir-translate not found at ${MLIR_BUILD_DIR}/build/bin/mlir-translate. Building MLIR/LLVM locally..."
        
        if [ ! -d "${MLIR_BUILD_DIR}" ]; then
            echo "   Cloning LLVM repository to ${MLIR_BUILD_DIR}..."
            git clone https://github.com/llvm/llvm-project.git "${MLIR_BUILD_DIR}"
        fi
        
        cd "${MLIR_BUILD_DIR}" || exit 1
        echo "   Checking out branch llvmorg-22-init..."
        git checkout llvmorg-22-init
        
        mkdir -p build && cd build || exit 1
        
        echo "   Configuring build with CMake and Ninja..."
        cmake -G Ninja ../llvm \
          -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
          -DLLVM_BUILD_EXAMPLES=ON \
          -DLLVM_TARGETS_TO_BUILD="X86" \
          -DCMAKE_BUILD_TYPE=Release \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DLLVM_USE_SPLIT_DWARF=ON \
          -DCMAKE_INSTALL_PREFIX=./install
          
        echo "   Building MLIR components..."
        cmake --build . --target mlir-translate -j6
        
        MLIR_TRANSLATE_PATH=$(realpath ./bin/mlir-translate)
        cd - > /dev/null || exit 1
        echo "✅ mlir-translate built locally at ${MLIR_TRANSLATE_PATH}"
    else
        echo "✅ mlir-translate already built at ${MLIR_BUILD_DIR}/build/bin/mlir-translate"
    fi
    exit 0
fi

# ------------------------------------------------------------
# REMOTE PHASE: OpenBLAS / BLIS
# ------------------------------------------------------------
if [ "$REMOTE_ONLY" = true ] || [ "$LOCAL_MLIR_ONLY" = false ]; then
    # Set defaults if not provided via arguments
    if [ -z "$OPENBLAS_DIR_ENV" ]; then
        OPENBLAS_DIR_ENV="$HOME/${RISCV_WORKSPACE}/openblas_install"
    fi

    if [ -z "$BLIS_DIR_ENV" ]; then
        BLIS_DIR_ENV="$HOME/${RISCV_WORKSPACE}/blis_install"
    fi

    echo "Setting up dependencies on board with VLEN=${VLEN_BITS}"

    echo ""
    echo "============================================================"
    echo "0. Checking Remote Build Dependencies"
    echo "============================================================"
    require_cmd "gcc-14" "GCC 14"
    require_cmd "gfortran-14" "GFortran 14"
    require_cmd "ssh" "SSH (remote access)"

    mkdir -p "$HOME/${RISCV_WORKSPACE}"

# ------------------------------------------------------------
# 1. OpenBLAS (Local to board)
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "1. Setting up OpenBLAS"
echo "============================================================"
OPENBLAS_REPO="https://github.com/OpenMathLib/OpenBLAS.git"
OPENBLAS_BRANCH="v0.3.31"

if [ -d "${OPENBLAS_DIR_ENV}" ] && { [ -f "${OPENBLAS_DIR_ENV}/lib/libopenblas.a" ] || [ -f "${OPENBLAS_DIR_ENV}/lib64/libopenblas.a" ]; }; then
    echo "✅ OpenBLAS already installed at ${OPENBLAS_DIR_ENV}"
else
    echo "⚠️ OpenBLAS not found. Cloning and building tag ${OPENBLAS_BRANCH}..."
    cd "$HOME/${RISCV_WORKSPACE}"
    if [ ! -d "OpenBLAS" ]; then
        git clone -b ${OPENBLAS_BRANCH} ${OPENBLAS_REPO}
    fi
    cd OpenBLAS
    # If the previous build failed, clean might be necessary
    make clean || true
    make TARGET=RISCV64_ZVL${VLEN_BITS}B CC=gcc-14 FC=gfortran-14 BINARY=64 USE_THREAD=1 NUM_THREADS=4 MAKE_NB_JOBS=1
    make PREFIX=${OPENBLAS_DIR_ENV} install
    echo "✅ OpenBLAS installation complete."
fi

# ------------------------------------------------------------
# 2. BLIS (Local to board)
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "2. Setting up BLIS"
echo "============================================================"
BLIS_REPO="https://github.com/flame/blis.git"
BLIS_TAG="2.0"

if [ -d "${BLIS_DIR_ENV}" ] && [ -f "${BLIS_DIR_ENV}/lib/libblis.a" ]; then
    echo "✅ BLIS already installed at ${BLIS_DIR_ENV}"
else
    echo "⚠️ BLIS not found. Cloning and building tag ${BLIS_TAG}..."
    cd "$HOME/${RISCV_WORKSPACE}"
    if [ ! -d "blis" ]; then
        git clone --depth 1 -b ${BLIS_TAG} ${BLIS_REPO}
    fi
    cd blis
    ./configure -p "${BLIS_DIR_ENV}" auto
    make -j$(nproc)
    make install
    echo "✅ BLIS installation complete."
fi

echo ""
echo "============================================================"
echo " Dependencies setup complete."
echo "============================================================"

fi
