#!/bin/bash
set -e

# ============================================================
# compile.sh — Unified compilation and deployment script
# Usage:
#   ./compile.sh [OPTIONS]
#
# RISC-V connection options:
#   --riscv-user <user>   SSH user on the RISC-V board  (default: jlei)
#   --riscv-ip   <host>   Hostname or IP of the board   (default: jbpi2)
#   --riscv-port <port>   SSH port on the board         (default: 22)
#
# The script will return benchmarking results and store them in the tests/output directory
# ============================================================


# ============================================================
# Configuration Start
# ============================================================
# To set the kernel families, use the format "mr,nr"
# For example, "3,3" means mr=3 and nr=3, 
# the script will generate kernels covering the range of mr from 1 to 3 and nr from 1 to 3
# it will also search the best kernel for each matrix size in the dataset
FAMILIES="32,32" 
FAMILIES="4,4" 
# FAMILIES="4,4" 
VLEN_BITS="256"
# This KC parameter only be used for kc loop per kernel benchmarking, 
# It does not affect the BLIS test results
KC_PROFILE="256"



# The RISCV board expected to be connected to the same x86 machine you are using
# The authentication is done through ssh keys
# user name of the RISCV machine
RISCV_USER="RISCV_USER"
# this is the IP or hostname of the RISCV machine
RISCV_REMOTE_IP="RISCV_REMOTE_IP" 
RISCV_SSH_PORT="22"
# This folder will be created at the home directory of the RISCV machine
# This folder will contain all the generated project structure for benchmarking 
RISCV_WORKSPACE="xdsl_rvv_microkernel"

# At RISCV machine, Remote software paths
# RISCV_OPENBLAS_DIR_ENV="\$HOME/.local/openblas_zvl256b"
# RISCV_BLIS_DIR_ENV="\$HOME/opt/blis"
RISCV_OPENBLAS_DIR_ENV="\$HOME/${RISCV_WORKSPACE}/openblas_install"
RISCV_BLIS_DIR_ENV="\$HOME/${RISCV_WORKSPACE}/blis_install"

# ============================================================
# Datasets to evaluate
# Edit this list — both XDSL sweep and OpenBLAS use it.
# ============================================================
DATASETS=(
    # "cnn_models/trail.dat"
    "cnn_models/tiny.dat"
    # "cnn_models/gnn.dat"
    # "cnn_models/bert_large.dat"
    # "cnn_models/gpt2_large.dat"
    # "cnn_models/square.dat"
)
# ============================================================
# Configuration End 
# ============================================================





VERBOSE_FLAG=""
NO_REF_FLAG="--no-ref"
USE_DIRECT=true   
SETUP_ENV=false
MLIR_BUILD_DIR="../llvm-project"

# Local paths 
LOCAL_DIR="tests/api_gen_cpp"
LOCAL_REF_DIR="tests/rvv_ref"

# ============================================================
# Argument parsing
# ============================================================

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --families)        FAMILIES="$2";              shift ;;
        --vlen)            VLEN_BITS="$2";             shift ;;
        --mlir-translate)  MLIR_TRANSLATE_PATH="$2";   shift ;;
        --verbose)         VERBOSE_FLAG="--verbose"         ;;
        --kc-profile)      KC_PROFILE="$2";            shift ;;
        --no-ref)          NO_REF_FLAG="--no-ref"           ;;
        --direct)          USE_DIRECT=true                  ;;   
        --jump)            USE_DIRECT=false                 ;;  
        --openblas-dir)    RISCV_OPENBLAS_DIR_ENV="$2";      shift ;;
        --blis-dir)        RISCV_BLIS_DIR_ENV="$2";          shift ;;
        --riscv-workspace) RISCV_WORKSPACE="$2";       shift ;;
        --riscv-user)      RISCV_USER="$2";            shift ;;
        --riscv-ip)        RISCV_REMOTE_IP="$2";       shift ;;
        --riscv-port)      RISCV_SSH_PORT="$2";        shift ;;
        --setup-env)       SETUP_ENV=true                   ;;
        --mlir-build-dir)  MLIR_BUILD_DIR="$2";        shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# If MLIR_TRANSLATE_PATH is not set by user arguments, default to the MLIR_BUILD_DIR
if [ -z "${MLIR_TRANSLATE_PATH+x}" ]; then
    MLIR_TRANSLATE_PATH="${MLIR_BUILD_DIR}/build/bin/mlir-translate"
fi

VLEN=$((VLEN_BITS / 32))
REF_BASE="kernels_RVV_${VLEN_BITS}_fp32_ldx"




# ============================================================
# SSH / rsync helpers
# ============================================================

SEP="------------------------------------------------------------"

if [ "$USE_DIRECT" = true ]; then
    echo ""; echo "  Mode: Direct "; echo ""
    ssh_board() { ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no ${RISCV_USER}@${RISCV_REMOTE_IP} "$@"; }
    rsync_board() { rsync -a -e "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no" "$@"; }
    ssh_host()  { ssh_board "$@"; }
    RSYNC_TARGET="${RISCV_USER}@${RISCV_REMOTE_IP}"
else
    echo ""; echo "  Mode: Jump-host (x86 → fej → jbpi2)"; echo ""
    ssh_board() {
        local escaped_args
        escaped_args=$(printf '%q ' "$@")
        sshpass -p "${REMOTE_PASS}" ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
            -t ${RISCV_USER}@${REMOTE_HOST} \
            "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no -t ${RISCV_REMOTE_IP} ${escaped_args}"
    }
    rsync_board() {
        sshpass -p "${REMOTE_PASS}" rsync -a \
            -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" "$@"
    }
    ssh_host() {
        sshpass -p "${REMOTE_PASS}" ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
            ${RISCV_USER}@${REMOTE_HOST} "$@"
    }
    RSYNC_TARGET="${RISCV_USER}@${REMOTE_HOST}"
fi

source .venv/bin/activate

# ============================================================
# Dependency Check & Auto-Setup
# ============================================================
echo ""; echo "$SEP"; echo " Dependency Check"; echo "$SEP"
for cmd in python3 ssh rsync; do
    if command -v $cmd >/dev/null 2>&1; then
        echo "  [OK] $cmd: $(command -v $cmd)"
    else
        echo "  [FAIL] $cmd not found in PATH"
        exit 1
    fi
done

if [ "$USE_DIRECT" = false ]; then
    if command -v sshpass >/dev/null 2>&1; then
        echo "  [OK] sshpass: $(command -v sshpass)"
    else
        echo "  [FAIL] sshpass not found in PATH (required for jump-host mode)"
        exit 1
    fi
fi

# Auto-check all dependencies before deciding to setup
if [ -f "$MLIR_TRANSLATE_PATH" ]; then
    echo "  [OK] mlir-translate: $(realpath "$MLIR_TRANSLATE_PATH")"
else
    echo "  [INFO] mlir-translate not found at $MLIR_TRANSLATE_PATH! Auto-enabling setup..."
    SETUP_ENV=true
fi

if ssh_host "[ -f \"${RISCV_OPENBLAS_DIR_ENV}/lib/libopenblas.a\" ] || [ -f \"${RISCV_OPENBLAS_DIR_ENV}/lib64/libopenblas.a\" ]" >/dev/null 2>&1; then
    echo "  [OK] OpenBLAS (Remote): ${RISCV_OPENBLAS_DIR_ENV}"
else
    echo "  [INFO] OpenBLAS not found on remote at ${RISCV_OPENBLAS_DIR_ENV}! Auto-enabling setup..."
    SETUP_ENV=true
fi

if ssh_host "[ -f \"${RISCV_BLIS_DIR_ENV}/lib/libblis.a\" ]" >/dev/null 2>&1; then
    echo "  [OK] BLIS (Remote): ${RISCV_BLIS_DIR_ENV}"
else
    echo "  [INFO] BLIS not found on remote at ${RISCV_BLIS_DIR_ENV}! Auto-enabling setup..."
    SETUP_ENV=true
fi

if [ "$SETUP_ENV" = true ]; then
    echo ""
    echo "$SEP"
    echo " Running MLIR/LLVM Local Setup"
    echo "$SEP"
    
    bash setup_env.sh --local-mlir-only --mlir-build-dir "$MLIR_BUILD_DIR"
    
    if [ ! -f "$MLIR_TRANSLATE_PATH" ] && [ -f "${MLIR_BUILD_DIR}/build/bin/mlir-translate" ]; then
        MLIR_TRANSLATE_PATH=$(realpath "${MLIR_BUILD_DIR}/build/bin/mlir-translate")
    fi

    echo ""
    echo "$SEP"
    echo " Syncing and Running Remote Environment Setup"
    echo "$SEP"
    
    ssh_host "mkdir -p ${RISCV_WORKSPACE}"
    rsync_board setup_env.sh ${RSYNC_TARGET}:${RISCV_WORKSPACE}/setup_env.sh
    ssh_board "bash ${RISCV_WORKSPACE}/setup_env.sh --remote-only --vlen ${VLEN_BITS} --riscv-workspace ${RISCV_WORKSPACE} --openblas-dir ${RISCV_OPENBLAS_DIR_ENV} --blis-dir ${RISCV_BLIS_DIR_ENV}"

    echo ""
    echo "$SEP"
    echo " Verifying Dependencies After Setup"
    echo "$SEP"
    if [ ! -f "$MLIR_TRANSLATE_PATH" ]; then
        echo "  [FAIL] mlir-translate not found at $MLIR_TRANSLATE_PATH"
        exit 1
    fi
    if ! ssh_host "[ -f \"${RISCV_OPENBLAS_DIR_ENV}/lib/libopenblas.a\" ] || [ -f \"${RISCV_OPENBLAS_DIR_ENV}/lib64/libopenblas.a\" ]" >/dev/null 2>&1; then
        echo "  [FAIL] OpenBLAS not found on remote!"
        exit 1
    fi
    if ! ssh_host "[ -f \"${RISCV_BLIS_DIR_ENV}/lib/libblis.a\" ]" >/dev/null 2>&1; then
        echo "  [FAIL] BLIS not found on remote!"
        exit 1
    fi
fi

# ============================================================
# Check Local and Remote RAM
# ============================================================
export USE_DIRECT RISCV_USER RISCV_REMOTE_IP RISCV_SSH_PORT REMOTE_PASS REMOTE_PORT REMOTE_HOST
bash ./check_ram.sh

# ============================================================
# Step 1 — Generate code locally
# ============================================================
echo ""; echo "$SEP"; echo " [1/8] Generate Code Locally"; echo "$SEP"

rm -rf tests/test_blis/*
echo " Families: $FAMILIES | VLEN: $VLEN | KC: $KC_PROFILE"
python3 tests/test_codeGeneration.py \
    --families "$FAMILIES" --vlen "$VLEN" \
    --mlir-translate "$MLIR_TRANSLATE_PATH" \
    $VERBOSE_FLAG --kc-profile "$KC_PROFILE" $NO_REF_FLAG


# ============================================================
# Step 2 — Create remote directories
# ============================================================
echo ""; echo "$SEP"; echo " [2/8] Create Remote Directories"; echo "$SEP"
ssh_host "mkdir -p ${RISCV_WORKSPACE}/kernels ${RISCV_WORKSPACE}/merged"


# ============================================================
# Step 3 — Reference kernel check / compile
# ============================================================
echo ""; echo "$SEP"; echo " [3/8] Reference Kernel Check"; echo "$SEP"
if [ -n "$NO_REF_FLAG" ]; then
    echo " Skipped (--no-ref)"
else
    files_to_sync=""
    for ext in cpp h md; do
        [ -f "${LOCAL_REF_DIR}/${REF_BASE}.${ext}" ] && files_to_sync+=" ${LOCAL_REF_DIR}/${REF_BASE}.${ext}"
    done
    [ -n "$files_to_sync" ] && rsync_board $files_to_sync ${RSYNC_TARGET}:${RISCV_WORKSPACE}/ || echo " [warn] No reference files found"

    if ssh_host "test -f ${RISCV_WORKSPACE}/${REF_BASE}.o"; then
        echo " ${REF_BASE}.o already on remote — skipping compile"
    else
        echo " Compiling ${REF_BASE}.cpp on board (large file, please wait)..."
        ssh_board "cd ${RISCV_WORKSPACE} && riscv64-linux-gnu-g++ -march=rv64gcv -mabi=lp64d -O3 -mcmodel=medany -lm -c ${REF_BASE}.cpp -o ${REF_BASE}.o"
    fi
fi


# ============================================================
# Step 4 — Clean remote output
# ============================================================
echo ""; echo "$SEP"; echo " [4/8] Clean Remote Output Directory"; echo "$SEP"
ssh_host "find ${RISCV_WORKSPACE}/kernels -type f -delete 2>/dev/null || true; \
     find ${RISCV_WORKSPACE}/merged -type f -delete 2>/dev/null || true; \
     find ${RISCV_WORKSPACE} -maxdepth 1 -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.c' \) ! -name '${REF_BASE}.*' -delete 2>/dev/null || true; \
     rm -f ${RISCV_WORKSPACE}/test_gemm ${RISCV_WORKSPACE}/test_xdsl_api_gemm_kernels.o ${RISCV_WORKSPACE}/Makefile"


# ============================================================
# Step 5 — Sync files to remote
# ============================================================
echo ""; echo "$SEP"; echo " [5/8] Sync Files to Remote"; echo "$SEP"
rsync_board ${LOCAL_DIR}/ ${RSYNC_TARGET}:${RISCV_WORKSPACE}/
rsync_board tests/test_openblas/ ${RSYNC_TARGET}:${RISCV_WORKSPACE}/test_openblas/


# ============================================================
# Step 6 — Compile and run unit test on board
# ============================================================
echo ""; echo "$SEP"; echo " [6/8] Compile & Run Unit Test (test_gemm)"; echo "$SEP"
ssh_board "cd ${RISCV_WORKSPACE} && make clean && make -j8"

MAX_MR=0; MAX_NR=0
for fam in $FAMILIES; do
    mr=$(echo $fam | cut -d',' -f1); nr=$(echo $fam | cut -d',' -f2)
    [ "$mr" -gt "$MAX_MR" ] && MAX_MR=$mr
    [ "$nr" -gt "$MAX_NR" ] && MAX_NR=$nr
done
MAX_DIM_STR="${MAX_MR}x${MAX_NR}"

mkdir -p tests/output
OUT_FILE="tests/output/uK_${MAX_DIM_STR}_${VLEN_BITS}bits_${KC_PROFILE}.txt"
echo " Saving output to ${OUT_FILE}..."
ssh_board "cd ${RISCV_WORKSPACE} && ./test_gemm" > "${OUT_FILE}"
python "tests/output/plot_perf.py" "${OUT_FILE}" \
    --families "${MAX_DIM_STR}" --vlen "${VLEN_BITS}" --kc "${KC_PROFILE}"
if [ -z "$NO_REF_FLAG" ]; then
    python "tests/output/plot_ref_comparison.py" "${OUT_FILE}"
fi

# Extract best kernel (highest GFLOP/s in the b0 section)
BEST_UK=$(grep '_b0' "${OUT_FILE}" | grep 'GFLOP/s' | awk '
    {
        for (i=1; i<=NF; i++) {
            if ($i == "GFLOP/s") gval = $(i-1)
        }
        if (gval+0 > best+0) { best = gval; name = $1 }
    }
    END { sub(/_b0$/, "", name); print name }'
)
echo " Best kc-loop kernel: ${BEST_UK}  (from ${OUT_FILE})"


# ============================================================
# Step 7 — XDSL kernel sweep
# ============================================================
echo ""; echo "$SEP"; echo " [7/8] XDSL Kernel Sweep (MR=1..${MAX_MR}, NR=1..${MAX_NR})"; echo "$SEP"

# Clone benchmark repo if needed
ssh_board "cd ${RISCV_WORKSPACE} && if [ ! -d 'RVV_ukernels_benchmark' ]; then git clone https://github.com/adcastel/RVV_ukernels_benchmark.git; else echo 'RVV_ukernels_benchmark already present.'; fi"
ssh_board "rm -rf ${RISCV_WORKSPACE}/RVV_ukernels_benchmark/kernels/RVV_${VLEN_BITS}_XDSL"
ssh_host "mkdir -p ${RISCV_WORKSPACE}/RVV_ukernels_benchmark/kernels/"
rsync_board tests/test_blis/RVV_${VLEN_BITS}_XDSL ${RSYNC_TARGET}:${RISCV_WORKSPACE}/RVV_ukernels_benchmark/kernels/
ssh_host "mkdir -p ${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/"
rsync_board tests/test_openblas/cnn_models ${RSYNC_TARGET}:${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/
rsync_board xdsl_sweep_driver/xdsl_gemm_blis.c ${RSYNC_TARGET}:${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/src/gemm_blis.c
rsync_board xdsl_sweep_driver/xdsl_gemm_blis.h ${RSYNC_TARGET}:${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/src/gemm_blis.h
rsync_board xdsl_sweep_driver/xdsl_test_gemm.c  ${RSYNC_TARGET}:${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/src/test_gemm.c

# Render DATASETS into a bash array declaration for the remote script
DATASETS_BASH_DECL="DATASETS=("
for ds in "${DATASETS[@]}"; do DATASETS_BASH_DECL+="\"$ds\" "; done
DATASETS_BASH_DECL+=")"

cat > /tmp/xdsl_sweep.sh << SWEEP_EOF
#!/bin/bash
BLIS_HOME=${RISCV_BLIS_DIR_ENV}
BENCH_DIR=\$HOME/${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family

cd \$BENCH_DIR
mkdir -p build out

FIXED_KPATH="../kernels/RVV_${VLEN_BITS}_XDSL/fp32/${MAX_DIM_STR}/"
FIXED_FILE="kernels_RVV_${MAX_DIM_STR}_fp32"

${DATASETS_BASH_DECL}

echo "Compiling monolithic driver binary..."
make clean > /dev/null 2>&1 || true

export CFLAGS="-DLIB_DIR=\"\$FIXED_KPATH\" -DCOMPILER_FLAGS=\"-O3 -mcmodel=medany -march=rv64gcv\""

if ! make MR=1 NR=1 KPATH="\$FIXED_KPATH" FILE="\$FIXED_FILE" \
          OPT=XDSL GATHER=2 SWAP=0 SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO \
          BLIS_HOME=\$BLIS_HOME > /tmp/make_monolithic.log 2>&1; then
    echo "ERROR: Compilation failed:"
    cat /tmp/make_monolithic.log
    exit 1
fi
echo "Binary built successfully."

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=\$BLIS_HOME/lib:\$LD_LIBRARY_PATH

for DATASET in "\${DATASETS[@]}"; do
    DS_NAME=\$(basename \$DATASET .dat)
    OUTFILE="out/result_\${DS_NAME}_XDSL_sweep_macc.dat"
    LOGFILE="out/sweep_log_\${DS_NAME}.txt"
    echo ""
    echo " >>> Dataset: \$DATASET"
    build/test_gemm.x "" "C" "C" "C" "N" "N" 1.0 0.0 0 0 0 0 0 0 0 0 0 0 3.0 F \$DATASET \$OUTFILE ${MAX_MR} ${MAX_NR} | tee "\$LOGFILE"
done
SWEEP_EOF

chmod +x /tmp/xdsl_sweep.sh
rsync_board /tmp/xdsl_sweep.sh ${RSYNC_TARGET}:${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/xdsl_sweep.sh
ssh_board "bash ${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/xdsl_sweep.sh"

echo " Collecting XDSL sweep CSVs to tests/output/..."
mkdir -p tests/output
for DS in "${DATASETS[@]}"; do
    DS_NAME=$(basename "$DS" .dat)
    REMOTE_CSV="${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/out/result_${DS_NAME}_XDSL_sweep_macc.dat"
    LOCAL_CSV="tests/output/xdsl_${DS_NAME}_${MAX_MR}x${MAX_NR}.csv"
    REMOTE_LOG="${RISCV_WORKSPACE}/RVV_ukernels_benchmark/gemm_blis_family/out/sweep_log_${DS_NAME}.txt"
    LOCAL_LOG="tests/output/sweep_log_${DS_NAME}_${MAX_MR}x${MAX_NR}.txt"
    rsync_board "${RSYNC_TARGET}:${REMOTE_CSV}" "${LOCAL_CSV}" || echo " [warn] Could not fetch ${REMOTE_CSV}"
    rsync_board "${RSYNC_TARGET}:${REMOTE_LOG}" "${LOCAL_LOG}" || echo " [warn] Could not fetch ${REMOTE_LOG}"
    echo " Saved: ${LOCAL_CSV}  +  ${LOCAL_LOG}"
done


# ============================================================
# Step 8 — OpenBLAS reference comparison + plot
# ============================================================
echo ""; echo "$SEP"; echo " [8/8] OpenBLAS Reference + Comparison Plots"; echo "$SEP"

for DS in "${DATASETS[@]}"; do
    DS_NAME=$(basename "$DS" .dat)
    LOCAL_OBLAS_OUT="tests/output/openblas_${DS_NAME}.txt"
    echo " Running OpenBLAS for: $DS"
    ssh_board "cd ${RISCV_WORKSPACE} && export OPENBLAS_NUM_THREADS=1 && \
        bash test_openblas/openblassDesign/compile_openblas.sh --dir \"${RISCV_OPENBLAS_DIR_ENV}\" --zvl test_openblas/${DS}" \
        | tee "${LOCAL_OBLAS_OUT}"

    LOCAL_CSV="tests/output/xdsl_${DS_NAME}_${MAX_MR}x${MAX_NR}.csv"
    OUT_PREFIX="tests/output/${DS_NAME}_${MAX_MR}x${MAX_NR}"
    if [ -f "${LOCAL_CSV}" ] && [ -f "${LOCAL_OBLAS_OUT}" ]; then
        echo " Plotting: $DS_NAME"
        LOCAL_LOG="tests/output/sweep_log_${DS_NAME}_${MAX_MR}x${MAX_NR}.txt"
        python tests/output/plot_comparison.py \
            --xdsl "${LOCAL_CSV}" --oblas "${LOCAL_OBLAS_OUT}" \
            --model "${DS_NAME}" --out "${OUT_PREFIX}" \
            --vlen "${VLEN_BITS}" --kc "${KC_PROFILE}" \
            --families "${MAX_DIM_STR}" \
            ${BEST_UK:+--best-kernel "${BEST_UK}"} \
            ${LOCAL_LOG:+--sweep-log "${LOCAL_LOG}"}
    else
        echo " [skip] Missing data for $DS_NAME"
    fi
done

echo ""
echo "============================================================"
echo " Run Summary"
echo "============================================================"
echo "  Families  : ${FAMILIES}  (max kernel: ${MAX_DIM_STR})"
echo "  VLEN      : ${VLEN_BITS} bits"
echo "  KC profile: ${KC_PROFILE}"
echo "  Datasets  : ${DATASETS[*]}"
echo "  Best kc-loop kernel: ${BEST_UK:-N/A}"
echo "  Outputs saved to: tests/output/"
echo "============================================================"
echo ""
