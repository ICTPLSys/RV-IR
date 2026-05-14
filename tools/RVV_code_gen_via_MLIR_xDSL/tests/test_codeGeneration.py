import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from glob import glob
from io import StringIO

# ============ Verbosity Control ============
# Set to control output detail level
VERBOSE = False  # Set to True for detailed output
SHOW_MLIR = False  # Set to True to see generated MLIR
SHOW_PASSES = False  # Set to True to see pass execution

# We update these based on the args later.

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.dialects.builtin import (
    Float32Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.printer import Printer

# Import xDSL-Template passes and dialects
from xdsltemplate.dialects.rvv import (
    RVV,
    RVVFloat32M1Type,
    vfmacc_vf_f32m1Op,
    vfmv_v_f_f32m1Op,
    vle32_v_f32m1Op,
    vse32_v_f32m1Op,
)
from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
from xdsltemplate.transforms.gemm_to_arith import GemmToArithPass
from xdsltemplate.transforms.memref_load_to_emitc import MemrefLoadToEmitcPass
from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass
from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
from xdsltemplate.transforms.rvv_to_emitc import RVVToEmitCPass
from xdsltemplate.transforms.scf_to_emitc import SCFToEmitCPass


def _sanitize_printed_mlir_for_mlir_translate(text: str) -> str:
    """Adjust generic-printed IR so stock mlir-translate can parse and emit it.

    - xDSL may print dynamic memref dims as -1; MLIR text uses '?'.
    - Local dialect name emitc_ext.constant is replaced by upstream emitc.constant.
    """

    def repl_memref(m: re.Match[str]) -> str:
        return "memref<" + m.group(1).replace("-1", "?") + ">"

    text = re.sub(r"memref<([^>]*)>", repl_memref, text)
    text = text.replace('"emitc_ext.constant"', '"emitc.constant"')
    return text


# ============ Configuration ============
class BetaMode:
    B0 = 0
    B1 = 1


@dataclass
class GEMMConfig:
    mr: int
    nr: int
    vlen: int = 8
    beta: int = BetaMode.B0

    @property
    def num_m_vecs(self) -> int:
        return (self.mr + self.vlen - 1) // self.vlen

    @property
    def num_c_regs(self) -> int:
        return self.nr * self.num_m_vecs

    @property
    def kernel_name(self) -> str:
        beta_str = "b0" if self.beta == BetaMode.B0 else "b1"
        return f"xdsl_api_gemm_RVV_{self.mr}x{self.nr}_{beta_str}_col_fp32"

    @property
    def register_efficiency(self) -> float:
        """Percentage of vector register elements actually used"""
        total_elements = self.nr * self.num_m_vecs * self.vlen
        used_elements = self.nr * self.mr
        return used_elements / total_elements if total_elements > 0 else 0.0


# ============ IRBuilder Helper ============
class IRBuilder:
    def __init__(self, block: Block, vlen: int = 8, config: GEMMConfig = None):
        self.block = block
        self.vlen = vlen
        self.config = config
        self.index_t = IndexType()
        self.f32_t = Float32Type()
        self._const_cache = {}

    def const_f32(self, val: float) -> SSAValue:
        op = arith.ConstantOp(FloatAttr(val, self.f32_t))
        self.block.add_op(op)
        return op.result

    def const_index(self, val: int) -> SSAValue:
        if val not in self._const_cache:
            op = arith.ConstantOp(IntegerAttr(val, self.index_t))
            self.block.add_op(op)
            self._const_cache[val] = op.result
        return self._const_cache[val]

    def muli(self, a: SSAValue, b: SSAValue) -> SSAValue:
        op = arith.MuliOp(a, b)
        self.block.add_op(op)
        return op.result

    def addi(self, a: SSAValue, b: SSAValue) -> SSAValue:
        op = arith.AddiOp(a, b)
        self.block.add_op(op)
        return op.result

    def vfmv_zero(self, vl: SSAValue, f0: SSAValue) -> SSAValue:
        op = vfmv_v_f_f32m1Op(f0, vl)
        self.block.add_op(op)
        return op.result

    def vle32(self, mem: SSAValue, offset: SSAValue, vl: SSAValue) -> SSAValue:
        op = vle32_v_f32m1Op(mem, offset, vl)
        self.block.add_op(op)
        return op.vl

    def vfmacc(
        self,
        acc: SSAValue,
        b_mem: SSAValue,
        b_off: SSAValue,
        a_vec: SSAValue,
        vl: SSAValue,
    ) -> SSAValue:
        op = vfmacc_vf_f32m1Op(acc, b_mem, b_off, a_vec, vl)
        self.block.add_op(op)
        return op.result

    def vse32(self, mem: SSAValue, offset: SSAValue, vec: SSAValue, vl: SSAValue):
        op = vse32_v_f32m1Op(mem, offset, vec, vl)
        self.block.add_op(op)

    def get_vl_for_m_vec(self, m_idx: int) -> SSAValue:
        # Check if last vector is partial
        if self.config and (m_idx == self.config.num_m_vecs - 1):
            remainder = self.config.mr % self.vlen
            if remainder != 0:
                return self.const_index(remainder)
        return self.const_index(self.vlen)


def check_kernels(kernels, vlen=8):
    """Quick check of kernel configurations. Returns list of valid kernels."""
    print(f"\n{'─' * 50}")
    print(f" Validating {len(kernels)} Kernels (VLEN={vlen})")
    print(f"{'─' * 50}")

    valid = []
    output_items = []

    for mr, nr in kernels:
        num_m_vecs = (mr + vlen - 1) // vlen
        total_regs = nr * num_m_vecs + num_m_vecs

        if total_regs > 32:
            status = f"⚠️  {total_regs:2d}/32 (SPILL)"
            valid.append((mr, nr))
        elif total_regs > 28:
            status = f"⚠️  {total_regs:2d}/32 (TIGHT)"
            valid.append((mr, nr))
        else:
            status = f"✅ {total_regs:2d}/32"
            valid.append((mr, nr))

        output_items.append(f"({mr:2d}×{nr:2d}): {status}")

    # Print in 4 columns
    for i in range(0, len(output_items), 4):
        chunk = output_items[i : i + 4]
        print("  " + "  ".join(f"{item:<25}" for item in chunk))

    print(f"{'─' * 50}")
    print(f" {len(valid)}/{len(kernels)} valid")
    return valid


# GEMM Builder
def build_gemm(
    mr: int = 8,
    nr: int = 4,
    vlen: int = 4,
    beta: BetaMode = BetaMode.B0,
    name: str = None,
) -> ModuleOp:
    """
    Build MRxNR GEMM microkernel using xDSL API.

    Generates function with signature:
    void gemm_...(kc, alpha, A, lda, B, ldb, beta, C, ldc)
    """
    config = GEMMConfig(mr=mr, nr=nr, vlen=vlen, beta=beta)
    if name is None:
        name = config.kernel_name
    num_m_vecs = config.num_m_vecs
    # Types
    index_t = IndexType()
    f32_t = Float32Type()
    memref_f32 = MemRefType(f32_t, [-1])
    rvv_vec_t = RVVFloat32M1Type()

    # Function arguments: (kc, alpha, A, lda, B, ldb, beta, C, ldc) -> ()
    input_types = [
        index_t,  # kc
        memref_f32,  # alpha
        memref_f32,  # A
        index_t,  # lda
        memref_f32,  # B
        index_t,  # ldb
        memref_f32,  # beta
        memref_f32,  # C
        index_t,  # ldc
    ]
    func_type = FunctionType.from_lists(input_types, [])
    entry_block = Block(arg_types=input_types)
    gemm_func = func.FuncOp(name, func_type, Region([entry_block]))
    # Unpack arguments
    kc, alpha, A, lda, B, ldb, beta_arg, C, ldc = entry_block.args

    # ============ Pass config to IRBuilder ============
    eb = IRBuilder(entry_block, vlen, config=config)
    # Constants
    f0 = eb.const_f32(0.0)

    # Pre-create all needed index constants
    for i in range(max(nr, num_m_vecs * vlen) + 1):
        eb.const_index(i)
    # ============ Dual register initialization ============
    # Initialize C registers (beta=0: zeros, beta=1: load from C)
    c_reg_init = {}
    if beta == BetaMode.B0:
        # For each column, create separate full and tail registers
        for n in range(nr):
            for m in range(num_m_vecs):
                # Get appropriate VL for this vector
                vl = eb.get_vl_for_m_vec(m)
                c_reg_init[(n, m)] = eb.vfmv_zero(vl, f0)

    elif beta == BetaMode.B1:
        for n in range(nr):
            # Calculate column base offset
            if n == 0:
                col_base = eb.const_index(0)
            elif n == 1:
                col_base = ldc
            else:
                col_base = eb.muli(eb.const_index(n), ldc)

            for m in range(num_m_vecs):
                if m == 0:
                    offset = col_base
                else:
                    offset = eb.addi(col_base, eb.const_index(m * vlen))

                vl = eb.get_vl_for_m_vec(m)
                c_reg_init[(n, m)] = eb.vle32(C, offset, vl)
    # Build k-loop body
    loop_body_types = [index_t] + [rvv_vec_t] * config.num_c_regs
    loop_body = Block(arg_types=loop_body_types)
    lb = IRBuilder(loop_body, vlen, config=config)
    k = loop_body.args[0]
    # Map loop args to C registers
    c_regs = {}
    idx = 1
    for n in range(nr):
        for m in range(num_m_vecs):
            c_regs[(n, m)] = loop_body.args[idx]
            idx += 1
    # ============ Load A with dynamic VL ============
    k_mul_lda = lb.muli(k, lda)
    A_regs = []
    for m in range(num_m_vecs):
        if m == 0:
            a_offset = k_mul_lda
        else:
            a_offset = lb.addi(k_mul_lda, eb.const_index(m * vlen))

        # Use appropriate VL for this vector
        vl = lb.get_vl_for_m_vec(m)
        A_regs.append(lb.vle32(A, a_offset, vl))
    # B offsets: k * ldb + n
    k_mul_ldb = lb.muli(k, ldb)
    b_offsets = []
    for n in range(nr):
        if n == 0:
            b_offsets.append(k_mul_ldb)
        else:
            b_offsets.append(lb.addi(k_mul_ldb, eb.const_index(n)))
    # ============ FMA with dynamic VL ============
    c_new = {}
    for n in range(nr):
        for m in range(num_m_vecs):
            # Use appropriate VL for this vector
            vl = lb.get_vl_for_m_vec(m)

            c_new[(n, m)] = lb.vfmacc(
                c_regs[(n, m)],  # accumulator
                B,  # B memory
                b_offsets[n],  # B offset
                A_regs[m],  # A vector
                vl,  # vector length (4 or 2)
            )
    # Yield updated C registers
    yield_vals = [c_new[(n, m)] for n in range(nr) for m in range(num_m_vecs)]
    loop_body.add_op(scf.YieldOp(*yield_vals))
    # Create scf.for loop
    iter_inits = [c_reg_init[(n, m)] for n in range(nr) for m in range(num_m_vecs)]
    for_op = scf.ForOp(
        eb.const_index(0), kc, eb.const_index(1), iter_inits, Region([loop_body])
    )
    entry_block.add_op(for_op)
    # Get final C values from loop results
    c_final = {}
    idx = 0
    for n in range(nr):
        for m in range(num_m_vecs):
            c_final[(n, m)] = for_op.results[idx]
            idx += 1
    # ============ Store C with dynamic VL ============
    for n in range(nr):
        # Calculate column base offset
        if n == 0:
            col_base = eb.const_index(0)
        elif n == 1:
            col_base = ldc
        else:
            col_base = eb.muli(eb.const_index(n), ldc)

        for m in range(num_m_vecs):
            if m == 0:
                offset = col_base
            else:
                offset = eb.addi(col_base, eb.const_index(m * vlen))

            # Use appropriate VL for this vector
            vl = eb.get_vl_for_m_vec(m)

            eb.vse32(C, offset, c_final[(n, m)], vl)
    # Return
    entry_block.add_op(func.ReturnOp())
    # Wrap in module
    return ModuleOp([gemm_func])


# C++ Header and Test Generation
CPP_HEADER = """\
#include <stdio.h>
#include <riscv_vector.h>
#include <cstdlib>
#include <math.h>
#include "gemm_profile.h"

"""


def generate_header_file(
    kernels: list[tuple[int, int]], output_path: str, beta_modes: list[BetaMode] = None
):
    """
    Generate a C++ header file with function declarations for all beta modes.
    """
    if beta_modes is None:
        beta_modes = [BetaMode.B0]

    code = """\
#ifndef xdsl_api_gemm_H
#define xdsl_api_gemm_H

#include <cstddef>

// Generated GEMM microkernel declarations
// Arguments: kc, alpha, A, lda, B, ldb, beta, C, ldc

"""

    for beta in beta_modes:
        beta_str = "b0" if beta == BetaMode.B0 else "b1"
        code += f"// ---- Beta={beta_str} kernels ----\n\n"
        for mr, nr in kernels:
            code += f"void xdsl_api_gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32(\n"
            code += "    void* v0,         // ctxt (placeholder, unused)\n"
            code += "    int v1,           // kc: Loop count (K dimension)\n"
            code += "    float* v2,        // alpha: Scale factor \n"
            code += "    float* v3,        // A: Matrix A pointer\n"
            code += "    int v4,           // lda: Leading dimension of A\n"
            code += "    float* v5,        // B: Matrix B pointer\n"
            code += "    int v6,           // ldb: Leading dimension of B\n"
            code += "    float* v7,        // beta: Scale factor  \n"
            code += "    float* v8,        // C: Matrix C pointer (output)\n"
            code += "    int v9            // ldc: Leading dimension of C\n"
            code += ");\n\n"

    code += "#endif // xdsl_api_gemm_H\n"

    with open(output_path, "w") as f:
        f.write(code)

    if VERBOSE:
        print(f"  ✔ {os.path.basename(output_path)}")


def generate_kernel_family_dispatchers(
    kernel_families: list[tuple[int, int]],
    all_kernels: list[tuple[int, int]],
    beta_modes: list[BetaMode] = None,
    no_ref: bool = False,
) -> str:
    """
    Generate family dispatcher functions that select the appropriate microkernel
    based on actual tile dimensions, for each beta mode.

    Args:
        kernel_families: List of (max_mr, max_nr) tuples defining each family
        all_kernels: Complete list of all generated kernels
        beta_modes: List of beta modes to generate dispatchers for

    Returns:
        C++ code for all family dispatchers
    """
    if beta_modes is None:
        beta_modes = [BetaMode.B0]

    code = ""

    code += """
// ============================================================
// KERNEL FAMILY DISPATCHERS - Dynamic microkernel selection
// ============================================================

"""

    for beta_mode in beta_modes:
        beta_str = "b0" if beta_mode == BetaMode.B0 else "b1"
        beta_value = "0.0f" if beta_mode == BetaMode.B0 else "1.0f"

        code += f"""
// ---- Dispatchers for beta={beta_str} ----

"""

        for max_mr, max_nr in kernel_families:
            # Get all kernels in this family
            family_kernels = [
                (mr, nr) for mr, nr in all_kernels if mr <= max_mr and nr <= max_nr
            ]

            code += f"""// Dispatcher for {max_mr}×{max_nr} family (xDSL kernels, {beta_str})
// Optimization: Function Pointer Table for O(1) dispatch
typedef void (*xdsl_kernel_{beta_str}_t)(void*, int, float*, float*, int, float*, int, float*, float*, int);

static void call_kernel_family_{max_mr}x{max_nr}_{beta_str}(int mr_actual, int nr_actual, size_t K,
                                                   float* A, size_t lda,
                                                   float* B, size_t ldb,
                                                   float* C, size_t ldc) {{
    float alpha = 1.0f;
    float beta = {beta_value};
    
    // Static Lookup Table
    static const xdsl_kernel_{beta_str}_t kernel_table[{max_mr} + 1][{max_nr} + 1] = {{
"""

            # Initialize table with NULLs first
            table_rows = []
            sorted_family_kernels = sorted(family_kernels)

            for mr_idx in range(max_mr + 1):
                row_entries = []
                for nr_idx in range(max_nr + 1):
                    if (mr_idx, nr_idx) in sorted_family_kernels:
                        row_entries.append(
                            f"gemm_RVV_{mr_idx}x{nr_idx}_{beta_str}_col_fp32"
                        )
                    else:
                        row_entries.append("NULL")

                table_rows.append("        { " + ", ".join(row_entries) + " }")

            code += ",\n".join(table_rows)
            code += "\n    };\n\n"

            # Dispatch Logic
            code += f"""    // O(1) Dispatch
    if (mr_actual <= {max_mr} && nr_actual <= {max_nr}) {{
        xdsl_kernel_{beta_str}_t func = kernel_table[mr_actual][nr_actual];
        if (func) {{
             // xDSL signature: (kc, alpha, A, lda, B, ldb, beta, C, ldc)
            func(NULL, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
        }} else {{
            fprintf(stderr, "ERROR: No xDSL kernel ({beta_str}) in {max_mr}x{max_nr} family for %dx%d (Table Entry NULL)\\n", mr_actual, nr_actual);
            exit(1);
        }}
    }} else {{
        fprintf(stderr, "ERROR: Dimensions %dx%d out of bounds for family {max_mr}x{max_nr}\\n", mr_actual, nr_actual);
        exit(1);
    }}
}}

"""

            # Generate reference family dispatcher
            if not no_ref:
                code += f"""// Dispatcher for {max_mr}×{max_nr} family (reference kernels, {beta_str})
// Optimization: Function Pointer Table for O(1) dispatch
typedef void (*ref_kernel_{beta_str}_t)(void*, int_fast32_t, const float*, float*, int, float*, int, const float*, float*, int);

static void call_kernel_family_ref_{max_mr}x{max_nr}_{beta_str}(int mr_actual, int nr_actual, size_t K,
                                                       float* A, size_t lda,
                                                       float* B, size_t ldb,
                                                       float* C, size_t ldc) {{
    float alpha = 1.0f;
    float beta = {beta_value};
    
    // Static Lookup Table
    static const ref_kernel_{beta_str}_t kernel_table[{max_mr} + 1][{max_nr} + 1] = {{
"""

                table_rows = []
                sorted_family_kernels = sorted(family_kernels)

                for mr_idx in range(max_mr + 1):
                    row_entries = []
                    for nr_idx in range(max_nr + 1):
                        if (mr_idx, nr_idx) in sorted_family_kernels:
                            row_entries.append(
                                f"gemm_RVV_{mr_idx}x{nr_idx}_{beta_str}_col_fp32"
                            )
                        else:
                            row_entries.append("NULL")

                    table_rows.append("        { " + ", ".join(row_entries) + " }")

                code += ",\n".join(table_rows)
                code += "\n    };\n\n"

                # Dispatch Logic
                code += f"""    // O(1) Dispatch
    if (mr_actual <= {max_mr} && nr_actual <= {max_nr}) {{
        ref_kernel_{beta_str}_t func = kernel_table[mr_actual][nr_actual];
        if (func) {{
            // Correct reference signature: (uC, K, alpha, A, lda, B, ldb, beta, C, ldc)
            func(NULL, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
        }} else {{
            fprintf(stderr, "ERROR: No reference kernel ({beta_str}) in {max_mr}x{max_nr} family for %dx%d (Table Entry NULL)\\n", mr_actual, nr_actual);
            exit(1);
        }}
    }} else {{
        fprintf(stderr, "ERROR: Dimensions %dx%d out of bounds for family {max_mr}x{max_nr}\\n", mr_actual, nr_actual);
        exit(1);
    }}
}}

"""

    return code


def generate_profiling_section(
    kernels: list[tuple[int, int]],
    beta_modes: list[BetaMode] = None,
    kc_profile: int = 256,
    no_ref: bool = False,
) -> str:
    """
    Generate profiling code for each microkernel, for all beta modes.

    Args:
        kernels: List of unique (mr, nr) kernel configurations to profile
        beta_modes: List of beta modes to profile
        kc_profile: The size of the KC loop to block over in measuring
        no_ref: Disable generating profiling comparisons for reference kernel

    Returns:
        C++ code for profiling section
    """
    if beta_modes is None:
        beta_modes = [BetaMode.B0]

    # Remove duplicates and sort
    unique_kernels = list(set(kernels))
    unique_kernels.sort()

    code = """

    printf("kc Loop Performance Profiling\\n");

    
    // CPU frequency configuration
    // Set to your actual CPU frequency in GHz for time conversion
    // For QEMU or unknown frequency, set to 0.0 to show cycles only
    
    const int WARMUP_ITERS = 500;      // Warmup iterations
    const int MEASURE_ITERS = 1500;    // Measurement iterations
"""
    code += f"    const size_t KC_PROFILE = {kc_profile};     // Larger KC for meaningful profiling\n"

    code += """
    printf("Configuration:\\n");
    printf("  Warmup:        %d iterations\\n", WARMUP_ITERS);
    printf("  Measurement:   %d iterations\\n", MEASURE_ITERS);
    printf("  KC (K-dim):    %lu\\n\\n", KC_PROFILE);
"""

    for beta_mode in beta_modes:
        beta_str = "b0" if beta_mode == BetaMode.B0 else "b1"
        beta_val = "0.0f" if beta_mode == BetaMode.B0 else "1.0f"

        if no_ref:
            code += f"""
    printf("\\n--- Profiling {beta_str} kernels ---\\n");
    // Table header
    printf("%-15s %8s %8s\\n", 
           "Kernel", "xDSL_us", "xDSL_GF");
    printf("%-15s %8s %8s\\n",
           "---------------", "--------", "--------");

    {{
    float alpha_{beta_str} = 1.0f;
    float beta_{beta_str} = {beta_val};
"""
        else:
            code += f"""
    printf("\\n--- Profiling {beta_str} kernels ---\\n");
    // Table header - side-by-side comparison
    printf("%-15s %8s %8s   %8s %8s %7s\\n", 
           "Kernel", "xDSL_us", "Ref_us", "xDSL_GF", "Ref_GF", "Speedup");
    printf("%-15s %8s %8s   %8s %8s %7s\\n",
           "---------------", "--------", "--------", "--------", "--------", "-------");

    {{
    float alpha_{beta_str} = 1.0f;
    float beta_{beta_str} = {beta_val};
    float c_alpha_{beta_str} = 1.0f;
    float c_beta_{beta_str} = {beta_val};
"""

        for mr, nr in unique_kernels:
            code += f"""
    {{ // Scope for {mr}x{nr} {beta_str} kernel pair
    init_abc(A, B, C_kernel, M, {mr}, {nr});
    
    // Profile xDSL kernel
    uint64_t xdsl_flops_{beta_str} = (uint64_t){mr} * {nr} * KC_PROFILE * 2;
    profile_data_t *xdsl_prof = profile_init(MEASURE_ITERS, xdsl_flops_{beta_str});
    for (int _w = 0; _w < WARMUP_ITERS; _w++) {{
        gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32(NULL, KC_PROFILE, &alpha_{beta_str}, A, {mr}, B, {nr}, &beta_{beta_str}, C_kernel, {mr});
    }}
    for (int _i = 0; _i < MEASURE_ITERS; _i++) {{
        uint64_t _start = TIMER_READ();
        gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32(NULL, KC_PROFILE, &alpha_{beta_str}, A, {mr}, B, {nr}, &beta_{beta_str}, C_kernel, {mr});
        uint64_t _end = TIMER_READ();
        profile_add_sample(xdsl_prof, _end - _start);
    }}
"""
            if not no_ref:
                code += f'''
    // Profile reference kernel
    init_abc(A, B, C_ref, M, {mr}, {nr});
    profile_data_t *ref_prof = profile_init(MEASURE_ITERS, xdsl_flops_{beta_str});
    for (int _w = 0; _w < WARMUP_ITERS; _w++) {{
        gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32(NULL, KC_PROFILE, &c_alpha_{beta_str}, A, {mr}, B, {nr}, &c_beta_{beta_str}, C_ref, {mr});
    }}
    for (int _i = 0; _i < MEASURE_ITERS; _i++) {{
        uint64_t _start = TIMER_READ();
        gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32(NULL, KC_PROFILE, &c_alpha_{beta_str}, A, {mr}, B, {nr}, &c_beta_{beta_str}, C_ref, {mr});
        uint64_t _end = TIMER_READ();
        profile_add_sample(ref_prof, _end - _start);
    }}
    
    // Print comparison
    profile_print_comparison("{mr}x{nr}_{beta_str}", xdsl_prof, ref_prof);
    
    profile_free(xdsl_prof);
    profile_free(ref_prof);
    }} // End scope for {mr}x{nr} {beta_str}
'''
            else:
                code += f'''
    // Print stats
    profile_print_report("{mr}x{nr}_{beta_str}", xdsl_prof, 0.0);
    
    profile_free(xdsl_prof);
    }} // End scope for {mr}x{nr} {beta_str}
'''

        code += """
    } // End beta scope
"""

    return code


def generate_test_maincpp(
    beta_modes: list[BetaMode],
    kernels: list[tuple[int, int]],
    kernel_families: list[tuple[int, int]],
    output_path: str,
    header_name: str = "xdsl_api_gemm.h",
    ref_kernel_name: str = "kernels_RVV_16x15_fp32",
    kc_profile: int = 256,
    no_ref: bool = False,
):
    """
    Generate a test harness C++ file to call generated kernels and outer loop tests.

    Args:
        beta_modes: List of beta modes to test
        kernels: Complete list of all generated kernels
        kernel_families: List of (max_mr, max_nr) defining kernel families
        output_path: Path to write the generated C++ file
        header_name: Name of the header file to include
        ref_kernel_name: Name of the reference kernel file (without extension)
    """
    code = CPP_HEADER

    # Include merged kernel header (declares gemm_RVV_* names) instead of xdsl_api_gemm.h
    max_mr_k = max(k[0] for k in kernels)
    max_nr_k = max(k[1] for k in kernels)
    merged_header_name = f"kernels_RVV_{max_mr_k}x{max_nr_k}_fp32.h"

    code += "#include <cstddef>\n"
    code += f'#include "merged/{merged_header_name}"\n'
    code += "#include <stdio.h>\n"
    code += "#include <stdlib.h>\n"
    code += "#include <stdint.h>\n"
    code += "#include <time.h>\n"
    code += '#include "gemm_profile.h"\n'
    if not no_ref:
        code += f'#include "{ref_kernel_name}.h"\n'

    # Helper functions
    code += """\
// Initialize matrices with test values
void init_abc(float* A, float* B, float* C, int size, int r_seed_1, int r_seed_2) {
    for (int i = 0; i < size; i++) {
        A[i] = (float)(i % 10) * 0.1f * r_seed_1;
        B[i] = (float)((i + 3) % 10) * 0.1f * r_seed_2;
        C[i] = 123.1f;
    }
}

// Print matrix values
void print_matrix(const char* name, float* M, int rows, int cols) {
    printf("%s:\\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", M[i + j * rows]);  
        }
        printf("\\n");
    }
}

// Compare two C matrices
int compare_C(float* C_ref, float* C, int size, float tol) {
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = C_ref[i] - C[i];
        if (diff < 0) diff = -diff;
        if (diff > tol) {
            if (errors < 10) {
                printf("Mismatch at %d: ref=%f, got=%f\\n", i, C_ref[i], C[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("      OK √ ");

    } else {
        printf("========================\\n");
        printf("=        MISMATCHES    =\\n");
        printf("=        MISMATCHES    =\\n");
        printf("=        MISMATCHES    =\\n");
        printf("FAIL: %d mismatches\\n", errors);
        printf("========================\\n");
    }
    return errors;
}
"""

    # ============================================================
    # GENERATE KERNEL FAMILY DISPATCHERS
    # ============================================================
    code += generate_kernel_family_dispatchers(
        kernel_families, kernels, beta_modes, no_ref=no_ref
    )

    # ============================================================
    # GENERATE FAMILY-BASED OUTER LOOPS WITH EDGE HANDLING
    # ============================================================

    code += """
// ============================================================
// FAMILY-BASED OUTER LOOPS - With edge case handling
// ============================================================

"""

    for beta_mode in beta_modes:
        beta_str = "b0" if beta_mode == BetaMode.B0 else "b1"

        for max_mr, max_nr in kernel_families:
            # xDSL outer loop with edge handling
            code += f"""// Outer loop for {max_mr}×{max_nr} family (xDSL, {beta_str}) - WITH edge handling and K-blocking
static void gemm_outer_family_{max_mr}x{max_nr}_{beta_str}(size_t M, size_t N, size_t K,
                                                  float* A, size_t lda,
                                                  float* B, size_t ldb,
                                                  float* C, size_t ldc) {{
    const int MR = {max_mr};
    const int NR = {max_nr};

    // Tile with edge handling
    for (int i = 0; i < M; i += MR) {{
        int mr = (M - i) < MR ? (M - i) : MR;  // Actual rows for this tile
        
        for (int j = 0; j < N; j += NR) {{
            int nr = (N - j) < NR ? (N - j) : NR;  // Actual columns for this tile
            
            call_kernel_family_{max_mr}x{max_nr}_{beta_str}(mr, nr, K, 
                                                  &A[i], lda, 
                                                  &B[j], ldb, 
                                                  &C[i + j * ldc], ldc);
        }}
    }}
}}

"""

            # Reference outer loop with edge handling
            if not no_ref:
                code += f"""// Outer loop for {max_mr}×{max_nr} family (reference, {beta_str}) - WITH edge handling
static void gemm_outer_family_ref_{max_mr}x{max_nr}_{beta_str}(size_t M, size_t N, size_t K,
                                                      float* A, size_t lda,
                                                      float* B, size_t ldb,
                                                      float* C, size_t ldc) {{
    const int MR = {max_mr};
    const int NR = {max_nr};
    
    // Tile with edge handling
    for (int i = 0; i < M; i += MR) {{
        int mr = (M - i) < MR ? (M - i) : MR;  // Actual rows for this tile
        
        for (int j = 0; j < N; j += NR) {{
            int nr = (N - j) < NR ? (N - j) : NR;  // Actual columns for this tile
            
            // Call family dispatcher with actual dimensions
            call_kernel_family_ref_{max_mr}x{max_nr}_{beta_str}(mr, nr, K, 
                                                      &A[i], lda, 
                                                      &B[j], ldb, 
                                                      &C[i + j * ldc], ldc);
        }}
    }}
}}

"""

    # Calculate maximum MR and NR to set appropriate lda, ldb, ldc
    max_mr = max(mr for mr, nr in kernels)
    max_nr = max(nr for mr, nr in kernels)

    # Main function
    code += f"""
int main() {{
    const int M = 32768; // Large enough for KC_PROFILE
    size_t KC = 6;
    float alpha = 1.0f;
    float beta = 0.0f;
    float c_alpha = 1.0f;
    float c_beta = 0.0f;
    size_t lda = {max_mr};  // Must be >= max(mr)
    size_t ldb = {max(16, max_nr)};  // Must be >= max(nr)
    size_t ldc = {max_mr};  // Must be >= max(mr) to avoid column overwrites!

    float* A = (float*)malloc(M * sizeof(float));
    float* B = (float*)malloc(M * sizeof(float));
"""

    # Declare C arrays for unique kernels only (avoid duplicates from overlapping families)
    unique_kernels = list(set(kernels))  # Remove duplicates
    unique_kernels.sort()  # Sort for consistent ordering

    # Single shared C buffer for all per-kernel profiling.
    # Avoids data-cache pollution: 32x32 would otherwise allocate 1024 C matrices
    # totalling 128 MB, which evicts A/B from cache and inflates small-kernel timings by 1.4-2.8x.
    # init_abc() is called before every kernel's measurement window to reset the buffer.
    code += "    float* C_kernel = (float*)malloc(M * sizeof(float));\n"
    code += "\n"
    if not no_ref:
        code += "    float* C_ref = (float*)malloc(M * sizeof(float));\n\n"

    if not no_ref:
        code += """
       printf("========================================\\n");
       printf("Part 1: uC independent test\\n");
       printf("========================================\\n\\n");
   """
        # Call each unique kernel (Part 1 Micro-kernels) for each beta mode
        for beta_mode in beta_modes:
            beta_str = "b0" if beta_mode == BetaMode.B0 else "b1"
            beta_val = "0.0f" if beta_mode == BetaMode.B0 else "1.0f"

            code += f"""
    printf("\\n--- Part 1: {beta_str} kernels ---\\n");
    {{
    float alpha_{beta_str} = 1.0f;
    float beta_{beta_str} = {beta_val};
    float c_alpha_{beta_str} = 1.0f;
    float c_beta_{beta_str} = {beta_val};
"""
            for mr, nr in unique_kernels:
                code += f'    printf("Testing gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32...");\n'
                code += f"    init_abc(A, B, C_ref, M, {mr}, {nr});\n"
                code += f"    gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32(NULL, KC, &alpha_{beta_str}, A, {mr}, B, {nr}, &beta_{beta_str}, C_kernel, {mr});\n"
                code += f"    gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32(NULL, KC, &c_alpha_{beta_str}, A, {mr}, B, {nr}, &c_beta_{beta_str}, C_ref, {mr});\n"
                code += "    compare_C(C_ref, C_kernel, M, 0.1);\n"
                code += '    printf("\\n");\n'

            code += "    } // end beta scope\n"

        code += """
    printf("========================================\\n");
    printf("Part 2: Outer Loop Tests (Kernel Families)\\n");
    printf("========================================\\n\\n");

    // Test with various matrix sizes INCLUDING edge cases
    int test_sizes[][3] = {

      {128, 128, 128}, 

    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    // Constants for Outer Loop Profiling
    const int WARMUP_ITERS_OUTER = 10;
    const int MEASURE_ITERS_OUTER = 20;
"""

        # Generate test loop for each kernel family, for each beta mode
        for beta_mode in beta_modes:
            beta_str = "b0" if beta_mode == BetaMode.B0 else "b1"

            for max_mr, max_nr in kernel_families:
                code += f"""
    // ========================================
    // Testing {max_mr}×{max_nr} family in outer loop ({beta_str})
    // ========================================
    printf("\\n--- Testing {max_mr}×{max_nr} family in outer loop ({beta_str}) ---\\n");
    printf("%-20s %-10s %12s %12s %12s\\n", "Size(MxNxK)", "Status", "xDSL_GF", "Ref_GF", "Speedup");
    printf("%-20s %-10s %12s %12s %12s\\n", "-----------", "------", "--------", "------", "-------");
    
    for (int t = 0; t < num_tests; t++) {{
        int M_test = test_sizes[t][0];
        int N_test = test_sizes[t][1];
        int K_test = test_sizes[t][2];

        int lda_test = M_test + 8;
        int ldb_test = K_test + 8;
        int ldc_test = M_test;

        float* A_test = (float*)malloc(lda_test * K_test * sizeof(float));
        float* B_test = (float*)malloc(ldb_test * K_test * sizeof(float)); 
        float* C_test = (float*)malloc(  M_test * N_test * sizeof(float));
        float* C_ref_test = (float*)malloc(  M_test * N_test * sizeof(float));

        for (int i = 0; i < lda_test * K_test; i++) A_test[i] = (float)(i % 10) * 0.1f;
        for (int i = 0; i < ldb_test * K_test; i++) B_test[i] = (float)(i % 10) * 0.1f;
        for (int i = 0; i <   M_test * N_test; i++) C_test[i] = 1.1f;
        for (int i = 0; i <   M_test * N_test; i++) C_ref_test[i] = 1.1f;

        // xDSL Outer Loop using {max_mr}×{max_nr} family ({beta_str})
        gemm_outer_family_{max_mr}x{max_nr}_{beta_str}(M_test, N_test, K_test, A_test, lda_test, B_test, ldb_test, C_test, ldc_test);

        // Reference Outer Loop using {max_mr}×{max_nr} family ({beta_str})
        gemm_outer_family_ref_{max_mr}x{max_nr}_{beta_str}(M_test, N_test, K_test, A_test, lda_test, B_test, ldb_test, C_ref_test, ldc_test);

        // Compare
        int errors = 0;
        for (int i = 0; i < M_test * N_test; i++) {{
            float diff = C_ref_test[i] - C_test[i];
            if (diff < 0) diff = -diff;
            if (diff > 0.01f) {{
                errors++;
                if (errors < 5) {{
                    printf("C_ref_test[%d] = %f, C_test[%d] = %f\\n", i, C_ref_test[i], i, C_test[i]);
                }}
            }}
        }}

        if (errors == 0) {{
            uint64_t flops_outer = (uint64_t)M_test * N_test * K_test * 2;
            
            // 1. Profile xDSL
            profile_data_t *prof_xdsl = profile_init(MEASURE_ITERS_OUTER, flops_outer);
            for (int _w = 0; _w < WARMUP_ITERS_OUTER; _w++) {{
                gemm_outer_family_{max_mr}x{max_nr}_{beta_str}(M_test, N_test, K_test, A_test, lda_test, B_test, ldb_test, C_test, ldc_test);
            }}
            for (int _i = 0; _i < MEASURE_ITERS_OUTER; _i++) {{
                uint64_t _start = TIMER_READ();
                gemm_outer_family_{max_mr}x{max_nr}_{beta_str}(M_test, N_test, K_test, A_test, lda_test, B_test, ldb_test, C_test, ldc_test);
                uint64_t _end = TIMER_READ();
                profile_add_sample(prof_xdsl, _end - _start);
            }}

            // 2. Profile Reference
            profile_data_t *prof_ref = profile_init(MEASURE_ITERS_OUTER, flops_outer);
            for (int _w = 0; _w < WARMUP_ITERS_OUTER; _w++) {{
                gemm_outer_family_ref_{max_mr}x{max_nr}_{beta_str}(M_test, N_test, K_test, A_test, lda_test, B_test, ldb_test, C_ref_test, ldc_test);
            }}
            for (int _i = 0; _i < MEASURE_ITERS_OUTER; _i++) {{
                uint64_t _start = TIMER_READ();
                gemm_outer_family_ref_{max_mr}x{max_nr}_{beta_str}(M_test, N_test, K_test, A_test, lda_test, B_test, ldb_test, C_ref_test, ldc_test);
                uint64_t _end = TIMER_READ();
                profile_add_sample(prof_ref, _end - _start);
            }}

            double min_x, median_x, mean_x, stddev_x;
            profile_compute_stats(prof_xdsl, &min_x, &median_x, &mean_x, &stddev_x);
            double gflops_xdsl = (double)flops_outer / (median_x / 1e9) / 1e9;

            double min_r, median_r, mean_r, stddev_r;
            profile_compute_stats(prof_ref, &min_r, &median_r, &mean_r, &stddev_r);
            double gflops_ref = (double)flops_outer / (median_r / 1e9) / 1e9;
            
            double speedup = gflops_xdsl / gflops_ref;

            char size_str[32];
            snprintf(size_str, 32, "%dx%dx%d", M_test, N_test, K_test);
            printf("%-20s %-10s %12.3f %12.3f %11.2fx\\n", size_str, "OK", gflops_xdsl, gflops_ref, speedup);
            
            profile_free(prof_xdsl);
            profile_free(prof_ref);

        }} else {{
            char size_str[32];
            snprintf(size_str, 32, "%dx%dx%d", M_test, N_test, K_test);
            printf("%-20s %-10s %12s %12s %12s\\n", size_str, "FAIL", "-", "-", "-");
        }}

        free(A_test);
        free(B_test);
        free(C_test);
        free(C_ref_test);
    }}
"""

    # ============================================================
    # GENERATE PROFILING SECTION (Part 3)
    # ============================================================
    code += generate_profiling_section(
        unique_kernels, beta_modes, kc_profile=kc_profile, no_ref=no_ref
    )

    code += """
    printf("========================================\\n");
    printf("End of tests!\\n");
    printf("========================================\\n\\n");
    free(A); free(B);
"""
    code += "    free(C_kernel);\n"
    if not no_ref:
        code += "    free(C_ref);\n"

    code += """
    return 0;
}
"""

    with open(output_path, "w") as f:
        f.write(code)

    if VERBOSE:
        print(f"  ✔ {os.path.basename(output_path)}")


def add_header_to_cpp(cpp_path: str):
    """Add standard headers to generated C++ file."""
    with open(cpp_path) as f:
        content = f.read()

    with open(cpp_path, "w") as f:
        f.write(CPP_HEADER)
        f.write(content)


def generate_profiling_header(output_path: str):
    """Generate the gemm_profile.h profiling infrastructure header."""

    header_code = """#ifndef GEMM_PROFILE_H
#define GEMM_PROFILE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ============================================================
// Timer Functions - Using clock_gettime for compatibility
// ============================================================

// Use POSIX clock_gettime() - works on most RISC-V systems
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define TIMER_READ() get_time_ns()
#define TIMER_UNIT "ns"

// ============================================================
// Profiling Data Structures
// ============================================================

typedef struct {
    double *samples;       // Array of timing samples (double for sub-ns precision)
    int num_samples;       // Number of samples collected
    int capacity;          // Array capacity
    uint64_t total_flops;  // Total FLOPs for this operation
} profile_data_t;

// Initialize profiling data
static inline profile_data_t* profile_init(int capacity, uint64_t flops) {
    profile_data_t *p = (profile_data_t*)malloc(sizeof(profile_data_t));
    p->samples = (double*)malloc(capacity * sizeof(double));
    p->num_samples = 0;
    p->capacity = capacity;
    p->total_flops = flops;
    return p;
}

// Add a timing sample
static inline void profile_add_sample(profile_data_t *p, double time) {
    if (p->num_samples < p->capacity) {
        p->samples[p->num_samples++] = time;
    }
}

// Comparison function for qsort
static int compare_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

// Compute statistics
static inline void profile_compute_stats(profile_data_t *p,
                                         double *min, double *median,
                                         double *mean, double *stddev) {
    if (p->num_samples == 0) return;
    
    // Sort samples for median
    qsort(p->samples, p->num_samples, sizeof(double), compare_double);
    
    // Min
    *min = p->samples[0];
    
    // Median
    *median = p->samples[p->num_samples / 2];
    
    // Mean
    double sum = 0;
    for (int i = 0; i < p->num_samples; i++) {
        sum += p->samples[i];
    }
    *mean = sum / p->num_samples;
    
    // Standard deviation
    double variance = 0.0;
    for (int i = 0; i < p->num_samples; i++) {
        double diff = p->samples[i] - (*mean);
        variance += diff * diff;
    }
    *stddev = sqrt(variance / p->num_samples);
}

// Print performance report
static inline void profile_print_report(const char *kernel_name, 
                                        profile_data_t *p,
                                        double cpu_freq_ghz) {
    double min, median, mean;
    double stddev;
    
    profile_compute_stats(p, &min, &median, &mean, &stddev);
    
    // Convert to microseconds
    double time_min_us = min / 1000.0;
    double time_median_us = median / 1000.0;
    
    // FLOP/s calculation
    double time_sec = min / 1e9;
    double gflops_min = (double)p->total_flops / time_sec / 1e9;
    
    time_sec = median / 1e9;
    double gflops_median = (double)p->total_flops / time_sec / 1e9;
    
    // Compact one-line format: Kernel | Min(µs) | Median(µs) | Peak(GFLOP/s) | Median(GFLOP/s)
printf(\"%-15s %8.2f µs %8.2f µs  %8.3f GFLOP/s %8.3f GFLOP/s\\n\",
           kernel_name, time_min_us, time_median_us, gflops_min, gflops_median);
}

// Free profiling data
static inline void profile_free(profile_data_t *p) {
    free(p->samples);
    free(p);
}

// Print comparison of xDSL vs Reference
static inline void profile_print_comparison(const char *kernel_base_name,
                                           profile_data_t *xdsl_prof,
                                           profile_data_t *ref_prof) {
    double xdsl_min, xdsl_median, xdsl_mean;
    double ref_min, ref_median, ref_mean;
    double xdsl_stddev, ref_stddev;
    
    profile_compute_stats(xdsl_prof, &xdsl_min, &xdsl_median, &xdsl_mean, &xdsl_stddev);
    profile_compute_stats(ref_prof, &ref_min, &ref_median, &ref_mean, &ref_stddev);
    
    // Convert to microseconds
    double xdsl_min_us = xdsl_min / 1000.0;
    double ref_min_us = ref_min / 1000.0;
    
    // FLOP/s calculation
    double xdsl_gflops = (double)xdsl_prof->total_flops / (xdsl_min / 1e9) / 1e9;
    double ref_gflops = (double)ref_prof->total_flops / (ref_min / 1e9) / 1e9;
    
    // Speedup
    double speedup = ref_min_us / xdsl_min_us;
    
    // Single line: Kernel | xDSL time | Ref time | | xDSL GF/s | Ref GF/s | Speedup
    printf("%-15s %8.6f %8.6f   %8.6f %8.6f %7.4fx\\n",
           kernel_base_name, xdsl_min_us, ref_min_us, xdsl_gflops, ref_gflops, speedup);
}

// ============================================================
// Profiling Macros
// ============================================================

#define PROFILE_KERNEL(kernel_name, kernel_call, mr, nr, kc, warmup, iterations, cpu_freq) \\
do { \\
    uint64_t flops = (uint64_t)(mr) * (nr) * (kc) * 2; \\
    profile_data_t *prof = profile_init(iterations, flops); \\
    \\
    /* Warmup */ \\
    for (int _w = 0; _w < (warmup); _w++) { \\
        kernel_call; \\
    } \\
    \\
    /* Measurement */ \\
    for (int _i = 0; _i < (iterations); _i++) { \\
        uint64_t _start = TIMER_READ(); \\
        kernel_call; \\
        uint64_t _end = TIMER_READ(); \\
        profile_add_sample(prof, _end - _start); \\
    } \\
    \\
    profile_print_report(kernel_name, prof, cpu_freq); \\
    profile_free(prof); \\
} while(0)

#endif // GEMM_PROFILE_H
"""

    with open(output_path, "w") as f:
        f.write(header_code)

    if VERBOSE:
        print(f"  ✔ {os.path.basename(output_path)}")


def generate_merged_kernel_file(
    kernels: list[tuple[int, int]],
    output_dir: str,
    beta_modes: list[BetaMode],
    vlen_bits: int = 128,
    tests_dir: str = None,
):
    """
    Reads individually generated kernel files, modifies their signature to match
    reference API (int instead of size_t, void* ctxt), and merges them into
    single .c and .h files.
    """

    if not kernels:
        print("Warning: No kernels to merge.")
        return

    # Determine filename based on max dimensions
    max_mr = max(k[0] for k in kernels)
    max_nr = max(k[1] for k in kernels)

    merged_c_filename = f"kernels_RVV_{max_mr}x{max_nr}_fp32.c"
    merged_h_filename = f"kernels_RVV_{max_mr}x{max_nr}_fp32.h"

    merged_dir = os.path.join(output_dir, "merged")
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    else:
        # Clear existing files before dumping fresh merged outputs
        for f in glob(f"{merged_dir}/*"):
            if os.path.isfile(f):
                os.remove(f)

    print(f"  Merging kernels to {merged_c_filename} in merged/...", end="", flush=True)

    header_content = "#pragma once\n"
    header_content += "#include <stddef.h>\n"
    header_content += "#include <stdint.h>\n"
    header_content += '#ifdef __cplusplus\nextern "C" {\n#endif\n\n'

    source_content = f'#include "{merged_h_filename}"\n'
    source_content += "#include <riscv_vector.h>\n\n"

    for mr, nr in kernels:
        for beta_mode in beta_modes:
            beta_str = "b0" if beta_mode == BetaMode.B0 else "b1"

            # Original function name in generated file
            original_kernel_name = f"xdsl_api_gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32"

            # Rename to gemm_RVV_* in merged file (drop xdsl_api_ prefix)
            target_kernel_name = f"gemm_RVV_{mr}x{nr}_{beta_str}_col_fp32"

            filename = f"xdsl_api_gemm_{mr}x{nr}_{beta_str}.cpp"
            filepath = os.path.join(output_dir, "kernels", filename)

            if not os.path.exists(filepath):
                print(f"\nWarning: Kernel file {filename} not found, skipping.")
                continue

            with open(filepath) as f:
                c_func = f.read()

            # Filter unwanted headers
            lines = c_func.split("\n")
            lines = [
                line
                for line in lines
                if "#include <cstdlib>" not in line
                and '#include "gemm_profile.h"' not in line
            ]

            # Post-process signature to ensure correct C-compatible types
            sig_found = False
            for i, line in enumerate(lines):
                if f"void {original_kernel_name}" in line and "(" in line:
                    # Rename: xdsl_api_gemm_RVV_* → gemm_RVV_*
                    line = line.replace(original_kernel_name, target_kernel_name)

                    # Ensure void* ctxt argument is present
                    if "(void*" in line:
                        line = line.replace("(void*", "(void* ctxt", 1)
                    else:
                        line = line.replace("(", "(void* ctxt, ", 1)

                    # Replace size_t with int
                    line = line.replace("size_t", "int")

                    lines[i] = line
                    sig_found = True
                    break

            if sig_found:
                from datetime import datetime

                vlen = vlen_bits // 32
                beta_label = (
                    "b0  (beta=0: C = alpha*A*B)"
                    if beta_str == "b0"
                    else "b1  (beta=1: C = alpha*A*B + beta*C)"
                )
                comment = f"""// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
//   MR         : {mr}
//   NR         : {nr}
//   VLEN       : {vlen}  (fp32 elements per vector register)
//   VLEN_BITS  : {vlen_bits}  (bit-width, VLEN * 32)
//   Beta mode  : {beta_label}
// ============================================================
"""
                c_func = "\n".join(lines)
                source_content += comment
                source_content += f"// Merged from {filename}\n"
                source_content += c_func + "\n"

                # Header declaration with metadata comment
                header_content += comment
                decl = f"void {target_kernel_name}(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);"
                header_content += decl + "\n"

            else:
                print(
                    f"\nWarning: Signature for {original_kernel_name} not found in {filename}"
                )

    header_content += "\n#ifdef __cplusplus\n}\n#endif\n"

    merged_c_path = os.path.join(merged_dir, merged_c_filename)
    merged_h_path = os.path.join(merged_dir, merged_h_filename)

    with open(merged_c_path, "w") as f:
        f.write(source_content)

    with open(merged_h_path, "w") as f:
        f.write(header_content)

    # ── Copy merged files into test_blis/RVV_<VLEN>_XDSL/loadAB/macc ──
    # compile_jbpi.sh rsyncs this folder to RVV_ukernels_benchmark automatically.
    if tests_dir is not None:
        xdsl_blis_dest = os.path.join(
            tests_dir,
            "test_blis",
            f"RVV_{vlen_bits}_XDSL",
            "fp32",
            f"{max_mr}x{max_nr}",
            "loadAB",
            "macc",
        )
        os.makedirs(xdsl_blis_dest, exist_ok=True)
        shutil.copy2(merged_c_path, xdsl_blis_dest)
        shutil.copy2(merged_h_path, xdsl_blis_dest)
        if VERBOSE:
            print(
                f"  ✔ Copied {merged_c_filename} → test_blis/RVV_{vlen_bits}_XDSL/fp32/{max_mr}x{max_nr}/loadAB/macc/"
            )

    if not VERBOSE:
        print("✅")
    else:
        print(f"  ✔ {merged_c_filename}")
        print(f"  ✔ {merged_h_filename}")


def generate_exo_matrix_files(
    kernels: list[tuple[int, int]],
    output_dir: str,
    vlen_bits: int,
    tests_dir: str = None,
):
    """
    Auto-generates exo_matrix_RVV_fp32.c and exo_matrix_RVV_fp32.h
    containing the ukrmatrix dispatcher for dynamic benchmark execution.
    """
    if not kernels:
        return

    max_mr = max(k[0] for k in kernels)
    max_nr = max(k[1] for k in kernels)

    unique_kernels = set(kernels)

    c_filename = "exo_matrix_RVV_fp32.c"
    h_filename = "exo_matrix_RVV_fp32.h"

    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    c_path = os.path.join(merged_dir, c_filename)
    h_path = os.path.join(merged_dir, h_filename)

    merged_h_filename = f"kernels_RVV_{max_mr}x{max_nr}_fp32.h"

    # Write Header
    h_content = f"""#pragma once
#include <stddef.h>
#include <stdint.h>
#include "{merged_h_filename}"

#ifdef __cplusplus
extern "C" {{
#endif

typedef void (*ukrFunction)(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);

ukrFunction**** allocateMatrix();
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix);

#ifdef __cplusplus
}}
#endif
"""
    with open(h_path, "w") as f:
        f.write(h_content)

    # Write C File
    c_content = f"""#include <stdlib.h>
#include "{h_filename}"

ukrFunction**** allocateMatrix() {{
    ukrFunction**** matrix = (ukrFunction****)malloc(({max_mr} + 1) * sizeof(ukrFunction***));
    for (int i = 0; i < {max_mr} + 1; i++) {{
        matrix[i] = (ukrFunction***)malloc(({max_nr} + 1) * sizeof(ukrFunction**));
        for (int j = 0; j < {max_nr} + 1; j++) {{
            matrix[i][j] = (ukrFunction**)malloc(2 * sizeof(ukrFunction*));
            for (int b = 0; b < 2; b++) {{
                matrix[i][j][b] = (ukrFunction*)malloc(1 * sizeof(ukrFunction));
            }}
        }}
    }}
    return matrix;
}}

void fillMatrix(ukrFunction**** matrix) {{
"""
    for mr in range(max_mr + 1):
        for nr in range(max_nr + 1):
            if mr == 0 or nr == 0 or (mr, nr) not in unique_kernels:
                c_content += f"    *matrix[{mr}][{nr}][0] = \t(ukrFunction)NULL;\n"
                c_content += f"    *matrix[{mr}][{nr}][1] = \t(ukrFunction)NULL;\n"
            else:
                c_content += f"    *matrix[{mr}][{nr}][0] = \t(ukrFunction)gemm_RVV_{mr}x{nr}_b0_col_fp32;\n"
                c_content += f"    *matrix[{mr}][{nr}][1] = \t(ukrFunction)gemm_RVV_{mr}x{nr}_b1_col_fp32;\n"

    c_content += f"""}}

void freeMatrix(ukrFunction**** matrix) {{
    for (int i = 0; i < {max_mr} + 1; i++) {{
        for (int j = 0; j < {max_nr} + 1; j++) {{
            for (int b = 0; b < 2; b++) {{
                free(matrix[i][j][b]);
            }}
            free(matrix[i][j]);
        }}
        free(matrix[i]);
    }}
    free(matrix);
}}
"""

    with open(c_path, "w") as f:
        f.write(c_content)

    if tests_dir is not None:
        xdsl_blis_dest = os.path.join(
            tests_dir,
            "test_blis",
            f"RVV_{vlen_bits}_XDSL",
            "fp32",
            f"{max_mr}x{max_nr}",
            "loadAB",
            "macc",
        )
        os.makedirs(xdsl_blis_dest, exist_ok=True)
        shutil.copy2(c_path, xdsl_blis_dest)
        shutil.copy2(h_path, xdsl_blis_dest)
        if VERBOSE:
            print(
                f"  ✔ Copied {c_filename} → test_blis/RVV_{vlen_bits}_XDSL/fp32/{max_mr}x{max_nr}/loadAB/macc/"
            )


def generate_makefile(
    kernels: list[tuple[int, int]],
    output_dir: str,
    beta_modes: list[BetaMode],
    ref_kernel_name: str = "kernels_RVV_16x15_fp32",
    no_ref: bool = False,
):
    """Generate Makefile that compiles the single merged kernel C file for fast single-pass build."""

    # Compute merged file name (same logic as generate_merged_kernel_file)
    max_mr = max(k[0] for k in kernels)
    max_nr = max(k[1] for k in kernels)
    merged_c_name = f"kernels_RVV_{max_mr}x{max_nr}_fp32"

    makefile = f"""\
# Auto-generated Makefile for GEMM kernels
# Single merged kernel file = fast single-pass compilation

CXX = riscv64-linux-gnu-g++
CC  = riscv64-linux-gnu-gcc
CFLAGS = -march=rv64gcv -mabi=lp64d -O3 -mcmodel=medany -lm -I.
TARGET = test_gemm

# Merged xDSL kernel source (one file, one compilation step)
MERGED_KERNEL_SRC = merged/{merged_c_name}.c
MERGED_KERNEL_OBJ = merged/{merged_c_name}.o

# Test program source
TEST_SRC = test_xdsl_api_gemm_kernels.cpp
TEST_OBJ = test_xdsl_api_gemm_kernels.o

"""
    if no_ref:
        makefile += "REF_KERNEL_OBJ =\n"
    else:
        makefile += (
            f"# Reference kernel (pre-compiled)\nREF_KERNEL_OBJ = {ref_kernel_name}.o\n"
        )

    makefile += """# Default target: build test_gemm (do not run automatically)
all: $(TARGET)

"""
    if no_ref:
        makefile += """\
# Build the test executable (no reference kernel)
$(TARGET): $(TEST_OBJ) $(MERGED_KERNEL_OBJ)
\t@echo "Linking test executable..."
\t$(CXX) $(CFLAGS) $(TEST_OBJ) $(MERGED_KERNEL_OBJ) -o $(TARGET)

"""
    else:
        makefile += """\
# Build the test executable
$(TARGET): $(TEST_OBJ) $(MERGED_KERNEL_OBJ) $(REF_KERNEL_OBJ)
\t@echo "Linking test executable..."
\t$(CXX) $(CFLAGS) $(TEST_OBJ) $(MERGED_KERNEL_OBJ) $(REF_KERNEL_OBJ) -o $(TARGET)

"""

    makefile += """\
# Compile merged kernel with C compiler (plain C linkage — no C++ mangling)
$(MERGED_KERNEL_OBJ): $(MERGED_KERNEL_SRC)
\t@echo "Compiling merged kernel (single pass)..."
\t$(CC) $(CFLAGS) -c $(MERGED_KERNEL_SRC) -o $(MERGED_KERNEL_OBJ)

# Compile test program (C++)
$(TEST_OBJ): $(TEST_SRC)
\t@echo "Compiling test program..."
\t$(CXX) $(CFLAGS) -c $(TEST_SRC) -o $(TEST_OBJ)

# Build and run the test
run: $(TARGET)
\t@echo "Running tests..."
\t./$(TARGET)

# Clean everything
clean:
\trm -f $(TARGET) $(TEST_OBJ) $(MERGED_KERNEL_OBJ)

# Help target
help:
\t@echo "Targets: all, run, clean, help"
\t@echo "Tip: merged kernel = single compilation step, fast build."

.PHONY: all run clean help
"""

    makefile_path = os.path.join(output_dir, "Makefile")
    with open(makefile_path, "w") as f:
        f.write(makefile)

    if VERBOSE:
        print("  ✔ Makefile")

    return makefile_path


# Pipeline Runner
def run_pipeline(
    mr: int = 8,
    nr: int = 4,
    vlen: int = 8,
    beta: BetaMode = BetaMode.B0,
    output_dir: str = ".",
    mlir_translate_path: str = None,
    add_header: bool = True,
):
    """
    Complete pipeline: build GEMM -> apply passes -> dump MLIR -> translate to C++.
    """
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(RVV)

    # config = GEMMConfig(mr=mr, nr=nr, vlen=vlen)

    # Step 1: Build GEMM using xDSL API
    if VERBOSE:
        print(f"\n{'┌' + '─' * 58 + '┐'}")
        print(
            f"│ Building {mr}×{nr} Kernel{' ' * (58 - len(f'Building {mr}×{nr} Kernel'))}│"
        )
        print(f"{'└' + '─' * 58 + '┘'}")

    module = build_gemm(mr=mr, nr=nr, vlen=vlen, beta=beta)

    if SHOW_MLIR:
        if nr == 8 and mr == 8:
            print("\n  📄 Generated MLIR:")
            print(f"  {'─' * 56}")
            output = StringIO()
            Printer(stream=output).print_op(module)
            # Indent MLIR output
            for line in output.getvalue().split("\n"):
                print(f"  {line}")
            print(f"  {'─' * 56}")

    # Step 2: Apply transformation passes
    if VERBOSE:
        print("\n  ⚙️  Applying Transformation Passes:")
        print(f"  {'─' * 56}")

    pass_pipeline = [
        GemmToArithPass(),
        MemRefToEmitCPass(),
        ArithToEmitCPass(),
        MemrefLoadToEmitcPass(),
        MemrefStoreToEmitcPass(),
        SCFToEmitCPass(),
        RVVToEmitCPass(),
    ]

    for i, p in enumerate(pass_pipeline):
        try:
            p.apply(ctx, module)
            if SHOW_PASSES:
                print(f"    [{i + 1}/{len(pass_pipeline)}] ✓ {p.__class__.__name__}")
        except Exception as e:
            print(f"    [{i + 1}/{len(pass_pipeline)}] ✗ {p.__class__.__name__} - {e}")
            raise

    if VERBOSE and SHOW_PASSES:
        print(f"  {'─' * 56}")

    # Step 3: Dump MLIR to file
    kernels_dir = os.path.join(output_dir, "kernels")
    if beta == BetaMode.B0:
        output_mlir = f"{kernels_dir}/xdsl_api_gemm_{mr}x{nr}_b0.mlir"
    elif beta == BetaMode.B1:
        output_mlir = f"{kernels_dir}/xdsl_api_gemm_{mr}x{nr}_b1.mlir"

    buf = StringIO()
    Printer(stream=buf, print_generic_format=True).print_op(module)
    with open(output_mlir, "w") as f:
        f.write(_sanitize_printed_mlir_for_mlir_translate(buf.getvalue()))

    if VERBOSE:
        print(f"\n  ✔ MLIR → {os.path.basename(output_mlir)}")

    # Step 4: Translate to C++
    if mlir_translate_path:
        if beta == BetaMode.B0:
            output_cpp = f"{kernels_dir}/xdsl_api_gemm_{mr}x{nr}_b0.cpp"
        elif beta == BetaMode.B1:
            output_cpp = f"{kernels_dir}/xdsl_api_gemm_{mr}x{nr}_b1.cpp"

        subprocess.run(
            [
                mlir_translate_path,
                "-allow-unregistered-dialect",
                "-mlir-to-cpp",
                output_mlir,
                "-o",
                output_cpp,
            ],
            check=True,
            stdout=subprocess.DEVNULL if not VERBOSE else None,
            stderr=subprocess.DEVNULL if not VERBOSE else None,
        )

        # Post-process: match reference signature in generated C code
        # 1. Replace size_t with int in function signature
        # 2. Add void* ctxt as first placeholder parameter
        with open(output_cpp) as f:
            cpp_code = f.read()
        lines = cpp_code.split("\n")
        for i, line in enumerate(lines):
            if "xdsl_api_gemm_RVV_" in line and "(" in line:
                # Replace size_t with int in signature line
                lines[i] = line.replace("size_t", "int")
                # Add void* as first parameter: insert after "("
                lines[i] = lines[i].replace("(", "(void*, ", 1)
                break
        cpp_code = "\n".join(lines)

        # Prepend parameter banner so user can identify file at a glance
        beta_str = "b0" if beta == BetaMode.B0 else "b1"
        vlen_bits = vlen * 32
        import datetime

        gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner = (
            f"// ============================================================\n"
            f"// xDSL Generated Kernel\n"
            f"// ------------------------------------------------------------\n"
            f"//   Generated   : {gen_time}\n"
            f"//   MR         : {mr}\n"
            f"//   NR         : {nr}\n"
            f"//   VLEN       : {vlen}  (fp32 elements per vector register)\n"
            f"//   VLEN_BITS  : {vlen_bits}  (bit-width, VLEN * 32)\n"
            f"//   Beta mode  : {beta_str}  ({'beta=0: C = alpha*A*B' if beta == BetaMode.B0 else 'beta=1: C = alpha*A*B + beta*C'})\n"
            f"// ============================================================\n\n"
        )
        cpp_code = banner + cpp_code

        with open(output_cpp, "w") as f:
            f.write(cpp_code)

        if VERBOSE:
            print(f"  ✔ C++  → {os.path.basename(output_cpp)}")

        # Add headers
        if add_header:
            add_header_to_cpp(output_cpp)
            if VERBOSE:
                print("  ✔ Headers added")

        return output_mlir, output_cpp

    return output_mlir, None


# Main
def main():
    global VERBOSE, SHOW_MLIR, SHOW_PASSES

    import argparse

    parser = argparse.ArgumentParser(description="Generate xDSL Microkernels")
    parser.add_argument(
        "--families",
        type=str,
        default="4,4",
        help="Space-separated list of MR,NR pairs (e.g., '4,4 9,9')",
    )
    parser.add_argument(
        "--vlen", type=int, default=8, help="Vector length for the generated code"
    )
    parser.add_argument(
        "--mlir-translate",
        type=str,
        default="/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025/llvm-project/build/bin/mlir-translate",
        help="Path to mlir-translate executable",
    )
    parser.add_argument(
        "--kc-profile",
        type=int,
        default=256,
        help="The block size parameter KC for profiling section",
    )
    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="Disable generating comparison with reference kernel",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug and ir output"
    )
    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True
        SHOW_MLIR = True
        SHOW_PASSES = True

    # Configuration - adjust paths for your environment
    MLIR_TRANSLATE = args.mlir_translate

    # Resolving paths relative to script location
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)  # .../tests/
    project_root = os.path.dirname(script_dir)  # .../xdsl-template/

    # Output directory relative to project root
    OUTPUT_DIR = os.path.join(project_root, "tests", "api_gen_cpp")
    KERNELS_DIR = os.path.join(OUTPUT_DIR, "kernels")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(KERNELS_DIR):
        os.makedirs(KERNELS_DIR)

    # Clean output directory first
    print("-" * 70)
    print("Cleaning output directory")
    print("-" * 70)

    vlen_bits = args.vlen * 32

    # Use glob relative to OUTPUT_DIR
    for f in glob(f"{OUTPUT_DIR}/*"):
        if os.path.isfile(f) and not os.path.basename(f).startswith(
            f"kernels_RVV_{vlen_bits}_fp32_ldx"
        ):
            os.remove(f)

    for f in glob(f"{KERNELS_DIR}/*"):
        if os.path.isfile(f):
            os.remove(f)

    # ============================================================
    # KERNEL FAMILIES - Generate all variants for edge case handling
    # ============================================================
    # Define kernel families (max MR × max NR)

    KERNEL_FAMILIES = []
    for pair in args.families.split():
        try:
            mr, nr = map(int, pair.split(","))
            KERNEL_FAMILIES.append((mr, nr))
        except ValueError:
            print(f"Warning: Invalid kernel family format '{pair}'. Expected 'MR,NR'.")

    if not KERNEL_FAMILIES:
        print("Error: No valid kernel families provided. Defaulting to 4,4.")
        KERNEL_FAMILIES = [(4, 4)]

    print(f"Using Kernel Families: {KERNEL_FAMILIES}")

    # Generate all kernel variants for each family
    KERNELS = []
    for max_mr, max_nr in KERNEL_FAMILIES:
        for mr in range(1, max_mr + 1):
            for nr in range(1, max_nr + 1):
                KERNELS.append((mr, nr))

    # Deduplicate kernels to avoid generating the same kernel multiple times
    KERNELS = sorted(list(set(KERNELS)))

    # 02022026_1613.txt

    # Create KERNEL_FAMILIES from 1,1 to 9,9

    # Part 1: mr 1-8
    # KERNEL_FAMILIES = [(mr, nr) for mr in range(1, 9) for nr in range(1, 18)]

    # Part 2: mr 9-16
    # KERNEL_FAMILIES = [(mr, nr) for mr in range(9, 17) for nr in range(1, 18)]

    # Part 3: mr 17-23
    # KERNEL_FAMILIES = [(mr, nr) for mr in range(17, 24) for nr in range(1, 18)]

    # 02022026_1613.txt
    # KERNEL_FAMILIES = [(mr, nr) for mr in range(4, 16 + 1) for nr in range(4, 16 + 1)]

    # KERNEL_FAMILIES = [(mr, nr) for mr in range(15, 24 + 1) for nr in range(15, 18 + 1)]

    # KERNELS = [
    #     (4, 4),
    #     (8, 4),
    #     (5, 4),  # MR%4=1
    #     (6, 4),  # MR%4=2
    #     (7, 4),  # MR%4=3
    #     (6, 10),
    #     (6, 8),
    #     (6, 16),
    #     (8, 16),
    #     # (10, 8), (14, 4),
    # ]

    # beta_mode = BetaMode.B1
    # beta_mode = BetaMode.B0
    beta_modes = [BetaMode.B0, BetaMode.B1]
    KERNELS = check_kernels(KERNELS, vlen=args.vlen)

    # Filter KERNEL_FAMILIES to only include those where (max_mr, max_nr) is valid
    print(f"\n{'─' * 50}")
    print(" Validating Kernel Families")
    print(f"{'─' * 50}")
    VALID_FAMILIES = []
    for max_mr, max_nr in KERNEL_FAMILIES:
        if (max_mr, max_nr) in KERNELS:
            print(f"  ({max_mr:2d}×{max_nr:2d}): ✅ Valid")
            VALID_FAMILIES.append((max_mr, max_nr))
        else:
            print(
                f"  ({max_mr:2d}×{max_nr:2d}): ❌ Max kernel invalid/skipped - Dropping Family"
            )
    KERNEL_FAMILIES = VALID_FAMILIES

    # Generate all kernels
    print(f"{'─' * 60}")
    print(f" Generating {len(KERNELS)} Kernels x {len(beta_modes)} Beta Modes")
    print(f"{'─' * 60}")

    col_count = 0
    for i, (mr, nr) in enumerate(KERNELS, 1):
        for beta_mode in beta_modes:
            run_pipeline(
                mr=mr,
                nr=nr,
                vlen=args.vlen,
                beta=beta_mode,
                output_dir=OUTPUT_DIR,
                mlir_translate_path=MLIR_TRANSLATE,
                add_header=True,
            )

        config = GEMMConfig(mr=mr, nr=nr, beta=BetaMode.B0, vlen=args.vlen)
        eff = config.register_efficiency
        if eff < 1.0:
            eff_str = f"⚠️  ({config.num_c_regs} C regs, {eff:.0%} eff)"
        else:
            eff_str = f"✅ ({config.num_c_regs} C regs)"

        item = f"[{i:2d}/{len(KERNELS)}] {mr:2d}×{nr:2d} (b0,b1) {eff_str}"
        print(f"  {item:<38}", end="", flush=True)

        col_count += 1
        if col_count == 2:
            print()
            col_count = 0

    if col_count != 0:
        print()

    # Generate support files
    print(f"{'─' * 50}")
    print(" Support Files")
    print(f"{'─' * 50}")

    print("  Profiling header... ", end="", flush=True)
    profiling_header_path = f"{OUTPUT_DIR}/gemm_profile.h"
    generate_profiling_header(profiling_header_path)
    if not VERBOSE:
        print("✅")

    # Calculate max dimensions for reference kernel name
    # Ensure local calculations match what generate_merged_kernel_file does
    if KERNELS:
        max_mr = max(k[0] for k in KERNELS)
        max_nr = max(k[1] for k in KERNELS)
        # Use the compiled reference kernel from the destination directory
        ref_kernel_base_name = f"kernels_RVV_{vlen_bits}_fp32_ldx"
    else:
        ref_kernel_base_name = f"kernels_RVV_{vlen_bits}_fp32_ldx"  # Fallback

    print("  Test... ", end="", flush=True)
    test_harness_path = f"{OUTPUT_DIR}/test_xdsl_api_gemm_kernels.cpp"
    generate_test_maincpp(
        beta_modes,
        KERNELS,
        KERNEL_FAMILIES,
        test_harness_path,
        header_name="xdsl_api_gemm.h",
        ref_kernel_name=ref_kernel_base_name,
        kc_profile=args.kc_profile,
        no_ref=args.no_ref,
    )
    if not VERBOSE:
        print("✅")

    print("  Makefile... ", end="", flush=True)
    generate_makefile(
        KERNELS,
        OUTPUT_DIR,
        beta_modes,
        ref_kernel_name=ref_kernel_base_name,
        no_ref=args.no_ref,
    )
    if not VERBOSE:
        print("✅")

    print("  Merged Kernels... ", end="", flush=True)
    generate_merged_kernel_file(
        KERNELS, OUTPUT_DIR, beta_modes, vlen_bits=vlen_bits, tests_dir=script_dir
    )
    print("  Exo Matrix Arrays... ", end="", flush=True)
    generate_exo_matrix_files(KERNELS, OUTPUT_DIR, vlen_bits, tests_dir=script_dir)
    if not VERBOSE:
        print("✅")

    # Python generation complete - compile_jbpi.sh handles the rest


if __name__ == "__main__":
    main()
