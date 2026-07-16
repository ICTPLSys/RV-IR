"""
Unified Strategy-based EmitC Code Generator

This module provides a flexible, strategy-based architecture for generating C code
from EmitC MLIR. Different GEMM execution strategies can be plugged in without
modifying the core parsing logic.

Key Design:
- Separation of concerns: Parse once, generate with different strategies
- Strategy Pattern: Each GEMM execution mode is a separate strategy
- Extensible: Easy to add new strategies without modifying existing code

"""

import re
from abc import ABC, abstractmethod

# ============================================================================
# Code Emitter
# ============================================================================


class CodeEmitter:
    """Helper class to generate C code with proper indentation"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.indent_level = 0
        self.code = []

    def emit(self, line: str):
        """Emit a line of code with proper indentation"""
        if line.strip():
            indent = "  " * self.indent_level
            self.code.append(indent + line)
        else:
            self.code.append("")

    def indent(self):
        self.indent_level += 1

    def dedent(self):
        self.indent_level -= 1

    def get_code(self) -> str:
        return "\n".join(self.code)


# ============================================================================
# EmitC Operation Parser
# ============================================================================


class EmitCOperation:
    """Base class for EmitC operations"""

    def __init__(self, line: str):
        self.line = line
        self.op_name = None
        self.parse()

    def parse(self):
        """Parse the operation from MLIR line"""
        # Extract operation name
        if '"emitc.call_opaque"' in self.line:
            callee_match = re.search(r'callee\s*=\s*"([^"]+)"', self.line)
            if callee_match:
                self.op_name = callee_match.group(1)
        elif '"emitc_ext.constant"' in self.line:
            self.op_name = "constant"
        elif '"memref.alloc"' in self.line:
            self.op_name = "alloc"
        elif '"linalg.fill"' in self.line:
            self.op_name = "fill"

    def should_ignore(self) -> bool:
        """Check if this operation should be ignored in C code generation"""
        return self.op_name in ["alloc", "fill"]

    def is_tensor_create(self) -> bool:
        """Check if this is a tensor creation operation"""
        return self.op_name in [
            "create_tensor_A",
            "create_tensor_B",
            "create_tensor_C",
            "create_tensor_transpose_in",
            "create_tensor_transpose_out",
        ]

    def is_gemm_call(self) -> bool:
        """Check if this is the gemm_operator call"""
        return self.op_name == "gemm_operator"

    def is_transpose_call(self) -> bool:
        """Check if this is the transpose_operator call"""
        return self.op_name == "transpose_operator"

    def is_rmsnorm_call(self) -> bool:
        """Check if this is the rmsnorm_operator call"""
        return self.op_name == "rmsnorm_operator"


def parse_emitc_module(mlir_content: str, verbose: bool = False) -> dict:
    """
    Parse EmitC MLIR module and extract structured information

    This is the unified parsing function used by all strategies.

    Returns:
        Dict containing:
        - func_name: function name
        - args: function arguments
        - return_type: return type
        - operations: list of operations in order
        - constants: dict of constant values
        - tensor_creates: list of tensor creation operations
        - gemm_call: the gemm_operator call (if any)
        - transpose_call: the transpose_operator call (if any)
        - rmsnorm_call: the rmsnorm_operator call (if any)
    """
    lines = mlir_content.split("\n")

    result = {
        "func_name": None,
        "args": [],
        "return_type": None,
        "operations": [],
        "constants": {},
        "tensor_creates": [],
        "gemm_call": None,
        "transpose_call": None,
        "rmsnorm_call": None,
    }

    in_function_body = False
    brace_count = 0

    for i, line in enumerate(lines):
        # Extract function signature
        if '"func.func"' in line and "sym_name" in line:
            name_match = re.search(r'sym_name = "(\w+)"', line)
            if name_match:
                result["func_name"] = name_match.group(1)

            # Extract arguments
            arg_match = re.search(r"\^bb0\(([^)]+)\)", line)
            if arg_match:
                args_str = arg_match.group(1)
                for arg in args_str.split(","):
                    arg = arg.strip()
                    if arg:
                        # Extract arg name and type
                        parts = arg.split(":")
                        if len(parts) == 2:
                            arg_name = parts[0].strip().replace("%", "")
                            arg_type = parts[1].strip()
                            result["args"].append((arg_name, arg_type))

            # Extract return type
            return_match = re.search(r"->\s*(!emitc\.ptr<[^>]+)>", line)
            if return_match:
                result["return_type"] = return_match.group(1).strip()

        # Track function body
        if "^bb0" in line:
            in_function_body = True
            continue

        if in_function_body:
            brace_count += line.count("{") - line.count("}")
            if brace_count < 0 and "})" in line:
                in_function_body = False

        # Parse operations in function body
        if in_function_body and brace_count >= 0:
            op = EmitCOperation(line)

            # Parse constants
            if '"emitc_ext.constant"' in line:
                const_match = re.search(
                    r'(%\w+)\s*=\s*"emitc_ext\.constant"\(\)\s*\{value\s*=\s*([^}]+)\}\s*:\s*\(\)\s*->\s*(\w+)',
                    line,
                )
                if const_match:
                    var_name = const_match.group(1)
                    value_str = const_match.group(2).strip()

                    # Parse value (remove type suffix if present)
                    if ": " in value_str:
                        value = value_str.split(":")[0].strip()
                    else:
                        value = value_str

                    result_type = const_match.group(3)
                    result["constants"][var_name] = (result_type, value)

            # Parse tensor creation operations
            if op.is_tensor_create():
                # Extract: %4 = "emitc.call_opaque"(%arg0, %1, %3, %0) <{callee = "create_tensor_A"}>
                # or: %4 = "emitc.call_opaque"(%arg0, %1, %3, %0) <{callee = "create_tensor_transpose_in"}>
                match = re.search(
                    r'(%\w+)\s*=\s*"emitc\.call_opaque"\(([^)]+)\)\s*<{callee\s*=\s*"create_tensor_\w+"}>',
                    line,
                )
                if match:
                    result_var = match.group(1)
                    args_str = match.group(2)

                    # Parse arguments: %arg0, %1, %3, %0
                    args_list = [
                        a.strip().replace("%", "") for a in args_str.split(",")
                    ]

                    # Extract tensor type from callee name
                    tensor_type_match = re.search(r"create_tensor_([A-Za-z_]+)", line)
                    if tensor_type_match:
                        tensor_type = tensor_type_match.group(1)

                        result["tensor_creates"].append(
                            {
                                "result_var": result_var,
                                "args": args_list,
                                "tensor_type": tensor_type,
                                "original_line": line,
                            }
                        )

            # Parse gemm_operator call
            if op.is_gemm_call():
                match = re.search(
                    r'"emitc\.call_opaque"\(([^)]+)\)\s*<{callee\s*=\s*"gemm_operator"}>',
                    line,
                )
                if match:
                    args_str = match.group(1)
                    args_list = [
                        a.strip().replace("%", "") for a in args_str.split(",")
                    ]
                    result["gemm_call"] = {
                        "args": args_list,
                        "original_line": line,
                    }

            # Parse transpose_operator call
            if op.is_transpose_call():
                match = re.search(
                    r'"emitc\.call_opaque"\(([^)]+)\)\s*<{callee\s*=\s*"transpose_operator"}>',
                    line,
                )
                if match:
                    args_str = match.group(1)
                    args_list = [
                        a.strip().replace("%", "") for a in args_str.split(",")
                    ]
                    result["transpose_call"] = {
                        "args": args_list,
                        "original_line": line,
                    }

            # Parse rmsnorm_operator call
            if op.is_rmsnorm_call():
                match = re.search(
                    r'"emitc\.call_opaque"\(([^)]+)\)\s*<{callee\s*=\s*"rmsnorm_operator"}>',
                    line,
                )
                if match:
                    args_str = match.group(1)
                    args_list = [
                        a.strip().replace("%", "") for a in args_str.split(",")
                    ]
                    result["rmsnorm_call"] = {
                        "args": args_list,
                        "original_line": line,
                    }

            result["operations"].append(op)

    if verbose:
        print(f"[DEBUG] Parsed function: {result['func_name']}")
        print(f"[DEBUG] Arguments: {result['args']}")
        print(f"[DEBUG] Constants: {result['constants']}")
        print(f"[DEBUG] Tensor creations: {len(result['tensor_creates'])}")
        print(f"[DEBUG] GEMM call: {result['gemm_call'] is not None}")

    return result


def get_constant_value(var_name: str, constants: dict) -> int:
    """Get integer value of a constant variable"""
    if var_name in constants:
        const_type, const_val = constants[var_name]
        try:
            return int(const_val)
        except:
            pass
    return 0


# ============================================================================
# Tensor Descriptor
# ============================================================================


class TensorDescriptor:
    """Describes a tensor with its dimensions and properties"""

    def __init__(self, name: str, tensor_type: str, dim0: int, dim1: int, dim2: int):
        self.name = name  # e.g., "A", "B", "C"
        self.tensor_type = tensor_type  # "A", "B", or "C"
        self.dim0 = dim0  # K for A/C, N for B
        self.dim1 = dim1  # M for A/C, K for B
        self.dim2 = dim2  # batch

    def get_min_stride_arg(self):
        """Get the dimension argument for min_stride1"""
        if self.tensor_type in ["A", "C"]:
            return self.dim0  # K or N
        else:  # B
            return self.dim0  # N


# ============================================================================
# Tensor Descriptor
# ============================================================================

# class TensorDescriptor:


class GEMMStrategy(ABC):
    """Base class for GEMM execution strategies"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_prologue(
        self, emitter: CodeEmitter, tensors: dict[str, TensorDescriptor]
    ):
        """Generate code before the main computation"""
        pass

    @abstractmethod
    def generate_tensor_init(self, emitter: CodeEmitter, tensor: TensorDescriptor):
        """Generate tensor struct initialization"""
        pass

    @abstractmethod
    def generate_computation(
        self,
        emitter: CodeEmitter,
        tensors: dict[str, TensorDescriptor],
        accumulate: int,
        activate: int,
    ):
        """Generate the main GEMM computation"""
        pass

    @abstractmethod
    def generate_epilogue(
        self, emitter: CodeEmitter, tensors: dict[str, TensorDescriptor]
    ):
        """Generate code after the main computation"""
        pass


class SimpleGEMMStrategy(GEMMStrategy):
    """Simple GEMM: direct computation without data movement"""

    def __init__(self):
        super().__init__("simple")

    def generate_prologue(
        self, emitter: CodeEmitter, tensors: dict[str, TensorDescriptor]
    ):
        emitter.emit("// Simple GEMM strategy: direct computation")
        emitter.emit("")

    def generate_tensor_init(self, emitter: CodeEmitter, tensor: TensorDescriptor):
        emitter.emit(
            f"// Tensor {tensor.name}: shape (batch={tensor.dim2}, M={tensor.dim1}, K={tensor.dim0})"
        )
        emitter.emit(
            f"int min_stride_{tensor.name} = min_stride1({tensor.get_min_stride_arg()}, WIDTH_8);"
        )
        emitter.emit(f"Tensor tensor_{tensor.name} = (Tensor){{")
        emitter.indent()
        emitter.emit(".base_addr = -1,")
        emitter.emit(f".dim0      = {tensor.dim0},")
        emitter.emit(f".dim1      = {tensor.dim1},")
        emitter.emit(f".dim2      = {tensor.dim2},")
        emitter.emit(f".byte_stride1 = min_stride_{tensor.name},")
        emitter.emit(f".byte_stride2 = min_stride_{tensor.name} * {tensor.dim1},")
        emitter.emit(".wd_data      = WIDTH_8,")
        emitter.emit(".type_data    = TYPE_INT")
        emitter.dedent()
        emitter.emit("};")
        emitter.emit("")

    def generate_computation(
        self,
        emitter: CodeEmitter,
        tensors: dict[str, TensorDescriptor],
        accumulate: int,
        activate: int,
    ):
        # Set memory addresses
        emitter.emit("// Set memory addresses for tensors")
        emitter.emit(" uint32_t start_addr = 0xa0000000;   // psram")
        if "A" in tensors:
            emitter.emit("tensor_A.base_addr = 0x90000000;  // scratchpad0")
        if "B" in tensors:
            emitter.emit("tensor_B.base_addr = 0x80000;  // CIM")
        if "C" in tensors:
            emitter.emit("tensor_C.base_addr = 0x90020000;  // scratchpad1")
        emitter.emit("")

        # Direct gemm_operator call
        emitter.emit("// Perform GEMM operation: C = A @ B")
        emitter.emit(f"// accumulate={accumulate}, activate={activate}")
        emitter.emit(
            f"gemm_operator(&tensor_A, &tensor_B, &tensor_C, &tensor_C, {accumulate}, {activate});"
        )
        emitter.emit("")

    def generate_epilogue(
        self, emitter: CodeEmitter, tensors: dict[str, TensorDescriptor]
    ):
        if "C" in tensors:
            emitter.emit("// Return pointer to output tensor C")
            emitter.emit("return (void *)tensor_C.base_addr;")


class WorkloadGEMMStrategy(GEMMStrategy):
    """Workload GEMM: with tensor_load/tensor_store for data movement"""

    def __init__(self, num_iterations: int = 100):
        super().__init__("workload")
        self.num_iterations = num_iterations

    def generate_prologue(
        self, emitter: CodeEmitter, tensors: dict[str, TensorDescriptor]
    ):
        emitter.emit(
            f"// Workload GEMM strategy: {self.num_iterations} iterations with data movement"
        )
        emitter.emit("")

        # Calculate PSRAM addresses
        emitter.emit("uint32_t start_addr = 0xa0000000; // PSRAM base")
        emitter.emit("uint32_t spad_A_addr = 0x90000000;  // SPAD0 for A block")
        emitter.emit("uint32_t spad_B_addr = 0x80000;     // CIM for B block")
        emitter.emit("uint32_t spad_C_addr = 0x90020000;  // SPAD1 for C block")
        emitter.emit("")

    def generate_tensor_init(self, emitter: CodeEmitter, tensor: TensorDescriptor):
        # Generate PSRAM tensor (full tensor)
        emitter.emit(
            f"// Tensor {tensor.name}: shape (batch={tensor.dim2}, M={tensor.dim1}, K={tensor.dim0})"
        )
        emitter.emit(
            f"int min_stride_{tensor.name} = min_stride1({tensor.get_min_stride_arg()}, WIDTH_8);"
        )
        emitter.emit(f"Tensor tensor_{tensor.name} = (Tensor){{")
        emitter.indent()
        emitter.emit(".base_addr = -1,")
        emitter.emit(f".dim0      = {tensor.dim0},")
        emitter.emit(f".dim1      = {tensor.dim1},")
        emitter.emit(f".dim2      = {tensor.dim2},")
        emitter.emit(f".byte_stride1 = min_stride_{tensor.name},")
        emitter.emit(f".byte_stride2 = min_stride_{tensor.name} * {tensor.dim1},")
        emitter.emit(".wd_data      = WIDTH_8,")
        emitter.emit(".type_data    = TYPE_INT")
        emitter.dedent()
        emitter.emit("};")
        emitter.emit("")

    def generate_computation(
        self,
        emitter: CodeEmitter,
        tensors: dict[str, TensorDescriptor],
        accumulate: int,
        activate: int,
    ):
        # Calculate tensor sizes and set PSRAM addresses
        emitter.emit("// Allocate memory in PSRAM for all matrices")
        for name in ["A", "B", "C"]:
            if name in tensors:
                emitter.emit(
                    f"uint32_t tensor_size_{name} = getTensorSize(&tensor_{name});"
                )
        emitter.emit("")

        if "A" in tensors:
            emitter.emit("tensor_A.base_addr = start_addr;")
        if "B" in tensors:
            emitter.emit("tensor_B.base_addr = start_addr + tensor_size_A;")
        if "C" in tensors:
            emitter.emit(
                "tensor_C.base_addr = start_addr + tensor_size_A + tensor_size_B;"
            )
        emitter.emit(
            "start_addr = start_addr + tensor_size_A + tensor_size_B + tensor_size_C;"
        )
        emitter.emit("")

        # Generate computation loop
        emitter.emit(f"for (int i = 0; i < {self.num_iterations}; i++) {{")
        emitter.indent()

        # Create block tensors in scratchpad
        for name in ["A", "B", "C"]:
            if name not in tensors:
                continue
            tensor = tensors[name]
            spad_addr = f"spad_{name}_addr"
            emitter.emit(f"// Create block tensor for {name}")
            emitter.emit(f"Tensor tensor_{name}_block = (Tensor){{")
            emitter.indent()
            emitter.emit(f".base_addr = {spad_addr},")
            emitter.emit(f".dim0      = {tensor.dim0},")
            emitter.emit(f".dim1      = {tensor.dim1},")
            emitter.emit(f".dim2      = {tensor.dim2},")
            emitter.emit(f".byte_stride1 = min_stride_{name},")
            emitter.emit(f".byte_stride2 = min_stride_{name} * {tensor.dim1},")
            emitter.emit(".wd_data      = WIDTH_8,")
            emitter.emit(".type_data    = TYPE_INT")
            emitter.dedent()
            emitter.emit("};")
            emitter.emit("")

        # Load data
        emitter.emit("// Load A block from PSRAM to SPAD0")
        emitter.emit("tensor_load(&tensor_A, &tensor_A_block);")
        emitter.emit("")
        emitter.emit("// Load B block from PSRAM to CIM")
        emitter.emit("tensor_load(&tensor_B, &tensor_B_block);")
        emitter.emit("")

        # Perform GEMM
        emitter.emit("// Perform block GEMM: C_block += A_block @ B_block")
        emitter.emit(
            f"gemm_operator(&tensor_A_block, &tensor_B_block, &tensor_C_block, "
            f"&tensor_C_block, {accumulate}, {activate});"
        )
        emitter.emit("")

        # Store result
        emitter.emit("// Store C block back to PSRAM")
        emitter.emit("tensor_store(&tensor_C_block, &tensor_C);")

        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

    def generate_epilogue(
        self, emitter: CodeEmitter, tensors: dict[str, TensorDescriptor]
    ):
        emitter.emit("return (void *)tensor_C.base_addr;")


class BlockedGEMMStrategy(GEMMStrategy):
    """Blocked GEMM: with tiling for large matrices"""

    def __init__(self, block_m: int = 32, block_n: int = 32, block_k: int = 32):
        super().__init__("blocked")
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

    def generate_prologue(
        self, emitter: CodeEmitter, tensors: dict[str, TensorDescriptor]
    ):
        emitter.emit(
            f"// Blocked GEMM strategy: tiling with block sizes "
            f"(block_m={self.block_m}, block_n={self.block_n}, block_k={self.block_k})"
        )
        emitter.emit("")

        # Get dimensions
        if "A" in tensors:
            m = tensors["A"].dim1
            k = tensors["A"].dim0
            n = tensors["B"].dim0 if "B" in tensors else 32

            emitter.emit("// Block sizes for tiling")
            emitter.emit(f"uint32_t block_m = {self.block_m};")
            emitter.emit(f"uint32_t block_n = {self.block_n};")
            emitter.emit(f"uint32_t block_k = {self.block_k};")
            emitter.emit("")

            # Scratchpad addresses
            emitter.emit(" uint32_t start_addr = 0xa0000000;   // psram")
            emitter.emit("uint32_t spad_A_addr = 0x90000000;  // SPAD0 for A block")
            emitter.emit("uint32_t spad_B_addr = 0x80000;     // CIM for B block")
            emitter.emit("uint32_t spad_C_addr = 0x90020000;  // SPAD1 for C block")
            emitter.emit("")

    def generate_tensor_init(self, emitter: CodeEmitter, tensor: TensorDescriptor):
        emitter.emit(
            f"// Tensor {tensor.name}: shape (batch={tensor.dim2}, M={tensor.dim1}, K={tensor.dim0})"
        )
        emitter.emit(
            f"int min_stride_{tensor.name} = min_stride1({tensor.get_min_stride_arg()}, WIDTH_8);"
        )
        emitter.emit(f"Tensor tensor_{tensor.name} = (Tensor){{")
        emitter.indent()
        emitter.emit(".base_addr = -1,")
        emitter.emit(f".dim0      = {tensor.dim0},")
        emitter.emit(f".dim1      = {tensor.dim1},")
        emitter.emit(f".dim2      = {tensor.dim2},")
        emitter.emit(f".byte_stride1 = min_stride_{tensor.name},")
        emitter.emit(f".byte_stride2 = min_stride_{tensor.name} * {tensor.dim1},")
        emitter.emit(".wd_data      = WIDTH_8,")
        emitter.emit(".type_data    = TYPE_INT")
        emitter.dedent()
        emitter.emit("};")
        emitter.emit("")

    def generate_computation(
        self,
        emitter: CodeEmitter,
        tensors: dict[str, TensorDescriptor],
        accumulate: int,
        activate: int,
    ):
        if "A" not in tensors or "B" not in tensors or "C" not in tensors:
            return

        m = tensors["A"].dim1
        k = tensors["A"].dim0
        n = tensors["B"].dim0

        # Calculate PSRAM addresses
        emitter.emit("// Allocate memory in PSRAM for all matrices")
        emitter.emit("uint32_t tensor_size_A = getTensorSize(&tensor_A);")
        emitter.emit("uint32_t tensor_size_B = getTensorSize(&tensor_B);")
        emitter.emit("uint32_t tensor_size_C = getTensorSize(&tensor_C);")
        emitter.emit("")
        emitter.emit("tensor_A.base_addr = start_addr;")
        emitter.emit("tensor_B.base_addr = start_addr + tensor_size_A;")
        emitter.emit("tensor_C.base_addr = start_addr + tensor_size_A + tensor_size_B;")
        emitter.emit("")

        # Generate nested loops for blocking
        emitter.emit("// Loop over output blocks in C")
        emitter.emit(f"for (uint32_t bm = 0; bm < {m}; bm += block_m) {{")
        emitter.indent()
        emitter.emit(
            f"uint32_t cur_block_m = (bm + block_m <= {m}) ? block_m : ({m} - bm);"
        )
        emitter.emit("")
        emitter.emit(f"for (uint32_t bn = 0; bn < {n}; bn += block_n) {{")
        emitter.indent()
        emitter.emit(
            f"uint32_t cur_block_n = (bn + block_n <= {n}) ? block_n : ({n} - bn);"
        )
        emitter.emit("")
        emitter.emit("// Initialize accumulation for this output block")
        emitter.emit("uint32_t accumulate_flag = 0;")
        emitter.emit("")

        emitter.emit("int min_stride_C = min_stride1(cur_block_n, WIDTH_8);")
        emitter.emit("Tensor tensor_C_block = (Tensor){")
        emitter.indent()
        emitter.emit(".base_addr = spad_C_addr,")
        emitter.emit(".dim0      = cur_block_n,")
        emitter.emit(".dim1      = cur_block_m,")
        emitter.emit(".dim2      = 1,")
        emitter.emit(".byte_stride1 = min_stride_C,")
        emitter.emit(".byte_stride2 = min_stride_C * cur_block_m,")
        emitter.emit(".wd_data      = WIDTH_8,")
        emitter.emit(".type_data    = TYPE_INT")
        emitter.dedent()
        emitter.emit("};")
        emitter.emit("")

        emitter.emit("// Loop over k dimension (reduction dimension)")
        emitter.emit(f"for (uint32_t bk = 0; bk < {k}; bk += block_k) {{")
        emitter.indent()
        emitter.emit(
            f"uint32_t cur_block_k = (bk + block_k <= {k}) ? block_k : ({k} - bk);"
        )
        emitter.emit("")

        # Create block tensors
        emitter.emit("// Create block tensors")
        emitter.emit("int min_stride_A = min_stride1(cur_block_k, WIDTH_8);")
        emitter.emit("Tensor tensor_A_block = (Tensor){")
        emitter.indent()
        emitter.emit(".base_addr = spad_A_addr,")
        emitter.emit(".dim0      = cur_block_k,")
        emitter.emit(".dim1      = cur_block_m,")
        emitter.emit(".dim2      = 1,")
        emitter.emit(".byte_stride1 = min_stride_A,")
        emitter.emit(".byte_stride2 = min_stride_A * cur_block_m,")
        emitter.emit(".wd_data      = WIDTH_8,")
        emitter.emit(".type_data    = TYPE_INT")
        emitter.dedent()
        emitter.emit("};")
        emitter.emit("")

        emitter.emit("int min_stride_B = min_stride1(cur_block_n, WIDTH_8);")
        emitter.emit("Tensor tensor_B_block = (Tensor){")
        emitter.indent()
        emitter.emit(".base_addr = spad_B_addr,")
        emitter.emit(".dim0      = cur_block_n,")
        emitter.emit(".dim1      = cur_block_k,")
        emitter.emit(".dim2      = 1,")
        emitter.emit(".byte_stride1 = min_stride_B,")
        emitter.emit(".byte_stride2 = min_stride_B * cur_block_k,")
        emitter.emit(".wd_data      = WIDTH_8,")
        emitter.emit(".type_data    = TYPE_INT")
        emitter.dedent()
        emitter.emit("};")
        emitter.emit("")

        # Create tensor views (simplified - actual offset calculation would be here)
        emitter.emit("// Create views of blocks from full tensors")
        emitter.emit("Tensor tensor_A_view = tensor_A;")
        emitter.emit("tensor_A_view.base_addr += bm * tensor_A.byte_stride1 + bk * 8;")
        emitter.emit("tensor_A_view.dim0 = cur_block_k;")
        emitter.emit("tensor_A_view.dim1 = cur_block_m;")
        emitter.emit("")
        emitter.emit("Tensor tensor_B_view = tensor_B;")
        emitter.emit("tensor_B_view.base_addr += bn * 8 + bk * tensor_B.byte_stride1;")
        emitter.emit("tensor_B_view.dim0 = cur_block_n;")
        emitter.emit("tensor_B_view.dim1 = cur_block_k;")
        emitter.emit("")
        emitter.emit("Tensor tensor_C_view = tensor_C;")
        emitter.emit("tensor_C_view.base_addr += bm * tensor_C.byte_stride1 + bn * 8;")
        emitter.emit("tensor_C_view.dim0 = cur_block_n;")
        emitter.emit("tensor_C_view.dim1 = cur_block_m;")
        emitter.emit("")

        # Load and compute
        emitter.emit("// Load blocks from PSRAM to scratchpad")
        emitter.emit("tensor_load(&tensor_A_view, &tensor_A_block);")
        emitter.emit("tensor_load(&tensor_B_view, &tensor_B_block);")
        emitter.emit("")
        emitter.emit("// Perform block GEMM")
        emitter.emit(
            "gemm_operator(&tensor_A_block, &tensor_B_block, &tensor_C_block, "
            f"&tensor_C_block, accumulate_flag, {activate});"
        )
        emitter.emit("")
        emitter.emit("accumulate_flag = 1;")

        emitter.dedent()
        emitter.emit("}")  # end bk loop
        emitter.emit("")

        # Store result
        emitter.emit("// Store final C block back to PSRAM")
        emitter.emit("Tensor tensor_C_view_final = tensor_C;")
        emitter.emit(
            "tensor_C_view_final.base_addr += bm * tensor_C.byte_stride1 + bn * 8;"
        )
        emitter.emit("tensor_C_view_final.dim0 = cur_block_n;")
        emitter.emit("tensor_C_view_final.dim1 = cur_block_m;")
        emitter.emit("")
        emitter.emit("tensor_store(&tensor_C_block, &tensor_C_view_final);")

        emitter.dedent()
        emitter.emit("}")  # end bn loop
        emitter.dedent()
        emitter.emit("}")  # end bm loop
        emitter.emit("")

    def generate_epilogue(
        self, emitter: CodeEmitter, tensors: dict[str, TensorDescriptor]
    ):
        emitter.emit("return (void *)tensor_C.base_addr;")


class GemminiStrategy(GEMMStrategy):
    """Gemmini accelerator strategy: generates complete standalone C programs
    using Berkeley Gemmini low-level ISA (mvin/compute/mvout).

    The generated code is a self-contained baremetal program that can be
    compiled and run with ``spike --extension=gemmini``.
    """

    GEMMINI_DIM = 16
    ADDR_LEN = 32
    BANK_NUM = 4
    BANK_ROWS = 4096

    def __init__(self):
        super().__init__("gemmini")

    @property
    def is_standalone_program(self) -> bool:
        return True

    def generate_prologue(self, emitter, tensors):
        pass

    def generate_tensor_init(self, emitter, tensor):
        pass

    def generate_computation(self, emitter, tensors, accumulate, activate):
        pass

    def generate_epilogue(self, emitter, tensors):
        pass

    def generate_full_program(
        self,
        tensors: dict[str, "TensorDescriptor"],
        func_name: str | None = None,
    ) -> str:
        DIM = self.GEMMINI_DIM
        emitter = CodeEmitter()

        M = tensors["A"].dim1 if "A" in tensors else DIM
        K = tensors["A"].dim0 if "A" in tensors else DIM
        N = tensors["B"].dim0 if "B" in tensors else DIM

        emitter.emit("// Auto-generated Gemmini baremetal program from RAIR MLIR")
        emitter.emit(f"// matmul: C[{M}x{N}] = A[{M}x{K}] * B[{K}x{N}]")
        emitter.emit("")
        emitter.emit("#include <stdint.h>")
        emitter.emit("#include <stddef.h>")
        emitter.emit("#include <stdlib.h>")
        emitter.emit("#include <stdio.h>")
        emitter.emit("#ifndef BAREMETAL")
        emitter.emit("#include <sys/mman.h>")
        emitter.emit("#endif")
        emitter.emit('#include "include/gemmini_testutils.h"')
        emitter.emit("")
        emitter.emit(f"#define MAT_DIM_I {M}")
        emitter.emit(f"#define MAT_DIM_K {K}")
        emitter.emit(f"#define MAT_DIM_J {N}")
        emitter.emit("")

        emitter.emit("static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);")
        emitter.emit("static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);")
        emitter.emit("static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);")
        emitter.emit("static elem_t C_gold[MAT_DIM_I][MAT_DIM_J];")
        emitter.emit("")

        self._emit_helper_functions(emitter)

        emitter.emit("int main(void) {")
        emitter.indent()

        emitter.emit("#ifndef BAREMETAL")
        emitter.emit("if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {")
        emitter.indent()
        emitter.emit('perror("mlockall failed");')
        emitter.emit("exit(1);")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("#endif")
        emitter.emit("")

        emitter.emit('printf("Gemmini matmul: C[%d,%d] = A[%d,%d] * B[%d,%d]\\n",')
        emitter.emit("    MAT_DIM_I, MAT_DIM_J, MAT_DIM_I, MAT_DIM_K, MAT_DIM_K, MAT_DIM_J);")
        emitter.emit("")

        self._emit_init_matrices(emitter)
        self._emit_cpu_reference(emitter)

        emitter.emit("gemmini_flush(0);")
        emitter.emit("")
        emitter.emit("uint64_t start = read_cycles();")
        emitter.emit("")

        if M <= DIM and K <= DIM and N <= DIM:
            self._emit_single_tile_matmul(emitter, M, K, N)
        else:
            self._emit_tiled_matmul(emitter, M, K, N)

        emitter.emit("gemmini_fence();")
        emitter.emit("")
        emitter.emit("uint64_t end = read_cycles();")
        emitter.emit('printf("Gemmini matmul took %llu cycles\\n", end - start);')
        emitter.emit("")

        self._emit_verification(emitter)

        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

        return emitter.get_code()

    def _emit_helper_functions(self, emitter: CodeEmitter) -> None:
        emitter.emit("static elem_t saturate(full_t x) {")
        emitter.indent()
        emitter.emit("#ifndef ELEM_T_IS_FLOAT")
        emitter.emit("if (x > elem_t_max) return elem_t_max;")
        emitter.emit("if (x < elem_t_min) return elem_t_min;")
        emitter.emit("#endif")
        emitter.emit("return (elem_t)x;")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

    def _emit_init_matrices(self, emitter: CodeEmitter) -> None:
        emitter.emit("for (size_t i = 0; i < MAT_DIM_I; i++)")
        emitter.indent()
        emitter.emit("for (size_t k = 0; k < MAT_DIM_K; k++)")
        emitter.indent()
        emitter.emit("A[i][k] = (elem_t)((i + 2 * k) % 5 - 2);")
        emitter.dedent()
        emitter.dedent()
        emitter.emit("")

        emitter.emit("for (size_t k = 0; k < MAT_DIM_K; k++)")
        emitter.indent()
        emitter.emit("for (size_t j = 0; j < MAT_DIM_J; j++)")
        emitter.indent()
        emitter.emit("B[k][j] = (elem_t)((3 * k + j) % 7 - 3);")
        emitter.dedent()
        emitter.dedent()
        emitter.emit("")

        emitter.emit("for (size_t i = 0; i < MAT_DIM_I; i++)")
        emitter.indent()
        emitter.emit("for (size_t j = 0; j < MAT_DIM_J; j++)")
        emitter.indent()
        emitter.emit("C[i][j] = 0;")
        emitter.dedent()
        emitter.dedent()
        emitter.emit("")

    def _emit_cpu_reference(self, emitter: CodeEmitter) -> None:
        emitter.emit("uint64_t cpu_start = read_cycles();")
        emitter.emit("for (size_t i = 0; i < MAT_DIM_I; i++) {")
        emitter.indent()
        emitter.emit("for (size_t j = 0; j < MAT_DIM_J; j++) {")
        emitter.indent()
        emitter.emit("full_t sum = 0;")
        emitter.emit("for (size_t k = 0; k < MAT_DIM_K; k++)")
        emitter.indent()
        emitter.emit("sum += (full_t)A[i][k] * (full_t)B[k][j];")
        emitter.dedent()
        emitter.emit("C_gold[i][j] = saturate(sum);")
        emitter.dedent()
        emitter.emit("}")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("uint64_t cpu_end = read_cycles();")
        emitter.emit('printf("CPU reference took %llu cycles\\n", cpu_end - cpu_start);')
        emitter.emit("")

    def _emit_single_tile_matmul(self, emitter: CodeEmitter, M: int, K: int, N: int) -> None:
        """Emit code for a matmul where all dimensions fit in one DIM×DIM tile."""
        DIM = self.GEMMINI_DIM
        emitter.emit(f"// Single-tile matmul ({M}x{K} * {K}x{N}), fits in DIM={DIM}")
        emitter.emit("")

        emitter.emit("// Scratchpad addresses")
        emitter.emit("size_t A_sp_addr = 0;")
        emitter.emit("size_t B_sp_addr = DIM;")
        emitter.emit("size_t C_sp_addr = 2 * DIM;")
        emitter.emit("")

        emitter.emit("// Configure load stride and move A, B into scratchpad")
        emitter.emit("gemmini_config_ld(MAT_DIM_K * sizeof(elem_t));")
        emitter.emit("gemmini_mvin(A, A_sp_addr);")
        emitter.emit("")
        emitter.emit("gemmini_config_ld(MAT_DIM_J * sizeof(elem_t));")
        emitter.emit("gemmini_mvin(B, B_sp_addr);")
        emitter.emit("")

        emitter.emit("// Configure execution: output-stationary, no activation, no shift")
        emitter.emit("gemmini_config_ex(OUTPUT_STATIONARY, 0, 0);")
        emitter.emit("")

        emitter.emit("// Preload zeros (no bias) and compute C = A * B")
        emitter.emit("gemmini_preload_zeros(C_sp_addr);")
        emitter.emit("gemmini_compute_preloaded(A_sp_addr, B_sp_addr);")
        emitter.emit("")

        emitter.emit("// Move result from scratchpad to main memory")
        emitter.emit("gemmini_config_st(MAT_DIM_J * sizeof(elem_t));")
        emitter.emit("gemmini_mvout(C, C_sp_addr);")
        emitter.emit("")

    def _emit_tiled_matmul(self, emitter: CodeEmitter, M: int, K: int, N: int) -> None:
        """Emit code for a matmul with tiling over DIM-sized blocks.

        Follows the official ``sp_tiled_matmul_os`` pattern from gemmini.h:
        all A/B tiles are loaded into scratchpad upfront (A from the bottom,
        B from the top), then compute iterates over tile indices referencing
        the pre-loaded addresses.  Results accumulate in the accumulator
        address space and are moved out at the end.
        """
        DIM = self.GEMMINI_DIM
        ADDR_LEN = self.ADDR_LEN
        BANK_NUM = self.BANK_NUM
        BANK_ROWS = self.BANK_ROWS

        I = (M + DIM - 1) // DIM
        J = (N + DIM - 1) // DIM
        K_tiles = (K + DIM - 1) // DIM

        pad_I = I * DIM - M
        pad_J = J * DIM - N
        pad_K = K_tiles * DIM - K

        emitter.emit(f"// Tiled matmul ({M}x{K} * {K}x{N}), DIM={DIM}")
        emitter.emit(f"// Tile grid: I={I}, J={J}, K={K_tiles}  "
                     f"(pad_I={pad_I}, pad_J={pad_J}, pad_K={pad_K})")
        emitter.emit("// Following sp_tiled_matmul_os addressing: "
                     "A from bottom, B from top of scratchpad")
        emitter.emit("")

        emitter.emit(f"const size_t I = {I};")
        emitter.emit(f"const size_t J = {J};")
        emitter.emit(f"const size_t K_tiles = {K_tiles};")
        emitter.emit("")

        emitter.emit("// Scratchpad address layout (matches sp_tiled_matmul_os)")
        emitter.emit("const uint32_t A_sp_addr_start = 0;")
        emitter.emit(f"const uint32_t B_sp_addr_start = "
                     f"{BANK_NUM} * {BANK_ROWS} - K_tiles * J * DIM;")
        emitter.emit(f"const uint32_t C_sp_addr_start = "
                     f"(3u << ({ADDR_LEN} - 2));")
        emitter.emit("")

        emitter.emit("gemmini_config_ex(OUTPUT_STATIONARY, 0, 0);")
        emitter.emit("")

        # --- Move-in B ---
        emitter.emit("// ---- Move-in B (all tiles) ----")
        emitter.emit("gemmini_config_ld(MAT_DIM_J * sizeof(elem_t));")
        emitter.emit("for (size_t k = 0; k < K_tiles; k++) {")
        emitter.indent()
        emitter.emit("for (size_t j = 0; j < J; j++) {")
        emitter.indent()
        emitter.emit("const uint32_t B_sp_addr = B_sp_addr_start + (k * J + j) * DIM;")
        emitter.emit(f"size_t cols = DIM - (j == J - 1 ? {pad_J} : 0);")
        emitter.emit(f"size_t rows = DIM - (k == K_tiles - 1 ? {pad_K} : 0);")
        emitter.emit("gemmini_extended_mvin(&B[k * DIM][j * DIM], B_sp_addr, cols, rows);")
        emitter.dedent()
        emitter.emit("}")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

        # --- Move-in A ---
        emitter.emit("// ---- Move-in A (all tiles) ----")
        emitter.emit("gemmini_config_ld(MAT_DIM_K * sizeof(elem_t));")
        emitter.emit("for (size_t i = 0; i < I; i++) {")
        emitter.indent()
        emitter.emit("for (size_t k = 0; k < K_tiles; k++) {")
        emitter.indent()
        emitter.emit("const uint32_t A_sp_addr = A_sp_addr_start + (i * K_tiles + k) * DIM;")
        emitter.emit(f"size_t cols = DIM - (k == K_tiles - 1 ? {pad_K} : 0);")
        emitter.emit(f"size_t rows = DIM - (i == I - 1 ? {pad_I} : 0);")
        emitter.emit("gemmini_extended_mvin(&A[i * DIM][k * DIM], A_sp_addr, cols, rows);")
        emitter.dedent()
        emitter.emit("}")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

        # --- Compute ---
        emitter.emit("// ---- Compute C = A * B (output-stationary) ----")
        emitter.emit("for (size_t i = 0; i < I; i++) {")
        emitter.indent()
        emitter.emit("for (size_t j = 0; j < J; j++) {")
        emitter.indent()
        emitter.emit("const uint32_t C_sp_addr = C_sp_addr_start + (i * J + j) * DIM;")
        emitter.emit("")
        emitter.emit("for (size_t k = 0; k < K_tiles; k++) {")
        emitter.indent()
        emitter.emit("const uint32_t A_sp_addr = A_sp_addr_start + (i * K_tiles + k) * DIM;")
        emitter.emit("const uint32_t B_sp_addr = B_sp_addr_start + (k * J + j) * DIM;")
        emitter.emit("")

        emitter.emit(f"size_t A_cols = DIM - (k == K_tiles - 1 ? {pad_K} : 0);")
        emitter.emit(f"size_t A_rows = DIM - (i == I - 1 ? {pad_I} : 0);")
        emitter.emit(f"size_t B_cols = DIM - (j == J - 1 ? {pad_J} : 0);")
        emitter.emit(f"size_t B_rows = DIM - (k == K_tiles - 1 ? {pad_K} : 0);")
        emitter.emit(f"size_t C_cols = DIM - (j == J - 1 ? {pad_J} : 0);")
        emitter.emit(f"size_t C_rows = DIM - (i == I - 1 ? {pad_I} : 0);")
        emitter.emit("")

        emitter.emit("// Last k-tile outputs to C_sp_addr; others to GARBAGE_ADDR")
        emitter.emit("uint32_t out_sp_addr = (k == K_tiles - 1) ? C_sp_addr : GARBAGE_ADDR;")
        emitter.emit("")

        emitter.emit("gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, "
                     "DIM, DIM, C_cols, C_rows);")
        emitter.emit("")

        emitter.emit("if (k == 0) {")
        emitter.indent()
        emitter.emit("gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, "
                     "A_cols, A_rows, B_cols, B_rows);")
        emitter.dedent()
        emitter.emit("} else {")
        emitter.indent()
        emitter.emit("gemmini_extended_compute_accumulated(A_sp_addr, B_sp_addr, "
                     "A_cols, A_rows, B_cols, B_rows);")
        emitter.dedent()
        emitter.emit("}")

        emitter.dedent()
        emitter.emit("}  // k")
        emitter.dedent()
        emitter.emit("}  // j")
        emitter.dedent()
        emitter.emit("}  // i")
        emitter.emit("")

        # --- Move-out C ---
        emitter.emit("// ---- Move-out C (all tiles) ----")
        emitter.emit("gemmini_config_st(MAT_DIM_J * sizeof(elem_t));")
        emitter.emit("for (size_t i = 0; i < I; i++) {")
        emitter.indent()
        emitter.emit("for (size_t j = 0; j < J; j++) {")
        emitter.indent()
        emitter.emit("const uint32_t C_sp_addr = C_sp_addr_start + (i * J + j) * DIM;")
        emitter.emit(f"size_t C_cols = DIM - (j == J - 1 ? {pad_J} : 0);")
        emitter.emit(f"size_t C_rows = DIM - (i == I - 1 ? {pad_I} : 0);")
        emitter.emit("gemmini_extended_mvout(&C[i * DIM][j * DIM], C_sp_addr, C_cols, C_rows);")
        emitter.dedent()
        emitter.emit("}")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

    def _emit_verification(self, emitter: CodeEmitter) -> None:
        emitter.emit("int pass = 1;")
        emitter.emit("for (size_t i = 0; i < MAT_DIM_I; i++) {")
        emitter.indent()
        emitter.emit("for (size_t j = 0; j < MAT_DIM_J; j++) {")
        emitter.indent()
        emitter.emit("if (C[i][j] != C_gold[i][j]) {")
        emitter.indent()
        emitter.emit('printf("MISMATCH at C[%u][%u]: got %d, expected %d\\n",')
        emitter.emit("    (unsigned)i, (unsigned)j, C[i][j], C_gold[i][j]);")
        emitter.emit("pass = 0;")
        emitter.dedent()
        emitter.emit("}")
        emitter.dedent()
        emitter.emit("}")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")
        emitter.emit("if (pass) {")
        emitter.indent()
        emitter.emit('printf("PASSED\\n");')
        emitter.emit("exit(0);")
        emitter.dedent()
        emitter.emit("} else {")
        emitter.indent()
        emitter.emit('printf("FAILED\\n");')
        emitter.emit("exit(1);")
        emitter.dedent()
        emitter.emit("}")

    def generate_full_program_auto(
        self,
        tensors: dict[str, "TensorDescriptor"],
        func_name: str | None = None,
    ) -> str:
        """Generate a Gemmini program using the high-level ``tiled_matmul_auto`` API.

        This produces simpler code than ``generate_full_program`` because Gemmini's
        runtime handles all tiling, double-buffering, and scratchpad management
        internally.  Works for arbitrary matrix sizes.
        """
        DIM = self.GEMMINI_DIM
        emitter = CodeEmitter()

        M = tensors["A"].dim1 if "A" in tensors else DIM
        K = tensors["A"].dim0 if "A" in tensors else DIM
        N = tensors["B"].dim0 if "B" in tensors else DIM

        emitter.emit("// Auto-generated Gemmini program from RAIR MLIR (tiled_matmul_auto mode)")
        emitter.emit(f"// matmul: C[{M}x{N}] = A[{M}x{K}] * B[{K}x{N}]")
        emitter.emit("")
        emitter.emit("#include <stdint.h>")
        emitter.emit("#include <stddef.h>")
        emitter.emit("#include <stdlib.h>")
        emitter.emit("#include <stdio.h>")
        emitter.emit("#ifndef BAREMETAL")
        emitter.emit("#include <sys/mman.h>")
        emitter.emit("#endif")
        emitter.emit('#include "include/gemmini_testutils.h"')
        emitter.emit("")
        emitter.emit(f"#define MAT_DIM_I {M}")
        emitter.emit(f"#define MAT_DIM_K {K}")
        emitter.emit(f"#define MAT_DIM_J {N}")
        emitter.emit("")

        emitter.emit("static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);")
        emitter.emit("static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);")
        emitter.emit("static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);")
        emitter.emit("static elem_t C_gold[MAT_DIM_I][MAT_DIM_J] row_align(1);")
        emitter.emit("")

        self._emit_helper_functions(emitter)

        emitter.emit("int main(void) {")
        emitter.indent()

        emitter.emit("#ifndef BAREMETAL")
        emitter.emit("if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {")
        emitter.indent()
        emitter.emit('perror("mlockall failed");')
        emitter.emit("exit(1);")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("#endif")
        emitter.emit("")

        emitter.emit('printf("Gemmini matmul: C[%d,%d] = A[%d,%d] * B[%d,%d]\\n",')
        emitter.emit("    MAT_DIM_I, MAT_DIM_J, MAT_DIM_I, MAT_DIM_K, MAT_DIM_K, MAT_DIM_J);")
        emitter.emit("")

        self._emit_init_matrices(emitter)
        self._emit_cpu_reference(emitter)

        emitter.emit("gemmini_flush(0);")
        emitter.emit("")
        emitter.emit("uint64_t start = read_cycles();")
        emitter.emit("")

        emitter.emit("tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,")
        emitter.emit("    (elem_t*)A, (elem_t*)B, NULL, (elem_t*)C,")
        emitter.emit("    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,")
        emitter.emit("    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,")
        emitter.emit("    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,")
        emitter.emit("    false, false,")
        emitter.emit("    false, false,")
        emitter.emit("    0,")
        emitter.emit("    WS);")
        emitter.emit("")

        emitter.emit("gemmini_fence();")
        emitter.emit("")
        emitter.emit("uint64_t end = read_cycles();")
        emitter.emit('printf("Gemmini matmul took %llu cycles\\n", end - start);')
        emitter.emit("")

        self._emit_verification(emitter)

        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

        return emitter.get_code()

    def generate_multi_layer_program(
        self,
        layers: list[tuple[int, int, int]],
        mode: str = "auto",
    ) -> str:
        """Generate a Gemmini program for a multi-layer MLP.

        Each entry in *layers* is ``(M, K, N)`` for one matmul.  The output of
        layer *i* is fed as the input of layer *i+1* (chain constraint:
        ``layers[i] N == layers[i+1] K`` and rows match).
        """
        DIM = self.GEMMINI_DIM
        emitter = CodeEmitter()
        num_layers = len(layers)

        emitter.emit("// Auto-generated Gemmini multi-layer MLP from RAIR MLIR")
        emitter.emit(f"// {num_layers} layers:")
        for idx, (M, K, N) in enumerate(layers):
            emitter.emit(f"//   layer {idx}: [{M}x{K}] * [{K}x{N}] -> [{M}x{N}]")
        emitter.emit("")

        emitter.emit("#include <stdint.h>")
        emitter.emit("#include <stddef.h>")
        emitter.emit("#include <stdlib.h>")
        emitter.emit("#include <stdio.h>")
        emitter.emit("#ifndef BAREMETAL")
        emitter.emit("#include <sys/mman.h>")
        emitter.emit("#endif")
        emitter.emit('#include "include/gemmini.h"')
        emitter.emit('#include "include/gemmini_nn.h"')
        emitter.emit('#include "include/gemmini_testutils.h"')
        emitter.emit("")

        # Collect unique matrix shapes
        # We need: input to layer 0, weights for each layer, intermediates, final output
        # Layer i: out_i[M_i x N_i] = in_i[M_i x K_i] * W_i[K_i x N_i]
        # in_0 = input, in_{i+1} = out_i

        for idx, (M, K, N) in enumerate(layers):
            emitter.emit(f"#define L{idx}_DIM_I {M}")
            emitter.emit(f"#define L{idx}_DIM_K {K}")
            emitter.emit(f"#define L{idx}_DIM_J {N}")
        emitter.emit("")

        # Declare matrices
        M0, K0, _ = layers[0]
        emitter.emit(f"static elem_t input_mat[L0_DIM_I][L0_DIM_K] row_align(1);")
        for idx, (M, K, N) in enumerate(layers):
            emitter.emit(f"static elem_t weights{idx}[L{idx}_DIM_K][L{idx}_DIM_J] row_align(1);")
            if idx < num_layers - 1:
                emitter.emit(f"static elem_t inter{idx}[L{idx}_DIM_I][L{idx}_DIM_J] row_align(1);")
            else:
                emitter.emit(f"static elem_t output_mat[L{idx}_DIM_I][L{idx}_DIM_J] row_align(1);")
        emitter.emit("")

        # CPU reference arrays
        for idx, (M, K, N) in enumerate(layers):
            if idx < num_layers - 1:
                emitter.emit(f"static elem_t inter{idx}_gold[L{idx}_DIM_I][L{idx}_DIM_J];")
            else:
                emitter.emit(f"static elem_t output_gold[L{idx}_DIM_I][L{idx}_DIM_J];")
        emitter.emit("")

        self._emit_helper_functions(emitter)

        # Init function
        emitter.emit("static void init_data(void) {")
        emitter.indent()
        emitter.emit("for (size_t i = 0; i < L0_DIM_I; i++)")
        emitter.indent()
        emitter.emit("for (size_t j = 0; j < L0_DIM_K; j++)")
        emitter.indent()
        emitter.emit("input_mat[i][j] = (elem_t)((i + 3 * j) % 9 - 4);")
        emitter.dedent()
        emitter.dedent()
        emitter.emit("")

        for idx, (M, K, N) in enumerate(layers):
            emitter.emit(f"for (size_t i = 0; i < L{idx}_DIM_K; i++)")
            emitter.indent()
            emitter.emit(f"for (size_t j = 0; j < L{idx}_DIM_J; j++)")
            emitter.indent()
            emitter.emit(f"weights{idx}[i][j] = (elem_t)(({2 * idx + 2} * i + j) % 7 - 3);")
            emitter.dedent()
            emitter.dedent()
            emitter.emit("")

        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

        # CPU reference function
        emitter.emit("static void cpu_reference(void) {")
        emitter.indent()
        for idx, (M, K, N) in enumerate(layers):
            if idx == 0:
                in_name = "input_mat"
            else:
                in_name = f"inter{idx - 1}"
            if idx < num_layers - 1:
                out_name = f"inter{idx}_gold"
            else:
                out_name = "output_gold"

            emitter.emit(f"for (size_t i = 0; i < L{idx}_DIM_I; i++) {{")
            emitter.indent()
            emitter.emit(f"for (size_t j = 0; j < L{idx}_DIM_J; j++) {{")
            emitter.indent()
            emitter.emit("full_t sum = 0;")
            emitter.emit(f"for (size_t k = 0; k < L{idx}_DIM_K; k++)")
            emitter.indent()
            emitter.emit(f"sum += (full_t){in_name}[i][k] * (full_t)weights{idx}[k][j];")
            emitter.dedent()
            emitter.emit(f"{out_name}[i][j] = saturate(sum);")
            emitter.dedent()
            emitter.emit("}")
            emitter.dedent()
            emitter.emit("}")

            # Feed gold output as input for next layer CPU ref
            if idx < num_layers - 1:
                emitter.emit(f"// Copy gold to inter{idx} for next layer CPU ref")
                emitter.emit(f"for (size_t i = 0; i < L{idx}_DIM_I; i++)")
                emitter.indent()
                emitter.emit(f"for (size_t j = 0; j < L{idx}_DIM_J; j++)")
                emitter.indent()
                emitter.emit(f"inter{idx}[i][j] = inter{idx}_gold[i][j];")
                emitter.dedent()
                emitter.dedent()
            emitter.emit("")

        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

        # main
        emitter.emit("int main(void) {")
        emitter.indent()

        emitter.emit("#ifndef BAREMETAL")
        emitter.emit("if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {")
        emitter.indent()
        emitter.emit('perror("mlockall failed");')
        emitter.emit("exit(1);")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("#endif")
        emitter.emit("")

        emitter.emit(f'printf("MLP {num_layers}-layer Gemmini program\\n");')
        for idx, (M, K, N) in enumerate(layers):
            emitter.emit(f'printf("  layer {idx}: [%d x %d] * [%d x %d] -> [%d x %d]\\n",')
            emitter.emit(f"    L{idx}_DIM_I, L{idx}_DIM_K, L{idx}_DIM_K, L{idx}_DIM_J,")
            emitter.emit(f"    L{idx}_DIM_I, L{idx}_DIM_J);")
        emitter.emit("")

        emitter.emit("init_data();")
        emitter.emit("")

        emitter.emit('printf("Computing CPU reference...\\n");')
        emitter.emit("uint64_t cpu_start = read_cycles();")
        emitter.emit("cpu_reference();")
        emitter.emit("uint64_t cpu_end = read_cycles();")
        emitter.emit('printf("CPU reference took %llu cycles\\n", cpu_end - cpu_start);')
        emitter.emit("")

        # Re-init intermediates for Gemmini (CPU ref overwrote them)
        for idx, (M, K, N) in enumerate(layers):
            if idx < num_layers - 1:
                emitter.emit(f"for (size_t i = 0; i < L{idx}_DIM_I; i++)")
                emitter.indent()
                emitter.emit(f"for (size_t j = 0; j < L{idx}_DIM_J; j++)")
                emitter.indent()
                emitter.emit(f"inter{idx}[i][j] = 0;")
                emitter.dedent()
                emitter.dedent()
        emitter.emit("")

        emitter.emit("gemmini_flush(0);")
        emitter.emit("")

        emitter.emit('printf("Running on Gemmini...\\n");')
        emitter.emit("uint64_t gemmini_start = read_cycles();")
        emitter.emit("")

        # Generate Gemmini matmul for each layer
        for idx, (M, K, N) in enumerate(layers):
            if idx == 0:
                in_name = "input_mat"
            else:
                in_name = f"inter{idx - 1}"
            if idx < num_layers - 1:
                out_name = f"inter{idx}"
            else:
                out_name = "output_mat"

            emitter.emit(f"// ---- Layer {idx}: [{M}x{K}] * [{K}x{N}] ----")
            emitter.emit(f"uint64_t l{idx}_start = read_cycles();")

            if mode == "auto":
                emitter.emit(f"tiled_matmul_auto(L{idx}_DIM_I, L{idx}_DIM_J, L{idx}_DIM_K,")
                emitter.emit(f"    (elem_t*){in_name}, (elem_t*)weights{idx}, NULL, (elem_t*){out_name},")
                emitter.emit(f"    L{idx}_DIM_K, L{idx}_DIM_J, L{idx}_DIM_J, L{idx}_DIM_J,")
                emitter.emit("    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,")
                emitter.emit("    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,")
                emitter.emit("    false, false,")
                emitter.emit("    false, false,")
                emitter.emit("    0,")
                emitter.emit("    WS);")
            else:
                emitter.emit(f"tiled_matmul_auto(L{idx}_DIM_I, L{idx}_DIM_J, L{idx}_DIM_K,")
                emitter.emit(f"    (elem_t*){in_name}, (elem_t*)weights{idx}, NULL, (elem_t*){out_name},")
                emitter.emit(f"    L{idx}_DIM_K, L{idx}_DIM_J, L{idx}_DIM_J, L{idx}_DIM_J,")
                emitter.emit("    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,")
                emitter.emit("    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,")
                emitter.emit("    false, false,")
                emitter.emit("    false, false,")
                emitter.emit("    0,")
                emitter.emit("    OS);")

            emitter.emit("gemmini_fence();")
            emitter.emit(f"uint64_t l{idx}_end = read_cycles();")
            emitter.emit(f'printf("  Layer {idx} Gemmini: %llu cycles\\n", l{idx}_end - l{idx}_start);')
            emitter.emit("")

        emitter.emit("uint64_t gemmini_end = read_cycles();")
        emitter.emit('printf("Total Gemmini: %llu cycles\\n", gemmini_end - gemmini_start);')
        emitter.emit("")

        # Verification on final output
        last_M, _, last_N = layers[-1]
        emitter.emit("// Verify final output")
        emitter.emit("int pass = 1;")
        emitter.emit(f"for (size_t i = 0; i < L{num_layers - 1}_DIM_I; i++) {{")
        emitter.indent()
        emitter.emit(f"for (size_t j = 0; j < L{num_layers - 1}_DIM_J; j++) {{")
        emitter.indent()
        emitter.emit("if (output_mat[i][j] != output_gold[i][j]) {")
        emitter.indent()
        emitter.emit('printf("MISMATCH at output[%u][%u]: got %d, expected %d\\n",')
        emitter.emit("    (unsigned)i, (unsigned)j, output_mat[i][j], output_gold[i][j]);")
        emitter.emit("pass = 0;")
        emitter.dedent()
        emitter.emit("}")
        emitter.dedent()
        emitter.emit("}")
        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

        emitter.emit("if (pass) {")
        emitter.indent()
        emitter.emit('printf("PASSED\\n");')
        emitter.emit("exit(0);")
        emitter.dedent()
        emitter.emit("} else {")
        emitter.indent()
        emitter.emit('printf("FAILED\\n");')
        emitter.emit("exit(1);")
        emitter.dedent()
        emitter.emit("}")

        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

        return emitter.get_code()


# ============================================================================
# Strategy Registry
# ============================================================================


class StrategyRegistry:
    """Registry for GEMM execution strategies"""

    _strategies = {
        "simple": SimpleGEMMStrategy,
        "workload": lambda: WorkloadGEMMStrategy(num_iterations=100),
        "blocked": lambda: BlockedGEMMStrategy(block_m=32, block_n=32, block_k=32),
        "gemmini": GemminiStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_factory):
        """Register a new strategy"""
        cls._strategies[name] = strategy_factory

    @classmethod
    def get_strategy(cls, name: str) -> GEMMStrategy:
        """Get a strategy by name"""
        if name not in cls._strategies:
            raise ValueError(
                f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}"
            )
        strategy_factory = cls._strategies[name]
        if callable(strategy_factory):
            return strategy_factory()
        return strategy_factory

    @classmethod
    def available_strategies(cls) -> list[str]:
        """Return list of available strategy names"""
        return list(cls._strategies.keys())


# ============================================================================
# Unified Code Generator (using strategies)
# ============================================================================


def _extract_tensors(parsed: dict) -> dict[str, "TensorDescriptor"]:
    """Extract tensor descriptors from parsed EmitC module."""
    tensors = {}
    for tensor_create in parsed["tensor_creates"]:
        tensor_type = tensor_create["tensor_type"]
        args = tensor_create["args"]

        if tensor_type in ["A", "B", "C"]:
            dim0 = get_constant_value(f"%{args[1]}", parsed["constants"])
            dim1 = get_constant_value(f"%{args[2]}", parsed["constants"])
            dim2 = get_constant_value(f"%{args[3]}", parsed["constants"])
            tensors[tensor_type] = TensorDescriptor(
                tensor_type, tensor_type, dim0, dim1, dim2
            )
        elif tensor_type in ["transpose_in", "transpose_out"]:
            dims = [
                get_constant_value(f"%{arg}", parsed["constants"])
                for arg in args[1:]
            ]
            dim0 = dims[0] if len(dims) > 0 else 1
            dim1 = dims[1] if len(dims) > 1 else 1
            dim2 = dims[2] if len(dims) > 2 else 1
            tensors[tensor_type] = TensorDescriptor(
                tensor_type, tensor_type, dim0, dim1, dim2
            )
    return tensors


def generate_c_with_strategy(
    mlir_content: str, strategy_name: str = "simple", verbose: bool = False
) -> str:
    """
    Generate C code from EmitC MLIR using the specified strategy

    Args:
        mlir_content: The EmitC MLIR content
        strategy_name: Name of the strategy to use ("simple", "workload", "blocked")
        verbose: Enable verbose output

    Returns:
        Generated C code as a string
    """
    # Parse MLIR
    parsed = parse_emitc_module(mlir_content, verbose)

    # Get strategy
    strategy = StrategyRegistry.get_strategy(strategy_name)

    # Gemmini strategy generates a complete standalone program
    if getattr(strategy, "is_standalone_program", False):
        tensors = _extract_tensors(parsed)
        return strategy.generate_full_program(tensors, parsed.get("func_name"))

    emitter = CodeEmitter()

    # Generate header
    emitter.emit("// Auto-generated C code from EmitC MLIR")
    emitter.emit(f"// Using strategy: {strategy.name}")
    emitter.emit("#include <stdint.h>")
    emitter.emit("#include <stdio.h>")
    emitter.emit("#include <stdlib.h>")
    emitter.emit("#include <string.h>")
    emitter.emit("#include <npu_highlevel.h>")
    emitter.emit("#include <primitive.h>")
    emitter.emit("")

    # Generate helper types
    emitter.emit(
        "// ===================================================================="
    )
    emitter.emit("// Tensor Helper Types and Functions")
    emitter.emit(
        "// ===================================================================="
    )
    emitter.emit("")
    # emitter.emit("typedef struct {")
    # emitter.emit("    uint32_t base_addr;")
    # emitter.emit("    uint32_t dim0;")
    # emitter.emit("    uint32_t dim1;")
    # emitter.emit("    uint32_t dim2;")
    # emitter.emit("    uint32_t byte_stride1;")
    # emitter.emit("    uint32_t byte_stride2;")
    # emitter.emit("    uint32_t wd_data;")
    # emitter.emit("    uint32_t type_data;")
    # emitter.emit("} Tensor;")
    emitter.emit("")

    # Emit required function declarations based on strategy
    if strategy_name == "workload":
        emitter.emit("// Helper functions")
        # emitter.emit("extern void npu_mem_init(void);")
        # emitter.emit("extern uint32_t getTensorSize(Tensor *tensor);")
        # emitter.emit("extern void tensor_load(Tensor *src, Tensor *dst);")
        # emitter.emit("extern void tensor_store(Tensor *src, Tensor *dst);")
        # emitter.emit("extern void gemm_operator(Tensor *A, Tensor *B, Tensor *C, Tensor *D, "
        #             "int accumulate, int activate);")
        emitter.emit("")
    elif strategy_name == "blocked":
        emitter.emit("// Helper functions")
        # emitter.emit("extern void npu_mem_init(void);")
        # emitter.emit("extern uint32_t getTensorSize(Tensor *tensor);")
        # emitter.emit("extern void tensor_load(Tensor *src, Tensor *dst);")
        # emitter.emit("extern void tensor_store(Tensor *src, Tensor *dst);")
        # emitter.emit("extern void gemm_operator(Tensor *A, Tensor *B, Tensor *C, Tensor *D, "
        #             "int accumulate, int activate);")
        emitter.emit("")
    else:  # simple
        emitter.emit("// Helper functions")
        # emitter.emit("extern void gemm_operator(Tensor *A, Tensor *B, Tensor *C, Tensor *D, "
        #             "int accumulate, int activate);")
        # emitter.emit("extern void npu_mem_init(void);")
        emitter.emit("")

    emitter.emit("// Helper function to compute minimum stride")
    # emitter.emit("static inline int min_stride1(int dim, int width) {")
    # emitter.emit("    uint32_t size_dim0a = 256 >> width;")
    # emitter.emit("    uint32_t size_dim0b = (dim + size_dim0a - 1) / size_dim0a;")
    # emitter.emit("    return size_dim0b * 32;")
    # emitter.emit("}")
    emitter.emit("")

    # Generate main function
    if parsed["func_name"]:
        emitter.emit(
            "// ===================================================================="
        )
        emitter.emit(f"// Generated function: {parsed['func_name']}")
        emitter.emit(f"// Strategy: {strategy.name}")
        emitter.emit(
            "// ===================================================================="
        )
        emitter.emit("")
        emitter.emit(f"void *{parsed['func_name']}(void *arg0, void *arg1) {{")
        emitter.indent()

        emitter.emit("// Initialize NPU memory")
        emitter.emit("npu_mem_init();")
        emitter.emit("")

        # Build tensor descriptors
        tensors = {}
        for tensor_create in parsed["tensor_creates"]:
            tensor_type = tensor_create["tensor_type"]
            args = tensor_create["args"]

            # Handle GEMM tensors (A, B, C)
            if tensor_type in ["A", "B", "C"]:
                # Get dimension values from constants
                if tensor_type in ["A", "C"]:
                    dim0 = get_constant_value(f"%{args[1]}", parsed["constants"])
                    dim1 = get_constant_value(f"%{args[2]}", parsed["constants"])
                else:  # B
                    dim0 = get_constant_value(f"%{args[1]}", parsed["constants"])
                    dim1 = get_constant_value(f"%{args[2]}", parsed["constants"])

                dim2 = get_constant_value(f"%{args[3]}", parsed["constants"])

                tensors[tensor_type] = TensorDescriptor(
                    tensor_type, tensor_type, dim0, dim1, dim2
                )

            # Handle transpose tensors
            elif tensor_type in ["transpose_in", "transpose_out"]:
                # For transpose, we need to handle variable number of dimensions
                # args[0] is the pointer, args[1:] are dimensions
                dims = [
                    get_constant_value(f"%{arg}", parsed["constants"])
                    for arg in args[1:]
                ]

                # Create a simple tensor descriptor (using TensorDescriptor as a container)
                # For transpose, we'll use dim0, dim1, dim2 as first three dimensions
                dim0 = dims[0] if len(dims) > 0 else 1
                dim1 = dims[1] if len(dims) > 1 else 1
                dim2 = dims[2] if len(dims) > 2 else 1

                tensors[tensor_type] = TensorDescriptor(
                    tensor_type, tensor_type, dim0, dim1, dim2
                )

        # Check if this is a transpose operation
        if parsed["transpose_call"]:
            # Generate transpose-specific code
            emitter.emit("// Transpose operation detected")
            emitter.emit("")

            # Get transpose arguments
            transpose_args = parsed["transpose_call"]["args"]
            # Args: [tensor_in, tensor_out, dim_axis]

            # Get dim_axis value
            dim_axis = 0
            if len(transpose_args) >= 3:
                dim_axis = get_constant_value(
                    f"%{transpose_args[2]}", parsed["constants"]
                )

            # Generate tensor initialization for transpose
            if "transpose_in" in tensors:
                tensor_in = tensors["transpose_in"]
                emitter.emit(
                    f"// Input tensor: shape ({tensor_in.dim0}, {tensor_in.dim1}, {tensor_in.dim2}, ...)"
                )
                emitter.emit("Tensor tensor_transpose_in = (Tensor){")
                emitter.indent()
                emitter.emit(".base_addr = -1,")
                emitter.emit(f".dim0      = {tensor_in.dim0},")
                emitter.emit(f".dim1      = {tensor_in.dim1},")
                emitter.emit(f".dim2      = {tensor_in.dim2},")
                emitter.emit(".byte_stride1 = 0,")
                emitter.emit(".byte_stride2 = 0,")
                emitter.emit(".wd_data      = WIDTH_8,")
                emitter.emit(".type_data    = TYPE_INT")
                emitter.dedent()
                emitter.emit("};")
                emitter.emit("")

            if "transpose_out" in tensors:
                tensor_out = tensors["transpose_out"]
                emitter.emit(
                    f"// Output tensor: shape ({tensor_out.dim0}, {tensor_out.dim1}, {tensor_out.dim2}, ...)"
                )
                emitter.emit("Tensor tensor_transpose_out = (Tensor){")
                emitter.indent()
                emitter.emit(".base_addr = -1,")
                emitter.emit(f".dim0      = {tensor_out.dim0},")
                emitter.emit(f".dim1      = {tensor_out.dim1},")
                emitter.emit(f".dim2      = {tensor_out.dim2},")
                emitter.emit(".byte_stride1 = 0,")
                emitter.emit(".byte_stride2 = 0,")
                emitter.emit(".wd_data      = WIDTH_8,")
                emitter.emit(".type_data    = TYPE_INT")
                emitter.dedent()
                emitter.emit("};")
                emitter.emit("")

            # Set memory addresses
            emitter.emit("// Set memory addresses for transpose tensors")
            emitter.emit("tensor_transpose_in.base_addr = 0x90000000;  // scratchpad0")
            emitter.emit("tensor_transpose_out.base_addr = 0x90020000;  // scratchpad1")
            emitter.emit("")

            # Call transpose_operator
            emitter.emit(f"// Perform transpose operation with dim_axis={dim_axis}")
            emitter.emit(
                f"transpose_operator(&tensor_transpose_in, &tensor_transpose_out, {dim_axis});"
            )
            emitter.emit("")

            # Return pointer to output tensor
            emitter.emit("// Return pointer to output tensor")
            emitter.emit("return (void *)tensor_transpose_out.base_addr;")

        elif parsed["rmsnorm_call"]:
            # Generate RMSNorm-specific code
            emitter.emit("// RMSNorm operation detected")
            emitter.emit("")

            # Get RMSNorm arguments: [input, gamma, output, epsilon]
            rmsnorm_args = parsed["rmsnorm_call"]["args"]

            # Extract tensor information
            # For now, assume we have tensor_in, tensor_gamma, tensor_out in the tensors dict
            # In a full implementation, we would parse the tensor_creates to find these

            # Simple approach: generate the function call with parameters
            emitter.emit(
                "// RMSNorm: output = input / sqrt(mean(input^2) + epsilon) * gamma"
            )

            # Get epsilon value
            epsilon = 1e-5  # Default
            if len(rmsnorm_args) >= 4:
                epsilon = get_constant_value(f"%{rmsnorm_args[3]}", parsed["constants"])
                # Convert to float
                epsilon = epsilon / 1.0e5  # Assuming epsilon is stored as 1e-5

            emitter.emit(f"// Epsilon: {epsilon}")
            emitter.emit("")

            # For RMSNorm, we need to set up tensor descriptors
            # This is simplified - a full implementation would parse tensor_creates
            emitter.emit("// Note: Tensor setup would go here")
            emitter.emit("// Tensor tensor_rmsnorm_in = {...};")
            emitter.emit("// Tensor tensor_rmsnorm_gamma = {...};")
            emitter.emit("// Tensor tensor_rmsnorm_out = {...};")
            emitter.emit("")

            # Call rmsnorm_operator
            emitter.emit("// Perform RMSNorm operation")
            emitter.emit(
                f"rmsnorm_operator(&tensor_rmsnorm_in, &tensor_rmsnorm_gamma, &tensor_rmsnorm_out, {epsilon});"
            )
            emitter.emit("")

            # Return pointer to output tensor
            emitter.emit("// Return pointer to output tensor")
            emitter.emit("return (void *)tensor_rmsnorm_out.base_addr;")

        elif parsed["gemm_call"]:
            # Original GEMM code generation
            # Generate using strategy
            strategy.generate_prologue(emitter, tensors)

            for tensor_type, tensor in tensors.items():
                if tensor_type in ["A", "B", "C"]:  # Only generate for GEMM tensors
                    strategy.generate_tensor_init(emitter, tensor)

            # Get accumulate and activate from gemm call
            accumulate = 0
            activate = 0
            if parsed["gemm_call"]:
                gemm_args = parsed["gemm_call"]["args"]
                if len(gemm_args) >= 6:
                    accumulate = get_constant_value(
                        f"%{gemm_args[4]}", parsed["constants"]
                    )
                    activate = get_constant_value(
                        f"%{gemm_args[5]}", parsed["constants"]
                    )

            strategy.generate_computation(emitter, tensors, accumulate, activate)
            strategy.generate_epilogue(emitter, tensors)
        else:
            emitter.emit("// No recognized operation (GEMM or transpose) found")

        emitter.dedent()
        emitter.emit("}")
        emitter.emit("")

    # Generate main function
    emitter.emit(
        "// ===================================================================="
    )
    emitter.emit("// Main Function (for testing)")
    emitter.emit(
        "// ===================================================================="
    )
    emitter.emit("")
    emitter.emit("int main(void) {")
    emitter.indent()
    if parsed["func_name"]:
        emitter.emit(f"{parsed['func_name']}(NULL, NULL);")
    emitter.dedent()
    emitter.emit("    return 0;")
    emitter.emit("}")

    return emitter.get_code()


def get_constant_value(var_name: str, constants: dict) -> int:
    """Get integer value of a constant variable"""
    if var_name in constants:
        const_type, const_val = constants[var_name]
        try:
            return int(const_val)
        except:
            pass


# ============================================================================
# Backward Compatibility Layer
# ============================================================================


def generate_c_from_emitc(mlir_content: str, verbose: bool = False) -> str:
    """
    Args:
        mlir_content: The EmitC MLIR content
        verbose: Enable verbose output

    Returns:
        Generated C code as a string
    """
    return generate_c_with_strategy(mlir_content, "simple", verbose)
    return 0


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python strategy_code_generator.py <input.mlir> [strategy] [output.cpp]"
        )
        print("")
        print("Available strategies:")
        for strategy_name in StrategyRegistry.available_strategies():
            strategy = StrategyRegistry.get_strategy(strategy_name)
            print(f"  - {strategy_name}: {strategy.name}")
        sys.exit(1)

    mlir_file = sys.argv[1]
    strategy_name = sys.argv[2] if len(sys.argv) > 2 else "simple"
    output_file = (
        sys.argv[3]
        if len(sys.argv) > 3
        else mlir_file.replace(".mlir", f"_{strategy_name}.cpp")
    )

    with open(mlir_file) as f:
        mlir_content = f.read()

    c_code = generate_c_with_strategy(mlir_content, strategy_name, verbose=True)

    with open(output_file, "w") as f:
        f.write(c_code)

    print(f"\nGenerated: {output_file} (strategy: {strategy_name})")
