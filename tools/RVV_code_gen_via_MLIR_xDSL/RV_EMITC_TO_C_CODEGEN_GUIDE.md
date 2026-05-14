# EmitC 代码生成器使用说明

## 概述

我们成功实现了一个**基于 EmitC 的代码生成器**，它能够：

1. ✅ **利用 EmitC 转换流程**：从 EmitC 操作生成 C 代码
2. ✅ **智能处理 Tensor 创建**：自动解析维度并生成 Tensor 结构体初始化
3. ✅ **直接调用计算函数**：为 `gemm_operator` 等生成真实的 C 函数调用
4. ✅ **忽略无关操作**：自动跳过 `memref.alloc` 和 `linalg.fill` 等无法转换为 C 的操作

### 三层架构
```
┌─────────────────────────────────────────┐
│   MLIR Input (EmitC dialect)            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Parser Layer (emitc_code_generator)   │
│   - Parse MLIR once                     │
│   - Extract dimensions & constants      │
│   - Build tensor descriptors            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Strategy Layer                       │
│   ┌────────────┐ ┌────────────┐        │
│   │  Simple    │ │  Workload  │        │
│   │  Strategy  │ │  Strategy  │        │
│   └────────────┘ └────────────┘        │
│   ┌────────────┐ ┌────────────┐        │
│   │  Blocked   │ │  Custom??? │        │
│   │  Strategy  │ │  Strategy  │        │
│   └────────────┘ └────────────┘        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Generated C Code                      │
└─────────────────────────────────────────┘
```

## 🚀 快速开始

### 基本使用

```bash
# 使用 simple 策略（默认）
python strategy_code_generator.py input.mlir simple output.cpp

# 使用 blocked 策略
python strategy_code_generator.py input.mlir blocked output.cpp

# 使用 workload 策略
python strategy_code_generator.py input.mlir workload output.cpp
```

```bash
```

输出：
```
Usage: ./convert_riscv_mlir_to_c.sh <input.mlir> [strategy] [output.cpp]

Available strategies:
  - simple: Simple GEMM
  - workload: Workload GEMM
  - blocked: Blocked GEMM
```

---


---

## 📝 策略详解

### 1. Simple 策略

**生成的代码结构**：
```c
void *batch_matmul_1x128x64(void *arg0, void *arg1) {
    npu_mem_init();

    // Tensor A
    Tensor tensor_A = (Tensor){...};

    // Tensor B
    Tensor tensor_B = (Tensor){...};

    // Tensor C
    Tensor tensor_C = (Tensor){...};

    // 直接设置内存地址
    tensor_A.base_addr = 0x90000000;
    tensor_B.base_addr = 0x80000;
    tensor_C.base_addr = 0x90020000;

    // 直接调用 GEMM
    gemm_operator(&tensor_A, &tensor_B, &tensor_C, &tensor_C, 0, 0);

    return (void *)tensor_C.base_addr;
}
```

**特点**：
- ✅ 最简单，最快
- ✅ 适合小矩阵（能完全放入 scratchpad）
- ❌ 无法处理大于 scratchpad 的矩阵

---

### 2. Workload 策略

**生成的代码结构**：
```c
void *batch_matmul_1x128x64(void *arg0, void *arg1) {
    npu_mem_init();

    // 在 PSRAM 中分配大矩阵
    uint32_t start_addr = 0xa0000000;
    tensor_A.base_addr = start_addr;
    tensor_B.base_addr = start_addr + tensor_size_A;
    tensor_C.base_addr = start_addr + tensor_size_A + tensor_size_B;

    // 多次迭代（性能测试）
    for (int i = 0; i < 100; i++) {
        // 创建 block tensors（在 scratchpad 中）
        Tensor tensor_A_block = (Tensor){.base_addr = 0x90000000, ...};

        // Load: PSRAM -> Scratchpad
        tensor_load(&tensor_A, &tensor_A_block);
        tensor_load(&tensor_B, &tensor_B_block);

        // Compute
        gemm_operator(&tensor_A_block, &tensor_B_block, &tensor_C_block,
                      &tensor_C_block, 0, 0);

        // Store: Scratchpad -> PSRAM
        tensor_store(&tensor_C_block, &tensor_C);
    }
}
```

**特点**：
- ✅ 支持 PSRAM 中的大矩阵
- ✅ 显式控制数据移动
- ✅ 适合性能测试（可配置迭代次数）
- ⚠️ 需要链接 `tensor_load`/`tensor_store`/`getTensorSize`

---

### 3. Blocked 策略

**生成的代码结构**：
```c
void *batch_matmul_1x128x64(void *arg0, void *arg1) {
    npu_mem_init();

    // 初始化...
    uint32_t block_m = 32;
    uint32_t block_n = 32;
    uint32_t block_k = 32;

    // 三层嵌套循环（分块）
    for (uint32_t bm = 0; bm < m; bm += block_m) {
        for (uint32_t bn = 0; bn < n; bn += block_n) {
            uint32_t accumulate_flag = 0;

            for (uint32_t bk = 0; bk < k; bk += block_k) {
                // 创建当前块
                Tensor tensor_A_block = (Tensor){
                    .base_addr = 0x90000000,
                    .dim0 = cur_block_k,
                    .dim1 = cur_block_m,
                    ...
                };

                // 创建块的视图
                Tensor tensor_A_view = tensor_A;
                tensor_A_view.base_addr += bm * stride + bk * 8;
                tensor_A_view.dim0 = cur_block_k;
                tensor_A_view.dim1 = cur_block_m;

                // Load -> Compute -> Accumulate
                tensor_load(&tensor_A_view, &tensor_A_block);
                tensor_load(&tensor_B_view, &tensor_B_block);
                gemm_operator(..., accumulate_flag, 0);
                accumulate_flag = 1;  // 后续块需要累加
            }

            // Store final C block
            tensor_store(&tensor_C_block, &tensor_C_view);
        }
    }
}
```

**特点**：
- ✅ 适合超大矩阵（256x256, 512x512 等）
- ✅ 自动分块，减少内存占用
- ✅ 支持累加（accumulate_flag）
- ⚠️ 需要链接 `tensor_load`/`tensor_store`/`getTensorSize`

---

## 🔧 扩展：添加自定义策略

### 步骤 1：定义策略类

```python
class MyCustomStrategy(GEMMStrategy):
    """My custom GEMM execution strategy"""

    def __init__(self, param1: int, param2: int):
        super().__init__("my_custom")
        self.param1 = param1
        self.param2 = param2

    def generate_prologue(self, emitter, tensors):
        emitter.emit(f"// Custom strategy with params: {self.param1}, {self.param2}")
        # ... 你的初始化代码

    def generate_tensor_init(self, emitter, tensor):
        # ... 你的 tensor 初始化代码
        pass

    def generate_computation(self, emitter, tensors, accumulate, activate):
        # ... 你的计算代码
        pass

    def generate_epilogue(self, emitter, tensors):
        # ... 你的清理代码
        pass
```

### 步骤 2：注册策略

```python
# 在 StrategyRegistry._strategies 中添加
StrategyRegistry.register("my_custom", lambda: MyCustomStrategy(param1=42, param2=100))
```

### 步骤 3：使用新策略

```bash
python strategy_code_generator.py input.mlir my_custom output.cpp
```

---

## 🎨 实际案例

### 案例 1：小矩阵（32x32）

```bash
# 使用 simple 策略
python strategy_code_generator.py small_gemm.mlir simple small_gemm.cpp
```

生成的代码：直接在 scratchpad 中计算，无需数据移动。

---

### 案例 2：大矩阵（256x256）

```bash
# 使用 blocked 策略
python strategy_code_generator.py large_gemm.mlir blocked large_gemm.cpp
```

生成的代码：自动分块为 32x32 的小块，逐块计算。

---

### 案例 3：性能基准测试

```bash
# 使用 workload 策略
python strategy_code_generator.py benchmark.mlir workload benchmark.cpp
```

生成的代码：100 次迭代，测量吞吐量。

---

## 🔍 代码对比

### 相同的 MLIR 输入

```mlir
%4 = "emitc.call_opaque"(%arg0, %1, %3, %0) <{callee = "create_tensor_A"}>
%5 = "emitc.call_opaque"(%arg1, %3, %2, %0) <{callee = "create_tensor_B"}>
%6 = "emitc.call_opaque"(%result, %1, %2, %0) <{callee = "create_tensor_C"}>
"emitc.call_opaque"(%4, %5, %6, %6, %7, %7) <{callee = "gemm_operator"}>
```
