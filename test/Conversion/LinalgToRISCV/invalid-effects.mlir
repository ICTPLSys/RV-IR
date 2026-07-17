// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics

func.func @matmul_rank(
    %lhs: memref<4xf32>,
    %rhs: memref<4x4xf32>,
    %out: memref<4x4xf32>) {
  // expected-error @+1 {{'rair.matmul' op expects lhs, rhs, and output to be rank-2 memrefs}}
  rair.matmul
    ins(%lhs, %rhs : memref<4xf32>, memref<4x4xf32>)
    outs(%out : memref<4x4xf32>)
  return
}

// -----

func.func @matmul_contracting_dimension(
    %lhs: memref<4x3xf32>,
    %rhs: memref<5x4xf32>,
    %out: memref<4x4xf32>) {
  // expected-error @+1 {{'rair.matmul' op has incompatible contracting dimensions: lhs dimension 1 is 3 but rhs dimension 0 is 5}}
  rair.matmul
    ins(%lhs, %rhs : memref<4x3xf32>, memref<5x4xf32>)
    outs(%out : memref<4x4xf32>)
  return
}

// -----

func.func @matmul_output_shape(
    %lhs: memref<4x3xf32>,
    %rhs: memref<3x5xf32>,
    %out: memref<7x5xf32>) {
  // expected-error @+1 {{'rair.matmul' op has incompatible row dimensions: lhs dimension 0 is 4 but output dimension 0 is 7}}
  rair.matmul
    ins(%lhs, %rhs : memref<4x3xf32>, memref<3x5xf32>)
    outs(%out : memref<7x5xf32>)
  return
}

// -----

func.func @matmul_element_type(
    %lhs: memref<4x3xf32>,
    %rhs: memref<3x5xf32>,
    %out: memref<4x5xf64>) {
  // expected-error @+1 {{'rair.matmul' op expects matching element types, but got lhs 'f32', rhs 'f32', and output 'f64'}}
  rair.matmul
    ins(%lhs, %rhs : memref<4x3xf32>, memref<3x5xf32>)
    outs(%out : memref<4x5xf64>)
  return
}

// -----

func.func @batch_matmul_batch_dimension(
    %lhs: memref<2x4x3xf32>,
    %rhs: memref<3x3x5xf32>,
    %out: memref<2x4x5xf32>) {
  // expected-error @+1 {{'rair.batch_matmul' op has incompatible batch dimensions: lhs dimension 0 is 2, rhs dimension 0 is 3, and output dimension 0 is 2}}
  rair.batch_matmul
    ins(%lhs, %rhs : memref<2x4x3xf32>, memref<3x3x5xf32>)
    outs(%out : memref<2x4x5xf32>)
  return
}

// -----

func.func @batch_matmul_dynamic_batch_dimension(
    %lhs: memref<?x4x3xf32>,
    %rhs: memref<3x3x5xf32>,
    %out: memref<2x4x5xf32>) {
  // expected-error @+1 {{'rair.batch_matmul' op has incompatible batch dimensions}}
  rair.batch_matmul
    ins(%lhs, %rhs : memref<?x4x3xf32>, memref<3x3x5xf32>)
    outs(%out : memref<2x4x5xf32>)
  return
}

// -----

func.func @transfer_element_type(
    %src: memref<4x4xf32>,
    %dst: memref<4x4xi32>) {
  // expected-error @+1 {{'rair.transfer' op expects matching element types, but got source 'f32' and destination 'i32'}}
  rair.transfer %src to %dst : memref<4x4xf32>, memref<4x4xi32>
  return
}

// -----

func.func @transfer_rank(
    %src: memref<4x4xf32>,
    %dst: memref<16xf32>) {
  // expected-error @+1 {{'rair.transfer' op expects source and destination to have the same rank, but got 2 and 1}}
  rair.transfer %src to %dst : memref<4x4xf32>, memref<16xf32>
  return
}

// -----

func.func @transfer_static_shape(
    %src: memref<4x4xf32>,
    %dst: memref<4x5xf32>) {
  // expected-error @+1 {{'rair.transfer' op has incompatible static size at dimension 1: source is 4 but destination is 5}}
  rair.transfer %src to %dst : memref<4x4xf32>, memref<4x5xf32>
  return
}

// -----

func.func @transfer_legacy_string_memory_space(
    %src: memref<4x4xf32>,
    %dst: memref<4x4xf32>) {
  // expected-error @+1 {{'rair.transfer' op attribute 'src_memory_space' failed to satisfy constraint: RAIR memory space}}
  rair.transfer %src to %dst {src_memory_space = "GMEM"}
    : memref<4x4xf32>, memref<4x4xf32>
  return
}

// -----

func.func @alloc_buffer_memory_space_mismatch(%ctx: !rair.context) {
  // expected-error @+1 {{'rair.alloc_buffer' op has inconsistent memory_space: attribute is #rair.space<lmem> but result memref type uses #rair.space<spad0>}}
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4x4xf32, #rair.space<spad0>>
  return
}

// -----

func.func @transfer_source_memory_space_mismatch(
    %src: memref<4x4xf32, #rair.space<lmem>>,
    %dst: memref<4x4xf32, #rair.space<spad0>>) {
  // expected-error @+1 {{'rair.transfer' op has inconsistent src_memory_space: attribute is #rair.space<gmem> but source memref type uses #rair.space<lmem>}}
  rair.transfer %src to %dst {
    src_memory_space = #rair.space<gmem>,
    dst_memory_space = #rair.space<spad0>
  } : memref<4x4xf32, #rair.space<lmem>>, memref<4x4xf32, #rair.space<spad0>>
  return
}

// -----

func.func @transfer_destination_memory_space_mismatch(
    %src: memref<4x4xf32, #rair.space<gmem>>,
    %dst: memref<4x4xf32, #rair.space<spad0>>) {
  // expected-error @+1 {{'rair.transfer' op has inconsistent dst_memory_space: attribute is #rair.space<lmem> but destination memref type uses #rair.space<spad0>}}
  rair.transfer %src to %dst {
    src_memory_space = #rair.space<gmem>,
    dst_memory_space = #rair.space<lmem>
  } : memref<4x4xf32, #rair.space<gmem>>, memref<4x4xf32, #rair.space<spad0>>
  return
}
