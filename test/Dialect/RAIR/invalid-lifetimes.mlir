// RUN: torch-mlir-opt %s --rair-verify-lifetimes -split-input-file -verify-diagnostics

func.func @missing_release() {
  // expected-error @+1 {{'rair.acquire' op context must be released exactly once in the same straight-line function block, but found 0 'rair.release' operations}}
  %ctx = rair.acquire : !rair.context
  rair.await %ctx : !rair.context
  return
}

// -----

func.func @double_release() {
  // expected-error @+1 {{'rair.acquire' op context must be released exactly once in the same straight-line function block, but found 2 'rair.release' operations}}
  %ctx = rair.acquire : !rair.context
  rair.release %ctx : !rair.context
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @context_use_after_release() {
  %ctx = rair.acquire : !rair.context
  rair.release %ctx : !rair.context
  // expected-error @+1 {{'rair.await' op uses an acquired RAIR context after 'rair.release'}}
  rair.await %ctx : !rair.context
  return
}

// -----

func.func @missing_dealloc() {
  %ctx = rair.acquire : !rair.context
  // expected-error @+1 {{'rair.alloc_buffer' op owned buffer must be deallocated exactly once in the same straight-line function block, but found 0 'rair.dealloc_buffer' operations}}
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4xf32, #rair.space<lmem>>
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @double_dealloc() {
  %ctx = rair.acquire : !rair.context
  // expected-error @+1 {{'rair.alloc_buffer' op owned buffer must be deallocated exactly once in the same straight-line function block, but found 2 'rair.dealloc_buffer' operations}}
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4xf32, #rair.space<lmem>>
  rair.dealloc_buffer %ctx, %buf : memref<4xf32, #rair.space<lmem>>
  rair.dealloc_buffer %ctx, %buf : memref<4xf32, #rair.space<lmem>>
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @wrong_dealloc_context() {
  %ctx0 = rair.acquire : !rair.context
  %ctx1 = rair.acquire : !rair.context
  %buf = rair.alloc_buffer %ctx0 {memory_space = #rair.space<lmem>}
    : memref<4xf32, #rair.space<lmem>>
  // expected-error @+1 {{'rair.dealloc_buffer' op must use the same context as the corresponding 'rair.alloc_buffer'}}
  rair.dealloc_buffer %ctx1, %buf : memref<4xf32, #rair.space<lmem>>
  rair.release %ctx1 : !rair.context
  rair.release %ctx0 : !rair.context
  return
}

// -----

func.func @buffer_use_after_dealloc(
    %dst: memref<4xf32, #rair.space<lmem>>) {
  %ctx = rair.acquire : !rair.context
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4xf32, #rair.space<lmem>>
  rair.dealloc_buffer %ctx, %buf : memref<4xf32, #rair.space<lmem>>
  // expected-error @+1 {{'rair.transfer' op uses an owned RAIR buffer after 'rair.dealloc_buffer'}}
  rair.transfer %buf to %dst
    : memref<4xf32, #rair.space<lmem>>,
      memref<4xf32, #rair.space<lmem>>
  rair.release %ctx : !rair.context
  return
}

// -----

func.func private @consume_context(!rair.context)

func.func @context_escape_to_call() {
  %ctx = rair.acquire : !rair.context
  // expected-error @+1 {{'func.call' op may not consume an acquired RAIR context; passing owned contexts to calls or unknown operations is unsupported}}
  func.call @consume_context(%ctx) : (!rair.context) -> ()
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @context_escape_to_return() -> !rair.context {
  %ctx = rair.acquire : !rair.context
  rair.release %ctx : !rair.context
  // expected-error @+1 {{'func.return' op returns or yields an acquired RAIR context; context escape is unsupported}}
  return %ctx : !rair.context
}

// -----

func.func private @consume_buffer(memref<4xf32, #rair.space<lmem>>)

func.func @buffer_escape_to_call() {
  %ctx = rair.acquire : !rair.context
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4xf32, #rair.space<lmem>>
  // expected-error @+1 {{'func.call' op passes an owned RAIR buffer to a call; interprocedural ownership is unsupported}}
  func.call @consume_buffer(%buf)
    : (memref<4xf32, #rair.space<lmem>>) -> ()
  rair.dealloc_buffer %ctx, %buf : memref<4xf32, #rair.space<lmem>>
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @buffer_escape_to_return()
    -> memref<4xf32, #rair.space<lmem>> {
  %ctx = rair.acquire : !rair.context
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4xf32, #rair.space<lmem>>
  rair.release %ctx : !rair.context
  // expected-error @+1 {{'func.return' op returns or yields an owned RAIR buffer; buffer escape is unsupported}}
  return %buf : memref<4xf32, #rair.space<lmem>>
}

// -----

func.func @derived_buffer_alias() {
  %ctx = rair.acquire : !rair.context
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4xf32, #rair.space<lmem>>
  // expected-error @+1 {{'memref.cast' op creates a derived memref from an owned RAIR buffer; alias-aware lifetime verification is not yet supported}}
  %alias = memref.cast %buf
    : memref<4xf32, #rair.space<lmem>>
      to memref<?xf32, #rair.space<lmem>>
  rair.dealloc_buffer %ctx, %buf : memref<4xf32, #rair.space<lmem>>
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @resource_in_branch(%condition: i1) {
  %ctx = rair.acquire : !rair.context
  scf.if %condition {
    // expected-error @+1 {{'rair.await' op is nested in a region; RAIR lifetime verification does not yet support resources inside branches, loops, or nested regions}}
    rair.await %ctx : !rair.context
  }
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @resource_in_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ctx = rair.acquire : !rair.context
  scf.for %i = %c0 to %c1 step %c1 {
    // expected-error @+1 {{'rair.await' op is nested in a region; RAIR lifetime verification does not yet support resources inside branches, loops, or nested regions}}
    rair.await %ctx : !rair.context
  }
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @release_borrowed_context(%ctx: !rair.context) {
  // expected-error @+1 {{'rair.release' op must consume a context produced by 'rair.acquire' in the same straight-line function block}}
  rair.release %ctx : !rair.context
  return
}

// -----

func.func @dealloc_borrowed_buffer(
    %ctx: !rair.context,
    %buf: memref<4xf32, #rair.space<lmem>>) {
  // expected-error @+1 {{'rair.dealloc_buffer' op must consume a buffer produced by 'rair.alloc_buffer' in the same straight-line function block}}
  rair.dealloc_buffer %ctx, %buf : memref<4xf32, #rair.space<lmem>>
  return
}
