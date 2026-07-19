// RUN: torch-mlir-opt %s --rair-infer-effects -o /dev/null | FileCheck %s

// CHECK-LABEL: RAIR effect report: func @overlap_classification scope 0 target @test
// CHECK: r0 base=arg0 space=host offsets=[0] sizes=[4] strides=[1]
// CHECK: r1 base=arg0 space=host offsets=[4] sizes=[4] strides=[1]
// CHECK: r2 base=arg0 space=host offsets=[2] sizes=[4] strides=[1]
// CHECK: r3 base=arg0 space=host offsets=[0] sizes=[2] strides=[2]
// CHECK: r4 base=arg0 space=host offsets=[1] sizes=[2] strides=[2]
// CHECK: r5 base=arg1 space=host offsets=[0] sizes=[4] strides=[1]
// CHECK: r6 base=arg2 space=device offsets=[0] sizes=[4] strides=[1]
// CHECK: relations:
// CHECK: r0 vs r1 = disjoint
// CHECK: r0 vs r2 = overlap
// CHECK: r0 vs r5 = may_overlap
// CHECK: r0 vs r6 = disjoint
// CHECK: r3 vs r4 = may_overlap
// CHECK: summary: regions=7 actions=0 graph_edges=0

rair.target @test {kind = "analysis_test"}

func.func @overlap_classification(
    %base: memref<16xf32, #rair.space<host>>,
    %possibly_aliasing: memref<16xf32, #rair.space<host>>,
    %different_space: memref<16xf32, #rair.space<device>>) {
  rair.scope @test {
    %disjoint_left = rair.view %base {
      offsets = array<i64: 0>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<16xf32, #rair.space<host>> -> !rair.region
    %disjoint_right = rair.view %base {
      offsets = array<i64: 4>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<16xf32, #rair.space<host>> -> !rair.region
    %overlapping = rair.view %base {
      offsets = array<i64: 2>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<16xf32, #rair.space<host>> -> !rair.region
    %strided_even = rair.view %base {
      offsets = array<i64: 0>, sizes = array<i64: 2>,
      strides = array<i64: 2>
    } : memref<16xf32, #rair.space<host>> -> !rair.region
    %strided_odd = rair.view %base {
      offsets = array<i64: 1>, sizes = array<i64: 2>,
      strides = array<i64: 2>
    } : memref<16xf32, #rair.space<host>> -> !rair.region
    %unknown_alias = rair.view %possibly_aliasing {
      offsets = array<i64: 0>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<16xf32, #rair.space<host>> -> !rair.region
    %no_alias_by_space = rair.view %different_space {
      offsets = array<i64: 0>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<16xf32, #rair.space<device>> -> !rair.region
  }
  return
}
