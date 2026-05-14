#!/usr/bin/env python3
"""Test script to verify elementwise operations are converted correctly"""

from xdsl.context import Context
from xdsl.parser import Parser

from xdsltemplate.transforms.linalg_generic_to_emitc import LinalgGenericToEmitCPass

# Test MLIR for elementwise add
test_add_mlir = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_elementwise_add(%arg0: memref<1x128x2048xf32>, %arg1: memref<1x128x2048xf32>) -> memref<1x128x2048xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>

    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<1x128x2048xf32>, memref<1x128x2048xf32>)
      outs(%alloc : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %in_76: f32, %out: f32):
        %6 = arith.addf %in, %in_76 : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x2048xf32>
  }
}
"""

# Test MLIR for elementwise mul
test_mul_mlir = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_elementwise_mul(%arg0: memref<1x128x2048xf32>, %arg1: memref<1x128x2048xf32>) -> memref<1x128x2048xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>

    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<1x128x2048xf32>, memref<1x128x2048xf32>)
      outs(%alloc : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %in_76: f32, %out: f32):
        %6 = arith.mulf %in, %in_76 : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x2048xf32>
  }
}
"""

# Test MLIR for elementwise div
test_div_mlir = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_elementwise_div(%arg0: memref<1x128x2048xf32>, %arg1: memref<1x128x2048xf32>) -> memref<1x128x2048xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>

    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<1x128x2048xf32>, memref<1x128x2048xf32>)
      outs(%alloc : memref<1x128x2048xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out: f32):
        %6 = arith.divf %in0, %in1 : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x2048xf32>
  }
}
"""


def test_conversion(mlir_code, test_name):
    """Test if the conversion produces the expected output"""
    print(f"\n{'=' * 60}")
    print(f"Testing {test_name}")
    print(f"{'=' * 60}")

    ctx = Context()

    parser = Parser(ctx, mlir_code)
    program = parser.parse_module()

    print("\nBefore conversion:")
    print(program)

    # Apply the conversion pass
    pass_obj = LinalgGenericToEmitCPass()
    pass_obj.apply(ctx, program)

    print("\nAfter conversion:")
    print(program)

    # Check if tensor_tensor_operator is in the output
    output_str = str(program)
    if "tensor_tensor_operator" in output_str:
        print(f"\n✓ SUCCESS: {test_name} - Found tensor_tensor_operator call")
        return True
    else:
        print(f"\n✗ FAILURE: {test_name} - tensor_tensor_operator call not found")
        return False


if __name__ == "__main__":
    results = []
    results.append(test_conversion(test_add_mlir, "Elementwise Add"))
    results.append(test_conversion(test_mul_mlir, "Elementwise Mul"))
    results.append(test_conversion(test_div_mlir, "Elementwise Div"))

    print(f"\n\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print("\n✗ Some tests failed")
        exit(1)
