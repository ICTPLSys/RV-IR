#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "ResNet"} {
  memref.global "private" constant @__constant_1000x512xf32 : memref<1000x512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1000xf32 : memref<1000xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_512x256x1x1xf32 : memref<512x256x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_512x512x3x3xf32 : memref<512x512x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_512xf32 : memref<512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_512x256x3x3xf32 : memref<512x256x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256x128x1x1xf32 : memref<256x128x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256x256x3x3xf32 : memref<256x256x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256xf32 : memref<256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256x128x3x3xf32 : memref<256x128x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_128x64x1x1xf32 : memref<128x64x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_128x128x3x3xf32 : memref<128x128x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_128xf32 : memref<128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_64x64x3x3xf32 : memref<64x64x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @__constant_64x3x7x7xf32 : memref<64x3x7x7xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  func.func @forward(%arg0: memref<1x3x224x224xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x1000xf32> {
    %cst = arith.constant 4.900000e+01 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_64x3x7x7xf32 : memref<64x3x7x7xf32>
    %1 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %2 = memref.get_global @__constant_64x64x3x3xf32 : memref<64x64x3x3xf32>
    %3 = memref.get_global @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32>
    %4 = memref.get_global @__constant_128xf32 : memref<128xf32>
    %5 = memref.get_global @__constant_128x128x3x3xf32 : memref<128x128x3x3xf32>
    %6 = memref.get_global @__constant_128x64x1x1xf32 : memref<128x64x1x1xf32>
    %7 = memref.get_global @__constant_256x128x3x3xf32 : memref<256x128x3x3xf32>
    %8 = memref.get_global @__constant_256xf32 : memref<256xf32>
    %9 = memref.get_global @__constant_256x256x3x3xf32 : memref<256x256x3x3xf32>
    %10 = memref.get_global @__constant_256x128x1x1xf32 : memref<256x128x1x1xf32>
    %11 = memref.get_global @__constant_512x256x3x3xf32 : memref<512x256x3x3xf32>
    %12 = memref.get_global @__constant_512xf32 : memref<512xf32>
    %13 = memref.get_global @__constant_512x512x3x3xf32 : memref<512x512x3x3xf32>
    %14 = memref.get_global @__constant_512x256x1x1xf32 : memref<512x256x1x1xf32>
    %15 = memref.get_global @__constant_1000xf32 : memref<1000xf32>
    %16 = memref.get_global @__constant_1000x512xf32 : memref<1000x512xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x230x230xf32>
    linalg.map outs(%alloc : memref<1x3x230x230xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview = memref.subview %alloc[0, 0, 3, 3] [1, 3, 224, 224] [1, 1, 1, 1] : memref<1x3x230x230xf32> to memref<1x3x224x224xf32, strided<[158700, 52900, 230, 1], offset: 693>>
    memref.copy %arg0, %subview : memref<1x3x224x224xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x3x224x224xf32, strided<[158700, 52900, 230, 1], offset: 693>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x64x112x112xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_3 : memref<1x64x112x112xf32>)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x64x112x112xf32>
    memref.copy %alloc_3, %alloc_4 : memref<1x64x112x112xf32> to memref<1x64x112x112xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc, %0 : memref<1x3x230x230xf32>, memref<64x3x7x7xf32>) outs(%alloc_4 : memref<1x64x112x112xf32>)
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x64x112x112xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_4, %1, %1, %1, %1 : memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%alloc_5 : memref<1x64x112x112xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x64x112x112xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_5 : memref<1x64x112x112xf32>) outs(%alloc_6 : memref<1x64x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x64x114x114xf32>
    linalg.map outs(%alloc_7 : memref<1x64x114x114xf32>)
      () {
        linalg.yield %cst_1 : f32
      }
    %subview_8 = memref.subview %alloc_7[0, 0, 1, 1] [1, 64, 112, 112] [1, 1, 1, 1] : memref<1x64x114x114xf32> to memref<1x64x112x112xf32, strided<[831744, 12996, 114, 1], offset: 115>>
    memref.copy %alloc_6, %subview_8 : memref<1x64x112x112xf32> to memref<1x64x112x112xf32, strided<[831744, 12996, 114, 1], offset: 115>>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc_9 : memref<1x64x56x56xf32>)
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    memref.copy %alloc_9, %alloc_11 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32>
    linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc_7, %alloc_10 : memref<1x64x114x114xf32>, memref<3x3xf32>) outs(%alloc_11 : memref<1x64x56x56xf32>)
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x64x58x58xf32>
    linalg.map outs(%alloc_12 : memref<1x64x58x58xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_13 = memref.subview %alloc_12[0, 0, 1, 1] [1, 64, 56, 56] [1, 1, 1, 1] : memref<1x64x58x58xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    memref.copy %alloc_11, %subview_13 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_14 : memref<1x64x56x56xf32>)
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    memref.copy %alloc_14, %alloc_15 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_12, %2 : memref<1x64x58x58xf32>, memref<64x64x3x3xf32>) outs(%alloc_15 : memref<1x64x56x56xf32>)
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_15, %1, %1, %1, %1 : memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%alloc_16 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_16 : memref<1x64x56x56xf32>) outs(%alloc_17 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x64x58x58xf32>
    linalg.map outs(%alloc_18 : memref<1x64x58x58xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_19 = memref.subview %alloc_18[0, 0, 1, 1] [1, 64, 56, 56] [1, 1, 1, 1] : memref<1x64x58x58xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    memref.copy %alloc_17, %subview_19 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    memref.copy %alloc_14, %alloc_20 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_18, %2 : memref<1x64x58x58xf32>, memref<64x64x3x3xf32>) outs(%alloc_20 : memref<1x64x56x56xf32>)
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_20, %1, %1, %1, %1 : memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%alloc_21 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_21, %alloc_11 : memref<1x64x56x56xf32>, memref<1x64x56x56xf32>) outs(%alloc_22 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_22 : memref<1x64x56x56xf32>) outs(%alloc_23 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x64x58x58xf32>
    linalg.map outs(%alloc_24 : memref<1x64x58x58xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_25 = memref.subview %alloc_24[0, 0, 1, 1] [1, 64, 56, 56] [1, 1, 1, 1] : memref<1x64x58x58xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    memref.copy %alloc_23, %subview_25 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    memref.copy %alloc_14, %alloc_26 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_24, %2 : memref<1x64x58x58xf32>, memref<64x64x3x3xf32>) outs(%alloc_26 : memref<1x64x56x56xf32>)
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_26, %1, %1, %1, %1 : memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%alloc_27 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_27 : memref<1x64x56x56xf32>) outs(%alloc_28 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<1x64x58x58xf32>
    linalg.map outs(%alloc_29 : memref<1x64x58x58xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_30 = memref.subview %alloc_29[0, 0, 1, 1] [1, 64, 56, 56] [1, 1, 1, 1] : memref<1x64x58x58xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    memref.copy %alloc_28, %subview_30 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    memref.copy %alloc_14, %alloc_31 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_29, %2 : memref<1x64x58x58xf32>, memref<64x64x3x3xf32>) outs(%alloc_31 : memref<1x64x56x56xf32>)
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_31, %1, %1, %1, %1 : memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%alloc_32 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_32, %alloc_23 : memref<1x64x56x56xf32>, memref<1x64x56x56xf32>) outs(%alloc_33 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_33 : memref<1x64x56x56xf32>) outs(%alloc_34 : memref<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<1x64x58x58xf32>
    linalg.map outs(%alloc_35 : memref<1x64x58x58xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_36 = memref.subview %alloc_35[0, 0, 1, 1] [1, 64, 56, 56] [1, 1, 1, 1] : memref<1x64x58x58xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    memref.copy %alloc_34, %subview_36 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32, strided<[215296, 3364, 58, 1], offset: 59>>
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_37 : memref<1x128x28x28xf32>)
    %alloc_38 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    memref.copy %alloc_37, %alloc_38 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc_35, %3 : memref<1x64x58x58xf32>, memref<128x64x3x3xf32>) outs(%alloc_38 : memref<1x128x28x28xf32>)
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_38, %4, %4, %4, %4 : memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>) outs(%alloc_39 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_39 : memref<1x128x28x28xf32>) outs(%alloc_40 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<1x128x30x30xf32>
    linalg.map outs(%alloc_41 : memref<1x128x30x30xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_42 = memref.subview %alloc_41[0, 0, 1, 1] [1, 128, 28, 28] [1, 1, 1, 1] : memref<1x128x30x30xf32> to memref<1x128x28x28xf32, strided<[115200, 900, 30, 1], offset: 31>>
    memref.copy %alloc_40, %subview_42 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32, strided<[115200, 900, 30, 1], offset: 31>>
    %alloc_43 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    memref.copy %alloc_37, %alloc_43 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_41, %5 : memref<1x128x30x30xf32>, memref<128x128x3x3xf32>) outs(%alloc_43 : memref<1x128x28x28xf32>)
    %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_43, %4, %4, %4, %4 : memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>) outs(%alloc_44 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_45 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    memref.copy %alloc_37, %alloc_45 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc_34, %6 : memref<1x64x56x56xf32>, memref<128x64x1x1xf32>) outs(%alloc_45 : memref<1x128x28x28xf32>)
    %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_45, %4, %4, %4, %4 : memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>) outs(%alloc_46 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_47 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_44, %alloc_46 : memref<1x128x28x28xf32>, memref<1x128x28x28xf32>) outs(%alloc_47 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_47 : memref<1x128x28x28xf32>) outs(%alloc_48 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<1x128x30x30xf32>
    linalg.map outs(%alloc_49 : memref<1x128x30x30xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_50 = memref.subview %alloc_49[0, 0, 1, 1] [1, 128, 28, 28] [1, 1, 1, 1] : memref<1x128x30x30xf32> to memref<1x128x28x28xf32, strided<[115200, 900, 30, 1], offset: 31>>
    memref.copy %alloc_48, %subview_50 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32, strided<[115200, 900, 30, 1], offset: 31>>
    %alloc_51 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    memref.copy %alloc_37, %alloc_51 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_49, %5 : memref<1x128x30x30xf32>, memref<128x128x3x3xf32>) outs(%alloc_51 : memref<1x128x28x28xf32>)
    %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_51, %4, %4, %4, %4 : memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>) outs(%alloc_52 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_53 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_52 : memref<1x128x28x28xf32>) outs(%alloc_53 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<1x128x30x30xf32>
    linalg.map outs(%alloc_54 : memref<1x128x30x30xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_55 = memref.subview %alloc_54[0, 0, 1, 1] [1, 128, 28, 28] [1, 1, 1, 1] : memref<1x128x30x30xf32> to memref<1x128x28x28xf32, strided<[115200, 900, 30, 1], offset: 31>>
    memref.copy %alloc_53, %subview_55 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32, strided<[115200, 900, 30, 1], offset: 31>>
    %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    memref.copy %alloc_37, %alloc_56 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_54, %5 : memref<1x128x30x30xf32>, memref<128x128x3x3xf32>) outs(%alloc_56 : memref<1x128x28x28xf32>)
    %alloc_57 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_56, %4, %4, %4, %4 : memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>) outs(%alloc_57 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_58 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_57, %alloc_48 : memref<1x128x28x28xf32>, memref<1x128x28x28xf32>) outs(%alloc_58 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_58 : memref<1x128x28x28xf32>) outs(%alloc_59 : memref<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_60 = memref.alloc() {alignment = 64 : i64} : memref<1x128x30x30xf32>
    linalg.map outs(%alloc_60 : memref<1x128x30x30xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_61 = memref.subview %alloc_60[0, 0, 1, 1] [1, 128, 28, 28] [1, 1, 1, 1] : memref<1x128x30x30xf32> to memref<1x128x28x28xf32, strided<[115200, 900, 30, 1], offset: 31>>
    memref.copy %alloc_59, %subview_61 : memref<1x128x28x28xf32> to memref<1x128x28x28xf32, strided<[115200, 900, 30, 1], offset: 31>>
    %alloc_62 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_62 : memref<1x256x14x14xf32>)
    %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    memref.copy %alloc_62, %alloc_63 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc_60, %7 : memref<1x128x30x30xf32>, memref<256x128x3x3xf32>) outs(%alloc_63 : memref<1x256x14x14xf32>)
    %alloc_64 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_63, %8, %8, %8, %8 : memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>) outs(%alloc_64 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_65 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_64 : memref<1x256x14x14xf32>) outs(%alloc_65 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_66 = memref.alloc() {alignment = 64 : i64} : memref<1x256x16x16xf32>
    linalg.map outs(%alloc_66 : memref<1x256x16x16xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_67 = memref.subview %alloc_66[0, 0, 1, 1] [1, 256, 14, 14] [1, 1, 1, 1] : memref<1x256x16x16xf32> to memref<1x256x14x14xf32, strided<[65536, 256, 16, 1], offset: 17>>
    memref.copy %alloc_65, %subview_67 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32, strided<[65536, 256, 16, 1], offset: 17>>
    %alloc_68 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    memref.copy %alloc_62, %alloc_68 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_66, %9 : memref<1x256x16x16xf32>, memref<256x256x3x3xf32>) outs(%alloc_68 : memref<1x256x14x14xf32>)
    %alloc_69 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_68, %8, %8, %8, %8 : memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>) outs(%alloc_69 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    memref.copy %alloc_62, %alloc_70 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc_59, %10 : memref<1x128x28x28xf32>, memref<256x128x1x1xf32>) outs(%alloc_70 : memref<1x256x14x14xf32>)
    %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_70, %8, %8, %8, %8 : memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>) outs(%alloc_71 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_72 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_69, %alloc_71 : memref<1x256x14x14xf32>, memref<1x256x14x14xf32>) outs(%alloc_72 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %alloc_73 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_72 : memref<1x256x14x14xf32>) outs(%alloc_73 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_74 = memref.alloc() {alignment = 64 : i64} : memref<1x256x16x16xf32>
    linalg.map outs(%alloc_74 : memref<1x256x16x16xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_75 = memref.subview %alloc_74[0, 0, 1, 1] [1, 256, 14, 14] [1, 1, 1, 1] : memref<1x256x16x16xf32> to memref<1x256x14x14xf32, strided<[65536, 256, 16, 1], offset: 17>>
    memref.copy %alloc_73, %subview_75 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32, strided<[65536, 256, 16, 1], offset: 17>>
    %alloc_76 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    memref.copy %alloc_62, %alloc_76 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_74, %9 : memref<1x256x16x16xf32>, memref<256x256x3x3xf32>) outs(%alloc_76 : memref<1x256x14x14xf32>)
    %alloc_77 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_76, %8, %8, %8, %8 : memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>) outs(%alloc_77 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_78 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_77 : memref<1x256x14x14xf32>) outs(%alloc_78 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_79 = memref.alloc() {alignment = 64 : i64} : memref<1x256x16x16xf32>
    linalg.map outs(%alloc_79 : memref<1x256x16x16xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_80 = memref.subview %alloc_79[0, 0, 1, 1] [1, 256, 14, 14] [1, 1, 1, 1] : memref<1x256x16x16xf32> to memref<1x256x14x14xf32, strided<[65536, 256, 16, 1], offset: 17>>
    memref.copy %alloc_78, %subview_80 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32, strided<[65536, 256, 16, 1], offset: 17>>
    %alloc_81 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    memref.copy %alloc_62, %alloc_81 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_79, %9 : memref<1x256x16x16xf32>, memref<256x256x3x3xf32>) outs(%alloc_81 : memref<1x256x14x14xf32>)
    %alloc_82 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_81, %8, %8, %8, %8 : memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>) outs(%alloc_82 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_82, %alloc_73 : memref<1x256x14x14xf32>, memref<1x256x14x14xf32>) outs(%alloc_83 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %alloc_84 = memref.alloc() {alignment = 64 : i64} : memref<1x256x14x14xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_83 : memref<1x256x14x14xf32>) outs(%alloc_84 : memref<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_85 = memref.alloc() {alignment = 64 : i64} : memref<1x256x16x16xf32>
    linalg.map outs(%alloc_85 : memref<1x256x16x16xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_86 = memref.subview %alloc_85[0, 0, 1, 1] [1, 256, 14, 14] [1, 1, 1, 1] : memref<1x256x16x16xf32> to memref<1x256x14x14xf32, strided<[65536, 256, 16, 1], offset: 17>>
    memref.copy %alloc_84, %subview_86 : memref<1x256x14x14xf32> to memref<1x256x14x14xf32, strided<[65536, 256, 16, 1], offset: 17>>
    %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_87 : memref<1x512x7x7xf32>)
    %alloc_88 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    memref.copy %alloc_87, %alloc_88 : memref<1x512x7x7xf32> to memref<1x512x7x7xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc_85, %11 : memref<1x256x16x16xf32>, memref<512x256x3x3xf32>) outs(%alloc_88 : memref<1x512x7x7xf32>)
    %alloc_89 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_88, %12, %12, %12, %12 : memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>) outs(%alloc_89 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_90 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_89 : memref<1x512x7x7xf32>) outs(%alloc_90 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_91 = memref.alloc() {alignment = 64 : i64} : memref<1x512x9x9xf32>
    linalg.map outs(%alloc_91 : memref<1x512x9x9xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_92 = memref.subview %alloc_91[0, 0, 1, 1] [1, 512, 7, 7] [1, 1, 1, 1] : memref<1x512x9x9xf32> to memref<1x512x7x7xf32, strided<[41472, 81, 9, 1], offset: 10>>
    memref.copy %alloc_90, %subview_92 : memref<1x512x7x7xf32> to memref<1x512x7x7xf32, strided<[41472, 81, 9, 1], offset: 10>>
    %alloc_93 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    memref.copy %alloc_87, %alloc_93 : memref<1x512x7x7xf32> to memref<1x512x7x7xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_91, %13 : memref<1x512x9x9xf32>, memref<512x512x3x3xf32>) outs(%alloc_93 : memref<1x512x7x7xf32>)
    %alloc_94 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_93, %12, %12, %12, %12 : memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>) outs(%alloc_94 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    memref.copy %alloc_87, %alloc_95 : memref<1x512x7x7xf32> to memref<1x512x7x7xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc_84, %14 : memref<1x256x14x14xf32>, memref<512x256x1x1xf32>) outs(%alloc_95 : memref<1x512x7x7xf32>)
    %alloc_96 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_95, %12, %12, %12, %12 : memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>) outs(%alloc_96 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_97 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_94, %alloc_96 : memref<1x512x7x7xf32>, memref<1x512x7x7xf32>) outs(%alloc_97 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %alloc_98 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_97 : memref<1x512x7x7xf32>) outs(%alloc_98 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_99 = memref.alloc() {alignment = 64 : i64} : memref<1x512x9x9xf32>
    linalg.map outs(%alloc_99 : memref<1x512x9x9xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_100 = memref.subview %alloc_99[0, 0, 1, 1] [1, 512, 7, 7] [1, 1, 1, 1] : memref<1x512x9x9xf32> to memref<1x512x7x7xf32, strided<[41472, 81, 9, 1], offset: 10>>
    memref.copy %alloc_98, %subview_100 : memref<1x512x7x7xf32> to memref<1x512x7x7xf32, strided<[41472, 81, 9, 1], offset: 10>>
    %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    memref.copy %alloc_87, %alloc_101 : memref<1x512x7x7xf32> to memref<1x512x7x7xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_99, %13 : memref<1x512x9x9xf32>, memref<512x512x3x3xf32>) outs(%alloc_101 : memref<1x512x7x7xf32>)
    %alloc_102 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_101, %12, %12, %12, %12 : memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>) outs(%alloc_102 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_103 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_102 : memref<1x512x7x7xf32>) outs(%alloc_103 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_104 = memref.alloc() {alignment = 64 : i64} : memref<1x512x9x9xf32>
    linalg.map outs(%alloc_104 : memref<1x512x9x9xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview_105 = memref.subview %alloc_104[0, 0, 1, 1] [1, 512, 7, 7] [1, 1, 1, 1] : memref<1x512x9x9xf32> to memref<1x512x7x7xf32, strided<[41472, 81, 9, 1], offset: 10>>
    memref.copy %alloc_103, %subview_105 : memref<1x512x7x7xf32> to memref<1x512x7x7xf32, strided<[41472, 81, 9, 1], offset: 10>>
    %alloc_106 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    memref.copy %alloc_87, %alloc_106 : memref<1x512x7x7xf32> to memref<1x512x7x7xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_104, %13 : memref<1x512x9x9xf32>, memref<512x512x3x3xf32>) outs(%alloc_106 : memref<1x512x7x7xf32>)
    %alloc_107 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_106, %12, %12, %12, %12 : memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>) outs(%alloc_107 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %alloc_108 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_107, %alloc_98 : memref<1x512x7x7xf32>, memref<1x512x7x7xf32>) outs(%alloc_108 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %alloc_109 = memref.alloc() {alignment = 64 : i64} : memref<1x512x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_108 : memref<1x512x7x7xf32>) outs(%alloc_109 : memref<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %alloc_110 = memref.alloc() {alignment = 64 : i64} : memref<1x512x1x1xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_110 : memref<1x512x1x1xf32>)
    %alloc_111 = memref.alloc() {alignment = 64 : i64} : memref<7x7xf32>
    %alloc_112 = memref.alloc() {alignment = 64 : i64} : memref<1x512x1x1xf32>
    memref.copy %alloc_110, %alloc_112 : memref<1x512x1x1xf32> to memref<1x512x1x1xf32>
    linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<7> : vector<2xi64>} ins(%alloc_109, %alloc_111 : memref<1x512x7x7xf32>, memref<7x7xf32>) outs(%alloc_112 : memref<1x512x1x1xf32>)
    %alloc_113 = memref.alloc() {alignment = 64 : i64} : memref<1x512x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_112 : memref<1x512x1x1xf32>) outs(%alloc_113 : memref<1x512x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.divf %in, %cst : f32
      linalg.yield %17 : f32
    }
    %collapse_shape = memref.collapse_shape %alloc_113 [[0], [1, 2, 3]] : memref<1x512x1x1xf32> into memref<1x512xf32>
    %alloc_114 = memref.alloc() {alignment = 64 : i64} : memref<512x1000xf32>
    linalg.transpose ins(%16 : memref<1000x512xf32>) outs(%alloc_114 : memref<512x1000xf32>) permutation = [1, 0] 
    %alloc_115 = memref.alloc() {alignment = 64 : i64} : memref<1x1000xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_115 : memref<1x1000xf32>)
    %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<1x1000xf32>
    memref.copy %alloc_115, %alloc_116 : memref<1x1000xf32> to memref<1x1000xf32>
    linalg.matmul ins(%collapse_shape, %alloc_114 : memref<1x512xf32>, memref<512x1000xf32>) outs(%alloc_116 : memref<1x1000xf32>)
    %alloc_117 = memref.alloc() {alignment = 64 : i64} : memref<1x1000xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%alloc_116, %15 : memref<1x1000xf32>, memref<1000xf32>) outs(%alloc_117 : memref<1x1000xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    return %alloc_117 : memref<1x1000xf32>
  }
}
