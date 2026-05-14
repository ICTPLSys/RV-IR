#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "ResNet"} {
  func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<64x3x7x7xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.000000e-05 : f64
    %cst_3 = arith.constant 4.900000e+01 : f32
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_7 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_8 = arith.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_9 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_10 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_11 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_12 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_13 = arith.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_14 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_15 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_16 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_17 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_18 = arith.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_19 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_20 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_21 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_22 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_23 = arith.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_24 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_25 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_26 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_27 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_28 = arith.constant dense_resource<__elided__> : tensor<128x64x3x3xf32>
    %cst_29 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_30 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_31 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_32 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_33 = arith.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_34 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_35 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_36 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_37 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_38 = arith.constant dense_resource<__elided__> : tensor<128x64x1x1xf32>
    %cst_39 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_40 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_41 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_42 = arith.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_43 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_44 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_45 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_46 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_47 = arith.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_48 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_49 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_50 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_51 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_52 = arith.constant dense_resource<__elided__> : tensor<256x128x3x3xf32>
    %cst_53 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_54 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_55 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_56 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_57 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_58 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_59 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_60 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_61 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_62 = arith.constant dense_resource<__elided__> : tensor<256x128x1x1xf32>
    %cst_63 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_64 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_65 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_66 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_67 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_68 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_69 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_70 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_71 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_72 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_73 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_74 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_75 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_76 = arith.constant dense_resource<__elided__> : tensor<512x256x3x3xf32>
    %cst_77 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_78 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_79 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_80 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_81 = arith.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_82 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_83 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_84 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_85 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_86 = arith.constant dense_resource<__elided__> : tensor<512x256x1x1xf32>
    %cst_87 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_88 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_89 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_90 = arith.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_91 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_92 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_93 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_94 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_95 = arith.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_96 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_97 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_98 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_99 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_100 = arith.constant dense_resource<__elided__> : tensor<1000xf32>
    %cst_101 = arith.constant dense_resource<__elided__> : tensor<1000x512xf32>
    %padded = tensor.pad %arg0 low[0, 0, 3, 3] high[0, 0, 3, 3] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x230x230xf32>
    %0 = tensor.empty() : tensor<1x64x112x112xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %cst : tensor<1x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%1 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %cst_7, %cst_6, %cst_5, %cst_4 : tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%2 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x64x112x112xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x64x112x112xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x64x112x112xf32>
    %padded_102 = tensor.pad %4 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x64x112x112xf32> to tensor<1x64x114x114xf32>
    %5 = tensor.empty() : tensor<1x64x56x56xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %7 = tensor.empty() : tensor<3x3xf32>
    %8 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_102, %7 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%6 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %padded_103 = tensor.pad %8 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %10 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_103, %cst_8 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %cst_12, %cst_11, %cst_10, %cst_9 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%10 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x64x56x56xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_104 = tensor.pad %12 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %13 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_104, %cst_13 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %cst_17, %cst_16, %cst_15, %cst_14 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%13 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x64x56x56xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %8 : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x64x56x56xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_105 = tensor.pad %16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %17 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_105, %cst_18 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %cst_22, %cst_21, %cst_20, %cst_19 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%17 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x64x56x56xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_106 = tensor.pad %19 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %20 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_106, %cst_23 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %cst_27, %cst_26, %cst_25, %cst_24 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%20 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x64x56x56xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %16 : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x64x56x56xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x64x56x56xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_107 = tensor.pad %23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %24 = tensor.empty() : tensor<1x128x28x28xf32>
    %25 = linalg.fill ins(%cst_0 : f32) outs(%24 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %26 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_107, %cst_28 : tensor<1x64x58x58xf32>, tensor<128x64x3x3xf32>) outs(%25 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %cst_32, %cst_31, %cst_30, %cst_29 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%26 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x128x28x28xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<1x128x28x28xf32>) outs(%24 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_108 = tensor.pad %28 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %29 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_108, %cst_33 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%25 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29, %cst_37, %cst_36, %cst_35, %cst_34 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%29 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x128x28x28xf32>
    %31 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%23, %cst_38 : tensor<1x64x56x56xf32>, tensor<128x64x1x1xf32>) outs(%25 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %32 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %cst_41, %cst_36, %cst_40, %cst_39 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%31 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x128x28x28xf32>
    %33 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %32 : tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) outs(%24 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x128x28x28xf32>
    %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<1x128x28x28xf32>) outs(%24 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_109 = tensor.pad %34 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %35 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_109, %cst_42 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%25 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %36 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %cst_46, %cst_45, %cst_44, %cst_43 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%35 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x128x28x28xf32>
    %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<1x128x28x28xf32>) outs(%24 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_110 = tensor.pad %37 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %38 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_110, %cst_47 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%25 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %39 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38, %cst_51, %cst_50, %cst_49, %cst_48 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%38 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x128x28x28xf32>
    %40 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39, %34 : tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) outs(%24 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x128x28x28xf32>
    %41 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40 : tensor<1x128x28x28xf32>) outs(%24 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_111 = tensor.pad %41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %42 = tensor.empty() : tensor<1x256x14x14xf32>
    %43 = linalg.fill ins(%cst_0 : f32) outs(%42 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %44 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_111, %cst_52 : tensor<1x128x30x30xf32>, tensor<256x128x3x3xf32>) outs(%43 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %45 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %cst_56, %cst_55, %cst_54, %cst_53 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%44 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x256x14x14xf32>
    %46 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%45 : tensor<1x256x14x14xf32>) outs(%42 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_112 = tensor.pad %46 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %47 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_112, %cst_57 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%43 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %48 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47, %cst_61, %cst_60, %cst_59, %cst_58 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%47 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x256x14x14xf32>
    %49 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%41, %cst_62 : tensor<1x128x28x28xf32>, tensor<256x128x1x1xf32>) outs(%43 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %50 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49, %cst_65, %cst_60, %cst_64, %cst_63 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%49 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x256x14x14xf32>
    %51 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%48, %50 : tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) outs(%42 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x256x14x14xf32>
    %52 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51 : tensor<1x256x14x14xf32>) outs(%42 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_113 = tensor.pad %52 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %53 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_113, %cst_66 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%43 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %54 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53, %cst_70, %cst_69, %cst_68, %cst_67 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%53 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x256x14x14xf32>
    %55 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%54 : tensor<1x256x14x14xf32>) outs(%42 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_114 = tensor.pad %55 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %56 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_114, %cst_71 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%43 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %57 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%56, %cst_75, %cst_74, %cst_73, %cst_72 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%56 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x256x14x14xf32>
    %58 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57, %52 : tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) outs(%42 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x256x14x14xf32>
    %59 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%58 : tensor<1x256x14x14xf32>) outs(%42 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_115 = tensor.pad %59 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %60 = tensor.empty() : tensor<1x512x7x7xf32>
    %61 = linalg.fill ins(%cst_0 : f32) outs(%60 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %62 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_115, %cst_76 : tensor<1x256x16x16xf32>, tensor<512x256x3x3xf32>) outs(%61 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %63 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %cst_80, %cst_79, %cst_78, %cst_77 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%62 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x512x7x7xf32>
    %64 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63 : tensor<1x512x7x7xf32>) outs(%60 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_116 = tensor.pad %64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %65 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_116, %cst_81 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%61 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %66 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %cst_85, %cst_84, %cst_83, %cst_82 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%65 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x512x7x7xf32>
    %67 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%59, %cst_86 : tensor<1x256x14x14xf32>, tensor<512x256x1x1xf32>) outs(%61 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %68 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %cst_89, %cst_84, %cst_88, %cst_87 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%67 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x512x7x7xf32>
    %69 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %68 : tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) outs(%60 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x512x7x7xf32>
    %70 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69 : tensor<1x512x7x7xf32>) outs(%60 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_117 = tensor.pad %70 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %71 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_117, %cst_90 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%61 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %72 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %cst_94, %cst_93, %cst_92, %cst_91 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%71 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x512x7x7xf32>
    %73 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72 : tensor<1x512x7x7xf32>) outs(%60 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_118 = tensor.pad %73 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %74 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_118, %cst_95 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%61 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %75 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%74, %cst_99, %cst_98, %cst_97, %cst_96 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%74 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_119: f32, %in_120: f32, %in_121: f32, %in_122: f32, %out: f32):
      %88 = arith.truncf %cst_2 : f64 to f32
      %89 = arith.addf %in_122, %88 : f32
      %90 = math.rsqrt %89 : f32
      %91 = arith.subf %in, %in_121 : f32
      %92 = arith.mulf %91, %90 : f32
      %93 = arith.mulf %92, %in_119 : f32
      %94 = arith.addf %93, %in_120 : f32
      linalg.yield %94 : f32
    } -> tensor<1x512x7x7xf32>
    %76 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %70 : tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) outs(%60 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x512x7x7xf32>
    %77 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76 : tensor<1x512x7x7xf32>) outs(%60 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.cmpf ugt, %in, %cst_0 : f32
      %89 = arith.select %88, %in, %cst_0 : f32
      linalg.yield %89 : f32
    } -> tensor<1x512x7x7xf32>
    %78 = tensor.empty() : tensor<1x512x1x1xf32>
    %79 = linalg.fill ins(%cst_0 : f32) outs(%78 : tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %80 = tensor.empty() : tensor<7x7xf32>
    %81 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<7> : vector<2xi64>} ins(%77, %80 : tensor<1x512x7x7xf32>, tensor<7x7xf32>) outs(%79 : tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %82 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%81 : tensor<1x512x1x1xf32>) outs(%78 : tensor<1x512x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.divf %in, %cst_3 : f32
      linalg.yield %88 : f32
    } -> tensor<1x512x1x1xf32>
    %collapsed = tensor.collapse_shape %82 [[0], [1, 2, 3]] : tensor<1x512x1x1xf32> into tensor<1x512xf32>
    %83 = tensor.empty() : tensor<512x1000xf32>
    %transposed = linalg.transpose ins(%cst_101 : tensor<1000x512xf32>) outs(%83 : tensor<512x1000xf32>) permutation = [1, 0] 
    %84 = tensor.empty() : tensor<1x1000xf32>
    %85 = linalg.fill ins(%cst_0 : f32) outs(%84 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %86 = linalg.matmul ins(%collapsed, %transposed : tensor<1x512xf32>, tensor<512x1000xf32>) outs(%85 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %87 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%86, %cst_100 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%84 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_119: f32, %out: f32):
      %88 = arith.addf %in, %in_119 : f32
      linalg.yield %88 : f32
    } -> tensor<1x1000xf32>
    return %87 : tensor<1x1000xf32>
  }
}
