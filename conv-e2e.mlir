// *** IR Dump After mlir::iree_compiler::IREE::Flow::(anonymous namespace)::FlattenTuplesInCFGPass ***
module  {
  func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export} {
    %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
    return %0 : tensor<1x112x112x32xf32>
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::Flow::LegalizeInputTypesPass ***
module  {
  func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export} {
    %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
    return %0 : tensor<1x112x112x32xf32>
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::Flow::MaterializeExportedReflectionPass ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32> {iree.reflection = {f_partial = "I18!B14!d1d225d225d16"}}, %arg1: tensor<3x3x16x32xf32> {iree.reflection = {f_partial = "I15!B11!d3d3d16d32"}}) -> (tensor<1x112x112x32xf32> {iree.reflection = {f_partial = "R18!B14!d1d112d112d32"}}) attributes {iree.module.export} {
  %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After mlir::iree_compiler::Shape::(anonymous namespace)::ExpandFunctionDynamicDimsPass ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32> {iree.reflection = {f_partial = "I18!B14!d1d225d225d16"}}, %arg1: tensor<3x3x16x32xf32> {iree.reflection = {f_partial = "I15!B11!d3d3d16d32"}}) -> (tensor<1x112x112x32xf32> {iree.reflection = {f_partial = "R18!B14!d1d112d112d32"}}) attributes {iree.module.export} {
  %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After mlir::iree_compiler::IREE::Flow::MergeExportedReflectionPass ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::ConvertHLOToLinalgOnTensorsPass ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %cst = constant 0.000000e+00 : f32
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32> 
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) outs(%1 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  return %2 : tensor<1x112x112x32xf32>
}

// *** IR Dump After LinalgFoldUnitExtentDims ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32> 
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) outs(%1 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  return %2 : tensor<1x112x112x32xf32>
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32> 
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) outs(%1 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  return %2 : tensor<1x112x112x32xf32>
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::FusionOfTensorOpsPass ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32> 
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) outs(%1 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  return %2 : tensor<1x112x112x32xf32>
}

// *** IR Dump After mlir::iree_compiler::IREE::Flow::(anonymous namespace)::DispatchLinalgOnTensorsPass ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c1 = constant 1 : index
  %c32 = constant 32 : index
  %c112 = constant 112 : index
  %0 = flow.dispatch.workgroups[%c32, %c112, %c112, %c1] (%arg0, %arg1) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> = (%arg2 : !flow.dispatch.input<1x225x225x16xf32>, %arg3 : !flow.dispatch.input<3x3x16x32xf32>, %arg4 : !flow.dispatch.output<1x112x112x32xf32>) {
    %cst = constant 0.000000e+00 : f32
    %c32_0 = constant 32 : index
    %c112_1 = constant 112 : index
    %c3 = constant 3 : index
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %c1_2 = constant 1 : index
    %1 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
    %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
    %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
    %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
    %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
    %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
    %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
    %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
    %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
    %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
    %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
    %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
    %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
    %2 = muli %workgroup_size_3, %workgroup_id_3 : index
    %3 = muli %workgroup_size_3, %workgroup_count_3 : index
    scf.for %arg5 = %2 to %c1_2 step %3 {
      %4 = muli %workgroup_size_2, %workgroup_id_2 : index
      %5 = muli %workgroup_size_2, %workgroup_count_2 : index
      scf.for %arg6 = %4 to %c112_1 step %5 {
        %6 = muli %workgroup_size_1, %workgroup_id_1 : index
        %7 = muli %workgroup_size_1, %workgroup_count_1 : index
        scf.for %arg7 = %6 to %c112_1 step %7 {
          %8 = muli %workgroup_size_0, %workgroup_id_0 : index
          %9 = muli %workgroup_size_0, %workgroup_count_0 : index
          scf.for %arg8 = %8 to %c32_0 step %9 {
            %10 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 1)>(%arg5, %workgroup_size_3)
            %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg6)
            %12 = affine.min affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>(%workgroup_size_2, %arg6)
            %13 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg7)
            %14 = affine.min affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>(%workgroup_size_1, %arg7)
            %15 = flow.dispatch.input.load %arg2, offsets = [%arg5, %11, %13, %c0], sizes = [%10, %12, %14, %c16], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
            %16 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 32)>(%arg8, %workgroup_size_0)
            %17 = flow.dispatch.input.load %arg3, offsets = [%c0, %c0, %c0, %arg8], sizes = [%c3, %c3, %c16, %16], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
            %18 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 1)>(%arg5, %workgroup_size_3)
            %19 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 112)>(%arg6, %workgroup_size_2)
            %20 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 112)>(%arg7, %workgroup_size_1)
            %21 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 32)>(%arg8, %workgroup_size_0)
            %22 = subtensor %1[%arg5, %arg6, %arg7, %arg8] [%18, %19, %20, %21] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<?x?x?x?xf32>
            %23 = linalg.fill(%22, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
            %24 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%15, %17 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%23 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
            flow.dispatch.output.store %24, %arg4, offsets = [%arg5, %arg6, %arg7, %arg8], sizes = [%18, %19, %20, %21], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
          }
        }
      }
    }
    flow.return
  }
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After Canonicalizer ***
#map0 = affine_map<(d0, d1) -> (d1, -d0 + 1)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>
#map3 = affine_map<(d0, d1) -> (d1, -d0 + 32)>
#map4 = affine_map<(d0, d1) -> (d1, -d0 + 112)>
module  {
  func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c1 = constant 1 : index
    %c32 = constant 32 : index
    %c112 = constant 112 : index
    %0 = flow.dispatch.workgroups[%c32, %c112, %c112, %c1] (%arg0, %arg1) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> = (%arg2 : !flow.dispatch.input<1x225x225x16xf32>, %arg3 : !flow.dispatch.input<3x3x16x32xf32>, %arg4 : !flow.dispatch.output<1x112x112x32xf32>) {
      %cst = constant 0.000000e+00 : f32
      %c32_0 = constant 32 : index
      %c112_1 = constant 112 : index
      %c3 = constant 3 : index
      %c0 = constant 0 : index
      %c16 = constant 16 : index
      %c1_2 = constant 1 : index
      %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
      %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
      %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
      %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
      %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
      %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
      %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
      %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
      %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
      %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
      %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
      %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
      %1 = muli %workgroup_size_3, %workgroup_id_3 : index
      %2 = muli %workgroup_size_3, %workgroup_count_3 : index
      scf.for %arg5 = %1 to %c1_2 step %2 {
        %3 = muli %workgroup_size_2, %workgroup_id_2 : index
        %4 = muli %workgroup_size_2, %workgroup_count_2 : index
        scf.for %arg6 = %3 to %c112_1 step %4 {
          %5 = muli %workgroup_size_1, %workgroup_id_1 : index
          %6 = muli %workgroup_size_1, %workgroup_count_1 : index
          scf.for %arg7 = %5 to %c112_1 step %6 {
            %7 = muli %workgroup_size_0, %workgroup_id_0 : index
            %8 = muli %workgroup_size_0, %workgroup_count_0 : index
            scf.for %arg8 = %7 to %c32_0 step %8 {
              %9 = affine.min #map0(%arg5, %workgroup_size_3)
              %10 = affine.apply #map1(%arg6)
              %11 = affine.min #map2(%workgroup_size_2, %arg6)
              %12 = affine.apply #map1(%arg7)
              %13 = affine.min #map2(%workgroup_size_1, %arg7)
              %14 = flow.dispatch.input.load %arg2, offsets = [%arg5, %10, %12, %c0], sizes = [%9, %11, %13, %c16], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
              %15 = affine.min #map3(%arg8, %workgroup_size_0)
              %16 = flow.dispatch.input.load %arg3, offsets = [%c0, %c0, %c0, %arg8], sizes = [%c3, %c3, %c16, %15], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
              %17 = affine.min #map0(%arg5, %workgroup_size_3)
              %18 = affine.min #map4(%arg6, %workgroup_size_2)
              %19 = affine.min #map4(%arg7, %workgroup_size_1)
              %20 = affine.min #map3(%arg8, %workgroup_size_0)
              %21 = linalg.init_tensor [%17, %18, %19, %20] : tensor<?x?x?x?xf32>
              %22 = linalg.fill(%21, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
              %23 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%14, %16 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%22 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
              flow.dispatch.output.store %23, %arg4, offsets = [%arg5, %arg6, %arg7, %arg8], sizes = [%17, %18, %19, %20], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
            }
          }
        }
      }
      flow.return
    }
    return %0 : tensor<1x112x112x32xf32>
  }
}


// *** IR Dump After CSE ***
#map0 = affine_map<(d0, d1) -> (d1, -d0 + 1)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>
#map3 = affine_map<(d0, d1) -> (d1, -d0 + 32)>
#map4 = affine_map<(d0, d1) -> (d1, -d0 + 112)>
module  {
  func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c1 = constant 1 : index
    %c32 = constant 32 : index
    %c112 = constant 112 : index
    %0 = flow.dispatch.workgroups[%c32, %c112, %c112, %c1] (%arg0, %arg1) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> = (%arg2 : !flow.dispatch.input<1x225x225x16xf32>, %arg3 : !flow.dispatch.input<3x3x16x32xf32>, %arg4 : !flow.dispatch.output<1x112x112x32xf32>) {
      %cst = constant 0.000000e+00 : f32
      %c32_0 = constant 32 : index
      %c112_1 = constant 112 : index
      %c3 = constant 3 : index
      %c0 = constant 0 : index
      %c16 = constant 16 : index
      %c1_2 = constant 1 : index
      %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
      %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
      %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
      %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
      %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
      %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
      %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
      %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
      %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
      %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
      %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
      %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
      %1 = muli %workgroup_size_3, %workgroup_id_3 : index
      %2 = muli %workgroup_size_3, %workgroup_count_3 : index
      scf.for %arg5 = %1 to %c1_2 step %2 {
        %3 = muli %workgroup_size_2, %workgroup_id_2 : index
        %4 = muli %workgroup_size_2, %workgroup_count_2 : index
        scf.for %arg6 = %3 to %c112_1 step %4 {
          %5 = muli %workgroup_size_1, %workgroup_id_1 : index
          %6 = muli %workgroup_size_1, %workgroup_count_1 : index
          scf.for %arg7 = %5 to %c112_1 step %6 {
            %7 = muli %workgroup_size_0, %workgroup_id_0 : index
            %8 = muli %workgroup_size_0, %workgroup_count_0 : index
            scf.for %arg8 = %7 to %c32_0 step %8 {
              %9 = affine.min #map0(%arg5, %workgroup_size_3)
              %10 = affine.apply #map1(%arg6)
              %11 = affine.min #map2(%workgroup_size_2, %arg6)
              %12 = affine.apply #map1(%arg7)
              %13 = affine.min #map2(%workgroup_size_1, %arg7)
              %14 = flow.dispatch.input.load %arg2, offsets = [%arg5, %10, %12, %c0], sizes = [%9, %11, %13, %c16], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
              %15 = affine.min #map3(%arg8, %workgroup_size_0)
              %16 = flow.dispatch.input.load %arg3, offsets = [%c0, %c0, %c0, %arg8], sizes = [%c3, %c3, %c16, %15], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
              %17 = affine.min #map4(%arg6, %workgroup_size_2)
              %18 = affine.min #map4(%arg7, %workgroup_size_1)
              %19 = linalg.init_tensor [%9, %17, %18, %15] : tensor<?x?x?x?xf32>
              %20 = linalg.fill(%19, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
              %21 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%14, %16 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%20 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
              flow.dispatch.output.store %21, %arg4, offsets = [%arg5, %arg6, %arg7, %arg8], sizes = [%9, %17, %18, %15], strides = [%c1_2, %c1_2, %c1_2, %c1_2] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
            }
          }
        }
      }
      flow.return
    }
    return %0 : tensor<1x112x112x32xf32>
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::Flow::OutlineDispatchRegions2Pass ***
#map0 = affine_map<(d0, d1) -> (d1, -d0 + 1)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>
#map3 = affine_map<(d0, d1) -> (d1, -d0 + 32)>
#map4 = affine_map<(d0, d1) -> (d1, -d0 + 112)>
module  {
  flow.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    flow.dispatch.entry @predict_ex_dispatch_1_dispatch_0 attributes {signature = (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>, workgroup_rank = 4 : index}
    module  {
      func @predict_ex_dispatch_1_dispatch_0(%arg0: !flow.dispatch.input<1x225x225x16xf32>, %arg1: !flow.dispatch.input<3x3x16x32xf32>, %arg2: !flow.dispatch.output<1x112x112x32xf32>) {
        %cst = constant 0.000000e+00 : f32
        %c32 = constant 32 : index
        %c112 = constant 112 : index
        %c3 = constant 3 : index
        %c0 = constant 0 : index
        %c16 = constant 16 : index
        %c1 = constant 1 : index
        %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
        %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
        %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
        %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
        %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
        %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
        %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
        %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
        %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
        %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
        %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
        %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
        %0 = muli %workgroup_size_3, %workgroup_id_3 : index
        %1 = muli %workgroup_size_3, %workgroup_count_3 : index
        scf.for %arg3 = %0 to %c1 step %1 {
          %2 = muli %workgroup_size_2, %workgroup_id_2 : index
          %3 = muli %workgroup_size_2, %workgroup_count_2 : index
          scf.for %arg4 = %2 to %c112 step %3 {
            %4 = muli %workgroup_size_1, %workgroup_id_1 : index
            %5 = muli %workgroup_size_1, %workgroup_count_1 : index
            scf.for %arg5 = %4 to %c112 step %5 {
              %6 = muli %workgroup_size_0, %workgroup_id_0 : index
              %7 = muli %workgroup_size_0, %workgroup_count_0 : index
              scf.for %arg6 = %6 to %c32 step %7 {
                %8 = affine.min #map0(%arg3, %workgroup_size_3)
                %9 = affine.apply #map1(%arg4)
                %10 = affine.min #map2(%workgroup_size_2, %arg4)
                %11 = affine.apply #map1(%arg5)
                %12 = affine.min #map2(%workgroup_size_1, %arg5)
                %13 = flow.dispatch.input.load %arg0, offsets = [%arg3, %9, %11, %c0], sizes = [%8, %10, %12, %c16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
                %14 = affine.min #map3(%arg6, %workgroup_size_0)
                %15 = flow.dispatch.input.load %arg1, offsets = [%c0, %c0, %c0, %arg6], sizes = [%c3, %c3, %c16, %14], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
                %16 = affine.min #map4(%arg4, %workgroup_size_2)
                %17 = affine.min #map4(%arg5, %workgroup_size_1)
                %18 = linalg.init_tensor [%8, %16, %17, %14] : tensor<?x?x?x?xf32>
                %19 = linalg.fill(%18, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
                %20 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%19 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
                flow.dispatch.output.store %20, %arg2, offsets = [%arg3, %arg4, %arg5, %arg6], sizes = [%8, %16, %17, %14], strides = [%c1, %c1, %c1, %c1] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
              }
            }
          }
        }
        return
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c1 = constant 1 : index
    %c32 = constant 32 : index
    %c112 = constant 112 : index
    %0 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%c32, %c112, %c112, %c1] (%arg0, %arg1) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
    return %0 : tensor<1x112x112x32xf32>
  }
}


// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c1 = constant 1 : index
  %c32 = constant 32 : index
  %c112 = constant 112 : index
  %0 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%c32, %c112, %c112, %c1] (%arg0, %arg1) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c1 = constant 1 : index
  %c32 = constant 32 : index
  %c112 = constant 112 : index
  %0 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%c32, %c112, %c112, %c1] (%arg0, %arg1) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After mlir::iree_compiler::IREE::Flow::(anonymous namespace)::HoistUnstreamableOps ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %0 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%c32, %c112, %c112, %c1] (%arg0, %arg1) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %0 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%c32, %c112, %c112, %c1] (%arg0, %arg1) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After mlir::iree_compiler::IREE::Flow::FormStreamsPass ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %0 = flow.ex.stream.fragment(%arg2 = %c32 : index, %arg3 = %c112 : index, %arg4 = %c1 : index, %arg5 = %arg0 : tensor<1x225x225x16xf32>, %arg6 = %arg1 : tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> {
    %1 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%arg2, %arg3, %arg3, %arg4] (%arg5, %arg6) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
    flow.return %1 : tensor<1x112x112x32xf32>
  }
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %0 = flow.ex.stream.fragment(%arg2 = %c32 : index, %arg3 = %c112 : index, %arg4 = %c1 : index, %arg5 = %arg0 : tensor<1x225x225x16xf32>, %arg6 = %arg1 : tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> {
    %1 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%arg2, %arg3, %arg3, %arg4] (%arg5, %arg6) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
    flow.return %1 : tensor<1x112x112x32xf32>
  }
  return %0 : tensor<1x112x112x32xf32>
}

// *** IR Dump After Canonicalizer ***
#map0 = affine_map<(d0)[s0] -> (s0, -d0 + 1)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 32)>
#map4 = affine_map<(d0)[s0] -> (s0, -d0 + 112)>
module  {
  flow.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    flow.dispatch.entry @predict_ex_dispatch_1_dispatch_0 attributes {signature = (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>, workgroup_rank = 4 : index}
    module  {
      func @predict_ex_dispatch_1_dispatch_0(%arg0: !flow.dispatch.input<1x225x225x16xf32>, %arg1: !flow.dispatch.input<3x3x16x32xf32>, %arg2: !flow.dispatch.output<1x112x112x32xf32>) {
        %cst = constant 0.000000e+00 : f32
        %c32 = constant 32 : index
        %c112 = constant 112 : index
        %c3 = constant 3 : index
        %c0 = constant 0 : index
        %c16 = constant 16 : index
        %c1 = constant 1 : index
        %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
        %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
        %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
        %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
        %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
        %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
        %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
        %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
        %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
        %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
        %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
        %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
        %0 = muli %workgroup_size_3, %workgroup_id_3 : index
        %1 = muli %workgroup_size_3, %workgroup_count_3 : index
        scf.for %arg3 = %0 to %c1 step %1 {
          %2 = muli %workgroup_size_2, %workgroup_id_2 : index
          %3 = muli %workgroup_size_2, %workgroup_count_2 : index
          scf.for %arg4 = %2 to %c112 step %3 {
            %4 = muli %workgroup_size_1, %workgroup_id_1 : index
            %5 = muli %workgroup_size_1, %workgroup_count_1 : index
            scf.for %arg5 = %4 to %c112 step %5 {
              %6 = muli %workgroup_size_0, %workgroup_id_0 : index
              %7 = muli %workgroup_size_0, %workgroup_count_0 : index
              scf.for %arg6 = %6 to %c32 step %7 {
                %8 = affine.min #map0(%arg3)[%workgroup_size_3]
                %9 = affine.apply #map1(%arg4)
                %10 = affine.min #map2(%arg4)[%workgroup_size_2]
                %11 = affine.apply #map1(%arg5)
                %12 = affine.min #map2(%arg5)[%workgroup_size_1]
                %13 = flow.dispatch.input.load %arg0, offsets = [%arg3, %9, %11, %c0], sizes = [%8, %10, %12, %c16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
                %14 = affine.min #map3(%arg6)[%workgroup_size_0]
                %15 = flow.dispatch.input.load %arg1, offsets = [%c0, %c0, %c0, %arg6], sizes = [%c3, %c3, %c16, %14], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
                %16 = affine.min #map4(%arg4)[%workgroup_size_2]
                %17 = affine.min #map4(%arg5)[%workgroup_size_1]
                %18 = linalg.init_tensor [%8, %16, %17, %14] : tensor<?x?x?x?xf32>
                %19 = linalg.fill(%18, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
                %20 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%19 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
                flow.dispatch.output.store %20, %arg2, offsets = [%arg3, %arg4, %arg5, %arg6], sizes = [%8, %16, %17, %14], strides = [%c1, %c1, %c1, %c1] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
              }
            }
          }
        }
        return
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %0 = flow.ex.stream.fragment(%arg2 = %c32 : index, %arg3 = %c112 : index, %arg4 = %c1 : index, %arg5 = %arg0 : tensor<1x225x225x16xf32>, %arg6 = %arg1 : tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> {
      %1 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%arg2, %arg3, %arg3, %arg4] (%arg5, %arg6) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
      flow.return %1 : tensor<1x112x112x32xf32>
    }
    return %0 : tensor<1x112x112x32xf32>
  }
}


// *** IR Dump After Canonicalizer ***
#map0 = affine_map<(d0)[s0] -> (s0, -d0 + 1)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 32)>
#map4 = affine_map<(d0)[s0] -> (s0, -d0 + 112)>
module  {
  flow.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    flow.dispatch.entry @predict_ex_dispatch_1_dispatch_0 attributes {signature = (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>, workgroup_rank = 4 : index}
    module  {
      func @predict_ex_dispatch_1_dispatch_0(%arg0: !flow.dispatch.input<1x225x225x16xf32>, %arg1: !flow.dispatch.input<3x3x16x32xf32>, %arg2: !flow.dispatch.output<1x112x112x32xf32>) {
        %cst = constant 0.000000e+00 : f32
        %c32 = constant 32 : index
        %c112 = constant 112 : index
        %c3 = constant 3 : index
        %c0 = constant 0 : index
        %c16 = constant 16 : index
        %c1 = constant 1 : index
        %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
        %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
        %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
        %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
        %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
        %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
        %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
        %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
        %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
        %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
        %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
        %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
        %0 = muli %workgroup_size_3, %workgroup_id_3 : index
        %1 = muli %workgroup_size_3, %workgroup_count_3 : index
        scf.for %arg3 = %0 to %c1 step %1 {
          %2 = muli %workgroup_size_2, %workgroup_id_2 : index
          %3 = muli %workgroup_size_2, %workgroup_count_2 : index
          scf.for %arg4 = %2 to %c112 step %3 {
            %4 = muli %workgroup_size_1, %workgroup_id_1 : index
            %5 = muli %workgroup_size_1, %workgroup_count_1 : index
            scf.for %arg5 = %4 to %c112 step %5 {
              %6 = muli %workgroup_size_0, %workgroup_id_0 : index
              %7 = muli %workgroup_size_0, %workgroup_count_0 : index
              scf.for %arg6 = %6 to %c32 step %7 {
                %8 = affine.min #map0(%arg3)[%workgroup_size_3]
                %9 = affine.apply #map1(%arg4)
                %10 = affine.min #map2(%arg4)[%workgroup_size_2]
                %11 = affine.apply #map1(%arg5)
                %12 = affine.min #map2(%arg5)[%workgroup_size_1]
                %13 = flow.dispatch.input.load %arg0, offsets = [%arg3, %9, %11, %c0], sizes = [%8, %10, %12, %c16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
                %14 = affine.min #map3(%arg6)[%workgroup_size_0]
                %15 = flow.dispatch.input.load %arg1, offsets = [%c0, %c0, %c0, %arg6], sizes = [%c3, %c3, %c16, %14], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
                %16 = affine.min #map4(%arg4)[%workgroup_size_2]
                %17 = affine.min #map4(%arg5)[%workgroup_size_1]
                %18 = linalg.init_tensor [%8, %16, %17, %14] : tensor<?x?x?x?xf32>
                %19 = linalg.fill(%18, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
                %20 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%19 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
                flow.dispatch.output.store %20, %arg2, offsets = [%arg3, %arg4, %arg5, %arg6], sizes = [%8, %16, %17, %14], strides = [%c1, %c1, %c1, %c1] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
              }
            }
          }
        }
        return
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %0 = flow.ex.stream.fragment(%arg2 = %c32 : index, %arg3 = %c112 : index, %arg4 = %c1 : index, %arg5 = %arg0 : tensor<1x225x225x16xf32>, %arg6 = %arg1 : tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> {
      %1 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%arg2, %arg3, %arg3, %arg4] (%arg5, %arg6) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
      flow.return %1 : tensor<1x112x112x32xf32>
    }
    return %0 : tensor<1x112x112x32xf32>
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::HAL::MaterializeInterfacesPass ***
#map0 = affine_map<(d0)[s0] -> (s0, -d0 + 1)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 32)>
#map4 = affine_map<(d0)[s0] -> (s0, -d0 + 112)>
module  {
  hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vulkan_spirv, filter="vulkan*" {
      hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index}
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
        func @predict_ex_dispatch_1_dispatch_0() {
          %c0 = constant 0 : index
          %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.input<1x225x225x16xf32>
          %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.input<3x3x16x32xf32>
          %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<1x112x112x32xf32>
          %cst = constant 0.000000e+00 : f32
          %c32 = constant 32 : index
          %c112 = constant 112 : index
          %c3 = constant 3 : index
          %c0_0 = constant 0 : index
          %c16 = constant 16 : index
          %c1 = constant 1 : index
          %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
          %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
          %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
          %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
          %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
          %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
          %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
          %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
          %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
          %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
          %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
          %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
          %3 = muli %workgroup_size_3, %workgroup_id_3 : index
          %4 = muli %workgroup_size_3, %workgroup_count_3 : index
          scf.for %arg0 = %3 to %c1 step %4 {
            %5 = muli %workgroup_size_2, %workgroup_id_2 : index
            %6 = muli %workgroup_size_2, %workgroup_count_2 : index
            scf.for %arg1 = %5 to %c112 step %6 {
              %7 = muli %workgroup_size_1, %workgroup_id_1 : index
              %8 = muli %workgroup_size_1, %workgroup_count_1 : index
              scf.for %arg2 = %7 to %c112 step %8 {
                %9 = muli %workgroup_size_0, %workgroup_id_0 : index
                %10 = muli %workgroup_size_0, %workgroup_count_0 : index
                scf.for %arg3 = %9 to %c32 step %10 {
                  %11 = affine.min #map0(%arg0)[%workgroup_size_3]
                  %12 = affine.apply #map1(%arg1)
                  %13 = affine.min #map2(%arg1)[%workgroup_size_2]
                  %14 = affine.apply #map1(%arg2)
                  %15 = affine.min #map2(%arg2)[%workgroup_size_1]
                  %16 = flow.dispatch.input.load %0, offsets = [%arg0, %12, %14, %c0_0], sizes = [%11, %13, %15, %c16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
                  %17 = affine.min #map3(%arg3)[%workgroup_size_0]
                  %18 = flow.dispatch.input.load %1, offsets = [%c0_0, %c0_0, %c0_0, %arg3], sizes = [%c3, %c3, %c16, %17], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
                  %19 = affine.min #map4(%arg1)[%workgroup_size_2]
                  %20 = affine.min #map4(%arg2)[%workgroup_size_1]
                  %21 = linalg.init_tensor [%11, %19, %20, %17] : tensor<?x?x?x?xf32>
                  %22 = linalg.fill(%21, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
                  %23 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%16, %18 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%22 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
                  flow.dispatch.output.store %23, %2, offsets = [%arg0, %arg1, %arg2, %arg3], sizes = [%11, %19, %20, %17], strides = [%c1, %c1, %c1, %c1] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
                }
              }
            }
          }
          return
        }
        hal.interface @legacy_io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: tensor<1x225x225x16xf32>, %arg1: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %0 = flow.ex.stream.fragment(%arg2 = %c32 : index, %arg3 = %c112 : index, %arg4 = %c1 : index, %arg5 = %arg0 : tensor<1x225x225x16xf32>, %arg6 = %arg1 : tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> {
      %1 = flow.dispatch @predict_ex_dispatch_1_dispatch_0::@predict_ex_dispatch_1_dispatch_0[%arg2, %arg3, %arg3, %arg4] (%arg5, %arg6) : (tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32>
      flow.return %1 : tensor<1x112x112x32xf32>
    }
    return %0 : tensor<1x112x112x32xf32>
  }
}


// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1_dispatch_0() {
  %cst = constant 0.000000e+00 : f32
  %c32 = constant 32 : index
  %c112 = constant 112 : index
  %c3 = constant 3 : index
  %c0 = constant 0 : index
  %c16 = constant 16 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.input<1x225x225x16xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.input<3x3x16x32xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<1x112x112x32xf32>
  %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
  %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
  %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
  %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
  %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
  %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
  %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
  %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
  %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
  %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
  %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
  %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
  %3 = muli %workgroup_size_3, %workgroup_id_3 : index
  %4 = muli %workgroup_size_3, %workgroup_count_3 : index
  scf.for %arg0 = %3 to %c1 step %4 {
    %5 = muli %workgroup_size_2, %workgroup_id_2 : index
    %6 = muli %workgroup_size_2, %workgroup_count_2 : index
    scf.for %arg1 = %5 to %c112 step %6 {
      %7 = muli %workgroup_size_1, %workgroup_id_1 : index
      %8 = muli %workgroup_size_1, %workgroup_count_1 : index
      scf.for %arg2 = %7 to %c112 step %8 {
        %9 = muli %workgroup_size_0, %workgroup_id_0 : index
        %10 = muli %workgroup_size_0, %workgroup_count_0 : index
        scf.for %arg3 = %9 to %c32 step %10 {
          %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_3]
          %12 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
          %13 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg1)[%workgroup_size_2]
          %14 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg2)
          %15 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg2)[%workgroup_size_1]
          %16 = flow.dispatch.input.load %0, offsets = [%arg0, %12, %14, %c0], sizes = [%11, %13, %15, %c16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
          %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg3)[%workgroup_size_0]
          %18 = flow.dispatch.input.load %1, offsets = [%c0, %c0, %c0, %arg3], sizes = [%c3, %c3, %c16, %17], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
          %19 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_2]
          %20 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg2)[%workgroup_size_1]
          %21 = linalg.init_tensor [%11, %19, %20, %17] : tensor<?x?x?x?xf32>
          %22 = linalg.fill(%21, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
          %23 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%16, %18 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%22 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
          flow.dispatch.output.store %23, %2, offsets = [%arg0, %arg1, %arg2, %arg3], sizes = [%11, %19, %20, %17], strides = [%c1, %c1, %c1, %c1] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
        }
      }
    }
  }
  return
}

// *** IR Dump After Inliner ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() {
    %cst = constant 0.000000e+00 : f32
    %c32 = constant 32 : index
    %c112 = constant 112 : index
    %c3 = constant 3 : index
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %c1 = constant 1 : index
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.input<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.input<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<1x112x112x32xf32>
    %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
    %workgroup_size_1 = flow.dispatch.workgroup.size[1] : index
    %workgroup_size_2 = flow.dispatch.workgroup.size[2] : index
    %workgroup_size_3 = flow.dispatch.workgroup.size[3] : index
    %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
    %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
    %workgroup_id_1 = flow.dispatch.workgroup.id[1] : index
    %workgroup_count_1 = flow.dispatch.workgroup.count[1] : index
    %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
    %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
    %workgroup_id_3 = flow.dispatch.workgroup.id[3] : index
    %workgroup_count_3 = flow.dispatch.workgroup.count[3] : index
    %3 = muli %workgroup_size_3, %workgroup_id_3 : index
    %4 = muli %workgroup_size_3, %workgroup_count_3 : index
    scf.for %arg0 = %3 to %c1 step %4 {
      %5 = muli %workgroup_size_2, %workgroup_id_2 : index
      %6 = muli %workgroup_size_2, %workgroup_count_2 : index
      scf.for %arg1 = %5 to %c112 step %6 {
        %7 = muli %workgroup_size_1, %workgroup_id_1 : index
        %8 = muli %workgroup_size_1, %workgroup_count_1 : index
        scf.for %arg2 = %7 to %c112 step %8 {
          %9 = muli %workgroup_size_0, %workgroup_id_0 : index
          %10 = muli %workgroup_size_0, %workgroup_count_0 : index
          scf.for %arg3 = %9 to %c32 step %10 {
            %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_3]
            %12 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
            %13 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg1)[%workgroup_size_2]
            %14 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg2)
            %15 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg2)[%workgroup_size_1]
            %16 = flow.dispatch.input.load %0, offsets = [%arg0, %12, %14, %c0], sizes = [%11, %13, %15, %c16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
            %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg3)[%workgroup_size_0]
            %18 = flow.dispatch.input.load %1, offsets = [%c0, %c0, %c0, %arg3], sizes = [%c3, %c3, %c16, %17], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
            %19 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_2]
            %20 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg2)[%workgroup_size_1]
            %21 = linalg.init_tensor [%11, %19, %20, %17] : tensor<?x?x?x?xf32>
            %22 = linalg.fill(%21, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
            %23 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%16, %18 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%22 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
            flow.dispatch.output.store %23, %2, offsets = [%arg0, %arg1, %arg2, %arg3], sizes = [%11, %19, %20, %17], strides = [%c1, %c1, %c1, %c1] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
          }
        }
      }
    }
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::ConcretizeWorkloadOpsPass ***
hal.executable.target @vulkan_spirv, filter="vulkan*" {
  hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index}
  module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
    func @predict_ex_dispatch_1_dispatch_0() {
      %cst = constant 0.000000e+00 : f32
      %c3 = constant 3 : index
      %c16 = constant 16 : index
      %c4 = constant 4 : index
      %c1 = constant 1 : index
      %c0 = constant 0 : index
      %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.input<1x225x225x16xf32>
      %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.input<3x3x16x32xf32>
      %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<1x112x112x32xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %workgroup_id_z = hal.interface.workgroup.id[2] : index
      scf.for %arg0 = %c0 to %c1 step %c1 {
        %3 = muli %workgroup_id_z, %c4 : index
        %4 = muli %workgroup_id_y, %c4 : index
        %5 = muli %workgroup_id_x, %c16 : index
        %6 = affine.min affine_map<(d0)[s0] -> (1, -d0 + 1)>(%arg0)[%c1]
        %7 = affine.apply affine_map<(d0) -> (d0 * 2)>(%3)
        %8 = affine.min affine_map<(d0)[s0] -> (9, d0 * -2 + 225)>(%3)[%c4]
        %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%4)
        %10 = affine.min affine_map<(d0)[s0] -> (9, d0 * -2 + 225)>(%4)[%c4]
        %11 = flow.dispatch.input.load %0, offsets = [%arg0, %7, %9, %c0], sizes = [%6, %8, %10, %c16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
        %12 = affine.min affine_map<(d0)[s0] -> (16, -d0 + 32)>(%5)[%c16]
        %13 = flow.dispatch.input.load %1, offsets = [%c0, %c0, %c0, %5], sizes = [%c3, %c3, %c16, %12], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
        %14 = affine.min affine_map<(d0)[s0] -> (4, -d0 + 112)>(%3)[%c4]
        %15 = affine.min affine_map<(d0)[s0] -> (4, -d0 + 112)>(%4)[%c4]
        %16 = linalg.init_tensor [%6, %14, %15, %12] : tensor<?x?x?x?xf32>
        %17 = linalg.fill(%16, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
        %18 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%11, %13 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%17 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
        flow.dispatch.output.store %18, %2, offsets = [%arg0, %3, %4, %5], sizes = [%6, %14, %15, %12], strides = [%c1, %c1, %c1, %c1] : tensor<?x?x?x?xf32> -> !flow.dispatch.output<1x112x112x32xf32>
      }
      return
    }
    hal.interface @legacy_io attributes {sym_visibility = "private"} {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
  }
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::LinalgBufferizePass ***
func @predict_ex_dispatch_1_dispatch_0() {
  %cst = constant 0.000000e+00 : f32
  %c3 = constant 3 : index
  %c16 = constant 16 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.input<1x225x225x16xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
  %3 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.input<3x3x16x32xf32>
  %4 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
  %5 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<1x112x112x32xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  scf.for %arg0 = %c0 to %c1 step %c1 {
    %6 = muli %workgroup_id_z, %c4 : index
    %7 = muli %workgroup_id_y, %c4 : index
    %8 = muli %workgroup_id_x, %c16 : index
    %9 = affine.min affine_map<(d0)[s0] -> (1, -d0 + 1)>(%arg0)[%c1]
    %10 = affine.apply affine_map<(d0) -> (d0 * 2)>(%6)
    %11 = affine.min affine_map<(d0)[s0] -> (9, d0 * -2 + 225)>(%6)[%c4]
    %12 = affine.apply affine_map<(d0) -> (d0 * 2)>(%7)
    %13 = affine.min affine_map<(d0)[s0] -> (9, d0 * -2 + 225)>(%7)[%c4]
    %14 = subview %0[%arg0, %10, %12, %c0] [%9, %11, %13, %c16] [%c1, %c1, %c1, %c1] : memref<1x225x225x16xf32> to memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>>
    %15 = flow.dispatch.input.load %1, offsets = [%arg0, %10, %12, %c0], sizes = [%9, %11, %13, %c16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<1x225x225x16xf32> -> tensor<?x?x?x16xf32>
    %16 = affine.min affine_map<(d0)[s0] -> (16, -d0 + 32)>(%8)[%c16]
    %17 = subview %2[%c0, %c0, %c0, %8] [%c3, %c3, %c16, %16] [%c1, %c1, %c1, %c1] : memref<3x3x16x32xf32> to memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>>
    %18 = flow.dispatch.input.load %3, offsets = [%c0, %c0, %c0, %8], sizes = [%c3, %c3, %c16, %16], strides = [%c1, %c1, %c1, %c1] : !flow.dispatch.input<3x3x16x32xf32> -> tensor<3x3x16x?xf32>
    %19 = affine.min affine_map<(d0)[s0] -> (4, -d0 + 112)>(%6)[%c4]
    %20 = affine.min affine_map<(d0)[s0] -> (4, -d0 + 112)>(%7)[%c4]
    %21 = subview %4[%arg0, %6, %7, %8] [%9, %19, %20, %16] [%c1, %c1, %c1, %c1] : memref<1x112x112x32xf32> to memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>>
    %22 = linalg.init_tensor [%9, %19, %20, %16] : tensor<?x?x?x?xf32>
    linalg.fill(%21, %cst) {__internal_linalg_transform__ = "workgroup"} : memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>>, f32 
    %23 = linalg.fill(%22, %cst) {__internal_linalg_transform__ = "workgroup"} : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32> 
    linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%14, %17 : memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>>, memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>>) outs(%21 : memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>>)
    %24 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%15, %18 : tensor<?x?x?x16xf32>, tensor<3x3x16x?xf32>) outs(%23 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  }
  return
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1_dispatch_0() {
  %cst = constant 0.000000e+00 : f32
  %c16 = constant 16 : index
  %c4 = constant 4 : index
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.input<1x225x225x16xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
  %3 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.input<3x3x16x32xf32>
  %4 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
  %5 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<1x112x112x32xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %6 = muli %workgroup_id_z, %c4 : index
  %7 = muli %workgroup_id_y, %c4 : index
  %8 = muli %workgroup_id_x, %c16 : index
  %9 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%6]
  %10 = affine.min affine_map<()[s0] -> (9, s0 * -2 + 225)>()[%6]
  %11 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%7]
  %12 = affine.min affine_map<()[s0] -> (9, s0 * -2 + 225)>()[%7]
  %13 = subview %0[0, %9, %11, 0] [1, %10, %12, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
  %14 = affine.min affine_map<()[s0] -> (16, -s0 + 32)>()[%8]
  %15 = subview %2[0, 0, 0, %8] [3, 3, 16, %14] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
  %16 = affine.min affine_map<()[s0] -> (4, -s0 + 112)>()[%6]
  %17 = affine.min affine_map<()[s0] -> (4, -s0 + 112)>()[%7]
  %18 = subview %4[0, %6, %7, %8] [1, %16, %17, %14] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  linalg.fill(%18, %cst) {__internal_linalg_transform__ = "workgroup"} : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, f32 
  linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>) outs(%18 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>)
  return
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::RemoveDeadMemAllocsPass ***
func @predict_ex_dispatch_1_dispatch_0() {
  %cst = constant 0.000000e+00 : f32
  %c16 = constant 16 : index
  %c4 = constant 4 : index
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %3 = muli %workgroup_id_z, %c4 : index
  %4 = muli %workgroup_id_y, %c4 : index
  %5 = muli %workgroup_id_x, %c16 : index
  %6 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%3]
  %7 = affine.min affine_map<()[s0] -> (9, s0 * -2 + 225)>()[%3]
  %8 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%4]
  %9 = affine.min affine_map<()[s0] -> (9, s0 * -2 + 225)>()[%4]
  %10 = subview %0[0, %6, %8, 0] [1, %7, %9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
  %11 = affine.min affine_map<()[s0] -> (16, -s0 + 32)>()[%5]
  %12 = subview %1[0, 0, 0, %5] [3, 3, 16, %11] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
  %13 = affine.min affine_map<()[s0] -> (4, -s0 + 112)>()[%3]
  %14 = affine.min affine_map<()[s0] -> (4, -s0 + 112)>()[%4]
  %15 = subview %2[0, %3, %4, %5] [1, %13, %14, %11] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  linalg.fill(%15, %cst) {__internal_linalg_transform__ = "workgroup"} : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, f32 
  linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%10, %12 : memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>) outs(%15 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>)
  return
}

// *** IR Dump After Canonicalizer ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() {
    %cst = constant 0.000000e+00 : f32
    %c16 = constant 16 : index
    %c4 = constant 4 : index
    %c0 = constant 0 : index
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%3]
    %7 = affine.min affine_map<()[s0] -> (9, s0 * -2 + 225)>()[%3]
    %8 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%4]
    %9 = affine.min affine_map<()[s0] -> (9, s0 * -2 + 225)>()[%4]
    %10 = subview %0[0, %6, %8, 0] [1, %7, %9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %11 = affine.min affine_map<()[s0] -> (16, -s0 + 32)>()[%5]
    %12 = subview %1[0, 0, 0, %5] [3, 3, 16, %11] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %13 = affine.min affine_map<()[s0] -> (4, -s0 + 112)>()[%3]
    %14 = affine.min affine_map<()[s0] -> (4, -s0 + 112)>()[%4]
    %15 = subview %2[0, %3, %4, %5] [1, %13, %14, %11] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    linalg.fill(%15, %cst) {__internal_linalg_transform__ = "workgroup"} : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, f32 
    linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, iree.codegen.distribution.original_operand_types = [tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>, tensor<1x112x112x32xf32>], iree.codegen.distribution.original_result_types = [tensor<1x112x112x32xf32>], iree.codegen.fushion.root_op = 0 : i64, strides = dense<2> : tensor<2xi64>} ins(%10, %12 : memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>) outs(%15 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>)
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::LinalgTileAndFusePass ***
hal.executable.target @vulkan_spirv, filter="vulkan*" {
  hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index} {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):  // no predecessors
    %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg0]
    %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg1]
    %2 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg2]
    hal.return %0, %1, %2 : index, index, index
  }
  module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
    func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
      %c16 = constant 16 : index
      %cst = constant dense<0.000000e+00> : vector<1x4x1x4xf32>
      %c4 = constant 4 : index
      %c6 = constant 6 : index
      %c0 = constant 0 : index
      %cst_0 = constant 0.000000e+00 : f32
      %c1 = constant 1 : index
      %c2 = constant 2 : index
      %c3 = constant 3 : index
      %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
      %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
      %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %workgroup_id_z = hal.interface.workgroup.id[2] : index
      %3 = muli %workgroup_id_z, %c4 : index
      %4 = muli %workgroup_id_y, %c4 : index
      %5 = muli %workgroup_id_x, %c16 : index
      %6 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%3]
      %7 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%4]
      %8 = subview %0[0, %6, %7, 0] [1, 9, 9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
      %9 = subview %1[0, 0, 0, %5] [3, 3, 16, 16] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
      %10 = subview %2[0, %3, %4, %5] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      %11 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %12 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %13 = "gpu.thread_id"() {dimension = "z"} : () -> index
      %14 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%13]
      %15 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%11]
      %16 = subview %10[0, %14, %12, %15] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      %17 = vector.extract_strided_slice %cst {offsets = [0, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<1x4x1x4xf32> to vector<1x1x1x4xf32>
      %18 = vector.extract_strided_slice %cst {offsets = [0, 1, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<1x4x1x4xf32> to vector<1x1x1x4xf32>
      %19 = vector.extract_strided_slice %cst {offsets = [0, 2, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<1x4x1x4xf32> to vector<1x1x1x4xf32>
      %20 = vector.extract_strided_slice %cst {offsets = [0, 3, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<1x4x1x4xf32> to vector<1x1x1x4xf32>
      vector.transfer_write %17, %16[%c0, %c0, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      vector.transfer_write %18, %16[%c0, %c1, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      vector.transfer_write %19, %16[%c0, %c2, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      vector.transfer_write %20, %16[%c0, %c3, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      %21 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %22 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %23 = "gpu.thread_id"() {dimension = "z"} : () -> index
      %24 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%23]
      %25 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%22]
      %26 = subview %8[0, %24, %25, 0] [1, 9, 3, 16] [1, 1, 1, 1] : memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
      %27 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%21]
      %28 = subview %9[0, 0, 0, %27] [3, 3, 16, 4] [1, 1, 1, 1] : memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
      %29 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%23]
      %30 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%21]
      %31 = subview %10[0, %29, %22, %30] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      %32 = vector.transfer_read %31[%c0, %c0, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
      %33 = vector.transfer_read %31[%c0, %c1, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
      %34 = vector.transfer_read %31[%c0, %c2, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
      %35 = vector.transfer_read %31[%c0, %c3, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
      %36:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %32, %arg2 = %33, %arg3 = %34, %arg4 = %35) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %37:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %38:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
            %39 = subview %26[0, %arg0, %arg5, %arg10] [1, 7, 1, 4] [1, 1, 1, 1] : memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
            %40 = subview %28[%arg0, %arg5, %arg10, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
            %41 = vector.transfer_read %40[%c0, %c0, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
            %42 = vector.transfer_read %40[%c0, %c0, %c1, %c0], %cst_0 {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
            %43 = vector.transfer_read %40[%c0, %c0, %c2, %c0], %cst_0 {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
            %44 = vector.transfer_read %40[%c0, %c0, %c3, %c0], %cst_0 {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
            %45 = vector.transfer_read %39[%c0, %c0, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
            %46 = vector.extract_strided_slice %45 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %47 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %46, %41, %arg11 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %48 = vector.extract_strided_slice %45 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %49 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %48, %42, %47 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %50 = vector.extract_strided_slice %45 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %51 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %50, %43, %49 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %52 = vector.extract_strided_slice %45 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %53 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %52, %44, %51 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %54 = vector.transfer_read %39[%c0, %c2, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
            %55 = vector.extract_strided_slice %54 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %56 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %55, %41, %arg12 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %57 = vector.extract_strided_slice %54 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %58 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %57, %42, %56 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %59 = vector.extract_strided_slice %54 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %60 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %59, %43, %58 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %61 = vector.extract_strided_slice %54 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %62 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %61, %44, %60 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %63 = vector.transfer_read %39[%c0, %c4, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
            %64 = vector.extract_strided_slice %63 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %65 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %64, %41, %arg13 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %66 = vector.extract_strided_slice %63 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %67 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %66, %42, %65 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %68 = vector.extract_strided_slice %63 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %69 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %68, %43, %67 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %70 = vector.extract_strided_slice %63 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %71 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %70, %44, %69 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %72 = vector.transfer_read %39[%c0, %c6, %c0, %c0], %cst_0 {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
            %73 = vector.extract_strided_slice %72 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %74 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %73, %41, %arg14 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %75 = vector.extract_strided_slice %72 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %76 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %75, %42, %74 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %77 = vector.extract_strided_slice %72 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %78 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %77, %43, %76 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %79 = vector.extract_strided_slice %72 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %80 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %79, %44, %78 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            scf.yield %53, %62, %71, %80 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
          }
          scf.yield %38#0, %38#1, %38#2, %38#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %37#0, %37#1, %37#2, %37#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      vector.transfer_write %36#3, %31[%c0, %c3, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      vector.transfer_write %36#2, %31[%c0, %c2, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      vector.transfer_write %36#1, %31[%c0, %c1, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      vector.transfer_write %36#0, %31[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
      return
    }
    hal.interface @legacy_io attributes {sym_visibility = "private"} {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
  }
}

// *** IR Dump After Canonicalizer ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c4 = constant 4 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x1x1x4xf32>
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%3]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%4]
    %8 = subview %0[0, %6, %7, 0] [1, 9, 9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %9 = subview %1[0, 0, 0, %5] [3, 3, 16, 16] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %10 = subview %2[0, %3, %4, %5] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %11 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %12 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %13 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %14 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%13]
    %15 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%11]
    %16 = subview %10[0, %14, %12, %15] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c0, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c1, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c2, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c3, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %17 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %18 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %19 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %20 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%19]
    %21 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%18]
    %22 = subview %8[0, %20, %21, 0] [1, 9, 3, 16] [1, 1, 1, 1] : memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %23 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%17]
    %24 = subview %9[0, 0, 0, %23] [3, 3, 16, 4] [1, 1, 1, 1] : memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %25 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%19]
    %26 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%17]
    %27 = subview %10[0, %25, %18, %26] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %28 = vector.transfer_read %27[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %29 = vector.transfer_read %27[%c0, %c1, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %30 = vector.transfer_read %27[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %31 = vector.transfer_read %27[%c0, %c3, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %32:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %28, %arg2 = %29, %arg3 = %30, %arg4 = %31) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %33:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %34:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %35 = subview %22[0, %arg0, %arg5, %arg10] [1, 7, 1, 4] [1, 1, 1, 1] : memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
          %36 = subview %24[%arg0, %arg5, %arg10, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
          %37 = vector.transfer_read %36[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %38 = vector.transfer_read %36[%c0, %c0, %c1, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %39 = vector.transfer_read %36[%c0, %c0, %c2, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %40 = vector.transfer_read %36[%c0, %c0, %c3, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %41 = vector.transfer_read %35[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %42 = vector.extract_strided_slice %41 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %43 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %37, %arg11 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %44 = vector.extract_strided_slice %41 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %45 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %38, %43 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %46 = vector.extract_strided_slice %41 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %47 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %46, %39, %45 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %48 = vector.extract_strided_slice %41 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %49 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %48, %40, %47 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %50 = vector.transfer_read %35[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %51 = vector.extract_strided_slice %50 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %52 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %51, %37, %arg12 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %53 = vector.extract_strided_slice %50 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %54 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %53, %38, %52 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %55 = vector.extract_strided_slice %50 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %56 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %55, %39, %54 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %57 = vector.extract_strided_slice %50 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %58 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %57, %40, %56 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %59 = vector.transfer_read %35[%c0, %c4, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %60 = vector.extract_strided_slice %59 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %61 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %60, %37, %arg13 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %62 = vector.extract_strided_slice %59 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %63 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %62, %38, %61 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %64 = vector.extract_strided_slice %59 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %65 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %64, %39, %63 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %66 = vector.extract_strided_slice %59 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %67 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %66, %40, %65 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %68 = vector.transfer_read %35[%c0, %c6, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %69 = vector.extract_strided_slice %68 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %70 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %69, %37, %arg14 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %71 = vector.extract_strided_slice %68 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %72 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %71, %38, %70 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %73 = vector.extract_strided_slice %68 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %74 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %73, %39, %72 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          %75 = vector.extract_strided_slice %68 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
          %76 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %75, %40, %74 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
          scf.yield %49, %58, %67, %76 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %34#0, %34#1, %34#2, %34#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %33#0, %33#1, %33#2, %33#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    vector.transfer_write %32#3, %27[%c0, %c3, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#2, %27[%c0, %c2, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#1, %27[%c0, %c1, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#0, %27[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::ConvertVectorToGPUPass ***
func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
  %c16 = constant 16 : index
  %c4 = constant 4 : index
  %c6 = constant 6 : index
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %cst_0 = constant dense<0.000000e+00> : vector<1x1x1x4xf32>
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %3 = muli %workgroup_id_z, %c4 : index
  %4 = muli %workgroup_id_y, %c4 : index
  %5 = muli %workgroup_id_x, %c16 : index
  %6 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%3]
  %7 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%4]
  %8 = subview %0[0, %6, %7, 0] [1, 9, 9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
  %9 = subview %1[0, 0, 0, %5] [3, 3, 16, 16] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
  %10 = subview %2[0, %3, %4, %5] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %11 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %12 = "gpu.thread_id"() {dimension = "y"} : () -> index
  %13 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %14 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%13]
  %15 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%11]
  %16 = subview %10[0, %14, %12, %15] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %cst_0, %16[%c0, %c0, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %cst_0, %16[%c0, %c1, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %cst_0, %16[%c0, %c2, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %cst_0, %16[%c0, %c3, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %17 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %18 = "gpu.thread_id"() {dimension = "y"} : () -> index
  %19 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %20 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%19]
  %21 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%18]
  %22 = subview %8[0, %20, %21, 0] [1, 9, 3, 16] [1, 1, 1, 1] : memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
  %23 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%17]
  %24 = subview %9[0, 0, 0, %23] [3, 3, 16, 4] [1, 1, 1, 1] : memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
  %25 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%19]
  %26 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%17]
  %27 = subview %10[0, %25, %18, %26] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %28 = vector.transfer_read %27[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
  %29 = vector.transfer_read %27[%c0, %c1, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
  %30 = vector.transfer_read %27[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
  %31 = vector.transfer_read %27[%c0, %c3, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
  %32:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %28, %arg2 = %29, %arg3 = %30, %arg4 = %31) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
    %33:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %34:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %35 = subview %22[0, %arg0, %arg5, %arg10] [1, 7, 1, 4] [1, 1, 1, 1] : memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
        %36 = subview %24[%arg0, %arg5, %arg10, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
        %37 = vector.transfer_read %36[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
        %38 = vector.transfer_read %36[%c0, %c0, %c1, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
        %39 = vector.transfer_read %36[%c0, %c0, %c2, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
        %40 = vector.transfer_read %36[%c0, %c0, %c3, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
        %41 = vector.transfer_read %35[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
        %42 = vector.extract %41[0, 0] : vector<1x4xf32>
        %43 = vector.broadcast %42 : f32 to vector<4xf32>
        %44 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
        %45 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
        %46 = mulf %43, %44 : vector<4xf32>
        %47 = addf %46, %45 : vector<4xf32>
        %48 = vector.extract %41[0, 1] : vector<1x4xf32>
        %49 = vector.broadcast %48 : f32 to vector<4xf32>
        %50 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
        %51 = mulf %49, %50 : vector<4xf32>
        %52 = addf %51, %47 : vector<4xf32>
        %53 = vector.extract %41[0, 2] : vector<1x4xf32>
        %54 = vector.broadcast %53 : f32 to vector<4xf32>
        %55 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
        %56 = mulf %54, %55 : vector<4xf32>
        %57 = addf %56, %52 : vector<4xf32>
        %58 = vector.extract %41[0, 3] : vector<1x4xf32>
        %59 = vector.broadcast %58 : f32 to vector<4xf32>
        %60 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
        %61 = mulf %59, %60 : vector<4xf32>
        %62 = addf %61, %57 : vector<4xf32>
        %63 = vector.shape_cast %62 : vector<4xf32> to vector<1x4xf32>
        %64 = vector.transfer_read %35[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
        %65 = vector.extract %64[0, 0] : vector<1x4xf32>
        %66 = vector.broadcast %65 : f32 to vector<4xf32>
        %67 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
        %68 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
        %69 = mulf %66, %67 : vector<4xf32>
        %70 = addf %69, %68 : vector<4xf32>
        %71 = vector.extract %64[0, 1] : vector<1x4xf32>
        %72 = vector.broadcast %71 : f32 to vector<4xf32>
        %73 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
        %74 = mulf %72, %73 : vector<4xf32>
        %75 = addf %74, %70 : vector<4xf32>
        %76 = vector.extract %64[0, 2] : vector<1x4xf32>
        %77 = vector.broadcast %76 : f32 to vector<4xf32>
        %78 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
        %79 = mulf %77, %78 : vector<4xf32>
        %80 = addf %79, %75 : vector<4xf32>
        %81 = vector.extract %64[0, 3] : vector<1x4xf32>
        %82 = vector.broadcast %81 : f32 to vector<4xf32>
        %83 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
        %84 = mulf %82, %83 : vector<4xf32>
        %85 = addf %84, %80 : vector<4xf32>
        %86 = vector.shape_cast %85 : vector<4xf32> to vector<1x4xf32>
        %87 = vector.transfer_read %35[%c0, %c4, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
        %88 = vector.extract %87[0, 0] : vector<1x4xf32>
        %89 = vector.broadcast %88 : f32 to vector<4xf32>
        %90 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
        %91 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
        %92 = mulf %89, %90 : vector<4xf32>
        %93 = addf %92, %91 : vector<4xf32>
        %94 = vector.extract %87[0, 1] : vector<1x4xf32>
        %95 = vector.broadcast %94 : f32 to vector<4xf32>
        %96 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
        %97 = mulf %95, %96 : vector<4xf32>
        %98 = addf %97, %93 : vector<4xf32>
        %99 = vector.extract %87[0, 2] : vector<1x4xf32>
        %100 = vector.broadcast %99 : f32 to vector<4xf32>
        %101 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
        %102 = mulf %100, %101 : vector<4xf32>
        %103 = addf %102, %98 : vector<4xf32>
        %104 = vector.extract %87[0, 3] : vector<1x4xf32>
        %105 = vector.broadcast %104 : f32 to vector<4xf32>
        %106 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
        %107 = mulf %105, %106 : vector<4xf32>
        %108 = addf %107, %103 : vector<4xf32>
        %109 = vector.shape_cast %108 : vector<4xf32> to vector<1x4xf32>
        %110 = vector.transfer_read %35[%c0, %c6, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
        %111 = vector.extract %110[0, 0] : vector<1x4xf32>
        %112 = vector.broadcast %111 : f32 to vector<4xf32>
        %113 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
        %114 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
        %115 = mulf %112, %113 : vector<4xf32>
        %116 = addf %115, %114 : vector<4xf32>
        %117 = vector.extract %110[0, 1] : vector<1x4xf32>
        %118 = vector.broadcast %117 : f32 to vector<4xf32>
        %119 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
        %120 = mulf %118, %119 : vector<4xf32>
        %121 = addf %120, %116 : vector<4xf32>
        %122 = vector.extract %110[0, 2] : vector<1x4xf32>
        %123 = vector.broadcast %122 : f32 to vector<4xf32>
        %124 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
        %125 = mulf %123, %124 : vector<4xf32>
        %126 = addf %125, %121 : vector<4xf32>
        %127 = vector.extract %110[0, 3] : vector<1x4xf32>
        %128 = vector.broadcast %127 : f32 to vector<4xf32>
        %129 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
        %130 = mulf %128, %129 : vector<4xf32>
        %131 = addf %130, %126 : vector<4xf32>
        %132 = vector.shape_cast %131 : vector<4xf32> to vector<1x4xf32>
        scf.yield %63, %86, %109, %132 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %34#0, %34#1, %34#2, %34#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    scf.yield %33#0, %33#1, %33#2, %33#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
  }
  vector.transfer_write %32#3, %27[%c0, %c3, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %32#2, %27[%c0, %c2, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %32#1, %27[%c0, %c1, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %32#0, %27[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  return
}

// *** IR Dump After ConvertAffineToStandard ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c4 = constant 4 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x1x1x4xf32>
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %c2_1 = constant 2 : index
    %6 = muli %3, %c2_1 : index
    %c2_2 = constant 2 : index
    %7 = muli %4, %c2_2 : index
    %8 = subview %0[0, %6, %7, 0] [1, 9, 9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %9 = subview %1[0, 0, 0, %5] [3, 3, 16, 16] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %10 = subview %2[0, %3, %4, %5] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %11 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %12 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %13 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %c4_3 = constant 4 : index
    %14 = muli %13, %c4_3 : index
    %c4_4 = constant 4 : index
    %15 = muli %11, %c4_4 : index
    %16 = subview %10[0, %14, %12, %15] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c0, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c1, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c2, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c3, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %17 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %18 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %19 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %c8 = constant 8 : index
    %20 = muli %19, %c8 : index
    %c2_5 = constant 2 : index
    %21 = muli %18, %c2_5 : index
    %22 = subview %8[0, %20, %21, 0] [1, 9, 3, 16] [1, 1, 1, 1] : memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %c4_6 = constant 4 : index
    %23 = muli %17, %c4_6 : index
    %24 = subview %9[0, 0, 0, %23] [3, 3, 16, 4] [1, 1, 1, 1] : memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %c4_7 = constant 4 : index
    %25 = muli %19, %c4_7 : index
    %c4_8 = constant 4 : index
    %26 = muli %17, %c4_8 : index
    %27 = subview %10[0, %25, %18, %26] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %28 = vector.transfer_read %27[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %29 = vector.transfer_read %27[%c0, %c1, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %30 = vector.transfer_read %27[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %31 = vector.transfer_read %27[%c0, %c3, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %32:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %28, %arg2 = %29, %arg3 = %30, %arg4 = %31) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %33:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %34:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %35 = subview %22[0, %arg0, %arg5, %arg10] [1, 7, 1, 4] [1, 1, 1, 1] : memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
          %36 = subview %24[%arg0, %arg5, %arg10, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
          %37 = vector.transfer_read %36[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %38 = vector.transfer_read %36[%c0, %c0, %c1, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %39 = vector.transfer_read %36[%c0, %c0, %c2, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %40 = vector.transfer_read %36[%c0, %c0, %c3, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %41 = vector.transfer_read %35[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %42 = vector.extract %41[0, 0] : vector<1x4xf32>
          %43 = vector.broadcast %42 : f32 to vector<4xf32>
          %44 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
          %45 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
          %46 = mulf %43, %44 : vector<4xf32>
          %47 = addf %46, %45 : vector<4xf32>
          %48 = vector.extract %41[0, 1] : vector<1x4xf32>
          %49 = vector.broadcast %48 : f32 to vector<4xf32>
          %50 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
          %51 = mulf %49, %50 : vector<4xf32>
          %52 = addf %51, %47 : vector<4xf32>
          %53 = vector.extract %41[0, 2] : vector<1x4xf32>
          %54 = vector.broadcast %53 : f32 to vector<4xf32>
          %55 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
          %56 = mulf %54, %55 : vector<4xf32>
          %57 = addf %56, %52 : vector<4xf32>
          %58 = vector.extract %41[0, 3] : vector<1x4xf32>
          %59 = vector.broadcast %58 : f32 to vector<4xf32>
          %60 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
          %61 = mulf %59, %60 : vector<4xf32>
          %62 = addf %61, %57 : vector<4xf32>
          %63 = vector.shape_cast %62 : vector<4xf32> to vector<1x4xf32>
          %64 = vector.transfer_read %35[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %65 = vector.extract %64[0, 0] : vector<1x4xf32>
          %66 = vector.broadcast %65 : f32 to vector<4xf32>
          %67 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
          %68 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
          %69 = mulf %66, %67 : vector<4xf32>
          %70 = addf %69, %68 : vector<4xf32>
          %71 = vector.extract %64[0, 1] : vector<1x4xf32>
          %72 = vector.broadcast %71 : f32 to vector<4xf32>
          %73 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
          %74 = mulf %72, %73 : vector<4xf32>
          %75 = addf %74, %70 : vector<4xf32>
          %76 = vector.extract %64[0, 2] : vector<1x4xf32>
          %77 = vector.broadcast %76 : f32 to vector<4xf32>
          %78 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
          %79 = mulf %77, %78 : vector<4xf32>
          %80 = addf %79, %75 : vector<4xf32>
          %81 = vector.extract %64[0, 3] : vector<1x4xf32>
          %82 = vector.broadcast %81 : f32 to vector<4xf32>
          %83 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
          %84 = mulf %82, %83 : vector<4xf32>
          %85 = addf %84, %80 : vector<4xf32>
          %86 = vector.shape_cast %85 : vector<4xf32> to vector<1x4xf32>
          %87 = vector.transfer_read %35[%c0, %c4, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %88 = vector.extract %87[0, 0] : vector<1x4xf32>
          %89 = vector.broadcast %88 : f32 to vector<4xf32>
          %90 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
          %91 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
          %92 = mulf %89, %90 : vector<4xf32>
          %93 = addf %92, %91 : vector<4xf32>
          %94 = vector.extract %87[0, 1] : vector<1x4xf32>
          %95 = vector.broadcast %94 : f32 to vector<4xf32>
          %96 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
          %97 = mulf %95, %96 : vector<4xf32>
          %98 = addf %97, %93 : vector<4xf32>
          %99 = vector.extract %87[0, 2] : vector<1x4xf32>
          %100 = vector.broadcast %99 : f32 to vector<4xf32>
          %101 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
          %102 = mulf %100, %101 : vector<4xf32>
          %103 = addf %102, %98 : vector<4xf32>
          %104 = vector.extract %87[0, 3] : vector<1x4xf32>
          %105 = vector.broadcast %104 : f32 to vector<4xf32>
          %106 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
          %107 = mulf %105, %106 : vector<4xf32>
          %108 = addf %107, %103 : vector<4xf32>
          %109 = vector.shape_cast %108 : vector<4xf32> to vector<1x4xf32>
          %110 = vector.transfer_read %35[%c0, %c6, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %111 = vector.extract %110[0, 0] : vector<1x4xf32>
          %112 = vector.broadcast %111 : f32 to vector<4xf32>
          %113 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
          %114 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
          %115 = mulf %112, %113 : vector<4xf32>
          %116 = addf %115, %114 : vector<4xf32>
          %117 = vector.extract %110[0, 1] : vector<1x4xf32>
          %118 = vector.broadcast %117 : f32 to vector<4xf32>
          %119 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
          %120 = mulf %118, %119 : vector<4xf32>
          %121 = addf %120, %116 : vector<4xf32>
          %122 = vector.extract %110[0, 2] : vector<1x4xf32>
          %123 = vector.broadcast %122 : f32 to vector<4xf32>
          %124 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
          %125 = mulf %123, %124 : vector<4xf32>
          %126 = addf %125, %121 : vector<4xf32>
          %127 = vector.extract %110[0, 3] : vector<1x4xf32>
          %128 = vector.broadcast %127 : f32 to vector<4xf32>
          %129 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
          %130 = mulf %128, %129 : vector<4xf32>
          %131 = addf %130, %126 : vector<4xf32>
          %132 = vector.shape_cast %131 : vector<4xf32> to vector<1x4xf32>
          scf.yield %63, %86, %109, %132 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %34#0, %34#1, %34#2, %34#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %33#0, %33#1, %33#2, %33#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    vector.transfer_write %32#3, %27[%c0, %c3, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#2, %27[%c0, %c2, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#1, %27[%c0, %c1, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#0, %27[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After Canonicalizer ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x1x1x4xf32>
    %c8 = constant 8 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = muli %3, %c2 : index
    %7 = muli %4, %c2 : index
    %8 = subview %0[0, %6, %7, 0] [1, 9, 9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %9 = subview %1[0, 0, 0, %5] [3, 3, 16, 16] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %10 = subview %2[0, %3, %4, %5] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %11 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %12 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %13 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %14 = muli %13, %c4 : index
    %15 = muli %11, %c4 : index
    %16 = subview %10[0, %14, %12, %15] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c0, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c1, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c2, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c3, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %17 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %18 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %19 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %20 = muli %19, %c8 : index
    %21 = muli %18, %c2 : index
    %22 = subview %8[0, %20, %21, 0] [1, 9, 3, 16] [1, 1, 1, 1] : memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %23 = muli %17, %c4 : index
    %24 = subview %9[0, 0, 0, %23] [3, 3, 16, 4] [1, 1, 1, 1] : memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %25 = muli %19, %c4 : index
    %26 = muli %17, %c4 : index
    %27 = subview %10[0, %25, %18, %26] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %28 = vector.transfer_read %27[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %29 = vector.transfer_read %27[%c0, %c1, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %30 = vector.transfer_read %27[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %31 = vector.transfer_read %27[%c0, %c3, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %32:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %28, %arg2 = %29, %arg3 = %30, %arg4 = %31) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %33:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %34:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %35 = subview %22[0, %arg0, %arg5, %arg10] [1, 7, 1, 4] [1, 1, 1, 1] : memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
          %36 = subview %24[%arg0, %arg5, %arg10, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
          %37 = vector.transfer_read %36[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %38 = vector.transfer_read %36[%c0, %c0, %c1, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %39 = vector.transfer_read %36[%c0, %c0, %c2, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %40 = vector.transfer_read %36[%c0, %c0, %c3, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %41 = vector.transfer_read %35[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %42 = vector.extract %41[0, 0] : vector<1x4xf32>
          %43 = vector.broadcast %42 : f32 to vector<4xf32>
          %44 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
          %45 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
          %46 = mulf %43, %44 : vector<4xf32>
          %47 = addf %46, %45 : vector<4xf32>
          %48 = vector.extract %41[0, 1] : vector<1x4xf32>
          %49 = vector.broadcast %48 : f32 to vector<4xf32>
          %50 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
          %51 = mulf %49, %50 : vector<4xf32>
          %52 = addf %51, %47 : vector<4xf32>
          %53 = vector.extract %41[0, 2] : vector<1x4xf32>
          %54 = vector.broadcast %53 : f32 to vector<4xf32>
          %55 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
          %56 = mulf %54, %55 : vector<4xf32>
          %57 = addf %56, %52 : vector<4xf32>
          %58 = vector.extract %41[0, 3] : vector<1x4xf32>
          %59 = vector.broadcast %58 : f32 to vector<4xf32>
          %60 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
          %61 = mulf %59, %60 : vector<4xf32>
          %62 = addf %61, %57 : vector<4xf32>
          %63 = vector.shape_cast %62 : vector<4xf32> to vector<1x4xf32>
          %64 = vector.transfer_read %35[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %65 = vector.extract %64[0, 0] : vector<1x4xf32>
          %66 = vector.broadcast %65 : f32 to vector<4xf32>
          %67 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
          %68 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
          %69 = mulf %66, %67 : vector<4xf32>
          %70 = addf %69, %68 : vector<4xf32>
          %71 = vector.extract %64[0, 1] : vector<1x4xf32>
          %72 = vector.broadcast %71 : f32 to vector<4xf32>
          %73 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
          %74 = mulf %72, %73 : vector<4xf32>
          %75 = addf %74, %70 : vector<4xf32>
          %76 = vector.extract %64[0, 2] : vector<1x4xf32>
          %77 = vector.broadcast %76 : f32 to vector<4xf32>
          %78 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
          %79 = mulf %77, %78 : vector<4xf32>
          %80 = addf %79, %75 : vector<4xf32>
          %81 = vector.extract %64[0, 3] : vector<1x4xf32>
          %82 = vector.broadcast %81 : f32 to vector<4xf32>
          %83 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
          %84 = mulf %82, %83 : vector<4xf32>
          %85 = addf %84, %80 : vector<4xf32>
          %86 = vector.shape_cast %85 : vector<4xf32> to vector<1x4xf32>
          %87 = vector.transfer_read %35[%c0, %c4, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %88 = vector.extract %87[0, 0] : vector<1x4xf32>
          %89 = vector.broadcast %88 : f32 to vector<4xf32>
          %90 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
          %91 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
          %92 = mulf %89, %90 : vector<4xf32>
          %93 = addf %92, %91 : vector<4xf32>
          %94 = vector.extract %87[0, 1] : vector<1x4xf32>
          %95 = vector.broadcast %94 : f32 to vector<4xf32>
          %96 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
          %97 = mulf %95, %96 : vector<4xf32>
          %98 = addf %97, %93 : vector<4xf32>
          %99 = vector.extract %87[0, 2] : vector<1x4xf32>
          %100 = vector.broadcast %99 : f32 to vector<4xf32>
          %101 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
          %102 = mulf %100, %101 : vector<4xf32>
          %103 = addf %102, %98 : vector<4xf32>
          %104 = vector.extract %87[0, 3] : vector<1x4xf32>
          %105 = vector.broadcast %104 : f32 to vector<4xf32>
          %106 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
          %107 = mulf %105, %106 : vector<4xf32>
          %108 = addf %107, %103 : vector<4xf32>
          %109 = vector.shape_cast %108 : vector<4xf32> to vector<1x4xf32>
          %110 = vector.transfer_read %35[%c0, %c6, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %111 = vector.extract %110[0, 0] : vector<1x4xf32>
          %112 = vector.broadcast %111 : f32 to vector<4xf32>
          %113 = vector.shape_cast %37 : vector<1x4xf32> to vector<4xf32>
          %114 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
          %115 = mulf %112, %113 : vector<4xf32>
          %116 = addf %115, %114 : vector<4xf32>
          %117 = vector.extract %110[0, 1] : vector<1x4xf32>
          %118 = vector.broadcast %117 : f32 to vector<4xf32>
          %119 = vector.shape_cast %38 : vector<1x4xf32> to vector<4xf32>
          %120 = mulf %118, %119 : vector<4xf32>
          %121 = addf %120, %116 : vector<4xf32>
          %122 = vector.extract %110[0, 2] : vector<1x4xf32>
          %123 = vector.broadcast %122 : f32 to vector<4xf32>
          %124 = vector.shape_cast %39 : vector<1x4xf32> to vector<4xf32>
          %125 = mulf %123, %124 : vector<4xf32>
          %126 = addf %125, %121 : vector<4xf32>
          %127 = vector.extract %110[0, 3] : vector<1x4xf32>
          %128 = vector.broadcast %127 : f32 to vector<4xf32>
          %129 = vector.shape_cast %40 : vector<1x4xf32> to vector<4xf32>
          %130 = mulf %128, %129 : vector<4xf32>
          %131 = addf %130, %126 : vector<4xf32>
          %132 = vector.shape_cast %131 : vector<4xf32> to vector<1x4xf32>
          scf.yield %63, %86, %109, %132 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %34#0, %34#1, %34#2, %34#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %33#0, %33#1, %33#2, %33#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    vector.transfer_write %32#3, %27[%c0, %c3, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#2, %27[%c0, %c2, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#1, %27[%c0, %c1, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %32#0, %27[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After CSE ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x1x1x4xf32>
    %c8 = constant 8 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = muli %3, %c2 : index
    %7 = muli %4, %c2 : index
    %8 = subview %0[0, %6, %7, 0] [1, 9, 9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %9 = subview %1[0, 0, 0, %5] [3, 3, 16, 16] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %10 = subview %2[0, %3, %4, %5] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %11 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %12 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %13 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %14 = muli %13, %c4 : index
    %15 = muli %11, %c4 : index
    %16 = subview %10[0, %14, %12, %15] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c0, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c1, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c2, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %cst_0, %16[%c0, %c3, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    %17 = muli %13, %c8 : index
    %18 = muli %12, %c2 : index
    %19 = subview %8[0, %17, %18, 0] [1, 9, 3, 16] [1, 1, 1, 1] : memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
    %20 = subview %9[0, 0, 0, %15] [3, 3, 16, 4] [1, 1, 1, 1] : memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
    %21 = vector.transfer_read %16[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %22 = vector.transfer_read %16[%c0, %c1, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %23 = vector.transfer_read %16[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %24 = vector.transfer_read %16[%c0, %c3, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
    %25:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %21, %arg2 = %22, %arg3 = %23, %arg4 = %24) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %26:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %27:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %28 = subview %19[0, %arg0, %arg5, %arg10] [1, 7, 1, 4] [1, 1, 1, 1] : memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
          %29 = subview %20[%arg0, %arg5, %arg10, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
          %30 = vector.transfer_read %29[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %31 = vector.transfer_read %29[%c0, %c0, %c1, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %32 = vector.transfer_read %29[%c0, %c0, %c2, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %33 = vector.transfer_read %29[%c0, %c0, %c3, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
          %34 = vector.transfer_read %28[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %35 = vector.extract %34[0, 0] : vector<1x4xf32>
          %36 = vector.broadcast %35 : f32 to vector<4xf32>
          %37 = vector.shape_cast %30 : vector<1x4xf32> to vector<4xf32>
          %38 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
          %39 = mulf %36, %37 : vector<4xf32>
          %40 = addf %39, %38 : vector<4xf32>
          %41 = vector.extract %34[0, 1] : vector<1x4xf32>
          %42 = vector.broadcast %41 : f32 to vector<4xf32>
          %43 = vector.shape_cast %31 : vector<1x4xf32> to vector<4xf32>
          %44 = mulf %42, %43 : vector<4xf32>
          %45 = addf %44, %40 : vector<4xf32>
          %46 = vector.extract %34[0, 2] : vector<1x4xf32>
          %47 = vector.broadcast %46 : f32 to vector<4xf32>
          %48 = vector.shape_cast %32 : vector<1x4xf32> to vector<4xf32>
          %49 = mulf %47, %48 : vector<4xf32>
          %50 = addf %49, %45 : vector<4xf32>
          %51 = vector.extract %34[0, 3] : vector<1x4xf32>
          %52 = vector.broadcast %51 : f32 to vector<4xf32>
          %53 = vector.shape_cast %33 : vector<1x4xf32> to vector<4xf32>
          %54 = mulf %52, %53 : vector<4xf32>
          %55 = addf %54, %50 : vector<4xf32>
          %56 = vector.shape_cast %55 : vector<4xf32> to vector<1x4xf32>
          %57 = vector.transfer_read %28[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %58 = vector.extract %57[0, 0] : vector<1x4xf32>
          %59 = vector.broadcast %58 : f32 to vector<4xf32>
          %60 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
          %61 = mulf %59, %37 : vector<4xf32>
          %62 = addf %61, %60 : vector<4xf32>
          %63 = vector.extract %57[0, 1] : vector<1x4xf32>
          %64 = vector.broadcast %63 : f32 to vector<4xf32>
          %65 = mulf %64, %43 : vector<4xf32>
          %66 = addf %65, %62 : vector<4xf32>
          %67 = vector.extract %57[0, 2] : vector<1x4xf32>
          %68 = vector.broadcast %67 : f32 to vector<4xf32>
          %69 = mulf %68, %48 : vector<4xf32>
          %70 = addf %69, %66 : vector<4xf32>
          %71 = vector.extract %57[0, 3] : vector<1x4xf32>
          %72 = vector.broadcast %71 : f32 to vector<4xf32>
          %73 = mulf %72, %53 : vector<4xf32>
          %74 = addf %73, %70 : vector<4xf32>
          %75 = vector.shape_cast %74 : vector<4xf32> to vector<1x4xf32>
          %76 = vector.transfer_read %28[%c0, %c4, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %77 = vector.extract %76[0, 0] : vector<1x4xf32>
          %78 = vector.broadcast %77 : f32 to vector<4xf32>
          %79 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
          %80 = mulf %78, %37 : vector<4xf32>
          %81 = addf %80, %79 : vector<4xf32>
          %82 = vector.extract %76[0, 1] : vector<1x4xf32>
          %83 = vector.broadcast %82 : f32 to vector<4xf32>
          %84 = mulf %83, %43 : vector<4xf32>
          %85 = addf %84, %81 : vector<4xf32>
          %86 = vector.extract %76[0, 2] : vector<1x4xf32>
          %87 = vector.broadcast %86 : f32 to vector<4xf32>
          %88 = mulf %87, %48 : vector<4xf32>
          %89 = addf %88, %85 : vector<4xf32>
          %90 = vector.extract %76[0, 3] : vector<1x4xf32>
          %91 = vector.broadcast %90 : f32 to vector<4xf32>
          %92 = mulf %91, %53 : vector<4xf32>
          %93 = addf %92, %89 : vector<4xf32>
          %94 = vector.shape_cast %93 : vector<4xf32> to vector<1x4xf32>
          %95 = vector.transfer_read %28[%c0, %c6, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
          %96 = vector.extract %95[0, 0] : vector<1x4xf32>
          %97 = vector.broadcast %96 : f32 to vector<4xf32>
          %98 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
          %99 = mulf %97, %37 : vector<4xf32>
          %100 = addf %99, %98 : vector<4xf32>
          %101 = vector.extract %95[0, 1] : vector<1x4xf32>
          %102 = vector.broadcast %101 : f32 to vector<4xf32>
          %103 = mulf %102, %43 : vector<4xf32>
          %104 = addf %103, %100 : vector<4xf32>
          %105 = vector.extract %95[0, 2] : vector<1x4xf32>
          %106 = vector.broadcast %105 : f32 to vector<4xf32>
          %107 = mulf %106, %48 : vector<4xf32>
          %108 = addf %107, %104 : vector<4xf32>
          %109 = vector.extract %95[0, 3] : vector<1x4xf32>
          %110 = vector.broadcast %109 : f32 to vector<4xf32>
          %111 = mulf %110, %53 : vector<4xf32>
          %112 = addf %111, %108 : vector<4xf32>
          %113 = vector.shape_cast %112 : vector<4xf32> to vector<1x4xf32>
          scf.yield %56, %75, %94, %113 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %27#0, %27#1, %27#2, %27#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %26#0, %26#1, %26#2, %26#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    vector.transfer_write %25#3, %16[%c0, %c3, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %25#2, %16[%c0, %c2, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %25#1, %16[%c0, %c1, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    vector.transfer_write %25#0, %16[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::ResolveShapeOpsPass ***
func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
  %c16 = constant 16 : index
  %c6 = constant 6 : index
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %cst_0 = constant dense<0.000000e+00> : vector<1x1x1x4xf32>
  %c8 = constant 8 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %3 = muli %workgroup_id_z, %c4 : index
  %4 = muli %workgroup_id_y, %c4 : index
  %5 = muli %workgroup_id_x, %c16 : index
  %6 = muli %3, %c2 : index
  %7 = muli %4, %c2 : index
  %8 = subview %0[0, %6, %7, 0] [1, 9, 9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
  %9 = subview %1[0, 0, 0, %5] [3, 3, 16, 16] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
  %10 = subview %2[0, %3, %4, %5] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %11 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %12 = "gpu.thread_id"() {dimension = "y"} : () -> index
  %13 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %14 = muli %13, %c4 : index
  %15 = muli %11, %c4 : index
  %16 = subview %10[0, %14, %12, %15] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %cst_0, %16[%c0, %c0, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %cst_0, %16[%c0, %c1, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %cst_0, %16[%c0, %c2, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %cst_0, %16[%c0, %c3, %c0, %c0] {masked = [false, false, false, false]} : vector<1x1x1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %17 = muli %13, %c8 : index
  %18 = muli %12, %c2 : index
  %19 = subview %8[0, %17, %18, 0] [1, 9, 3, 16] [1, 1, 1, 1] : memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
  %20 = subview %9[0, 0, 0, %15] [3, 3, 16, 4] [1, 1, 1, 1] : memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
  %21 = vector.transfer_read %16[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
  %22 = vector.transfer_read %16[%c0, %c1, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
  %23 = vector.transfer_read %16[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
  %24 = vector.transfer_read %16[%c0, %c3, %c0, %c0], %cst {masked = [false, false]} : memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, vector<1x4xf32>
  %25:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %21, %arg2 = %22, %arg3 = %23, %arg4 = %24) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
    %26:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %27:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %28 = subview %19[0, %arg0, %arg5, %arg10] [1, 7, 1, 4] [1, 1, 1, 1] : memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
        %29 = subview %20[%arg0, %arg5, %arg10, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
        %30 = vector.transfer_read %29[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
        %31 = vector.transfer_read %29[%c0, %c0, %c1, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
        %32 = vector.transfer_read %29[%c0, %c0, %c2, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
        %33 = vector.transfer_read %29[%c0, %c0, %c3, %c0], %cst {masked = [false, false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<1x4xf32>
        %34 = vector.transfer_read %28[%c0, %c0, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
        %35 = vector.extract %34[0, 0] : vector<1x4xf32>
        %36 = vector.broadcast %35 : f32 to vector<4xf32>
        %37 = vector.shape_cast %30 : vector<1x4xf32> to vector<4xf32>
        %38 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
        %39 = mulf %36, %37 : vector<4xf32>
        %40 = addf %39, %38 : vector<4xf32>
        %41 = vector.extract %34[0, 1] : vector<1x4xf32>
        %42 = vector.broadcast %41 : f32 to vector<4xf32>
        %43 = vector.shape_cast %31 : vector<1x4xf32> to vector<4xf32>
        %44 = mulf %42, %43 : vector<4xf32>
        %45 = addf %44, %40 : vector<4xf32>
        %46 = vector.extract %34[0, 2] : vector<1x4xf32>
        %47 = vector.broadcast %46 : f32 to vector<4xf32>
        %48 = vector.shape_cast %32 : vector<1x4xf32> to vector<4xf32>
        %49 = mulf %47, %48 : vector<4xf32>
        %50 = addf %49, %45 : vector<4xf32>
        %51 = vector.extract %34[0, 3] : vector<1x4xf32>
        %52 = vector.broadcast %51 : f32 to vector<4xf32>
        %53 = vector.shape_cast %33 : vector<1x4xf32> to vector<4xf32>
        %54 = mulf %52, %53 : vector<4xf32>
        %55 = addf %54, %50 : vector<4xf32>
        %56 = vector.shape_cast %55 : vector<4xf32> to vector<1x4xf32>
        %57 = vector.transfer_read %28[%c0, %c2, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
        %58 = vector.extract %57[0, 0] : vector<1x4xf32>
        %59 = vector.broadcast %58 : f32 to vector<4xf32>
        %60 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
        %61 = mulf %59, %37 : vector<4xf32>
        %62 = addf %61, %60 : vector<4xf32>
        %63 = vector.extract %57[0, 1] : vector<1x4xf32>
        %64 = vector.broadcast %63 : f32 to vector<4xf32>
        %65 = mulf %64, %43 : vector<4xf32>
        %66 = addf %65, %62 : vector<4xf32>
        %67 = vector.extract %57[0, 2] : vector<1x4xf32>
        %68 = vector.broadcast %67 : f32 to vector<4xf32>
        %69 = mulf %68, %48 : vector<4xf32>
        %70 = addf %69, %66 : vector<4xf32>
        %71 = vector.extract %57[0, 3] : vector<1x4xf32>
        %72 = vector.broadcast %71 : f32 to vector<4xf32>
        %73 = mulf %72, %53 : vector<4xf32>
        %74 = addf %73, %70 : vector<4xf32>
        %75 = vector.shape_cast %74 : vector<4xf32> to vector<1x4xf32>
        %76 = vector.transfer_read %28[%c0, %c4, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
        %77 = vector.extract %76[0, 0] : vector<1x4xf32>
        %78 = vector.broadcast %77 : f32 to vector<4xf32>
        %79 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
        %80 = mulf %78, %37 : vector<4xf32>
        %81 = addf %80, %79 : vector<4xf32>
        %82 = vector.extract %76[0, 1] : vector<1x4xf32>
        %83 = vector.broadcast %82 : f32 to vector<4xf32>
        %84 = mulf %83, %43 : vector<4xf32>
        %85 = addf %84, %81 : vector<4xf32>
        %86 = vector.extract %76[0, 2] : vector<1x4xf32>
        %87 = vector.broadcast %86 : f32 to vector<4xf32>
        %88 = mulf %87, %48 : vector<4xf32>
        %89 = addf %88, %85 : vector<4xf32>
        %90 = vector.extract %76[0, 3] : vector<1x4xf32>
        %91 = vector.broadcast %90 : f32 to vector<4xf32>
        %92 = mulf %91, %53 : vector<4xf32>
        %93 = addf %92, %89 : vector<4xf32>
        %94 = vector.shape_cast %93 : vector<4xf32> to vector<1x4xf32>
        %95 = vector.transfer_read %28[%c0, %c6, %c0, %c0], %cst {masked = [false, false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<1x4xf32>
        %96 = vector.extract %95[0, 0] : vector<1x4xf32>
        %97 = vector.broadcast %96 : f32 to vector<4xf32>
        %98 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
        %99 = mulf %97, %37 : vector<4xf32>
        %100 = addf %99, %98 : vector<4xf32>
        %101 = vector.extract %95[0, 1] : vector<1x4xf32>
        %102 = vector.broadcast %101 : f32 to vector<4xf32>
        %103 = mulf %102, %43 : vector<4xf32>
        %104 = addf %103, %100 : vector<4xf32>
        %105 = vector.extract %95[0, 2] : vector<1x4xf32>
        %106 = vector.broadcast %105 : f32 to vector<4xf32>
        %107 = mulf %106, %48 : vector<4xf32>
        %108 = addf %107, %104 : vector<4xf32>
        %109 = vector.extract %95[0, 3] : vector<1x4xf32>
        %110 = vector.broadcast %109 : f32 to vector<4xf32>
        %111 = mulf %110, %53 : vector<4xf32>
        %112 = addf %111, %108 : vector<4xf32>
        %113 = vector.shape_cast %112 : vector<4xf32> to vector<1x4xf32>
        scf.yield %56, %75, %94, %113 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %27#0, %27#1, %27#2, %27#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    scf.yield %26#0, %26#1, %26#2, %26#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
  }
  vector.transfer_write %25#3, %16[%c0, %c3, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %25#2, %16[%c0, %c2, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %25#1, %16[%c0, %c1, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  vector.transfer_write %25#0, %16[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  return
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::VectorTransferOptimizationPass ***
func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
  %c16 = constant 16 : index
  %c6 = constant 6 : index
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %cst_0 = constant dense<0.000000e+00> : vector<1x1x1x4xf32>
  %c8 = constant 8 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %3 = muli %workgroup_id_z, %c4 : index
  %4 = muli %workgroup_id_y, %c4 : index
  %5 = muli %workgroup_id_x, %c16 : index
  %6 = muli %3, %c2 : index
  %7 = muli %4, %c2 : index
  %8 = subview %0[0, %6, %7, 0] [1, 9, 9, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
  %9 = subview %1[0, 0, 0, %5] [3, 3, 16, 16] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
  %10 = subview %2[0, %3, %4, %5] [1, 4, 4, 16] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %11 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %12 = "gpu.thread_id"() {dimension = "y"} : () -> index
  %13 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %14 = muli %13, %c4 : index
  %15 = muli %11, %c4 : index
  %16 = subview %10[0, %14, %12, %15] [1, 4, 1, 4] [1, 1, 1, 1] : memref<1x4x4x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>> to memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %17 = vector.shape_cast %cst_0 : vector<1x1x1x4xf32> to vector<4xf32>
  %18 = vector.shape_cast %cst_0 : vector<1x1x1x4xf32> to vector<4xf32>
  %19 = vector.shape_cast %cst_0 : vector<1x1x1x4xf32> to vector<4xf32>
  %20 = vector.shape_cast %cst_0 : vector<1x1x1x4xf32> to vector<4xf32>
  %21 = muli %13, %c8 : index
  %22 = muli %12, %c2 : index
  %23 = subview %8[0, %21, %22, 0] [1, 9, 3, 16] [1, 1, 1, 1] : memref<1x9x9x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
  %24 = subview %9[0, 0, 0, %15] [3, 3, 16, 4] [1, 1, 1, 1] : memref<3x3x16x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
  %25 = vector.shape_cast %17 : vector<4xf32> to vector<1x4xf32>
  %26 = vector.shape_cast %18 : vector<4xf32> to vector<1x4xf32>
  %27 = vector.shape_cast %19 : vector<4xf32> to vector<1x4xf32>
  %28 = vector.shape_cast %20 : vector<4xf32> to vector<1x4xf32>
  %29:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %25, %arg2 = %26, %arg3 = %27, %arg4 = %28) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
    %34:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %35:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %36 = subview %23[0, %arg0, %arg5, %arg10] [1, 7, 1, 4] [1, 1, 1, 1] : memref<1x9x3x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>> to memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
        %37 = subview %24[%arg0, %arg5, %arg10, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<3x3x16x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>> to memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
        %38 = vector.transfer_read %37[%c0, %c0, %c0, %c0], %cst {masked = [false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<4xf32>
        %39 = vector.transfer_read %37[%c0, %c0, %c1, %c0], %cst {masked = [false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<4xf32>
        %40 = vector.transfer_read %37[%c0, %c0, %c2, %c0], %cst {masked = [false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<4xf32>
        %41 = vector.transfer_read %37[%c0, %c0, %c3, %c0], %cst {masked = [false]} : memref<1x1x4x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>, vector<4xf32>
        %42 = vector.transfer_read %36[%c0, %c0, %c0, %c0], %cst {masked = [false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<4xf32>
        %43 = vector.extract %42[0] : vector<4xf32>
        %44 = vector.broadcast %43 : f32 to vector<4xf32>
        %45 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
        %46 = mulf %44, %38 : vector<4xf32>
        %47 = addf %46, %45 : vector<4xf32>
        %48 = vector.extract %42[1] : vector<4xf32>
        %49 = vector.broadcast %48 : f32 to vector<4xf32>
        %50 = mulf %49, %39 : vector<4xf32>
        %51 = addf %50, %47 : vector<4xf32>
        %52 = vector.extract %42[2] : vector<4xf32>
        %53 = vector.broadcast %52 : f32 to vector<4xf32>
        %54 = mulf %53, %40 : vector<4xf32>
        %55 = addf %54, %51 : vector<4xf32>
        %56 = vector.extract %42[3] : vector<4xf32>
        %57 = vector.broadcast %56 : f32 to vector<4xf32>
        %58 = mulf %57, %41 : vector<4xf32>
        %59 = addf %58, %55 : vector<4xf32>
        %60 = vector.shape_cast %59 : vector<4xf32> to vector<1x4xf32>
        %61 = vector.transfer_read %36[%c0, %c2, %c0, %c0], %cst {masked = [false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<4xf32>
        %62 = vector.extract %61[0] : vector<4xf32>
        %63 = vector.broadcast %62 : f32 to vector<4xf32>
        %64 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
        %65 = mulf %63, %38 : vector<4xf32>
        %66 = addf %65, %64 : vector<4xf32>
        %67 = vector.extract %61[1] : vector<4xf32>
        %68 = vector.broadcast %67 : f32 to vector<4xf32>
        %69 = mulf %68, %39 : vector<4xf32>
        %70 = addf %69, %66 : vector<4xf32>
        %71 = vector.extract %61[2] : vector<4xf32>
        %72 = vector.broadcast %71 : f32 to vector<4xf32>
        %73 = mulf %72, %40 : vector<4xf32>
        %74 = addf %73, %70 : vector<4xf32>
        %75 = vector.extract %61[3] : vector<4xf32>
        %76 = vector.broadcast %75 : f32 to vector<4xf32>
        %77 = mulf %76, %41 : vector<4xf32>
        %78 = addf %77, %74 : vector<4xf32>
        %79 = vector.shape_cast %78 : vector<4xf32> to vector<1x4xf32>
        %80 = vector.transfer_read %36[%c0, %c4, %c0, %c0], %cst {masked = [false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<4xf32>
        %81 = vector.extract %80[0] : vector<4xf32>
        %82 = vector.broadcast %81 : f32 to vector<4xf32>
        %83 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
        %84 = mulf %82, %38 : vector<4xf32>
        %85 = addf %84, %83 : vector<4xf32>
        %86 = vector.extract %80[1] : vector<4xf32>
        %87 = vector.broadcast %86 : f32 to vector<4xf32>
        %88 = mulf %87, %39 : vector<4xf32>
        %89 = addf %88, %85 : vector<4xf32>
        %90 = vector.extract %80[2] : vector<4xf32>
        %91 = vector.broadcast %90 : f32 to vector<4xf32>
        %92 = mulf %91, %40 : vector<4xf32>
        %93 = addf %92, %89 : vector<4xf32>
        %94 = vector.extract %80[3] : vector<4xf32>
        %95 = vector.broadcast %94 : f32 to vector<4xf32>
        %96 = mulf %95, %41 : vector<4xf32>
        %97 = addf %96, %93 : vector<4xf32>
        %98 = vector.shape_cast %97 : vector<4xf32> to vector<1x4xf32>
        %99 = vector.transfer_read %36[%c0, %c6, %c0, %c0], %cst {masked = [false]} : memref<1x7x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, vector<4xf32>
        %100 = vector.extract %99[0] : vector<4xf32>
        %101 = vector.broadcast %100 : f32 to vector<4xf32>
        %102 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
        %103 = mulf %101, %38 : vector<4xf32>
        %104 = addf %103, %102 : vector<4xf32>
        %105 = vector.extract %99[1] : vector<4xf32>
        %106 = vector.broadcast %105 : f32 to vector<4xf32>
        %107 = mulf %106, %39 : vector<4xf32>
        %108 = addf %107, %104 : vector<4xf32>
        %109 = vector.extract %99[2] : vector<4xf32>
        %110 = vector.broadcast %109 : f32 to vector<4xf32>
        %111 = mulf %110, %40 : vector<4xf32>
        %112 = addf %111, %108 : vector<4xf32>
        %113 = vector.extract %99[3] : vector<4xf32>
        %114 = vector.broadcast %113 : f32 to vector<4xf32>
        %115 = mulf %114, %41 : vector<4xf32>
        %116 = addf %115, %112 : vector<4xf32>
        %117 = vector.shape_cast %116 : vector<4xf32> to vector<1x4xf32>
        scf.yield %60, %79, %98, %117 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %35#0, %35#1, %35#2, %35#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    scf.yield %34#0, %34#1, %34#2, %34#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
  }
  %30 = vector.shape_cast %29#3 : vector<1x4xf32> to vector<4xf32>
  vector.transfer_write %30, %16[%c0, %c3, %c0, %c0] {masked = [false]} : vector<4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %31 = vector.shape_cast %29#2 : vector<1x4xf32> to vector<4xf32>
  vector.transfer_write %31, %16[%c0, %c2, %c0, %c0] {masked = [false]} : vector<4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %32 = vector.shape_cast %29#1 : vector<1x4xf32> to vector<4xf32>
  vector.transfer_write %32, %16[%c0, %c1, %c0, %c0] {masked = [false]} : vector<4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  %33 = vector.shape_cast %29#0 : vector<1x4xf32> to vector<4xf32>
  vector.transfer_write %33, %16[%c0, %c0, %c0, %c0] {masked = [false]} : vector<4xf32>, memref<1x4x1x4xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
  return
}

// *** IR Dump After LegalizeStandardForSPIRV ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x1x1x4xf32>
    %c8 = constant 8 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = muli %3, %c2 : index
    %7 = muli %4, %c2 : index
    %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %9 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %10 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %11 = muli %10, %c4 : index
    %12 = muli %8, %c4 : index
    %13 = muli %10, %c8 : index
    %14 = muli %9, %c2 : index
    %15 = vector.shape_cast %cst_0 : vector<1x1x1x4xf32> to vector<1x4xf32>
    %16 = vector.shape_cast %cst_0 : vector<1x1x1x4xf32> to vector<1x4xf32>
    %17 = vector.shape_cast %cst_0 : vector<1x1x1x4xf32> to vector<1x4xf32>
    %18 = vector.shape_cast %cst_0 : vector<1x1x1x4xf32> to vector<1x4xf32>
    %19:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %15, %arg2 = %16, %arg3 = %17, %arg4 = %18) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %39:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %40:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %41 = addi %5, %12 : index
          %42 = vector.transfer_read %1[%arg0, %arg5, %arg10, %41], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %43 = addi %arg10, %c1 : index
          %44 = addi %5, %12 : index
          %45 = vector.transfer_read %1[%arg0, %arg5, %43, %44], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %46 = addi %arg10, %c2 : index
          %47 = addi %5, %12 : index
          %48 = vector.transfer_read %1[%arg0, %arg5, %46, %47], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %49 = addi %arg10, %c3 : index
          %50 = addi %5, %12 : index
          %51 = vector.transfer_read %1[%arg0, %arg5, %49, %50], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %52 = addi %13, %arg0 : index
          %53 = addi %14, %arg5 : index
          %54 = addi %6, %52 : index
          %55 = addi %7, %53 : index
          %56 = vector.transfer_read %0[%c0, %54, %55, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %57 = vector.extract %56[0] : vector<4xf32>
          %58 = vector.broadcast %57 : f32 to vector<4xf32>
          %59 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
          %60 = mulf %58, %42 : vector<4xf32>
          %61 = addf %60, %59 : vector<4xf32>
          %62 = vector.extract %56[1] : vector<4xf32>
          %63 = vector.broadcast %62 : f32 to vector<4xf32>
          %64 = mulf %63, %45 : vector<4xf32>
          %65 = addf %64, %61 : vector<4xf32>
          %66 = vector.extract %56[2] : vector<4xf32>
          %67 = vector.broadcast %66 : f32 to vector<4xf32>
          %68 = mulf %67, %48 : vector<4xf32>
          %69 = addf %68, %65 : vector<4xf32>
          %70 = vector.extract %56[3] : vector<4xf32>
          %71 = vector.broadcast %70 : f32 to vector<4xf32>
          %72 = mulf %71, %51 : vector<4xf32>
          %73 = addf %72, %69 : vector<4xf32>
          %74 = vector.shape_cast %73 : vector<4xf32> to vector<1x4xf32>
          %75 = addi %arg0, %c2 : index
          %76 = addi %13, %75 : index
          %77 = addi %14, %arg5 : index
          %78 = addi %6, %76 : index
          %79 = addi %7, %77 : index
          %80 = vector.transfer_read %0[%c0, %78, %79, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %81 = vector.extract %80[0] : vector<4xf32>
          %82 = vector.broadcast %81 : f32 to vector<4xf32>
          %83 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
          %84 = mulf %82, %42 : vector<4xf32>
          %85 = addf %84, %83 : vector<4xf32>
          %86 = vector.extract %80[1] : vector<4xf32>
          %87 = vector.broadcast %86 : f32 to vector<4xf32>
          %88 = mulf %87, %45 : vector<4xf32>
          %89 = addf %88, %85 : vector<4xf32>
          %90 = vector.extract %80[2] : vector<4xf32>
          %91 = vector.broadcast %90 : f32 to vector<4xf32>
          %92 = mulf %91, %48 : vector<4xf32>
          %93 = addf %92, %89 : vector<4xf32>
          %94 = vector.extract %80[3] : vector<4xf32>
          %95 = vector.broadcast %94 : f32 to vector<4xf32>
          %96 = mulf %95, %51 : vector<4xf32>
          %97 = addf %96, %93 : vector<4xf32>
          %98 = vector.shape_cast %97 : vector<4xf32> to vector<1x4xf32>
          %99 = addi %arg0, %c4 : index
          %100 = addi %13, %99 : index
          %101 = addi %14, %arg5 : index
          %102 = addi %6, %100 : index
          %103 = addi %7, %101 : index
          %104 = vector.transfer_read %0[%c0, %102, %103, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %105 = vector.extract %104[0] : vector<4xf32>
          %106 = vector.broadcast %105 : f32 to vector<4xf32>
          %107 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
          %108 = mulf %106, %42 : vector<4xf32>
          %109 = addf %108, %107 : vector<4xf32>
          %110 = vector.extract %104[1] : vector<4xf32>
          %111 = vector.broadcast %110 : f32 to vector<4xf32>
          %112 = mulf %111, %45 : vector<4xf32>
          %113 = addf %112, %109 : vector<4xf32>
          %114 = vector.extract %104[2] : vector<4xf32>
          %115 = vector.broadcast %114 : f32 to vector<4xf32>
          %116 = mulf %115, %48 : vector<4xf32>
          %117 = addf %116, %113 : vector<4xf32>
          %118 = vector.extract %104[3] : vector<4xf32>
          %119 = vector.broadcast %118 : f32 to vector<4xf32>
          %120 = mulf %119, %51 : vector<4xf32>
          %121 = addf %120, %117 : vector<4xf32>
          %122 = vector.shape_cast %121 : vector<4xf32> to vector<1x4xf32>
          %123 = addi %arg0, %c6 : index
          %124 = addi %13, %123 : index
          %125 = addi %14, %arg5 : index
          %126 = addi %6, %124 : index
          %127 = addi %7, %125 : index
          %128 = vector.transfer_read %0[%c0, %126, %127, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %129 = vector.extract %128[0] : vector<4xf32>
          %130 = vector.broadcast %129 : f32 to vector<4xf32>
          %131 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
          %132 = mulf %130, %42 : vector<4xf32>
          %133 = addf %132, %131 : vector<4xf32>
          %134 = vector.extract %128[1] : vector<4xf32>
          %135 = vector.broadcast %134 : f32 to vector<4xf32>
          %136 = mulf %135, %45 : vector<4xf32>
          %137 = addf %136, %133 : vector<4xf32>
          %138 = vector.extract %128[2] : vector<4xf32>
          %139 = vector.broadcast %138 : f32 to vector<4xf32>
          %140 = mulf %139, %48 : vector<4xf32>
          %141 = addf %140, %137 : vector<4xf32>
          %142 = vector.extract %128[3] : vector<4xf32>
          %143 = vector.broadcast %142 : f32 to vector<4xf32>
          %144 = mulf %143, %51 : vector<4xf32>
          %145 = addf %144, %141 : vector<4xf32>
          %146 = vector.shape_cast %145 : vector<4xf32> to vector<1x4xf32>
          scf.yield %74, %98, %122, %146 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %40#0, %40#1, %40#2, %40#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %39#0, %39#1, %39#2, %39#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    %20 = vector.shape_cast %19#3 : vector<1x4xf32> to vector<4xf32>
    %21 = addi %11, %c3 : index
    %22 = addi %3, %21 : index
    %23 = addi %4, %9 : index
    %24 = addi %5, %12 : index
    vector.transfer_write %20, %2[%c0, %22, %23, %24] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %25 = vector.shape_cast %19#2 : vector<1x4xf32> to vector<4xf32>
    %26 = addi %11, %c2 : index
    %27 = addi %3, %26 : index
    %28 = addi %4, %9 : index
    %29 = addi %5, %12 : index
    vector.transfer_write %25, %2[%c0, %27, %28, %29] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %30 = vector.shape_cast %19#1 : vector<1x4xf32> to vector<4xf32>
    %31 = addi %11, %c1 : index
    %32 = addi %3, %31 : index
    %33 = addi %4, %9 : index
    %34 = addi %5, %12 : index
    vector.transfer_write %30, %2[%c0, %32, %33, %34] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %35 = vector.shape_cast %19#0 : vector<1x4xf32> to vector<4xf32>
    %36 = addi %3, %11 : index
    %37 = addi %4, %9 : index
    %38 = addi %5, %12 : index
    vector.transfer_write %35, %2[%c0, %36, %37, %38] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After Canonicalizer ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    %c8 = constant 8 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x4xf32>
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = muli %3, %c2 : index
    %7 = muli %4, %c2 : index
    %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %9 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %10 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %11 = muli %10, %c4 : index
    %12 = muli %8, %c4 : index
    %13 = muli %10, %c8 : index
    %14 = muli %9, %c2 : index
    %15:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %cst_0, %arg2 = %cst_0, %arg3 = %cst_0, %arg4 = %cst_0) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %35:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %36:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %37 = addi %5, %12 : index
          %38 = vector.transfer_read %1[%arg0, %arg5, %arg10, %37], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %39 = addi %arg10, %c1 : index
          %40 = addi %5, %12 : index
          %41 = vector.transfer_read %1[%arg0, %arg5, %39, %40], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %42 = addi %arg10, %c2 : index
          %43 = addi %5, %12 : index
          %44 = vector.transfer_read %1[%arg0, %arg5, %42, %43], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %45 = addi %arg10, %c3 : index
          %46 = addi %5, %12 : index
          %47 = vector.transfer_read %1[%arg0, %arg5, %45, %46], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %48 = addi %13, %arg0 : index
          %49 = addi %14, %arg5 : index
          %50 = addi %6, %48 : index
          %51 = addi %7, %49 : index
          %52 = vector.transfer_read %0[%c0, %50, %51, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %53 = vector.extract %52[0] : vector<4xf32>
          %54 = vector.broadcast %53 : f32 to vector<4xf32>
          %55 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
          %56 = mulf %54, %38 : vector<4xf32>
          %57 = addf %56, %55 : vector<4xf32>
          %58 = vector.extract %52[1] : vector<4xf32>
          %59 = vector.broadcast %58 : f32 to vector<4xf32>
          %60 = mulf %59, %41 : vector<4xf32>
          %61 = addf %60, %57 : vector<4xf32>
          %62 = vector.extract %52[2] : vector<4xf32>
          %63 = vector.broadcast %62 : f32 to vector<4xf32>
          %64 = mulf %63, %44 : vector<4xf32>
          %65 = addf %64, %61 : vector<4xf32>
          %66 = vector.extract %52[3] : vector<4xf32>
          %67 = vector.broadcast %66 : f32 to vector<4xf32>
          %68 = mulf %67, %47 : vector<4xf32>
          %69 = addf %68, %65 : vector<4xf32>
          %70 = vector.shape_cast %69 : vector<4xf32> to vector<1x4xf32>
          %71 = addi %arg0, %c2 : index
          %72 = addi %13, %71 : index
          %73 = addi %14, %arg5 : index
          %74 = addi %6, %72 : index
          %75 = addi %7, %73 : index
          %76 = vector.transfer_read %0[%c0, %74, %75, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %77 = vector.extract %76[0] : vector<4xf32>
          %78 = vector.broadcast %77 : f32 to vector<4xf32>
          %79 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
          %80 = mulf %78, %38 : vector<4xf32>
          %81 = addf %80, %79 : vector<4xf32>
          %82 = vector.extract %76[1] : vector<4xf32>
          %83 = vector.broadcast %82 : f32 to vector<4xf32>
          %84 = mulf %83, %41 : vector<4xf32>
          %85 = addf %84, %81 : vector<4xf32>
          %86 = vector.extract %76[2] : vector<4xf32>
          %87 = vector.broadcast %86 : f32 to vector<4xf32>
          %88 = mulf %87, %44 : vector<4xf32>
          %89 = addf %88, %85 : vector<4xf32>
          %90 = vector.extract %76[3] : vector<4xf32>
          %91 = vector.broadcast %90 : f32 to vector<4xf32>
          %92 = mulf %91, %47 : vector<4xf32>
          %93 = addf %92, %89 : vector<4xf32>
          %94 = vector.shape_cast %93 : vector<4xf32> to vector<1x4xf32>
          %95 = addi %arg0, %c4 : index
          %96 = addi %13, %95 : index
          %97 = addi %14, %arg5 : index
          %98 = addi %6, %96 : index
          %99 = addi %7, %97 : index
          %100 = vector.transfer_read %0[%c0, %98, %99, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %101 = vector.extract %100[0] : vector<4xf32>
          %102 = vector.broadcast %101 : f32 to vector<4xf32>
          %103 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
          %104 = mulf %102, %38 : vector<4xf32>
          %105 = addf %104, %103 : vector<4xf32>
          %106 = vector.extract %100[1] : vector<4xf32>
          %107 = vector.broadcast %106 : f32 to vector<4xf32>
          %108 = mulf %107, %41 : vector<4xf32>
          %109 = addf %108, %105 : vector<4xf32>
          %110 = vector.extract %100[2] : vector<4xf32>
          %111 = vector.broadcast %110 : f32 to vector<4xf32>
          %112 = mulf %111, %44 : vector<4xf32>
          %113 = addf %112, %109 : vector<4xf32>
          %114 = vector.extract %100[3] : vector<4xf32>
          %115 = vector.broadcast %114 : f32 to vector<4xf32>
          %116 = mulf %115, %47 : vector<4xf32>
          %117 = addf %116, %113 : vector<4xf32>
          %118 = vector.shape_cast %117 : vector<4xf32> to vector<1x4xf32>
          %119 = addi %arg0, %c6 : index
          %120 = addi %13, %119 : index
          %121 = addi %14, %arg5 : index
          %122 = addi %6, %120 : index
          %123 = addi %7, %121 : index
          %124 = vector.transfer_read %0[%c0, %122, %123, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %125 = vector.extract %124[0] : vector<4xf32>
          %126 = vector.broadcast %125 : f32 to vector<4xf32>
          %127 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
          %128 = mulf %126, %38 : vector<4xf32>
          %129 = addf %128, %127 : vector<4xf32>
          %130 = vector.extract %124[1] : vector<4xf32>
          %131 = vector.broadcast %130 : f32 to vector<4xf32>
          %132 = mulf %131, %41 : vector<4xf32>
          %133 = addf %132, %129 : vector<4xf32>
          %134 = vector.extract %124[2] : vector<4xf32>
          %135 = vector.broadcast %134 : f32 to vector<4xf32>
          %136 = mulf %135, %44 : vector<4xf32>
          %137 = addf %136, %133 : vector<4xf32>
          %138 = vector.extract %124[3] : vector<4xf32>
          %139 = vector.broadcast %138 : f32 to vector<4xf32>
          %140 = mulf %139, %47 : vector<4xf32>
          %141 = addf %140, %137 : vector<4xf32>
          %142 = vector.shape_cast %141 : vector<4xf32> to vector<1x4xf32>
          scf.yield %70, %94, %118, %142 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %36#0, %36#1, %36#2, %36#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %35#0, %35#1, %35#2, %35#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    %16 = vector.shape_cast %15#3 : vector<1x4xf32> to vector<4xf32>
    %17 = addi %11, %c3 : index
    %18 = addi %3, %17 : index
    %19 = addi %4, %9 : index
    %20 = addi %5, %12 : index
    vector.transfer_write %16, %2[%c0, %18, %19, %20] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %21 = vector.shape_cast %15#2 : vector<1x4xf32> to vector<4xf32>
    %22 = addi %11, %c2 : index
    %23 = addi %3, %22 : index
    %24 = addi %4, %9 : index
    %25 = addi %5, %12 : index
    vector.transfer_write %21, %2[%c0, %23, %24, %25] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %26 = vector.shape_cast %15#1 : vector<1x4xf32> to vector<4xf32>
    %27 = addi %11, %c1 : index
    %28 = addi %3, %27 : index
    %29 = addi %4, %9 : index
    %30 = addi %5, %12 : index
    vector.transfer_write %26, %2[%c0, %28, %29, %30] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %31 = vector.shape_cast %15#0 : vector<1x4xf32> to vector<4xf32>
    %32 = addi %3, %11 : index
    %33 = addi %4, %9 : index
    %34 = addi %5, %12 : index
    vector.transfer_write %31, %2[%c0, %32, %33, %34] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After CSE ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    %c8 = constant 8 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x4xf32>
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = muli %3, %c2 : index
    %7 = muli %4, %c2 : index
    %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %9 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %10 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %11 = muli %10, %c4 : index
    %12 = muli %8, %c4 : index
    %13 = muli %10, %c8 : index
    %14 = muli %9, %c2 : index
    %15:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %cst_0, %arg2 = %cst_0, %arg3 = %cst_0, %arg4 = %cst_0) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %29:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %30:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %31 = addi %5, %12 : index
          %32 = vector.transfer_read %1[%arg0, %arg5, %arg10, %31], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %33 = addi %arg10, %c1 : index
          %34 = vector.transfer_read %1[%arg0, %arg5, %33, %31], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %35 = addi %arg10, %c2 : index
          %36 = vector.transfer_read %1[%arg0, %arg5, %35, %31], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %37 = addi %arg10, %c3 : index
          %38 = vector.transfer_read %1[%arg0, %arg5, %37, %31], %cst {masked = [false]} : memref<3x3x16x32xf32>, vector<4xf32>
          %39 = addi %13, %arg0 : index
          %40 = addi %14, %arg5 : index
          %41 = addi %6, %39 : index
          %42 = addi %7, %40 : index
          %43 = vector.transfer_read %0[%c0, %41, %42, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %44 = vector.extract %43[0] : vector<4xf32>
          %45 = vector.broadcast %44 : f32 to vector<4xf32>
          %46 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
          %47 = mulf %45, %32 : vector<4xf32>
          %48 = addf %47, %46 : vector<4xf32>
          %49 = vector.extract %43[1] : vector<4xf32>
          %50 = vector.broadcast %49 : f32 to vector<4xf32>
          %51 = mulf %50, %34 : vector<4xf32>
          %52 = addf %51, %48 : vector<4xf32>
          %53 = vector.extract %43[2] : vector<4xf32>
          %54 = vector.broadcast %53 : f32 to vector<4xf32>
          %55 = mulf %54, %36 : vector<4xf32>
          %56 = addf %55, %52 : vector<4xf32>
          %57 = vector.extract %43[3] : vector<4xf32>
          %58 = vector.broadcast %57 : f32 to vector<4xf32>
          %59 = mulf %58, %38 : vector<4xf32>
          %60 = addf %59, %56 : vector<4xf32>
          %61 = vector.shape_cast %60 : vector<4xf32> to vector<1x4xf32>
          %62 = addi %arg0, %c2 : index
          %63 = addi %13, %62 : index
          %64 = addi %6, %63 : index
          %65 = vector.transfer_read %0[%c0, %64, %42, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %66 = vector.extract %65[0] : vector<4xf32>
          %67 = vector.broadcast %66 : f32 to vector<4xf32>
          %68 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
          %69 = mulf %67, %32 : vector<4xf32>
          %70 = addf %69, %68 : vector<4xf32>
          %71 = vector.extract %65[1] : vector<4xf32>
          %72 = vector.broadcast %71 : f32 to vector<4xf32>
          %73 = mulf %72, %34 : vector<4xf32>
          %74 = addf %73, %70 : vector<4xf32>
          %75 = vector.extract %65[2] : vector<4xf32>
          %76 = vector.broadcast %75 : f32 to vector<4xf32>
          %77 = mulf %76, %36 : vector<4xf32>
          %78 = addf %77, %74 : vector<4xf32>
          %79 = vector.extract %65[3] : vector<4xf32>
          %80 = vector.broadcast %79 : f32 to vector<4xf32>
          %81 = mulf %80, %38 : vector<4xf32>
          %82 = addf %81, %78 : vector<4xf32>
          %83 = vector.shape_cast %82 : vector<4xf32> to vector<1x4xf32>
          %84 = addi %arg0, %c4 : index
          %85 = addi %13, %84 : index
          %86 = addi %6, %85 : index
          %87 = vector.transfer_read %0[%c0, %86, %42, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %88 = vector.extract %87[0] : vector<4xf32>
          %89 = vector.broadcast %88 : f32 to vector<4xf32>
          %90 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
          %91 = mulf %89, %32 : vector<4xf32>
          %92 = addf %91, %90 : vector<4xf32>
          %93 = vector.extract %87[1] : vector<4xf32>
          %94 = vector.broadcast %93 : f32 to vector<4xf32>
          %95 = mulf %94, %34 : vector<4xf32>
          %96 = addf %95, %92 : vector<4xf32>
          %97 = vector.extract %87[2] : vector<4xf32>
          %98 = vector.broadcast %97 : f32 to vector<4xf32>
          %99 = mulf %98, %36 : vector<4xf32>
          %100 = addf %99, %96 : vector<4xf32>
          %101 = vector.extract %87[3] : vector<4xf32>
          %102 = vector.broadcast %101 : f32 to vector<4xf32>
          %103 = mulf %102, %38 : vector<4xf32>
          %104 = addf %103, %100 : vector<4xf32>
          %105 = vector.shape_cast %104 : vector<4xf32> to vector<1x4xf32>
          %106 = addi %arg0, %c6 : index
          %107 = addi %13, %106 : index
          %108 = addi %6, %107 : index
          %109 = vector.transfer_read %0[%c0, %108, %42, %arg10], %cst {masked = [false]} : memref<1x225x225x16xf32>, vector<4xf32>
          %110 = vector.extract %109[0] : vector<4xf32>
          %111 = vector.broadcast %110 : f32 to vector<4xf32>
          %112 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
          %113 = mulf %111, %32 : vector<4xf32>
          %114 = addf %113, %112 : vector<4xf32>
          %115 = vector.extract %109[1] : vector<4xf32>
          %116 = vector.broadcast %115 : f32 to vector<4xf32>
          %117 = mulf %116, %34 : vector<4xf32>
          %118 = addf %117, %114 : vector<4xf32>
          %119 = vector.extract %109[2] : vector<4xf32>
          %120 = vector.broadcast %119 : f32 to vector<4xf32>
          %121 = mulf %120, %36 : vector<4xf32>
          %122 = addf %121, %118 : vector<4xf32>
          %123 = vector.extract %109[3] : vector<4xf32>
          %124 = vector.broadcast %123 : f32 to vector<4xf32>
          %125 = mulf %124, %38 : vector<4xf32>
          %126 = addf %125, %122 : vector<4xf32>
          %127 = vector.shape_cast %126 : vector<4xf32> to vector<1x4xf32>
          scf.yield %61, %83, %105, %127 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %30#0, %30#1, %30#2, %30#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %29#0, %29#1, %29#2, %29#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    %16 = vector.shape_cast %15#3 : vector<1x4xf32> to vector<4xf32>
    %17 = addi %11, %c3 : index
    %18 = addi %3, %17 : index
    %19 = addi %4, %9 : index
    %20 = addi %5, %12 : index
    vector.transfer_write %16, %2[%c0, %18, %19, %20] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %21 = vector.shape_cast %15#2 : vector<1x4xf32> to vector<4xf32>
    %22 = addi %11, %c2 : index
    %23 = addi %3, %22 : index
    vector.transfer_write %21, %2[%c0, %23, %19, %20] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %24 = vector.shape_cast %15#1 : vector<1x4xf32> to vector<4xf32>
    %25 = addi %11, %c1 : index
    %26 = addi %3, %25 : index
    vector.transfer_write %24, %2[%c0, %26, %19, %20] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    %27 = vector.shape_cast %15#0 : vector<1x4xf32> to vector<4xf32>
    %28 = addi %3, %11 : index
    vector.transfer_write %27, %2[%c0, %28, %19, %20] {masked = [false]} : vector<4xf32>, memref<1x112x112x32xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::VectorizeMemRefPass ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    %c8 = constant 8 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x4xf32>
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x4xvector<4xf32>>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x8xvector<4xf32>>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x8xvector<4xf32>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = muli %3, %c2 : index
    %7 = muli %4, %c2 : index
    %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %9 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %10 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %11 = muli %10, %c4 : index
    %12 = muli %8, %c4 : index
    %13 = muli %10, %c8 : index
    %14 = muli %9, %c2 : index
    %15:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %cst_0, %arg2 = %cst_0, %arg3 = %cst_0, %arg4 = %cst_0) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
      %41:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
        %42:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) {
          %43 = addi %5, %12 : index
          %c4_5 = constant 4 : index
          %44 = divi_signed %43, %c4_5 : index
          %45 = load %1[%arg0, %arg5, %arg10, %44] : memref<3x3x16x8xvector<4xf32>>
          %46 = vector.bitcast %45 : vector<4xf32> to vector<4xf32>
          %47 = vector.shape_cast %46 : vector<4xf32> to vector<4xf32>
          %48 = addi %arg10, %c1 : index
          %c4_6 = constant 4 : index
          %49 = divi_signed %43, %c4_6 : index
          %50 = load %1[%arg0, %arg5, %48, %49] : memref<3x3x16x8xvector<4xf32>>
          %51 = vector.bitcast %50 : vector<4xf32> to vector<4xf32>
          %52 = vector.shape_cast %51 : vector<4xf32> to vector<4xf32>
          %53 = addi %arg10, %c2 : index
          %c4_7 = constant 4 : index
          %54 = divi_signed %43, %c4_7 : index
          %55 = load %1[%arg0, %arg5, %53, %54] : memref<3x3x16x8xvector<4xf32>>
          %56 = vector.bitcast %55 : vector<4xf32> to vector<4xf32>
          %57 = vector.shape_cast %56 : vector<4xf32> to vector<4xf32>
          %58 = addi %arg10, %c3 : index
          %c4_8 = constant 4 : index
          %59 = divi_signed %43, %c4_8 : index
          %60 = load %1[%arg0, %arg5, %58, %59] : memref<3x3x16x8xvector<4xf32>>
          %61 = vector.bitcast %60 : vector<4xf32> to vector<4xf32>
          %62 = vector.shape_cast %61 : vector<4xf32> to vector<4xf32>
          %63 = addi %13, %arg0 : index
          %64 = addi %14, %arg5 : index
          %65 = addi %6, %63 : index
          %66 = addi %7, %64 : index
          %c4_9 = constant 4 : index
          %67 = divi_signed %arg10, %c4_9 : index
          %68 = load %0[%c0, %65, %66, %67] : memref<1x225x225x4xvector<4xf32>>
          %69 = vector.bitcast %68 : vector<4xf32> to vector<4xf32>
          %70 = vector.shape_cast %69 : vector<4xf32> to vector<4xf32>
          %71 = vector.extract %70[0] : vector<4xf32>
          %72 = vector.broadcast %71 : f32 to vector<4xf32>
          %73 = vector.shape_cast %arg11 : vector<1x4xf32> to vector<4xf32>
          %74 = mulf %72, %47 : vector<4xf32>
          %75 = addf %74, %73 : vector<4xf32>
          %76 = vector.extract %70[1] : vector<4xf32>
          %77 = vector.broadcast %76 : f32 to vector<4xf32>
          %78 = mulf %77, %52 : vector<4xf32>
          %79 = addf %78, %75 : vector<4xf32>
          %80 = vector.extract %70[2] : vector<4xf32>
          %81 = vector.broadcast %80 : f32 to vector<4xf32>
          %82 = mulf %81, %57 : vector<4xf32>
          %83 = addf %82, %79 : vector<4xf32>
          %84 = vector.extract %70[3] : vector<4xf32>
          %85 = vector.broadcast %84 : f32 to vector<4xf32>
          %86 = mulf %85, %62 : vector<4xf32>
          %87 = addf %86, %83 : vector<4xf32>
          %88 = vector.shape_cast %87 : vector<4xf32> to vector<1x4xf32>
          %89 = addi %arg0, %c2 : index
          %90 = addi %13, %89 : index
          %91 = addi %6, %90 : index
          %c4_10 = constant 4 : index
          %92 = divi_signed %arg10, %c4_10 : index
          %93 = load %0[%c0, %91, %66, %92] : memref<1x225x225x4xvector<4xf32>>
          %94 = vector.bitcast %93 : vector<4xf32> to vector<4xf32>
          %95 = vector.shape_cast %94 : vector<4xf32> to vector<4xf32>
          %96 = vector.extract %95[0] : vector<4xf32>
          %97 = vector.broadcast %96 : f32 to vector<4xf32>
          %98 = vector.shape_cast %arg12 : vector<1x4xf32> to vector<4xf32>
          %99 = mulf %97, %47 : vector<4xf32>
          %100 = addf %99, %98 : vector<4xf32>
          %101 = vector.extract %95[1] : vector<4xf32>
          %102 = vector.broadcast %101 : f32 to vector<4xf32>
          %103 = mulf %102, %52 : vector<4xf32>
          %104 = addf %103, %100 : vector<4xf32>
          %105 = vector.extract %95[2] : vector<4xf32>
          %106 = vector.broadcast %105 : f32 to vector<4xf32>
          %107 = mulf %106, %57 : vector<4xf32>
          %108 = addf %107, %104 : vector<4xf32>
          %109 = vector.extract %95[3] : vector<4xf32>
          %110 = vector.broadcast %109 : f32 to vector<4xf32>
          %111 = mulf %110, %62 : vector<4xf32>
          %112 = addf %111, %108 : vector<4xf32>
          %113 = vector.shape_cast %112 : vector<4xf32> to vector<1x4xf32>
          %114 = addi %arg0, %c4 : index
          %115 = addi %13, %114 : index
          %116 = addi %6, %115 : index
          %c4_11 = constant 4 : index
          %117 = divi_signed %arg10, %c4_11 : index
          %118 = load %0[%c0, %116, %66, %117] : memref<1x225x225x4xvector<4xf32>>
          %119 = vector.bitcast %118 : vector<4xf32> to vector<4xf32>
          %120 = vector.shape_cast %119 : vector<4xf32> to vector<4xf32>
          %121 = vector.extract %120[0] : vector<4xf32>
          %122 = vector.broadcast %121 : f32 to vector<4xf32>
          %123 = vector.shape_cast %arg13 : vector<1x4xf32> to vector<4xf32>
          %124 = mulf %122, %47 : vector<4xf32>
          %125 = addf %124, %123 : vector<4xf32>
          %126 = vector.extract %120[1] : vector<4xf32>
          %127 = vector.broadcast %126 : f32 to vector<4xf32>
          %128 = mulf %127, %52 : vector<4xf32>
          %129 = addf %128, %125 : vector<4xf32>
          %130 = vector.extract %120[2] : vector<4xf32>
          %131 = vector.broadcast %130 : f32 to vector<4xf32>
          %132 = mulf %131, %57 : vector<4xf32>
          %133 = addf %132, %129 : vector<4xf32>
          %134 = vector.extract %120[3] : vector<4xf32>
          %135 = vector.broadcast %134 : f32 to vector<4xf32>
          %136 = mulf %135, %62 : vector<4xf32>
          %137 = addf %136, %133 : vector<4xf32>
          %138 = vector.shape_cast %137 : vector<4xf32> to vector<1x4xf32>
          %139 = addi %arg0, %c6 : index
          %140 = addi %13, %139 : index
          %141 = addi %6, %140 : index
          %c4_12 = constant 4 : index
          %142 = divi_signed %arg10, %c4_12 : index
          %143 = load %0[%c0, %141, %66, %142] : memref<1x225x225x4xvector<4xf32>>
          %144 = vector.bitcast %143 : vector<4xf32> to vector<4xf32>
          %145 = vector.shape_cast %144 : vector<4xf32> to vector<4xf32>
          %146 = vector.extract %145[0] : vector<4xf32>
          %147 = vector.broadcast %146 : f32 to vector<4xf32>
          %148 = vector.shape_cast %arg14 : vector<1x4xf32> to vector<4xf32>
          %149 = mulf %147, %47 : vector<4xf32>
          %150 = addf %149, %148 : vector<4xf32>
          %151 = vector.extract %145[1] : vector<4xf32>
          %152 = vector.broadcast %151 : f32 to vector<4xf32>
          %153 = mulf %152, %52 : vector<4xf32>
          %154 = addf %153, %150 : vector<4xf32>
          %155 = vector.extract %145[2] : vector<4xf32>
          %156 = vector.broadcast %155 : f32 to vector<4xf32>
          %157 = mulf %156, %57 : vector<4xf32>
          %158 = addf %157, %154 : vector<4xf32>
          %159 = vector.extract %145[3] : vector<4xf32>
          %160 = vector.broadcast %159 : f32 to vector<4xf32>
          %161 = mulf %160, %62 : vector<4xf32>
          %162 = addf %161, %158 : vector<4xf32>
          %163 = vector.shape_cast %162 : vector<4xf32> to vector<1x4xf32>
          scf.yield %88, %113, %138, %163 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
        }
        scf.yield %42#0, %42#1, %42#2, %42#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
      }
      scf.yield %41#0, %41#1, %41#2, %41#3 : vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>
    }
    %16 = vector.shape_cast %15#3 : vector<1x4xf32> to vector<4xf32>
    %17 = addi %11, %c3 : index
    %18 = addi %3, %17 : index
    %19 = addi %4, %9 : index
    %20 = addi %5, %12 : index
    %c4_1 = constant 4 : index
    %21 = divi_signed %20, %c4_1 : index
    %22 = vector.shape_cast %16 : vector<4xf32> to vector<4xf32>
    %23 = vector.bitcast %22 : vector<4xf32> to vector<4xf32>
    store %23, %2[%c0, %18, %19, %21] : memref<1x112x112x8xvector<4xf32>>
    %24 = vector.shape_cast %15#2 : vector<1x4xf32> to vector<4xf32>
    %25 = addi %11, %c2 : index
    %26 = addi %3, %25 : index
    %c4_2 = constant 4 : index
    %27 = divi_signed %20, %c4_2 : index
    %28 = vector.shape_cast %24 : vector<4xf32> to vector<4xf32>
    %29 = vector.bitcast %28 : vector<4xf32> to vector<4xf32>
    store %29, %2[%c0, %26, %19, %27] : memref<1x112x112x8xvector<4xf32>>
    %30 = vector.shape_cast %15#1 : vector<1x4xf32> to vector<4xf32>
    %31 = addi %11, %c1 : index
    %32 = addi %3, %31 : index
    %c4_3 = constant 4 : index
    %33 = divi_signed %20, %c4_3 : index
    %34 = vector.shape_cast %30 : vector<4xf32> to vector<4xf32>
    %35 = vector.bitcast %34 : vector<4xf32> to vector<4xf32>
    store %35, %2[%c0, %32, %19, %33] : memref<1x112x112x8xvector<4xf32>>
    %36 = vector.shape_cast %15#0 : vector<1x4xf32> to vector<4xf32>
    %37 = addi %3, %11 : index
    %c4_4 = constant 4 : index
    %38 = divi_signed %20, %c4_4 : index
    %39 = vector.shape_cast %36 : vector<4xf32> to vector<4xf32>
    %40 = vector.bitcast %39 : vector<4xf32> to vector<4xf32>
    store %40, %2[%c0, %37, %19, %38] : memref<1x112x112x8xvector<4xf32>>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::ForOpCanonicalizationPass ***
func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
  %c16 = constant 16 : index
  %c6 = constant 6 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c8 = constant 8 : index
  %c2 = constant 2 : index
  %cst = constant dense<0.000000e+00> : vector<1x4xf32>
  %c4 = constant 4 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x4xvector<4xf32>>
  %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x8xvector<4xf32>>
  %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x8xvector<4xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %3 = muli %workgroup_id_z, %c4 : index
  %4 = muli %workgroup_id_y, %c4 : index
  %5 = muli %workgroup_id_x, %c16 : index
  %6 = muli %3, %c2 : index
  %7 = muli %4, %c2 : index
  %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %9 = "gpu.thread_id"() {dimension = "y"} : () -> index
  %10 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %11 = muli %10, %c4 : index
  %12 = muli %8, %c4 : index
  %13 = muli %10, %c8 : index
  %14 = muli %9, %c2 : index
  %15 = vector.shape_cast %cst : vector<1x4xf32> to vector<4xf32>
  %16 = vector.shape_cast %cst : vector<1x4xf32> to vector<4xf32>
  %17 = vector.shape_cast %cst : vector<1x4xf32> to vector<4xf32>
  %18 = vector.shape_cast %cst : vector<1x4xf32> to vector<4xf32>
  %19:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %15, %arg2 = %16, %arg3 = %17, %arg4 = %18) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
    %33:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
      %34:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
        %35 = addi %5, %12 : index
        %36 = divi_signed %35, %c4 : index
        %37 = load %1[%arg0, %arg5, %arg10, %36] : memref<3x3x16x8xvector<4xf32>>
        %38 = addi %arg10, %c1 : index
        %39 = divi_signed %35, %c4 : index
        %40 = load %1[%arg0, %arg5, %38, %39] : memref<3x3x16x8xvector<4xf32>>
        %41 = addi %arg10, %c2 : index
        %42 = divi_signed %35, %c4 : index
        %43 = load %1[%arg0, %arg5, %41, %42] : memref<3x3x16x8xvector<4xf32>>
        %44 = addi %arg10, %c3 : index
        %45 = divi_signed %35, %c4 : index
        %46 = load %1[%arg0, %arg5, %44, %45] : memref<3x3x16x8xvector<4xf32>>
        %47 = addi %13, %arg0 : index
        %48 = addi %14, %arg5 : index
        %49 = addi %6, %47 : index
        %50 = addi %7, %48 : index
        %51 = divi_signed %arg10, %c4 : index
        %52 = load %0[%c0, %49, %50, %51] : memref<1x225x225x4xvector<4xf32>>
        %53 = vector.extract %52[0] : vector<4xf32>
        %54 = vector.broadcast %53 : f32 to vector<4xf32>
        %55 = mulf %54, %37 : vector<4xf32>
        %56 = addf %55, %arg11 : vector<4xf32>
        %57 = vector.extract %52[1] : vector<4xf32>
        %58 = vector.broadcast %57 : f32 to vector<4xf32>
        %59 = mulf %58, %40 : vector<4xf32>
        %60 = addf %59, %56 : vector<4xf32>
        %61 = vector.extract %52[2] : vector<4xf32>
        %62 = vector.broadcast %61 : f32 to vector<4xf32>
        %63 = mulf %62, %43 : vector<4xf32>
        %64 = addf %63, %60 : vector<4xf32>
        %65 = vector.extract %52[3] : vector<4xf32>
        %66 = vector.broadcast %65 : f32 to vector<4xf32>
        %67 = mulf %66, %46 : vector<4xf32>
        %68 = addf %67, %64 : vector<4xf32>
        %69 = addi %arg0, %c2 : index
        %70 = addi %13, %69 : index
        %71 = addi %6, %70 : index
        %72 = divi_signed %arg10, %c4 : index
        %73 = load %0[%c0, %71, %50, %72] : memref<1x225x225x4xvector<4xf32>>
        %74 = vector.extract %73[0] : vector<4xf32>
        %75 = vector.broadcast %74 : f32 to vector<4xf32>
        %76 = mulf %75, %37 : vector<4xf32>
        %77 = addf %76, %arg12 : vector<4xf32>
        %78 = vector.extract %73[1] : vector<4xf32>
        %79 = vector.broadcast %78 : f32 to vector<4xf32>
        %80 = mulf %79, %40 : vector<4xf32>
        %81 = addf %80, %77 : vector<4xf32>
        %82 = vector.extract %73[2] : vector<4xf32>
        %83 = vector.broadcast %82 : f32 to vector<4xf32>
        %84 = mulf %83, %43 : vector<4xf32>
        %85 = addf %84, %81 : vector<4xf32>
        %86 = vector.extract %73[3] : vector<4xf32>
        %87 = vector.broadcast %86 : f32 to vector<4xf32>
        %88 = mulf %87, %46 : vector<4xf32>
        %89 = addf %88, %85 : vector<4xf32>
        %90 = addi %arg0, %c4 : index
        %91 = addi %13, %90 : index
        %92 = addi %6, %91 : index
        %93 = divi_signed %arg10, %c4 : index
        %94 = load %0[%c0, %92, %50, %93] : memref<1x225x225x4xvector<4xf32>>
        %95 = vector.extract %94[0] : vector<4xf32>
        %96 = vector.broadcast %95 : f32 to vector<4xf32>
        %97 = mulf %96, %37 : vector<4xf32>
        %98 = addf %97, %arg13 : vector<4xf32>
        %99 = vector.extract %94[1] : vector<4xf32>
        %100 = vector.broadcast %99 : f32 to vector<4xf32>
        %101 = mulf %100, %40 : vector<4xf32>
        %102 = addf %101, %98 : vector<4xf32>
        %103 = vector.extract %94[2] : vector<4xf32>
        %104 = vector.broadcast %103 : f32 to vector<4xf32>
        %105 = mulf %104, %43 : vector<4xf32>
        %106 = addf %105, %102 : vector<4xf32>
        %107 = vector.extract %94[3] : vector<4xf32>
        %108 = vector.broadcast %107 : f32 to vector<4xf32>
        %109 = mulf %108, %46 : vector<4xf32>
        %110 = addf %109, %106 : vector<4xf32>
        %111 = addi %arg0, %c6 : index
        %112 = addi %13, %111 : index
        %113 = addi %6, %112 : index
        %114 = divi_signed %arg10, %c4 : index
        %115 = load %0[%c0, %113, %50, %114] : memref<1x225x225x4xvector<4xf32>>
        %116 = vector.extract %115[0] : vector<4xf32>
        %117 = vector.broadcast %116 : f32 to vector<4xf32>
        %118 = mulf %117, %37 : vector<4xf32>
        %119 = addf %118, %arg14 : vector<4xf32>
        %120 = vector.extract %115[1] : vector<4xf32>
        %121 = vector.broadcast %120 : f32 to vector<4xf32>
        %122 = mulf %121, %40 : vector<4xf32>
        %123 = addf %122, %119 : vector<4xf32>
        %124 = vector.extract %115[2] : vector<4xf32>
        %125 = vector.broadcast %124 : f32 to vector<4xf32>
        %126 = mulf %125, %43 : vector<4xf32>
        %127 = addf %126, %123 : vector<4xf32>
        %128 = vector.extract %115[3] : vector<4xf32>
        %129 = vector.broadcast %128 : f32 to vector<4xf32>
        %130 = mulf %129, %46 : vector<4xf32>
        %131 = addf %130, %127 : vector<4xf32>
        scf.yield %68, %89, %110, %131 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
      }
      scf.yield %34#0, %34#1, %34#2, %34#3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
    }
    scf.yield %33#0, %33#1, %33#2, %33#3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
  }
  %20 = addi %11, %c3 : index
  %21 = addi %3, %20 : index
  %22 = addi %4, %9 : index
  %23 = addi %5, %12 : index
  %24 = divi_signed %23, %c4 : index
  store %19#3, %2[%c0, %21, %22, %24] : memref<1x112x112x8xvector<4xf32>>
  %25 = addi %11, %c2 : index
  %26 = addi %3, %25 : index
  %27 = divi_signed %23, %c4 : index
  store %19#2, %2[%c0, %26, %22, %27] : memref<1x112x112x8xvector<4xf32>>
  %28 = addi %11, %c1 : index
  %29 = addi %3, %28 : index
  %30 = divi_signed %23, %c4 : index
  store %19#1, %2[%c0, %29, %22, %30] : memref<1x112x112x8xvector<4xf32>>
  %31 = addi %3, %11 : index
  %32 = divi_signed %23, %c4 : index
  store %19#0, %2[%c0, %31, %22, %32] : memref<1x112x112x8xvector<4xf32>>
  return
}

// *** IR Dump After Canonicalizer ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    %c8 = constant 8 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %cst = constant dense<0.000000e+00> : vector<4xf32>
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x4xvector<4xf32>>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x8xvector<4xf32>>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x8xvector<4xf32>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = muli %3, %c2 : index
    %7 = muli %4, %c2 : index
    %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %9 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %10 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %11 = muli %10, %c4 : index
    %12 = muli %8, %c4 : index
    %13 = muli %10, %c8 : index
    %14 = muli %9, %c2 : index
    %15:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %cst, %arg2 = %cst, %arg3 = %cst, %arg4 = %cst) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
      %29:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
        %30:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %31 = addi %5, %12 : index
          %32 = divi_signed %31, %c4 : index
          %33 = load %1[%arg0, %arg5, %arg10, %32] : memref<3x3x16x8xvector<4xf32>>
          %34 = addi %arg10, %c1 : index
          %35 = divi_signed %31, %c4 : index
          %36 = load %1[%arg0, %arg5, %34, %35] : memref<3x3x16x8xvector<4xf32>>
          %37 = addi %arg10, %c2 : index
          %38 = divi_signed %31, %c4 : index
          %39 = load %1[%arg0, %arg5, %37, %38] : memref<3x3x16x8xvector<4xf32>>
          %40 = addi %arg10, %c3 : index
          %41 = divi_signed %31, %c4 : index
          %42 = load %1[%arg0, %arg5, %40, %41] : memref<3x3x16x8xvector<4xf32>>
          %43 = addi %13, %arg0 : index
          %44 = addi %14, %arg5 : index
          %45 = addi %6, %43 : index
          %46 = addi %7, %44 : index
          %47 = divi_signed %arg10, %c4 : index
          %48 = load %0[%c0, %45, %46, %47] : memref<1x225x225x4xvector<4xf32>>
          %49 = vector.extract %48[0] : vector<4xf32>
          %50 = vector.broadcast %49 : f32 to vector<4xf32>
          %51 = mulf %50, %33 : vector<4xf32>
          %52 = addf %51, %arg11 : vector<4xf32>
          %53 = vector.extract %48[1] : vector<4xf32>
          %54 = vector.broadcast %53 : f32 to vector<4xf32>
          %55 = mulf %54, %36 : vector<4xf32>
          %56 = addf %55, %52 : vector<4xf32>
          %57 = vector.extract %48[2] : vector<4xf32>
          %58 = vector.broadcast %57 : f32 to vector<4xf32>
          %59 = mulf %58, %39 : vector<4xf32>
          %60 = addf %59, %56 : vector<4xf32>
          %61 = vector.extract %48[3] : vector<4xf32>
          %62 = vector.broadcast %61 : f32 to vector<4xf32>
          %63 = mulf %62, %42 : vector<4xf32>
          %64 = addf %63, %60 : vector<4xf32>
          %65 = addi %arg0, %c2 : index
          %66 = addi %13, %65 : index
          %67 = addi %6, %66 : index
          %68 = divi_signed %arg10, %c4 : index
          %69 = load %0[%c0, %67, %46, %68] : memref<1x225x225x4xvector<4xf32>>
          %70 = vector.extract %69[0] : vector<4xf32>
          %71 = vector.broadcast %70 : f32 to vector<4xf32>
          %72 = mulf %71, %33 : vector<4xf32>
          %73 = addf %72, %arg12 : vector<4xf32>
          %74 = vector.extract %69[1] : vector<4xf32>
          %75 = vector.broadcast %74 : f32 to vector<4xf32>
          %76 = mulf %75, %36 : vector<4xf32>
          %77 = addf %76, %73 : vector<4xf32>
          %78 = vector.extract %69[2] : vector<4xf32>
          %79 = vector.broadcast %78 : f32 to vector<4xf32>
          %80 = mulf %79, %39 : vector<4xf32>
          %81 = addf %80, %77 : vector<4xf32>
          %82 = vector.extract %69[3] : vector<4xf32>
          %83 = vector.broadcast %82 : f32 to vector<4xf32>
          %84 = mulf %83, %42 : vector<4xf32>
          %85 = addf %84, %81 : vector<4xf32>
          %86 = addi %arg0, %c4 : index
          %87 = addi %13, %86 : index
          %88 = addi %6, %87 : index
          %89 = divi_signed %arg10, %c4 : index
          %90 = load %0[%c0, %88, %46, %89] : memref<1x225x225x4xvector<4xf32>>
          %91 = vector.extract %90[0] : vector<4xf32>
          %92 = vector.broadcast %91 : f32 to vector<4xf32>
          %93 = mulf %92, %33 : vector<4xf32>
          %94 = addf %93, %arg13 : vector<4xf32>
          %95 = vector.extract %90[1] : vector<4xf32>
          %96 = vector.broadcast %95 : f32 to vector<4xf32>
          %97 = mulf %96, %36 : vector<4xf32>
          %98 = addf %97, %94 : vector<4xf32>
          %99 = vector.extract %90[2] : vector<4xf32>
          %100 = vector.broadcast %99 : f32 to vector<4xf32>
          %101 = mulf %100, %39 : vector<4xf32>
          %102 = addf %101, %98 : vector<4xf32>
          %103 = vector.extract %90[3] : vector<4xf32>
          %104 = vector.broadcast %103 : f32 to vector<4xf32>
          %105 = mulf %104, %42 : vector<4xf32>
          %106 = addf %105, %102 : vector<4xf32>
          %107 = addi %arg0, %c6 : index
          %108 = addi %13, %107 : index
          %109 = addi %6, %108 : index
          %110 = divi_signed %arg10, %c4 : index
          %111 = load %0[%c0, %109, %46, %110] : memref<1x225x225x4xvector<4xf32>>
          %112 = vector.extract %111[0] : vector<4xf32>
          %113 = vector.broadcast %112 : f32 to vector<4xf32>
          %114 = mulf %113, %33 : vector<4xf32>
          %115 = addf %114, %arg14 : vector<4xf32>
          %116 = vector.extract %111[1] : vector<4xf32>
          %117 = vector.broadcast %116 : f32 to vector<4xf32>
          %118 = mulf %117, %36 : vector<4xf32>
          %119 = addf %118, %115 : vector<4xf32>
          %120 = vector.extract %111[2] : vector<4xf32>
          %121 = vector.broadcast %120 : f32 to vector<4xf32>
          %122 = mulf %121, %39 : vector<4xf32>
          %123 = addf %122, %119 : vector<4xf32>
          %124 = vector.extract %111[3] : vector<4xf32>
          %125 = vector.broadcast %124 : f32 to vector<4xf32>
          %126 = mulf %125, %42 : vector<4xf32>
          %127 = addf %126, %123 : vector<4xf32>
          scf.yield %64, %85, %106, %127 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        scf.yield %30#0, %30#1, %30#2, %30#3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
      }
      scf.yield %29#0, %29#1, %29#2, %29#3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
    }
    %16 = addi %11, %c3 : index
    %17 = addi %3, %16 : index
    %18 = addi %4, %9 : index
    %19 = addi %5, %12 : index
    %20 = divi_signed %19, %c4 : index
    store %15#3, %2[%c0, %17, %18, %20] : memref<1x112x112x8xvector<4xf32>>
    %21 = addi %11, %c2 : index
    %22 = addi %3, %21 : index
    %23 = divi_signed %19, %c4 : index
    store %15#2, %2[%c0, %22, %18, %23] : memref<1x112x112x8xvector<4xf32>>
    %24 = addi %11, %c1 : index
    %25 = addi %3, %24 : index
    %26 = divi_signed %19, %c4 : index
    store %15#1, %2[%c0, %25, %18, %26] : memref<1x112x112x8xvector<4xf32>>
    %27 = addi %3, %11 : index
    %28 = divi_signed %19, %c4 : index
    store %15#0, %2[%c0, %27, %18, %28] : memref<1x112x112x8xvector<4xf32>>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After CSE ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  func @predict_ex_dispatch_1_dispatch_0() attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
    %c16 = constant 16 : index
    %c6 = constant 6 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c3 = constant 3 : index
    %c8 = constant 8 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %cst = constant dense<0.000000e+00> : vector<4xf32>
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x4xvector<4xf32>>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x8xvector<4xf32>>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x8xvector<4xf32>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %3 = muli %workgroup_id_z, %c4 : index
    %4 = muli %workgroup_id_y, %c4 : index
    %5 = muli %workgroup_id_x, %c16 : index
    %6 = muli %3, %c2 : index
    %7 = muli %4, %c2 : index
    %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
    %9 = "gpu.thread_id"() {dimension = "y"} : () -> index
    %10 = "gpu.thread_id"() {dimension = "z"} : () -> index
    %11 = muli %10, %c4 : index
    %12 = muli %8, %c4 : index
    %13 = muli %10, %c8 : index
    %14 = muli %9, %c2 : index
    %15:4 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %cst, %arg2 = %cst, %arg3 = %cst, %arg4 = %cst) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
      %26:4 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg1, %arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg4) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
        %27:4 = scf.for %arg10 = %c0 to %c16 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %28 = addi %5, %12 : index
          %29 = divi_signed %28, %c4 : index
          %30 = load %1[%arg0, %arg5, %arg10, %29] : memref<3x3x16x8xvector<4xf32>>
          %31 = addi %arg10, %c1 : index
          %32 = load %1[%arg0, %arg5, %31, %29] : memref<3x3x16x8xvector<4xf32>>
          %33 = addi %arg10, %c2 : index
          %34 = load %1[%arg0, %arg5, %33, %29] : memref<3x3x16x8xvector<4xf32>>
          %35 = addi %arg10, %c3 : index
          %36 = load %1[%arg0, %arg5, %35, %29] : memref<3x3x16x8xvector<4xf32>>
          %37 = addi %13, %arg0 : index
          %38 = addi %14, %arg5 : index
          %39 = addi %6, %37 : index
          %40 = addi %7, %38 : index
          %41 = divi_signed %arg10, %c4 : index
          %42 = load %0[%c0, %39, %40, %41] : memref<1x225x225x4xvector<4xf32>>
          %43 = vector.extract %42[0] : vector<4xf32>
          %44 = vector.broadcast %43 : f32 to vector<4xf32>
          %45 = mulf %44, %30 : vector<4xf32>
          %46 = addf %45, %arg11 : vector<4xf32>
          %47 = vector.extract %42[1] : vector<4xf32>
          %48 = vector.broadcast %47 : f32 to vector<4xf32>
          %49 = mulf %48, %32 : vector<4xf32>
          %50 = addf %49, %46 : vector<4xf32>
          %51 = vector.extract %42[2] : vector<4xf32>
          %52 = vector.broadcast %51 : f32 to vector<4xf32>
          %53 = mulf %52, %34 : vector<4xf32>
          %54 = addf %53, %50 : vector<4xf32>
          %55 = vector.extract %42[3] : vector<4xf32>
          %56 = vector.broadcast %55 : f32 to vector<4xf32>
          %57 = mulf %56, %36 : vector<4xf32>
          %58 = addf %57, %54 : vector<4xf32>
          %59 = addi %arg0, %c2 : index
          %60 = addi %13, %59 : index
          %61 = addi %6, %60 : index
          %62 = load %0[%c0, %61, %40, %41] : memref<1x225x225x4xvector<4xf32>>
          %63 = vector.extract %62[0] : vector<4xf32>
          %64 = vector.broadcast %63 : f32 to vector<4xf32>
          %65 = mulf %64, %30 : vector<4xf32>
          %66 = addf %65, %arg12 : vector<4xf32>
          %67 = vector.extract %62[1] : vector<4xf32>
          %68 = vector.broadcast %67 : f32 to vector<4xf32>
          %69 = mulf %68, %32 : vector<4xf32>
          %70 = addf %69, %66 : vector<4xf32>
          %71 = vector.extract %62[2] : vector<4xf32>
          %72 = vector.broadcast %71 : f32 to vector<4xf32>
          %73 = mulf %72, %34 : vector<4xf32>
          %74 = addf %73, %70 : vector<4xf32>
          %75 = vector.extract %62[3] : vector<4xf32>
          %76 = vector.broadcast %75 : f32 to vector<4xf32>
          %77 = mulf %76, %36 : vector<4xf32>
          %78 = addf %77, %74 : vector<4xf32>
          %79 = addi %arg0, %c4 : index
          %80 = addi %13, %79 : index
          %81 = addi %6, %80 : index
          %82 = load %0[%c0, %81, %40, %41] : memref<1x225x225x4xvector<4xf32>>
          %83 = vector.extract %82[0] : vector<4xf32>
          %84 = vector.broadcast %83 : f32 to vector<4xf32>
          %85 = mulf %84, %30 : vector<4xf32>
          %86 = addf %85, %arg13 : vector<4xf32>
          %87 = vector.extract %82[1] : vector<4xf32>
          %88 = vector.broadcast %87 : f32 to vector<4xf32>
          %89 = mulf %88, %32 : vector<4xf32>
          %90 = addf %89, %86 : vector<4xf32>
          %91 = vector.extract %82[2] : vector<4xf32>
          %92 = vector.broadcast %91 : f32 to vector<4xf32>
          %93 = mulf %92, %34 : vector<4xf32>
          %94 = addf %93, %90 : vector<4xf32>
          %95 = vector.extract %82[3] : vector<4xf32>
          %96 = vector.broadcast %95 : f32 to vector<4xf32>
          %97 = mulf %96, %36 : vector<4xf32>
          %98 = addf %97, %94 : vector<4xf32>
          %99 = addi %arg0, %c6 : index
          %100 = addi %13, %99 : index
          %101 = addi %6, %100 : index
          %102 = load %0[%c0, %101, %40, %41] : memref<1x225x225x4xvector<4xf32>>
          %103 = vector.extract %102[0] : vector<4xf32>
          %104 = vector.broadcast %103 : f32 to vector<4xf32>
          %105 = mulf %104, %30 : vector<4xf32>
          %106 = addf %105, %arg14 : vector<4xf32>
          %107 = vector.extract %102[1] : vector<4xf32>
          %108 = vector.broadcast %107 : f32 to vector<4xf32>
          %109 = mulf %108, %32 : vector<4xf32>
          %110 = addf %109, %106 : vector<4xf32>
          %111 = vector.extract %102[2] : vector<4xf32>
          %112 = vector.broadcast %111 : f32 to vector<4xf32>
          %113 = mulf %112, %34 : vector<4xf32>
          %114 = addf %113, %110 : vector<4xf32>
          %115 = vector.extract %102[3] : vector<4xf32>
          %116 = vector.broadcast %115 : f32 to vector<4xf32>
          %117 = mulf %116, %36 : vector<4xf32>
          %118 = addf %117, %114 : vector<4xf32>
          scf.yield %58, %78, %98, %118 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        scf.yield %27#0, %27#1, %27#2, %27#3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
      }
      scf.yield %26#0, %26#1, %26#2, %26#3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
    }
    %16 = addi %11, %c3 : index
    %17 = addi %3, %16 : index
    %18 = addi %4, %9 : index
    %19 = addi %5, %12 : index
    %20 = divi_signed %19, %c4 : index
    store %15#3, %2[%c0, %17, %18, %20] : memref<1x112x112x8xvector<4xf32>>
    %21 = addi %11, %c2 : index
    %22 = addi %3, %21 : index
    store %15#2, %2[%c0, %22, %18, %20] : memref<1x112x112x8xvector<4xf32>>
    %23 = addi %11, %c1 : index
    %24 = addi %3, %23 : index
    store %15#1, %2[%c0, %24, %18, %20] : memref<1x112x112x8xvector<4xf32>>
    %25 = addi %3, %11 : index
    store %15#0, %2[%c0, %25, %18, %20] : memref<1x112x112x8xvector<4xf32>>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After mlir::iree_compiler::(anonymous namespace)::ConvertToSPIRVPass ***
module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
  spv.module Logical GLSL450 {
    spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    spv.func @predict_ex_dispatch_1_dispatch_0() "None" attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
      %0 = spv.constant 16 : i32
      %1 = spv.constant 6 : i32
      %2 = spv.constant 0 : i32
      %3 = spv.constant 1 : i32
      %4 = spv.constant 3 : i32
      %5 = spv.constant 8 : i32
      %6 = spv.constant 2 : i32
      %7 = spv.constant 4 : i32
      %8 = spv.constant dense<0.000000e+00> : vector<4xf32>
      %9 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      %10 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      %11 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      %12 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
      %13 = spv.Load "Input" %12 : vector<3xi32>
      %14 = spv.CompositeExtract %13[0 : i32] : vector<3xi32>
      %15 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
      %16 = spv.Load "Input" %15 : vector<3xi32>
      %17 = spv.CompositeExtract %16[1 : i32] : vector<3xi32>
      %18 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
      %19 = spv.Load "Input" %18 : vector<3xi32>
      %20 = spv.CompositeExtract %19[2 : i32] : vector<3xi32>
      %21 = spv.IMul %20, %7 : i32
      %22 = spv.IMul %17, %7 : i32
      %23 = spv.IMul %14, %0 : i32
      %24 = spv.IMul %21, %6 : i32
      %25 = spv.IMul %22, %6 : i32
      %26 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
      %27 = spv.Load "Input" %26 : vector<3xi32>
      %28 = spv.CompositeExtract %27[0 : i32] : vector<3xi32>
      %29 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
      %30 = spv.Load "Input" %29 : vector<3xi32>
      %31 = spv.CompositeExtract %30[1 : i32] : vector<3xi32>
      %32 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
      %33 = spv.Load "Input" %32 : vector<3xi32>
      %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
      %35 = spv.IMul %34, %7 : i32
      %36 = spv.IMul %28, %7 : i32
      %37 = spv.IMul %34, %5 : i32
      %38 = spv.IMul %31, %6 : i32
      %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      spv.loop {
        spv.Branch ^bb1(%2, %8, %8, %8, %8 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb1(%117: i32, %118: vector<4xf32>, %119: vector<4xf32>, %120: vector<4xf32>, %121: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
        %122 = spv.SLessThan %117, %4 : i32
        spv.BranchConditional %122, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %123 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %124 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %125 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %126 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        spv.loop {
          spv.Branch ^bb1(%2, %118, %119, %120, %121 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb1(%132: i32, %133: vector<4xf32>, %134: vector<4xf32>, %135: vector<4xf32>, %136: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
          %137 = spv.SLessThan %132, %4 : i32
          spv.BranchConditional %137, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %138 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
          %139 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
          %140 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
          %141 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
          spv.loop {
            spv.Branch ^bb1(%2, %133, %134, %135, %136 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
          ^bb1(%147: i32, %148: vector<4xf32>, %149: vector<4xf32>, %150: vector<4xf32>, %151: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
            %152 = spv.SLessThan %147, %0 : i32
            spv.BranchConditional %152, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %153 = spv.IAdd %23, %36 : i32
            %154 = spv.SDiv %153, %7 : i32
            %155 = spv.constant 0 : i32
            %156 = spv.constant 0 : i32
            %157 = spv.constant 384 : i32
            %158 = spv.IMul %157, %117 : i32
            %159 = spv.IAdd %156, %158 : i32
            %160 = spv.constant 128 : i32
            %161 = spv.IMul %160, %132 : i32
            %162 = spv.IAdd %159, %161 : i32
            %163 = spv.constant 8 : i32
            %164 = spv.IMul %163, %147 : i32
            %165 = spv.IAdd %162, %164 : i32
            %166 = spv.constant 1 : i32
            %167 = spv.IMul %166, %154 : i32
            %168 = spv.IAdd %165, %167 : i32
            %169 = spv.AccessChain %10[%155, %168] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            %170 = spv.Load "StorageBuffer" %169 : vector<4xf32>
            %171 = spv.IAdd %147, %3 : i32
            %172 = spv.constant 0 : i32
            %173 = spv.constant 0 : i32
            %174 = spv.constant 384 : i32
            %175 = spv.IMul %174, %117 : i32
            %176 = spv.IAdd %173, %175 : i32
            %177 = spv.constant 128 : i32
            %178 = spv.IMul %177, %132 : i32
            %179 = spv.IAdd %176, %178 : i32
            %180 = spv.constant 8 : i32
            %181 = spv.IMul %180, %171 : i32
            %182 = spv.IAdd %179, %181 : i32
            %183 = spv.constant 1 : i32
            %184 = spv.IMul %183, %154 : i32
            %185 = spv.IAdd %182, %184 : i32
            %186 = spv.AccessChain %10[%172, %185] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            %187 = spv.Load "StorageBuffer" %186 : vector<4xf32>
            %188 = spv.IAdd %147, %6 : i32
            %189 = spv.constant 0 : i32
            %190 = spv.constant 0 : i32
            %191 = spv.constant 384 : i32
            %192 = spv.IMul %191, %117 : i32
            %193 = spv.IAdd %190, %192 : i32
            %194 = spv.constant 128 : i32
            %195 = spv.IMul %194, %132 : i32
            %196 = spv.IAdd %193, %195 : i32
            %197 = spv.constant 8 : i32
            %198 = spv.IMul %197, %188 : i32
            %199 = spv.IAdd %196, %198 : i32
            %200 = spv.constant 1 : i32
            %201 = spv.IMul %200, %154 : i32
            %202 = spv.IAdd %199, %201 : i32
            %203 = spv.AccessChain %10[%189, %202] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            %204 = spv.Load "StorageBuffer" %203 : vector<4xf32>
            %205 = spv.IAdd %147, %4 : i32
            %206 = spv.constant 0 : i32
            %207 = spv.constant 0 : i32
            %208 = spv.constant 384 : i32
            %209 = spv.IMul %208, %117 : i32
            %210 = spv.IAdd %207, %209 : i32
            %211 = spv.constant 128 : i32
            %212 = spv.IMul %211, %132 : i32
            %213 = spv.IAdd %210, %212 : i32
            %214 = spv.constant 8 : i32
            %215 = spv.IMul %214, %205 : i32
            %216 = spv.IAdd %213, %215 : i32
            %217 = spv.constant 1 : i32
            %218 = spv.IMul %217, %154 : i32
            %219 = spv.IAdd %216, %218 : i32
            %220 = spv.AccessChain %10[%206, %219] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            %221 = spv.Load "StorageBuffer" %220 : vector<4xf32>
            %222 = spv.IAdd %37, %117 : i32
            %223 = spv.IAdd %38, %132 : i32
            %224 = spv.IAdd %24, %222 : i32
            %225 = spv.IAdd %25, %223 : i32
            %226 = spv.SDiv %147, %7 : i32
            %227 = spv.constant 0 : i32
            %228 = spv.constant 0 : i32
            %229 = spv.constant 202500 : i32
            %230 = spv.IMul %229, %2 : i32
            %231 = spv.IAdd %228, %230 : i32
            %232 = spv.constant 900 : i32
            %233 = spv.IMul %232, %224 : i32
            %234 = spv.IAdd %231, %233 : i32
            %235 = spv.constant 4 : i32
            %236 = spv.IMul %235, %225 : i32
            %237 = spv.IAdd %234, %236 : i32
            %238 = spv.constant 1 : i32
            %239 = spv.IMul %238, %226 : i32
            %240 = spv.IAdd %237, %239 : i32
            %241 = spv.AccessChain %9[%227, %240] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            %242 = spv.Load "StorageBuffer" %241 : vector<4xf32>
            %243 = spv.CompositeExtract %242[0 : i32] : vector<4xf32>
            %244 = spv.CompositeConstruct %243, %243, %243, %243 : vector<4xf32>
            %245 = spv.FMul %244, %170 : vector<4xf32>
            %246 = spv.FAdd %245, %148 : vector<4xf32>
            %247 = spv.CompositeExtract %242[1 : i32] : vector<4xf32>
            %248 = spv.CompositeConstruct %247, %247, %247, %247 : vector<4xf32>
            %249 = spv.FMul %248, %187 : vector<4xf32>
            %250 = spv.FAdd %249, %246 : vector<4xf32>
            %251 = spv.CompositeExtract %242[2 : i32] : vector<4xf32>
            %252 = spv.CompositeConstruct %251, %251, %251, %251 : vector<4xf32>
            %253 = spv.FMul %252, %204 : vector<4xf32>
            %254 = spv.FAdd %253, %250 : vector<4xf32>
            %255 = spv.CompositeExtract %242[3 : i32] : vector<4xf32>
            %256 = spv.CompositeConstruct %255, %255, %255, %255 : vector<4xf32>
            %257 = spv.FMul %256, %221 : vector<4xf32>
            %258 = spv.FAdd %257, %254 : vector<4xf32>
            %259 = spv.IAdd %117, %6 : i32
            %260 = spv.IAdd %37, %259 : i32
            %261 = spv.IAdd %24, %260 : i32
            %262 = spv.constant 0 : i32
            %263 = spv.constant 0 : i32
            %264 = spv.constant 202500 : i32
            %265 = spv.IMul %264, %2 : i32
            %266 = spv.IAdd %263, %265 : i32
            %267 = spv.constant 900 : i32
            %268 = spv.IMul %267, %261 : i32
            %269 = spv.IAdd %266, %268 : i32
            %270 = spv.constant 4 : i32
            %271 = spv.IMul %270, %225 : i32
            %272 = spv.IAdd %269, %271 : i32
            %273 = spv.constant 1 : i32
            %274 = spv.IMul %273, %226 : i32
            %275 = spv.IAdd %272, %274 : i32
            %276 = spv.AccessChain %9[%262, %275] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            %277 = spv.Load "StorageBuffer" %276 : vector<4xf32>
            %278 = spv.CompositeExtract %277[0 : i32] : vector<4xf32>
            %279 = spv.CompositeConstruct %278, %278, %278, %278 : vector<4xf32>
            %280 = spv.FMul %279, %170 : vector<4xf32>
            %281 = spv.FAdd %280, %149 : vector<4xf32>
            %282 = spv.CompositeExtract %277[1 : i32] : vector<4xf32>
            %283 = spv.CompositeConstruct %282, %282, %282, %282 : vector<4xf32>
            %284 = spv.FMul %283, %187 : vector<4xf32>
            %285 = spv.FAdd %284, %281 : vector<4xf32>
            %286 = spv.CompositeExtract %277[2 : i32] : vector<4xf32>
            %287 = spv.CompositeConstruct %286, %286, %286, %286 : vector<4xf32>
            %288 = spv.FMul %287, %204 : vector<4xf32>
            %289 = spv.FAdd %288, %285 : vector<4xf32>
            %290 = spv.CompositeExtract %277[3 : i32] : vector<4xf32>
            %291 = spv.CompositeConstruct %290, %290, %290, %290 : vector<4xf32>
            %292 = spv.FMul %291, %221 : vector<4xf32>
            %293 = spv.FAdd %292, %289 : vector<4xf32>
            %294 = spv.IAdd %117, %7 : i32
            %295 = spv.IAdd %37, %294 : i32
            %296 = spv.IAdd %24, %295 : i32
            %297 = spv.constant 0 : i32
            %298 = spv.constant 0 : i32
            %299 = spv.constant 202500 : i32
            %300 = spv.IMul %299, %2 : i32
            %301 = spv.IAdd %298, %300 : i32
            %302 = spv.constant 900 : i32
            %303 = spv.IMul %302, %296 : i32
            %304 = spv.IAdd %301, %303 : i32
            %305 = spv.constant 4 : i32
            %306 = spv.IMul %305, %225 : i32
            %307 = spv.IAdd %304, %306 : i32
            %308 = spv.constant 1 : i32
            %309 = spv.IMul %308, %226 : i32
            %310 = spv.IAdd %307, %309 : i32
            %311 = spv.AccessChain %9[%297, %310] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            %312 = spv.Load "StorageBuffer" %311 : vector<4xf32>
            %313 = spv.CompositeExtract %312[0 : i32] : vector<4xf32>
            %314 = spv.CompositeConstruct %313, %313, %313, %313 : vector<4xf32>
            %315 = spv.FMul %314, %170 : vector<4xf32>
            %316 = spv.FAdd %315, %150 : vector<4xf32>
            %317 = spv.CompositeExtract %312[1 : i32] : vector<4xf32>
            %318 = spv.CompositeConstruct %317, %317, %317, %317 : vector<4xf32>
            %319 = spv.FMul %318, %187 : vector<4xf32>
            %320 = spv.FAdd %319, %316 : vector<4xf32>
            %321 = spv.CompositeExtract %312[2 : i32] : vector<4xf32>
            %322 = spv.CompositeConstruct %321, %321, %321, %321 : vector<4xf32>
            %323 = spv.FMul %322, %204 : vector<4xf32>
            %324 = spv.FAdd %323, %320 : vector<4xf32>
            %325 = spv.CompositeExtract %312[3 : i32] : vector<4xf32>
            %326 = spv.CompositeConstruct %325, %325, %325, %325 : vector<4xf32>
            %327 = spv.FMul %326, %221 : vector<4xf32>
            %328 = spv.FAdd %327, %324 : vector<4xf32>
            %329 = spv.IAdd %117, %1 : i32
            %330 = spv.IAdd %37, %329 : i32
            %331 = spv.IAdd %24, %330 : i32
            %332 = spv.constant 0 : i32
            %333 = spv.constant 0 : i32
            %334 = spv.constant 202500 : i32
            %335 = spv.IMul %334, %2 : i32
            %336 = spv.IAdd %333, %335 : i32
            %337 = spv.constant 900 : i32
            %338 = spv.IMul %337, %331 : i32
            %339 = spv.IAdd %336, %338 : i32
            %340 = spv.constant 4 : i32
            %341 = spv.IMul %340, %225 : i32
            %342 = spv.IAdd %339, %341 : i32
            %343 = spv.constant 1 : i32
            %344 = spv.IMul %343, %226 : i32
            %345 = spv.IAdd %342, %344 : i32
            %346 = spv.AccessChain %9[%332, %345] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            %347 = spv.Load "StorageBuffer" %346 : vector<4xf32>
            %348 = spv.CompositeExtract %347[0 : i32] : vector<4xf32>
            %349 = spv.CompositeConstruct %348, %348, %348, %348 : vector<4xf32>
            %350 = spv.FMul %349, %170 : vector<4xf32>
            %351 = spv.FAdd %350, %151 : vector<4xf32>
            %352 = spv.CompositeExtract %347[1 : i32] : vector<4xf32>
            %353 = spv.CompositeConstruct %352, %352, %352, %352 : vector<4xf32>
            %354 = spv.FMul %353, %187 : vector<4xf32>
            %355 = spv.FAdd %354, %351 : vector<4xf32>
            %356 = spv.CompositeExtract %347[2 : i32] : vector<4xf32>
            %357 = spv.CompositeConstruct %356, %356, %356, %356 : vector<4xf32>
            %358 = spv.FMul %357, %204 : vector<4xf32>
            %359 = spv.FAdd %358, %355 : vector<4xf32>
            %360 = spv.CompositeExtract %347[3 : i32] : vector<4xf32>
            %361 = spv.CompositeConstruct %360, %360, %360, %360 : vector<4xf32>
            %362 = spv.FMul %361, %221 : vector<4xf32>
            %363 = spv.FAdd %362, %359 : vector<4xf32>
            spv.Store "Function" %138, %258 : vector<4xf32>
            spv.Store "Function" %139, %293 : vector<4xf32>
            spv.Store "Function" %140, %328 : vector<4xf32>
            spv.Store "Function" %141, %363 : vector<4xf32>
            %364 = spv.IAdd %147, %7 : i32
            spv.Branch ^bb1(%364, %258, %293, %328, %363 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
          ^bb3:  // pred: ^bb1
            spv.mlir.merge
          }
          %142 = spv.Load "Function" %141 : vector<4xf32>
          %143 = spv.Load "Function" %140 : vector<4xf32>
          %144 = spv.Load "Function" %139 : vector<4xf32>
          %145 = spv.Load "Function" %138 : vector<4xf32>
          spv.Store "Function" %123, %145 : vector<4xf32>
          spv.Store "Function" %124, %144 : vector<4xf32>
          spv.Store "Function" %125, %143 : vector<4xf32>
          spv.Store "Function" %126, %142 : vector<4xf32>
          %146 = spv.IAdd %132, %3 : i32
          spv.Branch ^bb1(%146, %145, %144, %143, %142 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb3:  // pred: ^bb1
          spv.mlir.merge
        }
        %127 = spv.Load "Function" %126 : vector<4xf32>
        %128 = spv.Load "Function" %125 : vector<4xf32>
        %129 = spv.Load "Function" %124 : vector<4xf32>
        %130 = spv.Load "Function" %123 : vector<4xf32>
        spv.Store "Function" %39, %130 : vector<4xf32>
        spv.Store "Function" %40, %129 : vector<4xf32>
        spv.Store "Function" %41, %128 : vector<4xf32>
        spv.Store "Function" %42, %127 : vector<4xf32>
        %131 = spv.IAdd %117, %3 : i32
        spv.Branch ^bb1(%131, %130, %129, %128, %127 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb3:  // pred: ^bb1
        spv.mlir.merge
      }
      %43 = spv.Load "Function" %42 : vector<4xf32>
      %44 = spv.Load "Function" %41 : vector<4xf32>
      %45 = spv.Load "Function" %40 : vector<4xf32>
      %46 = spv.Load "Function" %39 : vector<4xf32>
      %47 = spv.IAdd %35, %4 : i32
      %48 = spv.IAdd %21, %47 : i32
      %49 = spv.IAdd %22, %31 : i32
      %50 = spv.IAdd %23, %36 : i32
      %51 = spv.SDiv %50, %7 : i32
      %52 = spv.constant 0 : i32
      %53 = spv.constant 0 : i32
      %54 = spv.constant 100352 : i32
      %55 = spv.IMul %54, %2 : i32
      %56 = spv.IAdd %53, %55 : i32
      %57 = spv.constant 896 : i32
      %58 = spv.IMul %57, %48 : i32
      %59 = spv.IAdd %56, %58 : i32
      %60 = spv.constant 8 : i32
      %61 = spv.IMul %60, %49 : i32
      %62 = spv.IAdd %59, %61 : i32
      %63 = spv.constant 1 : i32
      %64 = spv.IMul %63, %51 : i32
      %65 = spv.IAdd %62, %64 : i32
      %66 = spv.AccessChain %11[%52, %65] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
      spv.Store "StorageBuffer" %66, %43 : vector<4xf32>
      %67 = spv.IAdd %35, %6 : i32
      %68 = spv.IAdd %21, %67 : i32
      %69 = spv.constant 0 : i32
      %70 = spv.constant 0 : i32
      %71 = spv.constant 100352 : i32
      %72 = spv.IMul %71, %2 : i32
      %73 = spv.IAdd %70, %72 : i32
      %74 = spv.constant 896 : i32
      %75 = spv.IMul %74, %68 : i32
      %76 = spv.IAdd %73, %75 : i32
      %77 = spv.constant 8 : i32
      %78 = spv.IMul %77, %49 : i32
      %79 = spv.IAdd %76, %78 : i32
      %80 = spv.constant 1 : i32
      %81 = spv.IMul %80, %51 : i32
      %82 = spv.IAdd %79, %81 : i32
      %83 = spv.AccessChain %11[%69, %82] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
      spv.Store "StorageBuffer" %83, %44 : vector<4xf32>
      %84 = spv.IAdd %35, %3 : i32
      %85 = spv.IAdd %21, %84 : i32
      %86 = spv.constant 0 : i32
      %87 = spv.constant 0 : i32
      %88 = spv.constant 100352 : i32
      %89 = spv.IMul %88, %2 : i32
      %90 = spv.IAdd %87, %89 : i32
      %91 = spv.constant 896 : i32
      %92 = spv.IMul %91, %85 : i32
      %93 = spv.IAdd %90, %92 : i32
      %94 = spv.constant 8 : i32
      %95 = spv.IMul %94, %49 : i32
      %96 = spv.IAdd %93, %95 : i32
      %97 = spv.constant 1 : i32
      %98 = spv.IMul %97, %51 : i32
      %99 = spv.IAdd %96, %98 : i32
      %100 = spv.AccessChain %11[%86, %99] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
      spv.Store "StorageBuffer" %100, %45 : vector<4xf32>
      %101 = spv.IAdd %21, %35 : i32
      %102 = spv.constant 0 : i32
      %103 = spv.constant 0 : i32
      %104 = spv.constant 100352 : i32
      %105 = spv.IMul %104, %2 : i32
      %106 = spv.IAdd %103, %105 : i32
      %107 = spv.constant 896 : i32
      %108 = spv.IMul %107, %101 : i32
      %109 = spv.IAdd %106, %108 : i32
      %110 = spv.constant 8 : i32
      %111 = spv.IMul %110, %49 : i32
      %112 = spv.IAdd %109, %111 : i32
      %113 = spv.constant 1 : i32
      %114 = spv.IMul %113, %51 : i32
      %115 = spv.IAdd %112, %114 : i32
      %116 = spv.AccessChain %11[%102, %115] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
      spv.Store "StorageBuffer" %116, %46 : vector<4xf32>
      spv.Return
    }
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// *** IR Dump After SPIRVLowerABIAttributes ***
spv.module Logical GLSL450 {
  spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
    %0 = spv.constant 16 : i32
    %1 = spv.constant 6 : i32
    %2 = spv.constant 0 : i32
    %3 = spv.constant 1 : i32
    %4 = spv.constant 3 : i32
    %5 = spv.constant 8 : i32
    %6 = spv.constant 2 : i32
    %7 = spv.constant 4 : i32
    %8 = spv.constant dense<0.000000e+00> : vector<4xf32>
    %9 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %10 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %11 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %12 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %13 = spv.Load "Input" %12 : vector<3xi32>
    %14 = spv.CompositeExtract %13[0 : i32] : vector<3xi32>
    %15 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %16 = spv.Load "Input" %15 : vector<3xi32>
    %17 = spv.CompositeExtract %16[1 : i32] : vector<3xi32>
    %18 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %19 = spv.Load "Input" %18 : vector<3xi32>
    %20 = spv.CompositeExtract %19[2 : i32] : vector<3xi32>
    %21 = spv.IMul %20, %7 : i32
    %22 = spv.IMul %17, %7 : i32
    %23 = spv.IMul %14, %0 : i32
    %24 = spv.IMul %21, %6 : i32
    %25 = spv.IMul %22, %6 : i32
    %26 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %27 = spv.Load "Input" %26 : vector<3xi32>
    %28 = spv.CompositeExtract %27[0 : i32] : vector<3xi32>
    %29 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %30 = spv.Load "Input" %29 : vector<3xi32>
    %31 = spv.CompositeExtract %30[1 : i32] : vector<3xi32>
    %32 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %33 = spv.Load "Input" %32 : vector<3xi32>
    %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
    %35 = spv.IMul %34, %7 : i32
    %36 = spv.IMul %28, %7 : i32
    %37 = spv.IMul %34, %5 : i32
    %38 = spv.IMul %31, %6 : i32
    %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    spv.loop {
      spv.Branch ^bb1(%2, %8, %8, %8, %8 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
    ^bb1(%117: i32, %118: vector<4xf32>, %119: vector<4xf32>, %120: vector<4xf32>, %121: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
      %122 = spv.SLessThan %117, %4 : i32
      spv.BranchConditional %122, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %123 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %124 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %125 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %126 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      spv.loop {
        spv.Branch ^bb1(%2, %118, %119, %120, %121 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb1(%132: i32, %133: vector<4xf32>, %134: vector<4xf32>, %135: vector<4xf32>, %136: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
        %137 = spv.SLessThan %132, %4 : i32
        spv.BranchConditional %137, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %138 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %139 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %140 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %141 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        spv.loop {
          spv.Branch ^bb1(%2, %133, %134, %135, %136 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb1(%147: i32, %148: vector<4xf32>, %149: vector<4xf32>, %150: vector<4xf32>, %151: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
          %152 = spv.SLessThan %147, %0 : i32
          spv.BranchConditional %152, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %153 = spv.IAdd %23, %36 : i32
          %154 = spv.SDiv %153, %7 : i32
          %155 = spv.constant 0 : i32
          %156 = spv.constant 0 : i32
          %157 = spv.constant 384 : i32
          %158 = spv.IMul %157, %117 : i32
          %159 = spv.IAdd %156, %158 : i32
          %160 = spv.constant 128 : i32
          %161 = spv.IMul %160, %132 : i32
          %162 = spv.IAdd %159, %161 : i32
          %163 = spv.constant 8 : i32
          %164 = spv.IMul %163, %147 : i32
          %165 = spv.IAdd %162, %164 : i32
          %166 = spv.constant 1 : i32
          %167 = spv.IMul %166, %154 : i32
          %168 = spv.IAdd %165, %167 : i32
          %169 = spv.AccessChain %10[%155, %168] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %170 = spv.Load "StorageBuffer" %169 : vector<4xf32>
          %171 = spv.IAdd %147, %3 : i32
          %172 = spv.constant 0 : i32
          %173 = spv.constant 0 : i32
          %174 = spv.constant 384 : i32
          %175 = spv.IMul %174, %117 : i32
          %176 = spv.IAdd %173, %175 : i32
          %177 = spv.constant 128 : i32
          %178 = spv.IMul %177, %132 : i32
          %179 = spv.IAdd %176, %178 : i32
          %180 = spv.constant 8 : i32
          %181 = spv.IMul %180, %171 : i32
          %182 = spv.IAdd %179, %181 : i32
          %183 = spv.constant 1 : i32
          %184 = spv.IMul %183, %154 : i32
          %185 = spv.IAdd %182, %184 : i32
          %186 = spv.AccessChain %10[%172, %185] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %187 = spv.Load "StorageBuffer" %186 : vector<4xf32>
          %188 = spv.IAdd %147, %6 : i32
          %189 = spv.constant 0 : i32
          %190 = spv.constant 0 : i32
          %191 = spv.constant 384 : i32
          %192 = spv.IMul %191, %117 : i32
          %193 = spv.IAdd %190, %192 : i32
          %194 = spv.constant 128 : i32
          %195 = spv.IMul %194, %132 : i32
          %196 = spv.IAdd %193, %195 : i32
          %197 = spv.constant 8 : i32
          %198 = spv.IMul %197, %188 : i32
          %199 = spv.IAdd %196, %198 : i32
          %200 = spv.constant 1 : i32
          %201 = spv.IMul %200, %154 : i32
          %202 = spv.IAdd %199, %201 : i32
          %203 = spv.AccessChain %10[%189, %202] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %204 = spv.Load "StorageBuffer" %203 : vector<4xf32>
          %205 = spv.IAdd %147, %4 : i32
          %206 = spv.constant 0 : i32
          %207 = spv.constant 0 : i32
          %208 = spv.constant 384 : i32
          %209 = spv.IMul %208, %117 : i32
          %210 = spv.IAdd %207, %209 : i32
          %211 = spv.constant 128 : i32
          %212 = spv.IMul %211, %132 : i32
          %213 = spv.IAdd %210, %212 : i32
          %214 = spv.constant 8 : i32
          %215 = spv.IMul %214, %205 : i32
          %216 = spv.IAdd %213, %215 : i32
          %217 = spv.constant 1 : i32
          %218 = spv.IMul %217, %154 : i32
          %219 = spv.IAdd %216, %218 : i32
          %220 = spv.AccessChain %10[%206, %219] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %221 = spv.Load "StorageBuffer" %220 : vector<4xf32>
          %222 = spv.IAdd %37, %117 : i32
          %223 = spv.IAdd %38, %132 : i32
          %224 = spv.IAdd %24, %222 : i32
          %225 = spv.IAdd %25, %223 : i32
          %226 = spv.SDiv %147, %7 : i32
          %227 = spv.constant 0 : i32
          %228 = spv.constant 0 : i32
          %229 = spv.constant 202500 : i32
          %230 = spv.IMul %229, %2 : i32
          %231 = spv.IAdd %228, %230 : i32
          %232 = spv.constant 900 : i32
          %233 = spv.IMul %232, %224 : i32
          %234 = spv.IAdd %231, %233 : i32
          %235 = spv.constant 4 : i32
          %236 = spv.IMul %235, %225 : i32
          %237 = spv.IAdd %234, %236 : i32
          %238 = spv.constant 1 : i32
          %239 = spv.IMul %238, %226 : i32
          %240 = spv.IAdd %237, %239 : i32
          %241 = spv.AccessChain %9[%227, %240] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %242 = spv.Load "StorageBuffer" %241 : vector<4xf32>
          %243 = spv.CompositeExtract %242[0 : i32] : vector<4xf32>
          %244 = spv.CompositeConstruct %243, %243, %243, %243 : vector<4xf32>
          %245 = spv.FMul %244, %170 : vector<4xf32>
          %246 = spv.FAdd %245, %148 : vector<4xf32>
          %247 = spv.CompositeExtract %242[1 : i32] : vector<4xf32>
          %248 = spv.CompositeConstruct %247, %247, %247, %247 : vector<4xf32>
          %249 = spv.FMul %248, %187 : vector<4xf32>
          %250 = spv.FAdd %249, %246 : vector<4xf32>
          %251 = spv.CompositeExtract %242[2 : i32] : vector<4xf32>
          %252 = spv.CompositeConstruct %251, %251, %251, %251 : vector<4xf32>
          %253 = spv.FMul %252, %204 : vector<4xf32>
          %254 = spv.FAdd %253, %250 : vector<4xf32>
          %255 = spv.CompositeExtract %242[3 : i32] : vector<4xf32>
          %256 = spv.CompositeConstruct %255, %255, %255, %255 : vector<4xf32>
          %257 = spv.FMul %256, %221 : vector<4xf32>
          %258 = spv.FAdd %257, %254 : vector<4xf32>
          %259 = spv.IAdd %117, %6 : i32
          %260 = spv.IAdd %37, %259 : i32
          %261 = spv.IAdd %24, %260 : i32
          %262 = spv.constant 0 : i32
          %263 = spv.constant 0 : i32
          %264 = spv.constant 202500 : i32
          %265 = spv.IMul %264, %2 : i32
          %266 = spv.IAdd %263, %265 : i32
          %267 = spv.constant 900 : i32
          %268 = spv.IMul %267, %261 : i32
          %269 = spv.IAdd %266, %268 : i32
          %270 = spv.constant 4 : i32
          %271 = spv.IMul %270, %225 : i32
          %272 = spv.IAdd %269, %271 : i32
          %273 = spv.constant 1 : i32
          %274 = spv.IMul %273, %226 : i32
          %275 = spv.IAdd %272, %274 : i32
          %276 = spv.AccessChain %9[%262, %275] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %277 = spv.Load "StorageBuffer" %276 : vector<4xf32>
          %278 = spv.CompositeExtract %277[0 : i32] : vector<4xf32>
          %279 = spv.CompositeConstruct %278, %278, %278, %278 : vector<4xf32>
          %280 = spv.FMul %279, %170 : vector<4xf32>
          %281 = spv.FAdd %280, %149 : vector<4xf32>
          %282 = spv.CompositeExtract %277[1 : i32] : vector<4xf32>
          %283 = spv.CompositeConstruct %282, %282, %282, %282 : vector<4xf32>
          %284 = spv.FMul %283, %187 : vector<4xf32>
          %285 = spv.FAdd %284, %281 : vector<4xf32>
          %286 = spv.CompositeExtract %277[2 : i32] : vector<4xf32>
          %287 = spv.CompositeConstruct %286, %286, %286, %286 : vector<4xf32>
          %288 = spv.FMul %287, %204 : vector<4xf32>
          %289 = spv.FAdd %288, %285 : vector<4xf32>
          %290 = spv.CompositeExtract %277[3 : i32] : vector<4xf32>
          %291 = spv.CompositeConstruct %290, %290, %290, %290 : vector<4xf32>
          %292 = spv.FMul %291, %221 : vector<4xf32>
          %293 = spv.FAdd %292, %289 : vector<4xf32>
          %294 = spv.IAdd %117, %7 : i32
          %295 = spv.IAdd %37, %294 : i32
          %296 = spv.IAdd %24, %295 : i32
          %297 = spv.constant 0 : i32
          %298 = spv.constant 0 : i32
          %299 = spv.constant 202500 : i32
          %300 = spv.IMul %299, %2 : i32
          %301 = spv.IAdd %298, %300 : i32
          %302 = spv.constant 900 : i32
          %303 = spv.IMul %302, %296 : i32
          %304 = spv.IAdd %301, %303 : i32
          %305 = spv.constant 4 : i32
          %306 = spv.IMul %305, %225 : i32
          %307 = spv.IAdd %304, %306 : i32
          %308 = spv.constant 1 : i32
          %309 = spv.IMul %308, %226 : i32
          %310 = spv.IAdd %307, %309 : i32
          %311 = spv.AccessChain %9[%297, %310] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %312 = spv.Load "StorageBuffer" %311 : vector<4xf32>
          %313 = spv.CompositeExtract %312[0 : i32] : vector<4xf32>
          %314 = spv.CompositeConstruct %313, %313, %313, %313 : vector<4xf32>
          %315 = spv.FMul %314, %170 : vector<4xf32>
          %316 = spv.FAdd %315, %150 : vector<4xf32>
          %317 = spv.CompositeExtract %312[1 : i32] : vector<4xf32>
          %318 = spv.CompositeConstruct %317, %317, %317, %317 : vector<4xf32>
          %319 = spv.FMul %318, %187 : vector<4xf32>
          %320 = spv.FAdd %319, %316 : vector<4xf32>
          %321 = spv.CompositeExtract %312[2 : i32] : vector<4xf32>
          %322 = spv.CompositeConstruct %321, %321, %321, %321 : vector<4xf32>
          %323 = spv.FMul %322, %204 : vector<4xf32>
          %324 = spv.FAdd %323, %320 : vector<4xf32>
          %325 = spv.CompositeExtract %312[3 : i32] : vector<4xf32>
          %326 = spv.CompositeConstruct %325, %325, %325, %325 : vector<4xf32>
          %327 = spv.FMul %326, %221 : vector<4xf32>
          %328 = spv.FAdd %327, %324 : vector<4xf32>
          %329 = spv.IAdd %117, %1 : i32
          %330 = spv.IAdd %37, %329 : i32
          %331 = spv.IAdd %24, %330 : i32
          %332 = spv.constant 0 : i32
          %333 = spv.constant 0 : i32
          %334 = spv.constant 202500 : i32
          %335 = spv.IMul %334, %2 : i32
          %336 = spv.IAdd %333, %335 : i32
          %337 = spv.constant 900 : i32
          %338 = spv.IMul %337, %331 : i32
          %339 = spv.IAdd %336, %338 : i32
          %340 = spv.constant 4 : i32
          %341 = spv.IMul %340, %225 : i32
          %342 = spv.IAdd %339, %341 : i32
          %343 = spv.constant 1 : i32
          %344 = spv.IMul %343, %226 : i32
          %345 = spv.IAdd %342, %344 : i32
          %346 = spv.AccessChain %9[%332, %345] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %347 = spv.Load "StorageBuffer" %346 : vector<4xf32>
          %348 = spv.CompositeExtract %347[0 : i32] : vector<4xf32>
          %349 = spv.CompositeConstruct %348, %348, %348, %348 : vector<4xf32>
          %350 = spv.FMul %349, %170 : vector<4xf32>
          %351 = spv.FAdd %350, %151 : vector<4xf32>
          %352 = spv.CompositeExtract %347[1 : i32] : vector<4xf32>
          %353 = spv.CompositeConstruct %352, %352, %352, %352 : vector<4xf32>
          %354 = spv.FMul %353, %187 : vector<4xf32>
          %355 = spv.FAdd %354, %351 : vector<4xf32>
          %356 = spv.CompositeExtract %347[2 : i32] : vector<4xf32>
          %357 = spv.CompositeConstruct %356, %356, %356, %356 : vector<4xf32>
          %358 = spv.FMul %357, %204 : vector<4xf32>
          %359 = spv.FAdd %358, %355 : vector<4xf32>
          %360 = spv.CompositeExtract %347[3 : i32] : vector<4xf32>
          %361 = spv.CompositeConstruct %360, %360, %360, %360 : vector<4xf32>
          %362 = spv.FMul %361, %221 : vector<4xf32>
          %363 = spv.FAdd %362, %359 : vector<4xf32>
          spv.Store "Function" %138, %258 : vector<4xf32>
          spv.Store "Function" %139, %293 : vector<4xf32>
          spv.Store "Function" %140, %328 : vector<4xf32>
          spv.Store "Function" %141, %363 : vector<4xf32>
          %364 = spv.IAdd %147, %7 : i32
          spv.Branch ^bb1(%364, %258, %293, %328, %363 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb3:  // pred: ^bb1
          spv.mlir.merge
        }
        %142 = spv.Load "Function" %141 : vector<4xf32>
        %143 = spv.Load "Function" %140 : vector<4xf32>
        %144 = spv.Load "Function" %139 : vector<4xf32>
        %145 = spv.Load "Function" %138 : vector<4xf32>
        spv.Store "Function" %123, %145 : vector<4xf32>
        spv.Store "Function" %124, %144 : vector<4xf32>
        spv.Store "Function" %125, %143 : vector<4xf32>
        spv.Store "Function" %126, %142 : vector<4xf32>
        %146 = spv.IAdd %132, %3 : i32
        spv.Branch ^bb1(%146, %145, %144, %143, %142 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb3:  // pred: ^bb1
        spv.mlir.merge
      }
      %127 = spv.Load "Function" %126 : vector<4xf32>
      %128 = spv.Load "Function" %125 : vector<4xf32>
      %129 = spv.Load "Function" %124 : vector<4xf32>
      %130 = spv.Load "Function" %123 : vector<4xf32>
      spv.Store "Function" %39, %130 : vector<4xf32>
      spv.Store "Function" %40, %129 : vector<4xf32>
      spv.Store "Function" %41, %128 : vector<4xf32>
      spv.Store "Function" %42, %127 : vector<4xf32>
      %131 = spv.IAdd %117, %3 : i32
      spv.Branch ^bb1(%131, %130, %129, %128, %127 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
    ^bb3:  // pred: ^bb1
      spv.mlir.merge
    }
    %43 = spv.Load "Function" %42 : vector<4xf32>
    %44 = spv.Load "Function" %41 : vector<4xf32>
    %45 = spv.Load "Function" %40 : vector<4xf32>
    %46 = spv.Load "Function" %39 : vector<4xf32>
    %47 = spv.IAdd %35, %4 : i32
    %48 = spv.IAdd %21, %47 : i32
    %49 = spv.IAdd %22, %31 : i32
    %50 = spv.IAdd %23, %36 : i32
    %51 = spv.SDiv %50, %7 : i32
    %52 = spv.constant 0 : i32
    %53 = spv.constant 0 : i32
    %54 = spv.constant 100352 : i32
    %55 = spv.IMul %54, %2 : i32
    %56 = spv.IAdd %53, %55 : i32
    %57 = spv.constant 896 : i32
    %58 = spv.IMul %57, %48 : i32
    %59 = spv.IAdd %56, %58 : i32
    %60 = spv.constant 8 : i32
    %61 = spv.IMul %60, %49 : i32
    %62 = spv.IAdd %59, %61 : i32
    %63 = spv.constant 1 : i32
    %64 = spv.IMul %63, %51 : i32
    %65 = spv.IAdd %62, %64 : i32
    %66 = spv.AccessChain %11[%52, %65] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %66, %43 : vector<4xf32>
    %67 = spv.IAdd %35, %6 : i32
    %68 = spv.IAdd %21, %67 : i32
    %69 = spv.constant 0 : i32
    %70 = spv.constant 0 : i32
    %71 = spv.constant 100352 : i32
    %72 = spv.IMul %71, %2 : i32
    %73 = spv.IAdd %70, %72 : i32
    %74 = spv.constant 896 : i32
    %75 = spv.IMul %74, %68 : i32
    %76 = spv.IAdd %73, %75 : i32
    %77 = spv.constant 8 : i32
    %78 = spv.IMul %77, %49 : i32
    %79 = spv.IAdd %76, %78 : i32
    %80 = spv.constant 1 : i32
    %81 = spv.IMul %80, %51 : i32
    %82 = spv.IAdd %79, %81 : i32
    %83 = spv.AccessChain %11[%69, %82] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %83, %44 : vector<4xf32>
    %84 = spv.IAdd %35, %3 : i32
    %85 = spv.IAdd %21, %84 : i32
    %86 = spv.constant 0 : i32
    %87 = spv.constant 0 : i32
    %88 = spv.constant 100352 : i32
    %89 = spv.IMul %88, %2 : i32
    %90 = spv.IAdd %87, %89 : i32
    %91 = spv.constant 896 : i32
    %92 = spv.IMul %91, %85 : i32
    %93 = spv.IAdd %90, %92 : i32
    %94 = spv.constant 8 : i32
    %95 = spv.IMul %94, %49 : i32
    %96 = spv.IAdd %93, %95 : i32
    %97 = spv.constant 1 : i32
    %98 = spv.IMul %97, %51 : i32
    %99 = spv.IAdd %96, %98 : i32
    %100 = spv.AccessChain %11[%86, %99] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %100, %45 : vector<4xf32>
    %101 = spv.IAdd %21, %35 : i32
    %102 = spv.constant 0 : i32
    %103 = spv.constant 0 : i32
    %104 = spv.constant 100352 : i32
    %105 = spv.IMul %104, %2 : i32
    %106 = spv.IAdd %103, %105 : i32
    %107 = spv.constant 896 : i32
    %108 = spv.IMul %107, %101 : i32
    %109 = spv.IAdd %106, %108 : i32
    %110 = spv.constant 8 : i32
    %111 = spv.IMul %110, %49 : i32
    %112 = spv.IAdd %109, %111 : i32
    %113 = spv.constant 1 : i32
    %114 = spv.IMul %113, %51 : i32
    %115 = spv.IAdd %112, %114 : i32
    %116 = spv.AccessChain %11[%102, %115] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %116, %46 : vector<4xf32>
    spv.Return
  }
  spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
  spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
}

// *** IR Dump After Canonicalizer ***
spv.module Logical GLSL450 {
  spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
    %0 = spv.constant 16 : i32
    %1 = spv.constant 6 : i32
    %2 = spv.constant 1 : i32
    %3 = spv.constant 3 : i32
    %4 = spv.constant 2 : i32
    %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
    %6 = spv.constant 384 : i32
    %7 = spv.constant 128 : i32
    %8 = spv.constant 900 : i32
    %9 = spv.constant 4 : i32
    %10 = spv.constant 0 : i32
    %11 = spv.constant 896 : i32
    %12 = spv.constant 8 : i32
    %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %17 = spv.Load "Input" %16 : vector<3xi32>
    %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
    %19 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %20 = spv.Load "Input" %19 : vector<3xi32>
    %21 = spv.CompositeExtract %20[1 : i32] : vector<3xi32>
    %22 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %23 = spv.Load "Input" %22 : vector<3xi32>
    %24 = spv.CompositeExtract %23[2 : i32] : vector<3xi32>
    %25 = spv.IMul %24, %9 : i32
    %26 = spv.IMul %21, %9 : i32
    %27 = spv.IMul %18, %0 : i32
    %28 = spv.IMul %25, %4 : i32
    %29 = spv.IMul %26, %4 : i32
    %30 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %31 = spv.Load "Input" %30 : vector<3xi32>
    %32 = spv.CompositeExtract %31[0 : i32] : vector<3xi32>
    %33 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %34 = spv.Load "Input" %33 : vector<3xi32>
    %35 = spv.CompositeExtract %34[1 : i32] : vector<3xi32>
    %36 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %37 = spv.Load "Input" %36 : vector<3xi32>
    %38 = spv.CompositeExtract %37[2 : i32] : vector<3xi32>
    %39 = spv.IMul %38, %9 : i32
    %40 = spv.IMul %32, %9 : i32
    %41 = spv.IMul %38, %12 : i32
    %42 = spv.IMul %35, %4 : i32
    %43 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %44 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %45 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %46 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    spv.loop {
      spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
    ^bb1(%81: i32, %82: vector<4xf32>, %83: vector<4xf32>, %84: vector<4xf32>, %85: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
      %86 = spv.SLessThan %81, %3 : i32
      spv.BranchConditional %86, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %87 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %88 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %89 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %90 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      spv.loop {
        spv.Branch ^bb1(%10, %82, %83, %84, %85 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb1(%96: i32, %97: vector<4xf32>, %98: vector<4xf32>, %99: vector<4xf32>, %100: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
        %101 = spv.SLessThan %96, %3 : i32
        spv.BranchConditional %101, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %102 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %103 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %104 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %105 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        spv.loop {
          spv.Branch ^bb1(%10, %97, %98, %99, %100 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb1(%111: i32, %112: vector<4xf32>, %113: vector<4xf32>, %114: vector<4xf32>, %115: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
          %116 = spv.SLessThan %111, %0 : i32
          spv.BranchConditional %116, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %117 = spv.IAdd %27, %40 : i32
          %118 = spv.SDiv %117, %9 : i32
          %119 = spv.IMul %81, %6 : i32
          %120 = spv.IMul %96, %7 : i32
          %121 = spv.IAdd %119, %120 : i32
          %122 = spv.IMul %111, %12 : i32
          %123 = spv.IAdd %121, %122 : i32
          %124 = spv.IAdd %123, %118 : i32
          %125 = spv.AccessChain %14[%10, %124] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %126 = spv.Load "StorageBuffer" %125 : vector<4xf32>
          %127 = spv.IAdd %111, %2 : i32
          %128 = spv.IMul %81, %6 : i32
          %129 = spv.IMul %96, %7 : i32
          %130 = spv.IAdd %128, %129 : i32
          %131 = spv.IMul %127, %12 : i32
          %132 = spv.IAdd %130, %131 : i32
          %133 = spv.IAdd %132, %118 : i32
          %134 = spv.AccessChain %14[%10, %133] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %135 = spv.Load "StorageBuffer" %134 : vector<4xf32>
          %136 = spv.IAdd %111, %4 : i32
          %137 = spv.IMul %81, %6 : i32
          %138 = spv.IMul %96, %7 : i32
          %139 = spv.IAdd %137, %138 : i32
          %140 = spv.IMul %136, %12 : i32
          %141 = spv.IAdd %139, %140 : i32
          %142 = spv.IAdd %141, %118 : i32
          %143 = spv.AccessChain %14[%10, %142] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %144 = spv.Load "StorageBuffer" %143 : vector<4xf32>
          %145 = spv.IAdd %111, %3 : i32
          %146 = spv.IMul %81, %6 : i32
          %147 = spv.IMul %96, %7 : i32
          %148 = spv.IAdd %146, %147 : i32
          %149 = spv.IMul %145, %12 : i32
          %150 = spv.IAdd %148, %149 : i32
          %151 = spv.IAdd %150, %118 : i32
          %152 = spv.AccessChain %14[%10, %151] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %153 = spv.Load "StorageBuffer" %152 : vector<4xf32>
          %154 = spv.IAdd %41, %81 : i32
          %155 = spv.IAdd %42, %96 : i32
          %156 = spv.IAdd %28, %154 : i32
          %157 = spv.IAdd %29, %155 : i32
          %158 = spv.SDiv %111, %9 : i32
          %159 = spv.IMul %156, %8 : i32
          %160 = spv.IMul %157, %9 : i32
          %161 = spv.IAdd %159, %160 : i32
          %162 = spv.IAdd %161, %158 : i32
          %163 = spv.AccessChain %13[%10, %162] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %164 = spv.Load "StorageBuffer" %163 : vector<4xf32>
          %165 = spv.CompositeExtract %164[0 : i32] : vector<4xf32>
          %166 = spv.CompositeConstruct %165, %165, %165, %165 : vector<4xf32>
          %167 = spv.FMul %166, %126 : vector<4xf32>
          %168 = spv.FAdd %167, %112 : vector<4xf32>
          %169 = spv.CompositeExtract %164[1 : i32] : vector<4xf32>
          %170 = spv.CompositeConstruct %169, %169, %169, %169 : vector<4xf32>
          %171 = spv.FMul %170, %135 : vector<4xf32>
          %172 = spv.FAdd %171, %168 : vector<4xf32>
          %173 = spv.CompositeExtract %164[2 : i32] : vector<4xf32>
          %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
          %175 = spv.FMul %174, %144 : vector<4xf32>
          %176 = spv.FAdd %175, %172 : vector<4xf32>
          %177 = spv.CompositeExtract %164[3 : i32] : vector<4xf32>
          %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
          %179 = spv.FMul %178, %153 : vector<4xf32>
          %180 = spv.FAdd %179, %176 : vector<4xf32>
          %181 = spv.IAdd %81, %4 : i32
          %182 = spv.IAdd %41, %181 : i32
          %183 = spv.IAdd %28, %182 : i32
          %184 = spv.IMul %183, %8 : i32
          %185 = spv.IMul %157, %9 : i32
          %186 = spv.IAdd %184, %185 : i32
          %187 = spv.IAdd %186, %158 : i32
          %188 = spv.AccessChain %13[%10, %187] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %189 = spv.Load "StorageBuffer" %188 : vector<4xf32>
          %190 = spv.CompositeExtract %189[0 : i32] : vector<4xf32>
          %191 = spv.CompositeConstruct %190, %190, %190, %190 : vector<4xf32>
          %192 = spv.FMul %191, %126 : vector<4xf32>
          %193 = spv.FAdd %192, %113 : vector<4xf32>
          %194 = spv.CompositeExtract %189[1 : i32] : vector<4xf32>
          %195 = spv.CompositeConstruct %194, %194, %194, %194 : vector<4xf32>
          %196 = spv.FMul %195, %135 : vector<4xf32>
          %197 = spv.FAdd %196, %193 : vector<4xf32>
          %198 = spv.CompositeExtract %189[2 : i32] : vector<4xf32>
          %199 = spv.CompositeConstruct %198, %198, %198, %198 : vector<4xf32>
          %200 = spv.FMul %199, %144 : vector<4xf32>
          %201 = spv.FAdd %200, %197 : vector<4xf32>
          %202 = spv.CompositeExtract %189[3 : i32] : vector<4xf32>
          %203 = spv.CompositeConstruct %202, %202, %202, %202 : vector<4xf32>
          %204 = spv.FMul %203, %153 : vector<4xf32>
          %205 = spv.FAdd %204, %201 : vector<4xf32>
          %206 = spv.IAdd %81, %9 : i32
          %207 = spv.IAdd %41, %206 : i32
          %208 = spv.IAdd %28, %207 : i32
          %209 = spv.IMul %208, %8 : i32
          %210 = spv.IMul %157, %9 : i32
          %211 = spv.IAdd %209, %210 : i32
          %212 = spv.IAdd %211, %158 : i32
          %213 = spv.AccessChain %13[%10, %212] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %214 = spv.Load "StorageBuffer" %213 : vector<4xf32>
          %215 = spv.CompositeExtract %214[0 : i32] : vector<4xf32>
          %216 = spv.CompositeConstruct %215, %215, %215, %215 : vector<4xf32>
          %217 = spv.FMul %216, %126 : vector<4xf32>
          %218 = spv.FAdd %217, %114 : vector<4xf32>
          %219 = spv.CompositeExtract %214[1 : i32] : vector<4xf32>
          %220 = spv.CompositeConstruct %219, %219, %219, %219 : vector<4xf32>
          %221 = spv.FMul %220, %135 : vector<4xf32>
          %222 = spv.FAdd %221, %218 : vector<4xf32>
          %223 = spv.CompositeExtract %214[2 : i32] : vector<4xf32>
          %224 = spv.CompositeConstruct %223, %223, %223, %223 : vector<4xf32>
          %225 = spv.FMul %224, %144 : vector<4xf32>
          %226 = spv.FAdd %225, %222 : vector<4xf32>
          %227 = spv.CompositeExtract %214[3 : i32] : vector<4xf32>
          %228 = spv.CompositeConstruct %227, %227, %227, %227 : vector<4xf32>
          %229 = spv.FMul %228, %153 : vector<4xf32>
          %230 = spv.FAdd %229, %226 : vector<4xf32>
          %231 = spv.IAdd %81, %1 : i32
          %232 = spv.IAdd %41, %231 : i32
          %233 = spv.IAdd %28, %232 : i32
          %234 = spv.IMul %233, %8 : i32
          %235 = spv.IMul %157, %9 : i32
          %236 = spv.IAdd %234, %235 : i32
          %237 = spv.IAdd %236, %158 : i32
          %238 = spv.AccessChain %13[%10, %237] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %239 = spv.Load "StorageBuffer" %238 : vector<4xf32>
          %240 = spv.CompositeExtract %239[0 : i32] : vector<4xf32>
          %241 = spv.CompositeConstruct %240, %240, %240, %240 : vector<4xf32>
          %242 = spv.FMul %241, %126 : vector<4xf32>
          %243 = spv.FAdd %242, %115 : vector<4xf32>
          %244 = spv.CompositeExtract %239[1 : i32] : vector<4xf32>
          %245 = spv.CompositeConstruct %244, %244, %244, %244 : vector<4xf32>
          %246 = spv.FMul %245, %135 : vector<4xf32>
          %247 = spv.FAdd %246, %243 : vector<4xf32>
          %248 = spv.CompositeExtract %239[2 : i32] : vector<4xf32>
          %249 = spv.CompositeConstruct %248, %248, %248, %248 : vector<4xf32>
          %250 = spv.FMul %249, %144 : vector<4xf32>
          %251 = spv.FAdd %250, %247 : vector<4xf32>
          %252 = spv.CompositeExtract %239[3 : i32] : vector<4xf32>
          %253 = spv.CompositeConstruct %252, %252, %252, %252 : vector<4xf32>
          %254 = spv.FMul %253, %153 : vector<4xf32>
          %255 = spv.FAdd %254, %251 : vector<4xf32>
          spv.Store "Function" %102, %180 : vector<4xf32>
          spv.Store "Function" %103, %205 : vector<4xf32>
          spv.Store "Function" %104, %230 : vector<4xf32>
          spv.Store "Function" %105, %255 : vector<4xf32>
          %256 = spv.IAdd %111, %9 : i32
          spv.Branch ^bb1(%256, %180, %205, %230, %255 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb3:  // pred: ^bb1
          spv.mlir.merge
        }
        %106 = spv.Load "Function" %105 : vector<4xf32>
        %107 = spv.Load "Function" %104 : vector<4xf32>
        %108 = spv.Load "Function" %103 : vector<4xf32>
        %109 = spv.Load "Function" %102 : vector<4xf32>
        spv.Store "Function" %87, %109 : vector<4xf32>
        spv.Store "Function" %88, %108 : vector<4xf32>
        spv.Store "Function" %89, %107 : vector<4xf32>
        spv.Store "Function" %90, %106 : vector<4xf32>
        %110 = spv.IAdd %96, %2 : i32
        spv.Branch ^bb1(%110, %109, %108, %107, %106 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb3:  // pred: ^bb1
        spv.mlir.merge
      }
      %91 = spv.Load "Function" %90 : vector<4xf32>
      %92 = spv.Load "Function" %89 : vector<4xf32>
      %93 = spv.Load "Function" %88 : vector<4xf32>
      %94 = spv.Load "Function" %87 : vector<4xf32>
      spv.Store "Function" %43, %94 : vector<4xf32>
      spv.Store "Function" %44, %93 : vector<4xf32>
      spv.Store "Function" %45, %92 : vector<4xf32>
      spv.Store "Function" %46, %91 : vector<4xf32>
      %95 = spv.IAdd %81, %2 : i32
      spv.Branch ^bb1(%95, %94, %93, %92, %91 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
    ^bb3:  // pred: ^bb1
      spv.mlir.merge
    }
    %47 = spv.Load "Function" %46 : vector<4xf32>
    %48 = spv.Load "Function" %45 : vector<4xf32>
    %49 = spv.Load "Function" %44 : vector<4xf32>
    %50 = spv.Load "Function" %43 : vector<4xf32>
    %51 = spv.IAdd %39, %3 : i32
    %52 = spv.IAdd %25, %51 : i32
    %53 = spv.IAdd %26, %35 : i32
    %54 = spv.IAdd %27, %40 : i32
    %55 = spv.SDiv %54, %9 : i32
    %56 = spv.IMul %52, %11 : i32
    %57 = spv.IMul %53, %12 : i32
    %58 = spv.IAdd %56, %57 : i32
    %59 = spv.IAdd %58, %55 : i32
    %60 = spv.AccessChain %15[%10, %59] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %60, %47 : vector<4xf32>
    %61 = spv.IAdd %39, %4 : i32
    %62 = spv.IAdd %25, %61 : i32
    %63 = spv.IMul %62, %11 : i32
    %64 = spv.IMul %53, %12 : i32
    %65 = spv.IAdd %63, %64 : i32
    %66 = spv.IAdd %65, %55 : i32
    %67 = spv.AccessChain %15[%10, %66] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %67, %48 : vector<4xf32>
    %68 = spv.IAdd %39, %2 : i32
    %69 = spv.IAdd %25, %68 : i32
    %70 = spv.IMul %69, %11 : i32
    %71 = spv.IMul %53, %12 : i32
    %72 = spv.IAdd %70, %71 : i32
    %73 = spv.IAdd %72, %55 : i32
    %74 = spv.AccessChain %15[%10, %73] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %74, %49 : vector<4xf32>
    %75 = spv.IAdd %25, %39 : i32
    %76 = spv.IMul %75, %11 : i32
    %77 = spv.IMul %53, %12 : i32
    %78 = spv.IAdd %76, %77 : i32
    %79 = spv.IAdd %78, %55 : i32
    %80 = spv.AccessChain %15[%10, %79] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %80, %50 : vector<4xf32>
    spv.Return
  }
  spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
  spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
}

// *** IR Dump After CSE ***
spv.module Logical GLSL450 {
  spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
    %0 = spv.constant 16 : i32
    %1 = spv.constant 6 : i32
    %2 = spv.constant 1 : i32
    %3 = spv.constant 3 : i32
    %4 = spv.constant 2 : i32
    %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
    %6 = spv.constant 384 : i32
    %7 = spv.constant 128 : i32
    %8 = spv.constant 900 : i32
    %9 = spv.constant 4 : i32
    %10 = spv.constant 0 : i32
    %11 = spv.constant 896 : i32
    %12 = spv.constant 8 : i32
    %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %17 = spv.Load "Input" %16 : vector<3xi32>
    %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
    %19 = spv.Load "Input" %16 : vector<3xi32>
    %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
    %21 = spv.Load "Input" %16 : vector<3xi32>
    %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
    %23 = spv.IMul %22, %9 : i32
    %24 = spv.IMul %20, %9 : i32
    %25 = spv.IMul %18, %0 : i32
    %26 = spv.IMul %23, %4 : i32
    %27 = spv.IMul %24, %4 : i32
    %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %29 = spv.Load "Input" %28 : vector<3xi32>
    %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
    %31 = spv.Load "Input" %28 : vector<3xi32>
    %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
    %33 = spv.Load "Input" %28 : vector<3xi32>
    %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
    %35 = spv.IMul %34, %9 : i32
    %36 = spv.IMul %30, %9 : i32
    %37 = spv.IMul %34, %12 : i32
    %38 = spv.IMul %32, %4 : i32
    %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    spv.loop {
      spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
    ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
      %79 = spv.SLessThan %74, %3 : i32
      spv.BranchConditional %79, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      spv.loop {
        spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
        %94 = spv.SLessThan %89, %3 : i32
        spv.BranchConditional %94, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        spv.loop {
          spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
          %109 = spv.SLessThan %104, %0 : i32
          spv.BranchConditional %109, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %110 = spv.IAdd %25, %36 : i32
          %111 = spv.SDiv %110, %9 : i32
          %112 = spv.IMul %74, %6 : i32
          %113 = spv.IMul %89, %7 : i32
          %114 = spv.IAdd %112, %113 : i32
          %115 = spv.IMul %104, %12 : i32
          %116 = spv.IAdd %114, %115 : i32
          %117 = spv.IAdd %116, %111 : i32
          %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
          %120 = spv.IAdd %104, %2 : i32
          %121 = spv.IMul %120, %12 : i32
          %122 = spv.IAdd %114, %121 : i32
          %123 = spv.IAdd %122, %111 : i32
          %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
          %126 = spv.IAdd %104, %4 : i32
          %127 = spv.IMul %126, %12 : i32
          %128 = spv.IAdd %114, %127 : i32
          %129 = spv.IAdd %128, %111 : i32
          %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
          %132 = spv.IAdd %104, %3 : i32
          %133 = spv.IMul %132, %12 : i32
          %134 = spv.IAdd %114, %133 : i32
          %135 = spv.IAdd %134, %111 : i32
          %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
          %138 = spv.IAdd %37, %74 : i32
          %139 = spv.IAdd %38, %89 : i32
          %140 = spv.IAdd %26, %138 : i32
          %141 = spv.IAdd %27, %139 : i32
          %142 = spv.SDiv %104, %9 : i32
          %143 = spv.IMul %140, %8 : i32
          %144 = spv.IMul %141, %9 : i32
          %145 = spv.IAdd %143, %144 : i32
          %146 = spv.IAdd %145, %142 : i32
          %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
          %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
          %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
          %151 = spv.FMul %150, %119 : vector<4xf32>
          %152 = spv.FAdd %151, %105 : vector<4xf32>
          %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
          %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
          %155 = spv.FMul %154, %125 : vector<4xf32>
          %156 = spv.FAdd %155, %152 : vector<4xf32>
          %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
          %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
          %159 = spv.FMul %158, %131 : vector<4xf32>
          %160 = spv.FAdd %159, %156 : vector<4xf32>
          %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
          %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
          %163 = spv.FMul %162, %137 : vector<4xf32>
          %164 = spv.FAdd %163, %160 : vector<4xf32>
          %165 = spv.IAdd %74, %4 : i32
          %166 = spv.IAdd %37, %165 : i32
          %167 = spv.IAdd %26, %166 : i32
          %168 = spv.IMul %167, %8 : i32
          %169 = spv.IAdd %168, %144 : i32
          %170 = spv.IAdd %169, %142 : i32
          %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
          %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
          %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
          %175 = spv.FMul %174, %119 : vector<4xf32>
          %176 = spv.FAdd %175, %106 : vector<4xf32>
          %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
          %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
          %179 = spv.FMul %178, %125 : vector<4xf32>
          %180 = spv.FAdd %179, %176 : vector<4xf32>
          %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
          %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
          %183 = spv.FMul %182, %131 : vector<4xf32>
          %184 = spv.FAdd %183, %180 : vector<4xf32>
          %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
          %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
          %187 = spv.FMul %186, %137 : vector<4xf32>
          %188 = spv.FAdd %187, %184 : vector<4xf32>
          %189 = spv.IAdd %74, %9 : i32
          %190 = spv.IAdd %37, %189 : i32
          %191 = spv.IAdd %26, %190 : i32
          %192 = spv.IMul %191, %8 : i32
          %193 = spv.IAdd %192, %144 : i32
          %194 = spv.IAdd %193, %142 : i32
          %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
          %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
          %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
          %199 = spv.FMul %198, %119 : vector<4xf32>
          %200 = spv.FAdd %199, %107 : vector<4xf32>
          %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
          %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
          %203 = spv.FMul %202, %125 : vector<4xf32>
          %204 = spv.FAdd %203, %200 : vector<4xf32>
          %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
          %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
          %207 = spv.FMul %206, %131 : vector<4xf32>
          %208 = spv.FAdd %207, %204 : vector<4xf32>
          %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
          %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
          %211 = spv.FMul %210, %137 : vector<4xf32>
          %212 = spv.FAdd %211, %208 : vector<4xf32>
          %213 = spv.IAdd %74, %1 : i32
          %214 = spv.IAdd %37, %213 : i32
          %215 = spv.IAdd %26, %214 : i32
          %216 = spv.IMul %215, %8 : i32
          %217 = spv.IAdd %216, %144 : i32
          %218 = spv.IAdd %217, %142 : i32
          %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
          %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
          %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
          %223 = spv.FMul %222, %119 : vector<4xf32>
          %224 = spv.FAdd %223, %108 : vector<4xf32>
          %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
          %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
          %227 = spv.FMul %226, %125 : vector<4xf32>
          %228 = spv.FAdd %227, %224 : vector<4xf32>
          %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
          %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
          %231 = spv.FMul %230, %131 : vector<4xf32>
          %232 = spv.FAdd %231, %228 : vector<4xf32>
          %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
          %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
          %235 = spv.FMul %234, %137 : vector<4xf32>
          %236 = spv.FAdd %235, %232 : vector<4xf32>
          spv.Store "Function" %95, %164 : vector<4xf32>
          spv.Store "Function" %96, %188 : vector<4xf32>
          spv.Store "Function" %97, %212 : vector<4xf32>
          spv.Store "Function" %98, %236 : vector<4xf32>
          %237 = spv.IAdd %104, %9 : i32
          spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb3:  // pred: ^bb1
          spv.mlir.merge
        }
        %99 = spv.Load "Function" %98 : vector<4xf32>
        %100 = spv.Load "Function" %97 : vector<4xf32>
        %101 = spv.Load "Function" %96 : vector<4xf32>
        %102 = spv.Load "Function" %95 : vector<4xf32>
        spv.Store "Function" %80, %102 : vector<4xf32>
        spv.Store "Function" %81, %101 : vector<4xf32>
        spv.Store "Function" %82, %100 : vector<4xf32>
        spv.Store "Function" %83, %99 : vector<4xf32>
        %103 = spv.IAdd %89, %2 : i32
        spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb3:  // pred: ^bb1
        spv.mlir.merge
      }
      %84 = spv.Load "Function" %83 : vector<4xf32>
      %85 = spv.Load "Function" %82 : vector<4xf32>
      %86 = spv.Load "Function" %81 : vector<4xf32>
      %87 = spv.Load "Function" %80 : vector<4xf32>
      spv.Store "Function" %39, %87 : vector<4xf32>
      spv.Store "Function" %40, %86 : vector<4xf32>
      spv.Store "Function" %41, %85 : vector<4xf32>
      spv.Store "Function" %42, %84 : vector<4xf32>
      %88 = spv.IAdd %74, %2 : i32
      spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
    ^bb3:  // pred: ^bb1
      spv.mlir.merge
    }
    %43 = spv.Load "Function" %42 : vector<4xf32>
    %44 = spv.Load "Function" %41 : vector<4xf32>
    %45 = spv.Load "Function" %40 : vector<4xf32>
    %46 = spv.Load "Function" %39 : vector<4xf32>
    %47 = spv.IAdd %35, %3 : i32
    %48 = spv.IAdd %23, %47 : i32
    %49 = spv.IAdd %24, %32 : i32
    %50 = spv.IAdd %25, %36 : i32
    %51 = spv.SDiv %50, %9 : i32
    %52 = spv.IMul %48, %11 : i32
    %53 = spv.IMul %49, %12 : i32
    %54 = spv.IAdd %52, %53 : i32
    %55 = spv.IAdd %54, %51 : i32
    %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
    %57 = spv.IAdd %35, %4 : i32
    %58 = spv.IAdd %23, %57 : i32
    %59 = spv.IMul %58, %11 : i32
    %60 = spv.IAdd %59, %53 : i32
    %61 = spv.IAdd %60, %51 : i32
    %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
    %63 = spv.IAdd %35, %2 : i32
    %64 = spv.IAdd %23, %63 : i32
    %65 = spv.IMul %64, %11 : i32
    %66 = spv.IAdd %65, %53 : i32
    %67 = spv.IAdd %66, %51 : i32
    %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
    %69 = spv.IAdd %23, %35 : i32
    %70 = spv.IMul %69, %11 : i32
    %71 = spv.IAdd %70, %53 : i32
    %72 = spv.IAdd %71, %51 : i32
    %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
    spv.Return
  }
  spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
  spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
}

// *** IR Dump After SPIRVUpdateVCE ***
spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
  spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
    %0 = spv.constant 16 : i32
    %1 = spv.constant 6 : i32
    %2 = spv.constant 1 : i32
    %3 = spv.constant 3 : i32
    %4 = spv.constant 2 : i32
    %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
    %6 = spv.constant 384 : i32
    %7 = spv.constant 128 : i32
    %8 = spv.constant 900 : i32
    %9 = spv.constant 4 : i32
    %10 = spv.constant 0 : i32
    %11 = spv.constant 896 : i32
    %12 = spv.constant 8 : i32
    %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %17 = spv.Load "Input" %16 : vector<3xi32>
    %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
    %19 = spv.Load "Input" %16 : vector<3xi32>
    %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
    %21 = spv.Load "Input" %16 : vector<3xi32>
    %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
    %23 = spv.IMul %22, %9 : i32
    %24 = spv.IMul %20, %9 : i32
    %25 = spv.IMul %18, %0 : i32
    %26 = spv.IMul %23, %4 : i32
    %27 = spv.IMul %24, %4 : i32
    %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %29 = spv.Load "Input" %28 : vector<3xi32>
    %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
    %31 = spv.Load "Input" %28 : vector<3xi32>
    %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
    %33 = spv.Load "Input" %28 : vector<3xi32>
    %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
    %35 = spv.IMul %34, %9 : i32
    %36 = spv.IMul %30, %9 : i32
    %37 = spv.IMul %34, %12 : i32
    %38 = spv.IMul %32, %4 : i32
    %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
    spv.loop {
      spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
    ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
      %79 = spv.SLessThan %74, %3 : i32
      spv.BranchConditional %79, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
      spv.loop {
        spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
        %94 = spv.SLessThan %89, %3 : i32
        spv.BranchConditional %94, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        spv.loop {
          spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
          %109 = spv.SLessThan %104, %0 : i32
          spv.BranchConditional %109, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %110 = spv.IAdd %25, %36 : i32
          %111 = spv.SDiv %110, %9 : i32
          %112 = spv.IMul %74, %6 : i32
          %113 = spv.IMul %89, %7 : i32
          %114 = spv.IAdd %112, %113 : i32
          %115 = spv.IMul %104, %12 : i32
          %116 = spv.IAdd %114, %115 : i32
          %117 = spv.IAdd %116, %111 : i32
          %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
          %120 = spv.IAdd %104, %2 : i32
          %121 = spv.IMul %120, %12 : i32
          %122 = spv.IAdd %114, %121 : i32
          %123 = spv.IAdd %122, %111 : i32
          %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
          %126 = spv.IAdd %104, %4 : i32
          %127 = spv.IMul %126, %12 : i32
          %128 = spv.IAdd %114, %127 : i32
          %129 = spv.IAdd %128, %111 : i32
          %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
          %132 = spv.IAdd %104, %3 : i32
          %133 = spv.IMul %132, %12 : i32
          %134 = spv.IAdd %114, %133 : i32
          %135 = spv.IAdd %134, %111 : i32
          %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
          %138 = spv.IAdd %37, %74 : i32
          %139 = spv.IAdd %38, %89 : i32
          %140 = spv.IAdd %26, %138 : i32
          %141 = spv.IAdd %27, %139 : i32
          %142 = spv.SDiv %104, %9 : i32
          %143 = spv.IMul %140, %8 : i32
          %144 = spv.IMul %141, %9 : i32
          %145 = spv.IAdd %143, %144 : i32
          %146 = spv.IAdd %145, %142 : i32
          %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
          %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
          %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
          %151 = spv.FMul %150, %119 : vector<4xf32>
          %152 = spv.FAdd %151, %105 : vector<4xf32>
          %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
          %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
          %155 = spv.FMul %154, %125 : vector<4xf32>
          %156 = spv.FAdd %155, %152 : vector<4xf32>
          %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
          %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
          %159 = spv.FMul %158, %131 : vector<4xf32>
          %160 = spv.FAdd %159, %156 : vector<4xf32>
          %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
          %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
          %163 = spv.FMul %162, %137 : vector<4xf32>
          %164 = spv.FAdd %163, %160 : vector<4xf32>
          %165 = spv.IAdd %74, %4 : i32
          %166 = spv.IAdd %37, %165 : i32
          %167 = spv.IAdd %26, %166 : i32
          %168 = spv.IMul %167, %8 : i32
          %169 = spv.IAdd %168, %144 : i32
          %170 = spv.IAdd %169, %142 : i32
          %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
          %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
          %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
          %175 = spv.FMul %174, %119 : vector<4xf32>
          %176 = spv.FAdd %175, %106 : vector<4xf32>
          %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
          %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
          %179 = spv.FMul %178, %125 : vector<4xf32>
          %180 = spv.FAdd %179, %176 : vector<4xf32>
          %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
          %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
          %183 = spv.FMul %182, %131 : vector<4xf32>
          %184 = spv.FAdd %183, %180 : vector<4xf32>
          %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
          %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
          %187 = spv.FMul %186, %137 : vector<4xf32>
          %188 = spv.FAdd %187, %184 : vector<4xf32>
          %189 = spv.IAdd %74, %9 : i32
          %190 = spv.IAdd %37, %189 : i32
          %191 = spv.IAdd %26, %190 : i32
          %192 = spv.IMul %191, %8 : i32
          %193 = spv.IAdd %192, %144 : i32
          %194 = spv.IAdd %193, %142 : i32
          %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
          %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
          %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
          %199 = spv.FMul %198, %119 : vector<4xf32>
          %200 = spv.FAdd %199, %107 : vector<4xf32>
          %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
          %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
          %203 = spv.FMul %202, %125 : vector<4xf32>
          %204 = spv.FAdd %203, %200 : vector<4xf32>
          %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
          %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
          %207 = spv.FMul %206, %131 : vector<4xf32>
          %208 = spv.FAdd %207, %204 : vector<4xf32>
          %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
          %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
          %211 = spv.FMul %210, %137 : vector<4xf32>
          %212 = spv.FAdd %211, %208 : vector<4xf32>
          %213 = spv.IAdd %74, %1 : i32
          %214 = spv.IAdd %37, %213 : i32
          %215 = spv.IAdd %26, %214 : i32
          %216 = spv.IMul %215, %8 : i32
          %217 = spv.IAdd %216, %144 : i32
          %218 = spv.IAdd %217, %142 : i32
          %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
          %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
          %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
          %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
          %223 = spv.FMul %222, %119 : vector<4xf32>
          %224 = spv.FAdd %223, %108 : vector<4xf32>
          %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
          %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
          %227 = spv.FMul %226, %125 : vector<4xf32>
          %228 = spv.FAdd %227, %224 : vector<4xf32>
          %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
          %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
          %231 = spv.FMul %230, %131 : vector<4xf32>
          %232 = spv.FAdd %231, %228 : vector<4xf32>
          %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
          %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
          %235 = spv.FMul %234, %137 : vector<4xf32>
          %236 = spv.FAdd %235, %232 : vector<4xf32>
          spv.Store "Function" %95, %164 : vector<4xf32>
          spv.Store "Function" %96, %188 : vector<4xf32>
          spv.Store "Function" %97, %212 : vector<4xf32>
          spv.Store "Function" %98, %236 : vector<4xf32>
          %237 = spv.IAdd %104, %9 : i32
          spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb3:  // pred: ^bb1
          spv.mlir.merge
        }
        %99 = spv.Load "Function" %98 : vector<4xf32>
        %100 = spv.Load "Function" %97 : vector<4xf32>
        %101 = spv.Load "Function" %96 : vector<4xf32>
        %102 = spv.Load "Function" %95 : vector<4xf32>
        spv.Store "Function" %80, %102 : vector<4xf32>
        spv.Store "Function" %81, %101 : vector<4xf32>
        spv.Store "Function" %82, %100 : vector<4xf32>
        spv.Store "Function" %83, %99 : vector<4xf32>
        %103 = spv.IAdd %89, %2 : i32
        spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      ^bb3:  // pred: ^bb1
        spv.mlir.merge
      }
      %84 = spv.Load "Function" %83 : vector<4xf32>
      %85 = spv.Load "Function" %82 : vector<4xf32>
      %86 = spv.Load "Function" %81 : vector<4xf32>
      %87 = spv.Load "Function" %80 : vector<4xf32>
      spv.Store "Function" %39, %87 : vector<4xf32>
      spv.Store "Function" %40, %86 : vector<4xf32>
      spv.Store "Function" %41, %85 : vector<4xf32>
      spv.Store "Function" %42, %84 : vector<4xf32>
      %88 = spv.IAdd %74, %2 : i32
      spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
    ^bb3:  // pred: ^bb1
      spv.mlir.merge
    }
    %43 = spv.Load "Function" %42 : vector<4xf32>
    %44 = spv.Load "Function" %41 : vector<4xf32>
    %45 = spv.Load "Function" %40 : vector<4xf32>
    %46 = spv.Load "Function" %39 : vector<4xf32>
    %47 = spv.IAdd %35, %3 : i32
    %48 = spv.IAdd %23, %47 : i32
    %49 = spv.IAdd %24, %32 : i32
    %50 = spv.IAdd %25, %36 : i32
    %51 = spv.SDiv %50, %9 : i32
    %52 = spv.IMul %48, %11 : i32
    %53 = spv.IMul %49, %12 : i32
    %54 = spv.IAdd %52, %53 : i32
    %55 = spv.IAdd %54, %51 : i32
    %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
    %57 = spv.IAdd %35, %4 : i32
    %58 = spv.IAdd %23, %57 : i32
    %59 = spv.IMul %58, %11 : i32
    %60 = spv.IAdd %59, %53 : i32
    %61 = spv.IAdd %60, %51 : i32
    %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
    %63 = spv.IAdd %35, %2 : i32
    %64 = spv.IAdd %23, %63 : i32
    %65 = spv.IMul %64, %11 : i32
    %66 = spv.IAdd %65, %53 : i32
    %67 = spv.IAdd %66, %51 : i32
    %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
    %69 = spv.IAdd %23, %35 : i32
    %70 = spv.IMul %69, %11 : i32
    %71 = spv.IAdd %70, %53 : i32
    %72 = spv.IAdd %71, %51 : i32
    %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
    spv.Return
  }
  spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
  spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
}

// *** IR Dump After mlir::iree_compiler::IREE::HAL::TranslateExecutablesPass ***
hal.executable.target @vulkan_spirv, filter="vulkan*" {
  hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index} {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):  // no predecessors
    %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg0]
    %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg1]
    %2 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg2]
    hal.return %0, %1, %2 : index, index, index
  }
  module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
    spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
      spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
      spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
      spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
        %0 = spv.constant 16 : i32
        %1 = spv.constant 6 : i32
        %2 = spv.constant 1 : i32
        %3 = spv.constant 3 : i32
        %4 = spv.constant 2 : i32
        %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
        %6 = spv.constant 384 : i32
        %7 = spv.constant 128 : i32
        %8 = spv.constant 900 : i32
        %9 = spv.constant 4 : i32
        %10 = spv.constant 0 : i32
        %11 = spv.constant 896 : i32
        %12 = spv.constant 8 : i32
        %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
        %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
        %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
        %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
        %17 = spv.Load "Input" %16 : vector<3xi32>
        %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
        %19 = spv.Load "Input" %16 : vector<3xi32>
        %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
        %21 = spv.Load "Input" %16 : vector<3xi32>
        %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
        %23 = spv.IMul %22, %9 : i32
        %24 = spv.IMul %20, %9 : i32
        %25 = spv.IMul %18, %0 : i32
        %26 = spv.IMul %23, %4 : i32
        %27 = spv.IMul %24, %4 : i32
        %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
        %29 = spv.Load "Input" %28 : vector<3xi32>
        %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
        %31 = spv.Load "Input" %28 : vector<3xi32>
        %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
        %33 = spv.Load "Input" %28 : vector<3xi32>
        %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
        %35 = spv.IMul %34, %9 : i32
        %36 = spv.IMul %30, %9 : i32
        %37 = spv.IMul %34, %12 : i32
        %38 = spv.IMul %32, %4 : i32
        %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
        spv.loop {
          spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
          %79 = spv.SLessThan %74, %3 : i32
          spv.BranchConditional %79, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
          %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
          %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
          %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
          spv.loop {
            spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
          ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
            %94 = spv.SLessThan %89, %3 : i32
            spv.BranchConditional %94, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            spv.loop {
              spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
              %109 = spv.SLessThan %104, %0 : i32
              spv.BranchConditional %109, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
              %110 = spv.IAdd %25, %36 : i32
              %111 = spv.SDiv %110, %9 : i32
              %112 = spv.IMul %74, %6 : i32
              %113 = spv.IMul %89, %7 : i32
              %114 = spv.IAdd %112, %113 : i32
              %115 = spv.IMul %104, %12 : i32
              %116 = spv.IAdd %114, %115 : i32
              %117 = spv.IAdd %116, %111 : i32
              %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
              %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
              %120 = spv.IAdd %104, %2 : i32
              %121 = spv.IMul %120, %12 : i32
              %122 = spv.IAdd %114, %121 : i32
              %123 = spv.IAdd %122, %111 : i32
              %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
              %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
              %126 = spv.IAdd %104, %4 : i32
              %127 = spv.IMul %126, %12 : i32
              %128 = spv.IAdd %114, %127 : i32
              %129 = spv.IAdd %128, %111 : i32
              %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
              %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
              %132 = spv.IAdd %104, %3 : i32
              %133 = spv.IMul %132, %12 : i32
              %134 = spv.IAdd %114, %133 : i32
              %135 = spv.IAdd %134, %111 : i32
              %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
              %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
              %138 = spv.IAdd %37, %74 : i32
              %139 = spv.IAdd %38, %89 : i32
              %140 = spv.IAdd %26, %138 : i32
              %141 = spv.IAdd %27, %139 : i32
              %142 = spv.SDiv %104, %9 : i32
              %143 = spv.IMul %140, %8 : i32
              %144 = spv.IMul %141, %9 : i32
              %145 = spv.IAdd %143, %144 : i32
              %146 = spv.IAdd %145, %142 : i32
              %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
              %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
              %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
              %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
              %151 = spv.FMul %150, %119 : vector<4xf32>
              %152 = spv.FAdd %151, %105 : vector<4xf32>
              %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
              %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
              %155 = spv.FMul %154, %125 : vector<4xf32>
              %156 = spv.FAdd %155, %152 : vector<4xf32>
              %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
              %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
              %159 = spv.FMul %158, %131 : vector<4xf32>
              %160 = spv.FAdd %159, %156 : vector<4xf32>
              %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
              %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
              %163 = spv.FMul %162, %137 : vector<4xf32>
              %164 = spv.FAdd %163, %160 : vector<4xf32>
              %165 = spv.IAdd %74, %4 : i32
              %166 = spv.IAdd %37, %165 : i32
              %167 = spv.IAdd %26, %166 : i32
              %168 = spv.IMul %167, %8 : i32
              %169 = spv.IAdd %168, %144 : i32
              %170 = spv.IAdd %169, %142 : i32
              %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
              %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
              %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
              %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
              %175 = spv.FMul %174, %119 : vector<4xf32>
              %176 = spv.FAdd %175, %106 : vector<4xf32>
              %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
              %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
              %179 = spv.FMul %178, %125 : vector<4xf32>
              %180 = spv.FAdd %179, %176 : vector<4xf32>
              %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
              %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
              %183 = spv.FMul %182, %131 : vector<4xf32>
              %184 = spv.FAdd %183, %180 : vector<4xf32>
              %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
              %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
              %187 = spv.FMul %186, %137 : vector<4xf32>
              %188 = spv.FAdd %187, %184 : vector<4xf32>
              %189 = spv.IAdd %74, %9 : i32
              %190 = spv.IAdd %37, %189 : i32
              %191 = spv.IAdd %26, %190 : i32
              %192 = spv.IMul %191, %8 : i32
              %193 = spv.IAdd %192, %144 : i32
              %194 = spv.IAdd %193, %142 : i32
              %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
              %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
              %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
              %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
              %199 = spv.FMul %198, %119 : vector<4xf32>
              %200 = spv.FAdd %199, %107 : vector<4xf32>
              %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
              %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
              %203 = spv.FMul %202, %125 : vector<4xf32>
              %204 = spv.FAdd %203, %200 : vector<4xf32>
              %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
              %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
              %207 = spv.FMul %206, %131 : vector<4xf32>
              %208 = spv.FAdd %207, %204 : vector<4xf32>
              %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
              %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
              %211 = spv.FMul %210, %137 : vector<4xf32>
              %212 = spv.FAdd %211, %208 : vector<4xf32>
              %213 = spv.IAdd %74, %1 : i32
              %214 = spv.IAdd %37, %213 : i32
              %215 = spv.IAdd %26, %214 : i32
              %216 = spv.IMul %215, %8 : i32
              %217 = spv.IAdd %216, %144 : i32
              %218 = spv.IAdd %217, %142 : i32
              %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
              %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
              %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
              %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
              %223 = spv.FMul %222, %119 : vector<4xf32>
              %224 = spv.FAdd %223, %108 : vector<4xf32>
              %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
              %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
              %227 = spv.FMul %226, %125 : vector<4xf32>
              %228 = spv.FAdd %227, %224 : vector<4xf32>
              %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
              %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
              %231 = spv.FMul %230, %131 : vector<4xf32>
              %232 = spv.FAdd %231, %228 : vector<4xf32>
              %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
              %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
              %235 = spv.FMul %234, %137 : vector<4xf32>
              %236 = spv.FAdd %235, %232 : vector<4xf32>
              spv.Store "Function" %95, %164 : vector<4xf32>
              spv.Store "Function" %96, %188 : vector<4xf32>
              spv.Store "Function" %97, %212 : vector<4xf32>
              spv.Store "Function" %98, %236 : vector<4xf32>
              %237 = spv.IAdd %104, %9 : i32
              spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb3:  // pred: ^bb1
              spv.mlir.merge
            }
            %99 = spv.Load "Function" %98 : vector<4xf32>
            %100 = spv.Load "Function" %97 : vector<4xf32>
            %101 = spv.Load "Function" %96 : vector<4xf32>
            %102 = spv.Load "Function" %95 : vector<4xf32>
            spv.Store "Function" %80, %102 : vector<4xf32>
            spv.Store "Function" %81, %101 : vector<4xf32>
            spv.Store "Function" %82, %100 : vector<4xf32>
            spv.Store "Function" %83, %99 : vector<4xf32>
            %103 = spv.IAdd %89, %2 : i32
            spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
          ^bb3:  // pred: ^bb1
            spv.mlir.merge
          }
          %84 = spv.Load "Function" %83 : vector<4xf32>
          %85 = spv.Load "Function" %82 : vector<4xf32>
          %86 = spv.Load "Function" %81 : vector<4xf32>
          %87 = spv.Load "Function" %80 : vector<4xf32>
          spv.Store "Function" %39, %87 : vector<4xf32>
          spv.Store "Function" %40, %86 : vector<4xf32>
          spv.Store "Function" %41, %85 : vector<4xf32>
          spv.Store "Function" %42, %84 : vector<4xf32>
          %88 = spv.IAdd %74, %2 : i32
          spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
        ^bb3:  // pred: ^bb1
          spv.mlir.merge
        }
        %43 = spv.Load "Function" %42 : vector<4xf32>
        %44 = spv.Load "Function" %41 : vector<4xf32>
        %45 = spv.Load "Function" %40 : vector<4xf32>
        %46 = spv.Load "Function" %39 : vector<4xf32>
        %47 = spv.IAdd %35, %3 : i32
        %48 = spv.IAdd %23, %47 : i32
        %49 = spv.IAdd %24, %32 : i32
        %50 = spv.IAdd %25, %36 : i32
        %51 = spv.SDiv %50, %9 : i32
        %52 = spv.IMul %48, %11 : i32
        %53 = spv.IMul %49, %12 : i32
        %54 = spv.IAdd %52, %53 : i32
        %55 = spv.IAdd %54, %51 : i32
        %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
        spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
        %57 = spv.IAdd %35, %4 : i32
        %58 = spv.IAdd %23, %57 : i32
        %59 = spv.IMul %58, %11 : i32
        %60 = spv.IAdd %59, %53 : i32
        %61 = spv.IAdd %60, %51 : i32
        %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
        spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
        %63 = spv.IAdd %35, %2 : i32
        %64 = spv.IAdd %23, %63 : i32
        %65 = spv.IMul %64, %11 : i32
        %66 = spv.IAdd %65, %53 : i32
        %67 = spv.IAdd %66, %51 : i32
        %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
        spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
        %69 = spv.IAdd %23, %35 : i32
        %70 = spv.IMul %69, %11 : i32
        %71 = spv.IAdd %70, %53 : i32
        %72 = spv.IAdd %71, %51 : i32
        %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
        spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
        spv.Return
      }
      spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
      spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
    }
    hal.interface @legacy_io attributes {sym_visibility = "private"} {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
  }
}

// *** IR Dump After mlir::iree_compiler::IREE::HAL::(anonymous namespace)::ConvertToHALPass ***
#map0 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 4)>
module  {
  hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vulkan_spirv, filter="vulkan*" {
      hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index} {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):  // no predecessors
        %0 = affine.apply #map0()[%arg0]
        %1 = affine.apply #map1()[%arg1]
        %2 = affine.apply #map1()[%arg2]
        hal.return %0, %1, %2 : index, index, index
      }
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
        spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
          spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
            %0 = spv.constant 16 : i32
            %1 = spv.constant 6 : i32
            %2 = spv.constant 1 : i32
            %3 = spv.constant 3 : i32
            %4 = spv.constant 2 : i32
            %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
            %6 = spv.constant 384 : i32
            %7 = spv.constant 128 : i32
            %8 = spv.constant 900 : i32
            %9 = spv.constant 4 : i32
            %10 = spv.constant 0 : i32
            %11 = spv.constant 896 : i32
            %12 = spv.constant 8 : i32
            %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
            %17 = spv.Load "Input" %16 : vector<3xi32>
            %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
            %19 = spv.Load "Input" %16 : vector<3xi32>
            %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
            %21 = spv.Load "Input" %16 : vector<3xi32>
            %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
            %23 = spv.IMul %22, %9 : i32
            %24 = spv.IMul %20, %9 : i32
            %25 = spv.IMul %18, %0 : i32
            %26 = spv.IMul %23, %4 : i32
            %27 = spv.IMul %24, %4 : i32
            %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
            %29 = spv.Load "Input" %28 : vector<3xi32>
            %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
            %31 = spv.Load "Input" %28 : vector<3xi32>
            %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
            %33 = spv.Load "Input" %28 : vector<3xi32>
            %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
            %35 = spv.IMul %34, %9 : i32
            %36 = spv.IMul %30, %9 : i32
            %37 = spv.IMul %34, %12 : i32
            %38 = spv.IMul %32, %4 : i32
            %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            spv.loop {
              spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
              %79 = spv.SLessThan %74, %3 : i32
              spv.BranchConditional %79, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
              %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              spv.loop {
                spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                %94 = spv.SLessThan %89, %3 : i32
                spv.BranchConditional %94, ^bb2, ^bb3
              ^bb2:  // pred: ^bb1
                %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                spv.loop {
                  spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                  %109 = spv.SLessThan %104, %0 : i32
                  spv.BranchConditional %109, ^bb2, ^bb3
                ^bb2:  // pred: ^bb1
                  %110 = spv.IAdd %25, %36 : i32
                  %111 = spv.SDiv %110, %9 : i32
                  %112 = spv.IMul %74, %6 : i32
                  %113 = spv.IMul %89, %7 : i32
                  %114 = spv.IAdd %112, %113 : i32
                  %115 = spv.IMul %104, %12 : i32
                  %116 = spv.IAdd %114, %115 : i32
                  %117 = spv.IAdd %116, %111 : i32
                  %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
                  %120 = spv.IAdd %104, %2 : i32
                  %121 = spv.IMul %120, %12 : i32
                  %122 = spv.IAdd %114, %121 : i32
                  %123 = spv.IAdd %122, %111 : i32
                  %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
                  %126 = spv.IAdd %104, %4 : i32
                  %127 = spv.IMul %126, %12 : i32
                  %128 = spv.IAdd %114, %127 : i32
                  %129 = spv.IAdd %128, %111 : i32
                  %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
                  %132 = spv.IAdd %104, %3 : i32
                  %133 = spv.IMul %132, %12 : i32
                  %134 = spv.IAdd %114, %133 : i32
                  %135 = spv.IAdd %134, %111 : i32
                  %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
                  %138 = spv.IAdd %37, %74 : i32
                  %139 = spv.IAdd %38, %89 : i32
                  %140 = spv.IAdd %26, %138 : i32
                  %141 = spv.IAdd %27, %139 : i32
                  %142 = spv.SDiv %104, %9 : i32
                  %143 = spv.IMul %140, %8 : i32
                  %144 = spv.IMul %141, %9 : i32
                  %145 = spv.IAdd %143, %144 : i32
                  %146 = spv.IAdd %145, %142 : i32
                  %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
                  %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
                  %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
                  %151 = spv.FMul %150, %119 : vector<4xf32>
                  %152 = spv.FAdd %151, %105 : vector<4xf32>
                  %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
                  %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
                  %155 = spv.FMul %154, %125 : vector<4xf32>
                  %156 = spv.FAdd %155, %152 : vector<4xf32>
                  %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
                  %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
                  %159 = spv.FMul %158, %131 : vector<4xf32>
                  %160 = spv.FAdd %159, %156 : vector<4xf32>
                  %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
                  %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
                  %163 = spv.FMul %162, %137 : vector<4xf32>
                  %164 = spv.FAdd %163, %160 : vector<4xf32>
                  %165 = spv.IAdd %74, %4 : i32
                  %166 = spv.IAdd %37, %165 : i32
                  %167 = spv.IAdd %26, %166 : i32
                  %168 = spv.IMul %167, %8 : i32
                  %169 = spv.IAdd %168, %144 : i32
                  %170 = spv.IAdd %169, %142 : i32
                  %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
                  %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
                  %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
                  %175 = spv.FMul %174, %119 : vector<4xf32>
                  %176 = spv.FAdd %175, %106 : vector<4xf32>
                  %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
                  %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
                  %179 = spv.FMul %178, %125 : vector<4xf32>
                  %180 = spv.FAdd %179, %176 : vector<4xf32>
                  %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
                  %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
                  %183 = spv.FMul %182, %131 : vector<4xf32>
                  %184 = spv.FAdd %183, %180 : vector<4xf32>
                  %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
                  %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
                  %187 = spv.FMul %186, %137 : vector<4xf32>
                  %188 = spv.FAdd %187, %184 : vector<4xf32>
                  %189 = spv.IAdd %74, %9 : i32
                  %190 = spv.IAdd %37, %189 : i32
                  %191 = spv.IAdd %26, %190 : i32
                  %192 = spv.IMul %191, %8 : i32
                  %193 = spv.IAdd %192, %144 : i32
                  %194 = spv.IAdd %193, %142 : i32
                  %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
                  %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
                  %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
                  %199 = spv.FMul %198, %119 : vector<4xf32>
                  %200 = spv.FAdd %199, %107 : vector<4xf32>
                  %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
                  %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
                  %203 = spv.FMul %202, %125 : vector<4xf32>
                  %204 = spv.FAdd %203, %200 : vector<4xf32>
                  %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
                  %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
                  %207 = spv.FMul %206, %131 : vector<4xf32>
                  %208 = spv.FAdd %207, %204 : vector<4xf32>
                  %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
                  %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
                  %211 = spv.FMul %210, %137 : vector<4xf32>
                  %212 = spv.FAdd %211, %208 : vector<4xf32>
                  %213 = spv.IAdd %74, %1 : i32
                  %214 = spv.IAdd %37, %213 : i32
                  %215 = spv.IAdd %26, %214 : i32
                  %216 = spv.IMul %215, %8 : i32
                  %217 = spv.IAdd %216, %144 : i32
                  %218 = spv.IAdd %217, %142 : i32
                  %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
                  %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
                  %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
                  %223 = spv.FMul %222, %119 : vector<4xf32>
                  %224 = spv.FAdd %223, %108 : vector<4xf32>
                  %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
                  %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
                  %227 = spv.FMul %226, %125 : vector<4xf32>
                  %228 = spv.FAdd %227, %224 : vector<4xf32>
                  %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
                  %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
                  %231 = spv.FMul %230, %131 : vector<4xf32>
                  %232 = spv.FAdd %231, %228 : vector<4xf32>
                  %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
                  %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
                  %235 = spv.FMul %234, %137 : vector<4xf32>
                  %236 = spv.FAdd %235, %232 : vector<4xf32>
                  spv.Store "Function" %95, %164 : vector<4xf32>
                  spv.Store "Function" %96, %188 : vector<4xf32>
                  spv.Store "Function" %97, %212 : vector<4xf32>
                  spv.Store "Function" %98, %236 : vector<4xf32>
                  %237 = spv.IAdd %104, %9 : i32
                  spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb3:  // pred: ^bb1
                  spv.mlir.merge
                }
                %99 = spv.Load "Function" %98 : vector<4xf32>
                %100 = spv.Load "Function" %97 : vector<4xf32>
                %101 = spv.Load "Function" %96 : vector<4xf32>
                %102 = spv.Load "Function" %95 : vector<4xf32>
                spv.Store "Function" %80, %102 : vector<4xf32>
                spv.Store "Function" %81, %101 : vector<4xf32>
                spv.Store "Function" %82, %100 : vector<4xf32>
                spv.Store "Function" %83, %99 : vector<4xf32>
                %103 = spv.IAdd %89, %2 : i32
                spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb3:  // pred: ^bb1
                spv.mlir.merge
              }
              %84 = spv.Load "Function" %83 : vector<4xf32>
              %85 = spv.Load "Function" %82 : vector<4xf32>
              %86 = spv.Load "Function" %81 : vector<4xf32>
              %87 = spv.Load "Function" %80 : vector<4xf32>
              spv.Store "Function" %39, %87 : vector<4xf32>
              spv.Store "Function" %40, %86 : vector<4xf32>
              spv.Store "Function" %41, %85 : vector<4xf32>
              spv.Store "Function" %42, %84 : vector<4xf32>
              %88 = spv.IAdd %74, %2 : i32
              spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb3:  // pred: ^bb1
              spv.mlir.merge
            }
            %43 = spv.Load "Function" %42 : vector<4xf32>
            %44 = spv.Load "Function" %41 : vector<4xf32>
            %45 = spv.Load "Function" %40 : vector<4xf32>
            %46 = spv.Load "Function" %39 : vector<4xf32>
            %47 = spv.IAdd %35, %3 : i32
            %48 = spv.IAdd %23, %47 : i32
            %49 = spv.IAdd %24, %32 : i32
            %50 = spv.IAdd %25, %36 : i32
            %51 = spv.SDiv %50, %9 : i32
            %52 = spv.IMul %48, %11 : i32
            %53 = spv.IMul %49, %12 : i32
            %54 = spv.IAdd %52, %53 : i32
            %55 = spv.IAdd %54, %51 : i32
            %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
            %57 = spv.IAdd %35, %4 : i32
            %58 = spv.IAdd %23, %57 : i32
            %59 = spv.IMul %58, %11 : i32
            %60 = spv.IAdd %59, %53 : i32
            %61 = spv.IAdd %60, %51 : i32
            %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
            %63 = spv.IAdd %35, %2 : i32
            %64 = spv.IAdd %23, %63 : i32
            %65 = spv.IMul %64, %11 : i32
            %66 = spv.IAdd %65, %53 : i32
            %67 = spv.IAdd %66, %51 : i32
            %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
            %69 = spv.IAdd %23, %35 : i32
            %70 = spv.IMul %69, %11 : i32
            %71 = spv.IAdd %70, %53 : i32
            %72 = spv.IAdd %71, %51 : i32
            %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
            spv.Return
          }
          spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
          spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
        }
        hal.interface @legacy_io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %dev = hal.ex.shared_device : !hal.device
    %allocator = hal.device.allocator %dev : !hal.allocator
    %c1_0 = constant 1 : index
    %c112_1 = constant 112 : index
    %c112_2 = constant 112 : index
    %c32_3 = constant 32 : index
    %c50331680_i32 = constant 50331680 : i32
    %sz = hal.allocator.compute_size %allocator, shape = [%c1_0, %c112_1, %c112_2, %c32_3], element_type = %c50331680_i32
    %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %sz : !hal.buffer
    %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
    hal.command_buffer.begin %cmd
    %executable_layout = hal.executable_layout.lookup %dev, set_layouts = [[#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">]] : !hal.executable_layout
    %c0 = constant 0 : index
    %c1_4 = constant 1 : index
    %c225 = constant 225 : index
    %c225_5 = constant 225 : index
    %c16 = constant 16 : index
    %allocator_6 = hal.buffer.allocator %arg0 : !hal.allocator
    %c50331680_i32_7 = constant 50331680 : i32
    %sz_8 = hal.allocator.compute_size %allocator_6, shape = [%c1_4, %c225, %c225_5, %c16], element_type = %c50331680_i32_7
    %c0_9 = constant 0 : index
    %c3 = constant 3 : index
    %c3_10 = constant 3 : index
    %c16_11 = constant 16 : index
    %c32_12 = constant 32 : index
    %allocator_13 = hal.buffer.allocator %arg1 : !hal.allocator
    %c50331680_i32_14 = constant 50331680 : i32
    %sz_15 = hal.allocator.compute_size %allocator_13, shape = [%c3, %c3_10, %c16_11, %c32_12], element_type = %c50331680_i32_14
    %c1_16 = constant 1 : index
    %c1_17 = constant 1 : index
    %c112_18 = constant 112 : index
    %c112_19 = constant 112 : index
    %c32_20 = constant 32 : index
    %allocator_21 = hal.buffer.allocator %buffer : !hal.allocator
    %c50331680_i32_22 = constant 50331680 : i32
    %sz_23 = hal.allocator.compute_size %allocator_21, shape = [%c1_17, %c112_18, %c112_19, %c32_20], element_type = %c50331680_i32_22
    %c2 = constant 2 : index
    %c0_24 = constant 0 : index
    hal.command_buffer.push_descriptor_set %cmd, %executable_layout, set = %c0_24, bindings = [%c0_9 = (%arg0, %c0, %sz_8), %c1_16 = (%arg1, %c0, %sz_15), %c2 = (%buffer, %c0, %sz_23)]
    hal.device.switch(%dev : !hal.device)
    #hal.device.match.id<"vulkan*">(%arg2 = %cmd : !hal.command_buffer, %arg3 = %c32 : index, %arg4 = %c112 : index, %arg5 = %c112 : index, %arg6 = %c1 : index) {
      %0 = affine.apply #map0()[%arg3]
      %1 = affine.apply #map1()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      hal.command_buffer.dispatch.symbol %arg2, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv::@predict_ex_dispatch_1_dispatch_0, workgroup_xyz = [%0, %1, %2]
      hal.return
    }
    hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
    hal.command_buffer.end %cmd
    hal.ex.submit_and_wait %dev, %cmd
    return %buffer : !hal.buffer
  }
}


// *** IR Dump After mlir::iree_compiler::Shape::(anonymous namespace)::ExpandFunctionRankedShapeDimsPass ***
func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator %dev : !hal.allocator
  %c1_0 = constant 1 : index
  %c112_1 = constant 112 : index
  %c112_2 = constant 112 : index
  %c32_3 = constant 32 : index
  %c50331680_i32 = constant 50331680 : i32
  %sz = hal.allocator.compute_size %allocator, shape = [%c1_0, %c112_1, %c112_2, %c32_3], element_type = %c50331680_i32
  %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %sz : !hal.buffer
  %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  %executable_layout = hal.executable_layout.lookup %dev, set_layouts = [[#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">]] : !hal.executable_layout
  %c0 = constant 0 : index
  %c1_4 = constant 1 : index
  %c225 = constant 225 : index
  %c225_5 = constant 225 : index
  %c16 = constant 16 : index
  %allocator_6 = hal.buffer.allocator %arg0 : !hal.allocator
  %c50331680_i32_7 = constant 50331680 : i32
  %sz_8 = hal.allocator.compute_size %allocator_6, shape = [%c1_4, %c225, %c225_5, %c16], element_type = %c50331680_i32_7
  %c0_9 = constant 0 : index
  %c3 = constant 3 : index
  %c3_10 = constant 3 : index
  %c16_11 = constant 16 : index
  %c32_12 = constant 32 : index
  %allocator_13 = hal.buffer.allocator %arg1 : !hal.allocator
  %c50331680_i32_14 = constant 50331680 : i32
  %sz_15 = hal.allocator.compute_size %allocator_13, shape = [%c3, %c3_10, %c16_11, %c32_12], element_type = %c50331680_i32_14
  %c1_16 = constant 1 : index
  %c1_17 = constant 1 : index
  %c112_18 = constant 112 : index
  %c112_19 = constant 112 : index
  %c32_20 = constant 32 : index
  %allocator_21 = hal.buffer.allocator %buffer : !hal.allocator
  %c50331680_i32_22 = constant 50331680 : i32
  %sz_23 = hal.allocator.compute_size %allocator_21, shape = [%c1_17, %c112_18, %c112_19, %c32_20], element_type = %c50331680_i32_22
  %c2 = constant 2 : index
  %c0_24 = constant 0 : index
  hal.command_buffer.push_descriptor_set %cmd, %executable_layout, set = %c0_24, bindings = [%c0_9 = (%arg0, %c0, %sz_8), %c1_16 = (%arg1, %c0, %sz_15), %c2 = (%buffer, %c0, %sz_23)]
  hal.device.switch(%dev : !hal.device)
    #hal.device.match.id<"vulkan*">(%arg2 = %cmd : !hal.command_buffer, %arg3 = %c32 : index, %arg4 = %c112 : index, %arg5 = %c112 : index, %arg6 = %c1 : index) {
    %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg3]
    %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg4]
    %2 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg5]
    hal.command_buffer.dispatch.symbol %arg2, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv::@predict_ex_dispatch_1_dispatch_0, workgroup_xyz = [%0, %1, %2]
    hal.return
  }
  hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
  hal.command_buffer.end %cmd
  hal.ex.submit_and_wait %dev, %cmd
  return %buffer : !hal.buffer
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export, iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c3240000 = constant 3240000 : index
  %c18432 = constant 18432 : index
  %c1 = constant 1 : index
  %c1605632 = constant 1605632 : index
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator %dev : !hal.allocator
  %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
  %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  %executable_layout = hal.executable_layout.lookup %dev, set_layouts = [[#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">]] : !hal.executable_layout
  hal.command_buffer.push_descriptor_set %cmd, %executable_layout, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
  hal.device.switch(%dev : !hal.device)
    #hal.device.match.id<"vulkan*">(%arg2 = %cmd : !hal.command_buffer, %arg3 = %c32 : index, %arg4 = %c112 : index, %arg5 = %c112 : index, %arg6 = %c1 : index) {
    %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg3]
    %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg4]
    %2 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg5]
    hal.command_buffer.dispatch.symbol %arg2, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv::@predict_ex_dispatch_1_dispatch_0, workgroup_xyz = [%0, %1, %2]
    hal.return
  }
  hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
  hal.command_buffer.end %cmd
  hal.ex.submit_and_wait %dev, %cmd
  return %buffer : !hal.buffer
}

// *** IR Dump After mlir::iree_compiler::IREE::HAL::(anonymous namespace)::PublicABIGenerationPass ***
#map0 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 4)>
module  {
  hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vulkan_spirv, filter="vulkan*" {
      hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index} {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):  // no predecessors
        %0 = affine.apply #map0()[%arg0]
        %1 = affine.apply #map1()[%arg1]
        %2 = affine.apply #map1()[%arg2]
        hal.return %0, %1, %2 : index, index, index
      }
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
        spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
          spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
            %0 = spv.constant 16 : i32
            %1 = spv.constant 6 : i32
            %2 = spv.constant 1 : i32
            %3 = spv.constant 3 : i32
            %4 = spv.constant 2 : i32
            %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
            %6 = spv.constant 384 : i32
            %7 = spv.constant 128 : i32
            %8 = spv.constant 900 : i32
            %9 = spv.constant 4 : i32
            %10 = spv.constant 0 : i32
            %11 = spv.constant 896 : i32
            %12 = spv.constant 8 : i32
            %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
            %17 = spv.Load "Input" %16 : vector<3xi32>
            %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
            %19 = spv.Load "Input" %16 : vector<3xi32>
            %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
            %21 = spv.Load "Input" %16 : vector<3xi32>
            %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
            %23 = spv.IMul %22, %9 : i32
            %24 = spv.IMul %20, %9 : i32
            %25 = spv.IMul %18, %0 : i32
            %26 = spv.IMul %23, %4 : i32
            %27 = spv.IMul %24, %4 : i32
            %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
            %29 = spv.Load "Input" %28 : vector<3xi32>
            %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
            %31 = spv.Load "Input" %28 : vector<3xi32>
            %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
            %33 = spv.Load "Input" %28 : vector<3xi32>
            %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
            %35 = spv.IMul %34, %9 : i32
            %36 = spv.IMul %30, %9 : i32
            %37 = spv.IMul %34, %12 : i32
            %38 = spv.IMul %32, %4 : i32
            %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            spv.loop {
              spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
              %79 = spv.SLessThan %74, %3 : i32
              spv.BranchConditional %79, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
              %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              spv.loop {
                spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                %94 = spv.SLessThan %89, %3 : i32
                spv.BranchConditional %94, ^bb2, ^bb3
              ^bb2:  // pred: ^bb1
                %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                spv.loop {
                  spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                  %109 = spv.SLessThan %104, %0 : i32
                  spv.BranchConditional %109, ^bb2, ^bb3
                ^bb2:  // pred: ^bb1
                  %110 = spv.IAdd %25, %36 : i32
                  %111 = spv.SDiv %110, %9 : i32
                  %112 = spv.IMul %74, %6 : i32
                  %113 = spv.IMul %89, %7 : i32
                  %114 = spv.IAdd %112, %113 : i32
                  %115 = spv.IMul %104, %12 : i32
                  %116 = spv.IAdd %114, %115 : i32
                  %117 = spv.IAdd %116, %111 : i32
                  %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
                  %120 = spv.IAdd %104, %2 : i32
                  %121 = spv.IMul %120, %12 : i32
                  %122 = spv.IAdd %114, %121 : i32
                  %123 = spv.IAdd %122, %111 : i32
                  %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
                  %126 = spv.IAdd %104, %4 : i32
                  %127 = spv.IMul %126, %12 : i32
                  %128 = spv.IAdd %114, %127 : i32
                  %129 = spv.IAdd %128, %111 : i32
                  %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
                  %132 = spv.IAdd %104, %3 : i32
                  %133 = spv.IMul %132, %12 : i32
                  %134 = spv.IAdd %114, %133 : i32
                  %135 = spv.IAdd %134, %111 : i32
                  %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
                  %138 = spv.IAdd %37, %74 : i32
                  %139 = spv.IAdd %38, %89 : i32
                  %140 = spv.IAdd %26, %138 : i32
                  %141 = spv.IAdd %27, %139 : i32
                  %142 = spv.SDiv %104, %9 : i32
                  %143 = spv.IMul %140, %8 : i32
                  %144 = spv.IMul %141, %9 : i32
                  %145 = spv.IAdd %143, %144 : i32
                  %146 = spv.IAdd %145, %142 : i32
                  %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
                  %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
                  %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
                  %151 = spv.FMul %150, %119 : vector<4xf32>
                  %152 = spv.FAdd %151, %105 : vector<4xf32>
                  %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
                  %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
                  %155 = spv.FMul %154, %125 : vector<4xf32>
                  %156 = spv.FAdd %155, %152 : vector<4xf32>
                  %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
                  %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
                  %159 = spv.FMul %158, %131 : vector<4xf32>
                  %160 = spv.FAdd %159, %156 : vector<4xf32>
                  %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
                  %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
                  %163 = spv.FMul %162, %137 : vector<4xf32>
                  %164 = spv.FAdd %163, %160 : vector<4xf32>
                  %165 = spv.IAdd %74, %4 : i32
                  %166 = spv.IAdd %37, %165 : i32
                  %167 = spv.IAdd %26, %166 : i32
                  %168 = spv.IMul %167, %8 : i32
                  %169 = spv.IAdd %168, %144 : i32
                  %170 = spv.IAdd %169, %142 : i32
                  %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
                  %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
                  %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
                  %175 = spv.FMul %174, %119 : vector<4xf32>
                  %176 = spv.FAdd %175, %106 : vector<4xf32>
                  %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
                  %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
                  %179 = spv.FMul %178, %125 : vector<4xf32>
                  %180 = spv.FAdd %179, %176 : vector<4xf32>
                  %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
                  %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
                  %183 = spv.FMul %182, %131 : vector<4xf32>
                  %184 = spv.FAdd %183, %180 : vector<4xf32>
                  %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
                  %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
                  %187 = spv.FMul %186, %137 : vector<4xf32>
                  %188 = spv.FAdd %187, %184 : vector<4xf32>
                  %189 = spv.IAdd %74, %9 : i32
                  %190 = spv.IAdd %37, %189 : i32
                  %191 = spv.IAdd %26, %190 : i32
                  %192 = spv.IMul %191, %8 : i32
                  %193 = spv.IAdd %192, %144 : i32
                  %194 = spv.IAdd %193, %142 : i32
                  %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
                  %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
                  %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
                  %199 = spv.FMul %198, %119 : vector<4xf32>
                  %200 = spv.FAdd %199, %107 : vector<4xf32>
                  %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
                  %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
                  %203 = spv.FMul %202, %125 : vector<4xf32>
                  %204 = spv.FAdd %203, %200 : vector<4xf32>
                  %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
                  %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
                  %207 = spv.FMul %206, %131 : vector<4xf32>
                  %208 = spv.FAdd %207, %204 : vector<4xf32>
                  %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
                  %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
                  %211 = spv.FMul %210, %137 : vector<4xf32>
                  %212 = spv.FAdd %211, %208 : vector<4xf32>
                  %213 = spv.IAdd %74, %1 : i32
                  %214 = spv.IAdd %37, %213 : i32
                  %215 = spv.IAdd %26, %214 : i32
                  %216 = spv.IMul %215, %8 : i32
                  %217 = spv.IAdd %216, %144 : i32
                  %218 = spv.IAdd %217, %142 : i32
                  %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
                  %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
                  %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
                  %223 = spv.FMul %222, %119 : vector<4xf32>
                  %224 = spv.FAdd %223, %108 : vector<4xf32>
                  %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
                  %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
                  %227 = spv.FMul %226, %125 : vector<4xf32>
                  %228 = spv.FAdd %227, %224 : vector<4xf32>
                  %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
                  %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
                  %231 = spv.FMul %230, %131 : vector<4xf32>
                  %232 = spv.FAdd %231, %228 : vector<4xf32>
                  %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
                  %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
                  %235 = spv.FMul %234, %137 : vector<4xf32>
                  %236 = spv.FAdd %235, %232 : vector<4xf32>
                  spv.Store "Function" %95, %164 : vector<4xf32>
                  spv.Store "Function" %96, %188 : vector<4xf32>
                  spv.Store "Function" %97, %212 : vector<4xf32>
                  spv.Store "Function" %98, %236 : vector<4xf32>
                  %237 = spv.IAdd %104, %9 : i32
                  spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb3:  // pred: ^bb1
                  spv.mlir.merge
                }
                %99 = spv.Load "Function" %98 : vector<4xf32>
                %100 = spv.Load "Function" %97 : vector<4xf32>
                %101 = spv.Load "Function" %96 : vector<4xf32>
                %102 = spv.Load "Function" %95 : vector<4xf32>
                spv.Store "Function" %80, %102 : vector<4xf32>
                spv.Store "Function" %81, %101 : vector<4xf32>
                spv.Store "Function" %82, %100 : vector<4xf32>
                spv.Store "Function" %83, %99 : vector<4xf32>
                %103 = spv.IAdd %89, %2 : i32
                spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb3:  // pred: ^bb1
                spv.mlir.merge
              }
              %84 = spv.Load "Function" %83 : vector<4xf32>
              %85 = spv.Load "Function" %82 : vector<4xf32>
              %86 = spv.Load "Function" %81 : vector<4xf32>
              %87 = spv.Load "Function" %80 : vector<4xf32>
              spv.Store "Function" %39, %87 : vector<4xf32>
              spv.Store "Function" %40, %86 : vector<4xf32>
              spv.Store "Function" %41, %85 : vector<4xf32>
              spv.Store "Function" %42, %84 : vector<4xf32>
              %88 = spv.IAdd %74, %2 : i32
              spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb3:  // pred: ^bb1
              spv.mlir.merge
            }
            %43 = spv.Load "Function" %42 : vector<4xf32>
            %44 = spv.Load "Function" %41 : vector<4xf32>
            %45 = spv.Load "Function" %40 : vector<4xf32>
            %46 = spv.Load "Function" %39 : vector<4xf32>
            %47 = spv.IAdd %35, %3 : i32
            %48 = spv.IAdd %23, %47 : i32
            %49 = spv.IAdd %24, %32 : i32
            %50 = spv.IAdd %25, %36 : i32
            %51 = spv.SDiv %50, %9 : i32
            %52 = spv.IMul %48, %11 : i32
            %53 = spv.IMul %49, %12 : i32
            %54 = spv.IAdd %52, %53 : i32
            %55 = spv.IAdd %54, %51 : i32
            %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
            %57 = spv.IAdd %35, %4 : i32
            %58 = spv.IAdd %23, %57 : i32
            %59 = spv.IMul %58, %11 : i32
            %60 = spv.IAdd %59, %53 : i32
            %61 = spv.IAdd %60, %51 : i32
            %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
            %63 = spv.IAdd %35, %2 : i32
            %64 = spv.IAdd %23, %63 : i32
            %65 = spv.IMul %64, %11 : i32
            %66 = spv.IAdd %65, %53 : i32
            %67 = spv.IAdd %66, %51 : i32
            %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
            %69 = spv.IAdd %23, %35 : i32
            %70 = spv.IMul %69, %11 : i32
            %71 = spv.IAdd %70, %53 : i32
            %72 = spv.IAdd %71, %51 : i32
            %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
            spv.Return
          }
          spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
          spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
        }
        hal.interface @legacy_io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c3240000 = constant 3240000 : index
    %c18432 = constant 18432 : index
    %c1 = constant 1 : index
    %c1605632 = constant 1605632 : index
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %dev = hal.ex.shared_device : !hal.device
    %allocator = hal.device.allocator %dev : !hal.allocator
    %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
    %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
    hal.command_buffer.begin %cmd
    %executable_layout = hal.executable_layout.lookup %dev, set_layouts = [[#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">]] : !hal.executable_layout
    hal.command_buffer.push_descriptor_set %cmd, %executable_layout, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
    hal.device.switch(%dev : !hal.device)
    #hal.device.match.id<"vulkan*">(%arg2 = %cmd : !hal.command_buffer, %arg3 = %c32 : index, %arg4 = %c112 : index, %arg5 = %c112 : index, %arg6 = %c1 : index) {
      %0 = affine.apply #map0()[%arg3]
      %1 = affine.apply #map1()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      hal.command_buffer.dispatch.symbol %arg2, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv::@predict_ex_dispatch_1_dispatch_0, workgroup_xyz = [%0, %1, %2]
      hal.return
    }
    hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
    hal.command_buffer.end %cmd
    hal.ex.submit_and_wait %dev, %cmd
    return %buffer : !hal.buffer
  }
  func @predict_ex_dispatch_1$async(%arg0: !hal.semaphore, %arg1: index, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.semaphore, %arg5: index) -> !hal.buffer_view attributes {iree.module.export = "predict_ex_dispatch_1$async"} {
    %0 = hal.semaphore.await %arg0, min_value = %arg1 : i32
    hal.check_success %0, "semaphore wait failed"
    %buffer = hal.buffer_view.buffer %arg2 : !hal.buffer
    %buffer_0 = hal.buffer_view.buffer %arg3 : !hal.buffer
    %1 = call @predict_ex_dispatch_1(%buffer, %buffer_0) : (!hal.buffer, !hal.buffer) -> !hal.buffer
    %c1 = constant 1 : index
    %c112 = constant 112 : index
    %c112_1 = constant 112 : index
    %c32 = constant 32 : index
    %c50331680_i32 = constant 50331680 : i32
    %view = hal.buffer_view.create %1, element_type = %c50331680_i32, shape = [%c1, %c112, %c112_1, %c32] : !hal.buffer_view
    hal.semaphore.signal %arg4, value = %arg5
    return %view : !hal.buffer_view
  }
  func @predict_ex_dispatch_1$sync(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export = "predict_ex_dispatch_1", iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dev = hal.ex.shared_device : !hal.device
    %semaphore = hal.semaphore.create %dev, initial_value = %c0 : !hal.semaphore
    %0 = call @predict_ex_dispatch_1$async(%semaphore, %c0, %arg0, %arg1, %semaphore, %c1) : (!hal.semaphore, index, !hal.buffer_view, !hal.buffer_view, !hal.semaphore, index) -> !hal.buffer_view
    %1 = hal.semaphore.await %semaphore, min_value = %c1 : i32
    hal.check_success %1, "semaphore wait failed"
    return %0 : !hal.buffer_view
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::HAL::ResolveEntryPointOrdinalsPass ***
#map0 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 4)>
module  {
  hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vulkan_spirv, filter="vulkan*" {
      hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index} {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):  // no predecessors
        %0 = affine.apply #map0()[%arg0]
        %1 = affine.apply #map1()[%arg1]
        %2 = affine.apply #map1()[%arg2]
        hal.return %0, %1, %2 : index, index, index
      }
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
        spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
          spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
            %0 = spv.constant 16 : i32
            %1 = spv.constant 6 : i32
            %2 = spv.constant 1 : i32
            %3 = spv.constant 3 : i32
            %4 = spv.constant 2 : i32
            %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
            %6 = spv.constant 384 : i32
            %7 = spv.constant 128 : i32
            %8 = spv.constant 900 : i32
            %9 = spv.constant 4 : i32
            %10 = spv.constant 0 : i32
            %11 = spv.constant 896 : i32
            %12 = spv.constant 8 : i32
            %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
            %17 = spv.Load "Input" %16 : vector<3xi32>
            %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
            %19 = spv.Load "Input" %16 : vector<3xi32>
            %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
            %21 = spv.Load "Input" %16 : vector<3xi32>
            %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
            %23 = spv.IMul %22, %9 : i32
            %24 = spv.IMul %20, %9 : i32
            %25 = spv.IMul %18, %0 : i32
            %26 = spv.IMul %23, %4 : i32
            %27 = spv.IMul %24, %4 : i32
            %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
            %29 = spv.Load "Input" %28 : vector<3xi32>
            %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
            %31 = spv.Load "Input" %28 : vector<3xi32>
            %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
            %33 = spv.Load "Input" %28 : vector<3xi32>
            %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
            %35 = spv.IMul %34, %9 : i32
            %36 = spv.IMul %30, %9 : i32
            %37 = spv.IMul %34, %12 : i32
            %38 = spv.IMul %32, %4 : i32
            %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            spv.loop {
              spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
              %79 = spv.SLessThan %74, %3 : i32
              spv.BranchConditional %79, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
              %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              spv.loop {
                spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                %94 = spv.SLessThan %89, %3 : i32
                spv.BranchConditional %94, ^bb2, ^bb3
              ^bb2:  // pred: ^bb1
                %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                spv.loop {
                  spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                  %109 = spv.SLessThan %104, %0 : i32
                  spv.BranchConditional %109, ^bb2, ^bb3
                ^bb2:  // pred: ^bb1
                  %110 = spv.IAdd %25, %36 : i32
                  %111 = spv.SDiv %110, %9 : i32
                  %112 = spv.IMul %74, %6 : i32
                  %113 = spv.IMul %89, %7 : i32
                  %114 = spv.IAdd %112, %113 : i32
                  %115 = spv.IMul %104, %12 : i32
                  %116 = spv.IAdd %114, %115 : i32
                  %117 = spv.IAdd %116, %111 : i32
                  %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
                  %120 = spv.IAdd %104, %2 : i32
                  %121 = spv.IMul %120, %12 : i32
                  %122 = spv.IAdd %114, %121 : i32
                  %123 = spv.IAdd %122, %111 : i32
                  %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
                  %126 = spv.IAdd %104, %4 : i32
                  %127 = spv.IMul %126, %12 : i32
                  %128 = spv.IAdd %114, %127 : i32
                  %129 = spv.IAdd %128, %111 : i32
                  %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
                  %132 = spv.IAdd %104, %3 : i32
                  %133 = spv.IMul %132, %12 : i32
                  %134 = spv.IAdd %114, %133 : i32
                  %135 = spv.IAdd %134, %111 : i32
                  %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
                  %138 = spv.IAdd %37, %74 : i32
                  %139 = spv.IAdd %38, %89 : i32
                  %140 = spv.IAdd %26, %138 : i32
                  %141 = spv.IAdd %27, %139 : i32
                  %142 = spv.SDiv %104, %9 : i32
                  %143 = spv.IMul %140, %8 : i32
                  %144 = spv.IMul %141, %9 : i32
                  %145 = spv.IAdd %143, %144 : i32
                  %146 = spv.IAdd %145, %142 : i32
                  %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
                  %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
                  %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
                  %151 = spv.FMul %150, %119 : vector<4xf32>
                  %152 = spv.FAdd %151, %105 : vector<4xf32>
                  %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
                  %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
                  %155 = spv.FMul %154, %125 : vector<4xf32>
                  %156 = spv.FAdd %155, %152 : vector<4xf32>
                  %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
                  %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
                  %159 = spv.FMul %158, %131 : vector<4xf32>
                  %160 = spv.FAdd %159, %156 : vector<4xf32>
                  %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
                  %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
                  %163 = spv.FMul %162, %137 : vector<4xf32>
                  %164 = spv.FAdd %163, %160 : vector<4xf32>
                  %165 = spv.IAdd %74, %4 : i32
                  %166 = spv.IAdd %37, %165 : i32
                  %167 = spv.IAdd %26, %166 : i32
                  %168 = spv.IMul %167, %8 : i32
                  %169 = spv.IAdd %168, %144 : i32
                  %170 = spv.IAdd %169, %142 : i32
                  %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
                  %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
                  %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
                  %175 = spv.FMul %174, %119 : vector<4xf32>
                  %176 = spv.FAdd %175, %106 : vector<4xf32>
                  %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
                  %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
                  %179 = spv.FMul %178, %125 : vector<4xf32>
                  %180 = spv.FAdd %179, %176 : vector<4xf32>
                  %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
                  %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
                  %183 = spv.FMul %182, %131 : vector<4xf32>
                  %184 = spv.FAdd %183, %180 : vector<4xf32>
                  %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
                  %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
                  %187 = spv.FMul %186, %137 : vector<4xf32>
                  %188 = spv.FAdd %187, %184 : vector<4xf32>
                  %189 = spv.IAdd %74, %9 : i32
                  %190 = spv.IAdd %37, %189 : i32
                  %191 = spv.IAdd %26, %190 : i32
                  %192 = spv.IMul %191, %8 : i32
                  %193 = spv.IAdd %192, %144 : i32
                  %194 = spv.IAdd %193, %142 : i32
                  %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
                  %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
                  %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
                  %199 = spv.FMul %198, %119 : vector<4xf32>
                  %200 = spv.FAdd %199, %107 : vector<4xf32>
                  %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
                  %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
                  %203 = spv.FMul %202, %125 : vector<4xf32>
                  %204 = spv.FAdd %203, %200 : vector<4xf32>
                  %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
                  %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
                  %207 = spv.FMul %206, %131 : vector<4xf32>
                  %208 = spv.FAdd %207, %204 : vector<4xf32>
                  %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
                  %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
                  %211 = spv.FMul %210, %137 : vector<4xf32>
                  %212 = spv.FAdd %211, %208 : vector<4xf32>
                  %213 = spv.IAdd %74, %1 : i32
                  %214 = spv.IAdd %37, %213 : i32
                  %215 = spv.IAdd %26, %214 : i32
                  %216 = spv.IMul %215, %8 : i32
                  %217 = spv.IAdd %216, %144 : i32
                  %218 = spv.IAdd %217, %142 : i32
                  %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
                  %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
                  %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
                  %223 = spv.FMul %222, %119 : vector<4xf32>
                  %224 = spv.FAdd %223, %108 : vector<4xf32>
                  %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
                  %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
                  %227 = spv.FMul %226, %125 : vector<4xf32>
                  %228 = spv.FAdd %227, %224 : vector<4xf32>
                  %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
                  %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
                  %231 = spv.FMul %230, %131 : vector<4xf32>
                  %232 = spv.FAdd %231, %228 : vector<4xf32>
                  %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
                  %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
                  %235 = spv.FMul %234, %137 : vector<4xf32>
                  %236 = spv.FAdd %235, %232 : vector<4xf32>
                  spv.Store "Function" %95, %164 : vector<4xf32>
                  spv.Store "Function" %96, %188 : vector<4xf32>
                  spv.Store "Function" %97, %212 : vector<4xf32>
                  spv.Store "Function" %98, %236 : vector<4xf32>
                  %237 = spv.IAdd %104, %9 : i32
                  spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb3:  // pred: ^bb1
                  spv.mlir.merge
                }
                %99 = spv.Load "Function" %98 : vector<4xf32>
                %100 = spv.Load "Function" %97 : vector<4xf32>
                %101 = spv.Load "Function" %96 : vector<4xf32>
                %102 = spv.Load "Function" %95 : vector<4xf32>
                spv.Store "Function" %80, %102 : vector<4xf32>
                spv.Store "Function" %81, %101 : vector<4xf32>
                spv.Store "Function" %82, %100 : vector<4xf32>
                spv.Store "Function" %83, %99 : vector<4xf32>
                %103 = spv.IAdd %89, %2 : i32
                spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb3:  // pred: ^bb1
                spv.mlir.merge
              }
              %84 = spv.Load "Function" %83 : vector<4xf32>
              %85 = spv.Load "Function" %82 : vector<4xf32>
              %86 = spv.Load "Function" %81 : vector<4xf32>
              %87 = spv.Load "Function" %80 : vector<4xf32>
              spv.Store "Function" %39, %87 : vector<4xf32>
              spv.Store "Function" %40, %86 : vector<4xf32>
              spv.Store "Function" %41, %85 : vector<4xf32>
              spv.Store "Function" %42, %84 : vector<4xf32>
              %88 = spv.IAdd %74, %2 : i32
              spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb3:  // pred: ^bb1
              spv.mlir.merge
            }
            %43 = spv.Load "Function" %42 : vector<4xf32>
            %44 = spv.Load "Function" %41 : vector<4xf32>
            %45 = spv.Load "Function" %40 : vector<4xf32>
            %46 = spv.Load "Function" %39 : vector<4xf32>
            %47 = spv.IAdd %35, %3 : i32
            %48 = spv.IAdd %23, %47 : i32
            %49 = spv.IAdd %24, %32 : i32
            %50 = spv.IAdd %25, %36 : i32
            %51 = spv.SDiv %50, %9 : i32
            %52 = spv.IMul %48, %11 : i32
            %53 = spv.IMul %49, %12 : i32
            %54 = spv.IAdd %52, %53 : i32
            %55 = spv.IAdd %54, %51 : i32
            %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
            %57 = spv.IAdd %35, %4 : i32
            %58 = spv.IAdd %23, %57 : i32
            %59 = spv.IMul %58, %11 : i32
            %60 = spv.IAdd %59, %53 : i32
            %61 = spv.IAdd %60, %51 : i32
            %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
            %63 = spv.IAdd %35, %2 : i32
            %64 = spv.IAdd %23, %63 : i32
            %65 = spv.IMul %64, %11 : i32
            %66 = spv.IAdd %65, %53 : i32
            %67 = spv.IAdd %66, %51 : i32
            %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
            %69 = spv.IAdd %23, %35 : i32
            %70 = spv.IMul %69, %11 : i32
            %71 = spv.IAdd %70, %53 : i32
            %72 = spv.IAdd %71, %51 : i32
            %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
            spv.Return
          }
          spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
          spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
        }
        hal.interface @legacy_io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c3240000 = constant 3240000 : index
    %c18432 = constant 18432 : index
    %c1 = constant 1 : index
    %c1605632 = constant 1605632 : index
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %dev = hal.ex.shared_device : !hal.device
    %allocator = hal.device.allocator %dev : !hal.allocator
    %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
    %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
    hal.command_buffer.begin %cmd
    %executable_layout = hal.executable_layout.lookup %dev, set_layouts = [[#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">]] : !hal.executable_layout
    hal.command_buffer.push_descriptor_set %cmd, %executable_layout, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
    hal.device.switch(%dev : !hal.device)
    #hal.device.match.id<"vulkan*">(%arg2 = %cmd : !hal.command_buffer, %arg3 = %c32 : index, %arg4 = %c112 : index, %arg5 = %c112 : index, %arg6 = %c1 : index) {
      %0 = affine.apply #map0()[%arg3]
      %1 = affine.apply #map1()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      %3 = hal.command_buffer.device %arg2 : !hal.device
      %exe = hal.executable.lookup %3, @predict_ex_dispatch_1_dispatch_0 : !hal.executable
      hal.command_buffer.dispatch %arg2, %exe, entry_point = 0, workgroup_xyz = [%0, %1, %2]
      hal.return
    }
    hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
    hal.command_buffer.end %cmd
    hal.ex.submit_and_wait %dev, %cmd
    return %buffer : !hal.buffer
  }
  func @predict_ex_dispatch_1$async(%arg0: !hal.semaphore, %arg1: index, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.semaphore, %arg5: index) -> !hal.buffer_view attributes {iree.module.export = "predict_ex_dispatch_1$async"} {
    %c1 = constant 1 : index
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c50331680_i32 = constant 50331680 : i32
    %0 = hal.semaphore.await %arg0, min_value = %arg1 : i32
    hal.check_success %0, "semaphore wait failed"
    %buffer = hal.buffer_view.buffer %arg2 : !hal.buffer
    %buffer_0 = hal.buffer_view.buffer %arg3 : !hal.buffer
    %1 = call @predict_ex_dispatch_1(%buffer, %buffer_0) : (!hal.buffer, !hal.buffer) -> !hal.buffer
    %view = hal.buffer_view.create %1, element_type = %c50331680_i32, shape = [%c1, %c112, %c112, %c32] : !hal.buffer_view
    hal.semaphore.signal %arg4, value = %arg5
    return %view : !hal.buffer_view
  }
  func @predict_ex_dispatch_1$sync(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export = "predict_ex_dispatch_1", iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dev = hal.ex.shared_device : !hal.device
    %semaphore = hal.semaphore.create %dev, initial_value = %c0 : !hal.semaphore
    %0 = call @predict_ex_dispatch_1$async(%semaphore, %c0, %arg0, %arg1, %semaphore, %c1) : (!hal.semaphore, index, !hal.buffer_view, !hal.buffer_view, !hal.semaphore, index) -> !hal.buffer_view
    %1 = hal.semaphore.await %semaphore, min_value = %c1 : i32
    hal.check_success %1, "semaphore wait failed"
    return %0 : !hal.buffer_view
  }
}


// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c3240000 = constant 3240000 : index
  %c18432 = constant 18432 : index
  %c1 = constant 1 : index
  %c1605632 = constant 1605632 : index
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator %dev : !hal.allocator
  %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
  %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  %executable_layout = hal.executable_layout.lookup %dev, set_layouts = [[#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">]] : !hal.executable_layout
  hal.command_buffer.push_descriptor_set %cmd, %executable_layout, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
  hal.device.switch(%dev : !hal.device)
    #hal.device.match.id<"vulkan*">(%arg2 = %cmd : !hal.command_buffer, %arg3 = %c32 : index, %arg4 = %c112 : index, %arg5 = %c112 : index, %arg6 = %c1 : index) {
    %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg3]
    %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg4]
    %2 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg5]
    %3 = hal.command_buffer.device %arg2 : !hal.device
    %exe = hal.executable.lookup %3, @predict_ex_dispatch_1_dispatch_0 : !hal.executable
    hal.command_buffer.dispatch %arg2, %exe, entry_point = 0, workgroup_xyz = [%0, %1, %2]
    hal.return
  }
  hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
  hal.command_buffer.end %cmd
  hal.ex.submit_and_wait %dev, %cmd
  return %buffer : !hal.buffer
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1$sync(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export = "predict_ex_dispatch_1", iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %dev = hal.ex.shared_device : !hal.device
  %semaphore = hal.semaphore.create %dev, initial_value = %c0 : !hal.semaphore
  %0 = call @predict_ex_dispatch_1$async(%semaphore, %c0, %arg0, %arg1, %semaphore, %c1) : (!hal.semaphore, index, !hal.buffer_view, !hal.buffer_view, !hal.semaphore, index) -> !hal.buffer_view
  %1 = hal.semaphore.await %semaphore, min_value = %c1 : i32
  hal.check_success %1, "semaphore wait failed"
  return %0 : !hal.buffer_view
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1$async(%arg0: !hal.semaphore, %arg1: index, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.semaphore, %arg5: index) -> !hal.buffer_view attributes {iree.module.export = "predict_ex_dispatch_1$async"} {
  %c1 = constant 1 : index
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c50331680_i32 = constant 50331680 : i32
  %0 = hal.semaphore.await %arg0, min_value = %arg1 : i32
  hal.check_success %0, "semaphore wait failed"
  %buffer = hal.buffer_view.buffer %arg2 : !hal.buffer
  %buffer_0 = hal.buffer_view.buffer %arg3 : !hal.buffer
  %1 = call @predict_ex_dispatch_1(%buffer, %buffer_0) : (!hal.buffer, !hal.buffer) -> !hal.buffer
  %view = hal.buffer_view.create %1, element_type = %c50331680_i32, shape = [%c1, %c112, %c112, %c32] : !hal.buffer_view
  hal.semaphore.signal %arg4, value = %arg5
  return %view : !hal.buffer_view
}

// *** IR Dump After mlir::iree_compiler::IREE::HAL::MaterializeResourceCachesPass ***
#map0 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 4)>
module  {
  hal.variable @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !hal.descriptor_set_layout attributes {sym_visibility = "private"}
  func private @_descriptor_set_layout_0_initializer() -> !hal.descriptor_set_layout {
    %dev = hal.ex.shared_device : !hal.device
    %descriptor_set_layout = hal.descriptor_set_layout.create %dev, PushOnly, bindings = [#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">] : !hal.descriptor_set_layout
    return %descriptor_set_layout : !hal.descriptor_set_layout
  }
  hal.variable @_executable_layout_0 init(@_executable_layout_0_initializer) : !hal.executable_layout attributes {sym_visibility = "private"}
  func private @_executable_layout_0_initializer() -> !hal.executable_layout {
    %0 = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
    %dev = hal.ex.shared_device : !hal.device
    %executable_layout = hal.executable_layout.create %dev, push_constants = 0, set_layouts = [%0] : !hal.executable_layout
    return %executable_layout : !hal.executable_layout
  }
  hal.variable @_executable_predict_ex_dispatch_1_dispatch_0 init(@_executable_predict_ex_dispatch_1_dispatch_0_initializer) : !hal.executable attributes {sym_visibility = "private"}
  func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !hal.executable {
    %dev = hal.ex.shared_device : !hal.device
    %0 = hal.device.switch(%dev : !hal.device) -> !hal.executable
    #hal.device.match.id<"vulkan*">(%arg0 = %dev : !hal.device) {
      %1 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
      %exe = hal.executable.create %arg0, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv, layouts = [%1] : !hal.executable
      hal.return %exe : !hal.executable
    },
    #hal.match.always() {
      %1 = iree.null : !hal.executable
      hal.return %1 : !hal.executable
    }
    return %0 : !hal.executable
  }
  hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vulkan_spirv, filter="vulkan*" {
      hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index} {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):  // no predecessors
        %0 = affine.apply #map0()[%arg0]
        %1 = affine.apply #map1()[%arg1]
        %2 = affine.apply #map1()[%arg2]
        hal.return %0, %1, %2 : index, index, index
      }
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
        spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
          spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
            %0 = spv.constant 16 : i32
            %1 = spv.constant 6 : i32
            %2 = spv.constant 1 : i32
            %3 = spv.constant 3 : i32
            %4 = spv.constant 2 : i32
            %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
            %6 = spv.constant 384 : i32
            %7 = spv.constant 128 : i32
            %8 = spv.constant 900 : i32
            %9 = spv.constant 4 : i32
            %10 = spv.constant 0 : i32
            %11 = spv.constant 896 : i32
            %12 = spv.constant 8 : i32
            %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
            %17 = spv.Load "Input" %16 : vector<3xi32>
            %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
            %19 = spv.Load "Input" %16 : vector<3xi32>
            %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
            %21 = spv.Load "Input" %16 : vector<3xi32>
            %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
            %23 = spv.IMul %22, %9 : i32
            %24 = spv.IMul %20, %9 : i32
            %25 = spv.IMul %18, %0 : i32
            %26 = spv.IMul %23, %4 : i32
            %27 = spv.IMul %24, %4 : i32
            %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
            %29 = spv.Load "Input" %28 : vector<3xi32>
            %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
            %31 = spv.Load "Input" %28 : vector<3xi32>
            %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
            %33 = spv.Load "Input" %28 : vector<3xi32>
            %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
            %35 = spv.IMul %34, %9 : i32
            %36 = spv.IMul %30, %9 : i32
            %37 = spv.IMul %34, %12 : i32
            %38 = spv.IMul %32, %4 : i32
            %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            spv.loop {
              spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
              %79 = spv.SLessThan %74, %3 : i32
              spv.BranchConditional %79, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
              %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              spv.loop {
                spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                %94 = spv.SLessThan %89, %3 : i32
                spv.BranchConditional %94, ^bb2, ^bb3
              ^bb2:  // pred: ^bb1
                %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                spv.loop {
                  spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                  %109 = spv.SLessThan %104, %0 : i32
                  spv.BranchConditional %109, ^bb2, ^bb3
                ^bb2:  // pred: ^bb1
                  %110 = spv.IAdd %25, %36 : i32
                  %111 = spv.SDiv %110, %9 : i32
                  %112 = spv.IMul %74, %6 : i32
                  %113 = spv.IMul %89, %7 : i32
                  %114 = spv.IAdd %112, %113 : i32
                  %115 = spv.IMul %104, %12 : i32
                  %116 = spv.IAdd %114, %115 : i32
                  %117 = spv.IAdd %116, %111 : i32
                  %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
                  %120 = spv.IAdd %104, %2 : i32
                  %121 = spv.IMul %120, %12 : i32
                  %122 = spv.IAdd %114, %121 : i32
                  %123 = spv.IAdd %122, %111 : i32
                  %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
                  %126 = spv.IAdd %104, %4 : i32
                  %127 = spv.IMul %126, %12 : i32
                  %128 = spv.IAdd %114, %127 : i32
                  %129 = spv.IAdd %128, %111 : i32
                  %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
                  %132 = spv.IAdd %104, %3 : i32
                  %133 = spv.IMul %132, %12 : i32
                  %134 = spv.IAdd %114, %133 : i32
                  %135 = spv.IAdd %134, %111 : i32
                  %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
                  %138 = spv.IAdd %37, %74 : i32
                  %139 = spv.IAdd %38, %89 : i32
                  %140 = spv.IAdd %26, %138 : i32
                  %141 = spv.IAdd %27, %139 : i32
                  %142 = spv.SDiv %104, %9 : i32
                  %143 = spv.IMul %140, %8 : i32
                  %144 = spv.IMul %141, %9 : i32
                  %145 = spv.IAdd %143, %144 : i32
                  %146 = spv.IAdd %145, %142 : i32
                  %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
                  %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
                  %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
                  %151 = spv.FMul %150, %119 : vector<4xf32>
                  %152 = spv.FAdd %151, %105 : vector<4xf32>
                  %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
                  %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
                  %155 = spv.FMul %154, %125 : vector<4xf32>
                  %156 = spv.FAdd %155, %152 : vector<4xf32>
                  %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
                  %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
                  %159 = spv.FMul %158, %131 : vector<4xf32>
                  %160 = spv.FAdd %159, %156 : vector<4xf32>
                  %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
                  %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
                  %163 = spv.FMul %162, %137 : vector<4xf32>
                  %164 = spv.FAdd %163, %160 : vector<4xf32>
                  %165 = spv.IAdd %74, %4 : i32
                  %166 = spv.IAdd %37, %165 : i32
                  %167 = spv.IAdd %26, %166 : i32
                  %168 = spv.IMul %167, %8 : i32
                  %169 = spv.IAdd %168, %144 : i32
                  %170 = spv.IAdd %169, %142 : i32
                  %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
                  %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
                  %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
                  %175 = spv.FMul %174, %119 : vector<4xf32>
                  %176 = spv.FAdd %175, %106 : vector<4xf32>
                  %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
                  %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
                  %179 = spv.FMul %178, %125 : vector<4xf32>
                  %180 = spv.FAdd %179, %176 : vector<4xf32>
                  %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
                  %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
                  %183 = spv.FMul %182, %131 : vector<4xf32>
                  %184 = spv.FAdd %183, %180 : vector<4xf32>
                  %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
                  %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
                  %187 = spv.FMul %186, %137 : vector<4xf32>
                  %188 = spv.FAdd %187, %184 : vector<4xf32>
                  %189 = spv.IAdd %74, %9 : i32
                  %190 = spv.IAdd %37, %189 : i32
                  %191 = spv.IAdd %26, %190 : i32
                  %192 = spv.IMul %191, %8 : i32
                  %193 = spv.IAdd %192, %144 : i32
                  %194 = spv.IAdd %193, %142 : i32
                  %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
                  %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
                  %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
                  %199 = spv.FMul %198, %119 : vector<4xf32>
                  %200 = spv.FAdd %199, %107 : vector<4xf32>
                  %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
                  %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
                  %203 = spv.FMul %202, %125 : vector<4xf32>
                  %204 = spv.FAdd %203, %200 : vector<4xf32>
                  %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
                  %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
                  %207 = spv.FMul %206, %131 : vector<4xf32>
                  %208 = spv.FAdd %207, %204 : vector<4xf32>
                  %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
                  %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
                  %211 = spv.FMul %210, %137 : vector<4xf32>
                  %212 = spv.FAdd %211, %208 : vector<4xf32>
                  %213 = spv.IAdd %74, %1 : i32
                  %214 = spv.IAdd %37, %213 : i32
                  %215 = spv.IAdd %26, %214 : i32
                  %216 = spv.IMul %215, %8 : i32
                  %217 = spv.IAdd %216, %144 : i32
                  %218 = spv.IAdd %217, %142 : i32
                  %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
                  %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
                  %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
                  %223 = spv.FMul %222, %119 : vector<4xf32>
                  %224 = spv.FAdd %223, %108 : vector<4xf32>
                  %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
                  %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
                  %227 = spv.FMul %226, %125 : vector<4xf32>
                  %228 = spv.FAdd %227, %224 : vector<4xf32>
                  %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
                  %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
                  %231 = spv.FMul %230, %131 : vector<4xf32>
                  %232 = spv.FAdd %231, %228 : vector<4xf32>
                  %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
                  %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
                  %235 = spv.FMul %234, %137 : vector<4xf32>
                  %236 = spv.FAdd %235, %232 : vector<4xf32>
                  spv.Store "Function" %95, %164 : vector<4xf32>
                  spv.Store "Function" %96, %188 : vector<4xf32>
                  spv.Store "Function" %97, %212 : vector<4xf32>
                  spv.Store "Function" %98, %236 : vector<4xf32>
                  %237 = spv.IAdd %104, %9 : i32
                  spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb3:  // pred: ^bb1
                  spv.mlir.merge
                }
                %99 = spv.Load "Function" %98 : vector<4xf32>
                %100 = spv.Load "Function" %97 : vector<4xf32>
                %101 = spv.Load "Function" %96 : vector<4xf32>
                %102 = spv.Load "Function" %95 : vector<4xf32>
                spv.Store "Function" %80, %102 : vector<4xf32>
                spv.Store "Function" %81, %101 : vector<4xf32>
                spv.Store "Function" %82, %100 : vector<4xf32>
                spv.Store "Function" %83, %99 : vector<4xf32>
                %103 = spv.IAdd %89, %2 : i32
                spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb3:  // pred: ^bb1
                spv.mlir.merge
              }
              %84 = spv.Load "Function" %83 : vector<4xf32>
              %85 = spv.Load "Function" %82 : vector<4xf32>
              %86 = spv.Load "Function" %81 : vector<4xf32>
              %87 = spv.Load "Function" %80 : vector<4xf32>
              spv.Store "Function" %39, %87 : vector<4xf32>
              spv.Store "Function" %40, %86 : vector<4xf32>
              spv.Store "Function" %41, %85 : vector<4xf32>
              spv.Store "Function" %42, %84 : vector<4xf32>
              %88 = spv.IAdd %74, %2 : i32
              spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb3:  // pred: ^bb1
              spv.mlir.merge
            }
            %43 = spv.Load "Function" %42 : vector<4xf32>
            %44 = spv.Load "Function" %41 : vector<4xf32>
            %45 = spv.Load "Function" %40 : vector<4xf32>
            %46 = spv.Load "Function" %39 : vector<4xf32>
            %47 = spv.IAdd %35, %3 : i32
            %48 = spv.IAdd %23, %47 : i32
            %49 = spv.IAdd %24, %32 : i32
            %50 = spv.IAdd %25, %36 : i32
            %51 = spv.SDiv %50, %9 : i32
            %52 = spv.IMul %48, %11 : i32
            %53 = spv.IMul %49, %12 : i32
            %54 = spv.IAdd %52, %53 : i32
            %55 = spv.IAdd %54, %51 : i32
            %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
            %57 = spv.IAdd %35, %4 : i32
            %58 = spv.IAdd %23, %57 : i32
            %59 = spv.IMul %58, %11 : i32
            %60 = spv.IAdd %59, %53 : i32
            %61 = spv.IAdd %60, %51 : i32
            %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
            %63 = spv.IAdd %35, %2 : i32
            %64 = spv.IAdd %23, %63 : i32
            %65 = spv.IMul %64, %11 : i32
            %66 = spv.IAdd %65, %53 : i32
            %67 = spv.IAdd %66, %51 : i32
            %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
            %69 = spv.IAdd %23, %35 : i32
            %70 = spv.IMul %69, %11 : i32
            %71 = spv.IAdd %70, %53 : i32
            %72 = spv.IAdd %71, %51 : i32
            %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
            spv.Return
          }
          spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
          spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
        }
        hal.interface @legacy_io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c3240000 = constant 3240000 : index
    %c18432 = constant 18432 : index
    %c1 = constant 1 : index
    %c1605632 = constant 1605632 : index
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %dev = hal.ex.shared_device : !hal.device
    %allocator = hal.device.allocator %dev : !hal.allocator
    %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
    %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
    hal.command_buffer.begin %cmd
    %0 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
    hal.command_buffer.push_descriptor_set %cmd, %0, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
    hal.device.switch(%dev : !hal.device)
    #hal.device.match.id<"vulkan*">(%arg2 = %cmd : !hal.command_buffer, %arg3 = %c32 : index, %arg4 = %c112 : index, %arg5 = %c112 : index, %arg6 = %c1 : index) {
      %1 = affine.apply #map0()[%arg3]
      %2 = affine.apply #map1()[%arg4]
      %3 = affine.apply #map1()[%arg5]
      %4 = hal.command_buffer.device %arg2 : !hal.device
      %5 = hal.variable.load @_executable_predict_ex_dispatch_1_dispatch_0 : !hal.executable
      hal.command_buffer.dispatch %arg2, %5, entry_point = 0, workgroup_xyz = [%1, %2, %3]
      hal.return
    }
    hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
    hal.command_buffer.end %cmd
    hal.ex.submit_and_wait %dev, %cmd
    return %buffer : !hal.buffer
  }
  func @predict_ex_dispatch_1$async(%arg0: !hal.semaphore, %arg1: index, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.semaphore, %arg5: index) -> !hal.buffer_view attributes {iree.module.export = "predict_ex_dispatch_1$async"} {
    %c1 = constant 1 : index
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c50331680_i32 = constant 50331680 : i32
    %0 = hal.semaphore.await %arg0, min_value = %arg1 : i32
    hal.check_success %0, "semaphore wait failed"
    %buffer = hal.buffer_view.buffer %arg2 : !hal.buffer
    %buffer_0 = hal.buffer_view.buffer %arg3 : !hal.buffer
    %1 = call @predict_ex_dispatch_1(%buffer, %buffer_0) : (!hal.buffer, !hal.buffer) -> !hal.buffer
    %view = hal.buffer_view.create %1, element_type = %c50331680_i32, shape = [%c1, %c112, %c112, %c32] : !hal.buffer_view
    hal.semaphore.signal %arg4, value = %arg5
    return %view : !hal.buffer_view
  }
  func @predict_ex_dispatch_1$sync(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export = "predict_ex_dispatch_1", iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dev = hal.ex.shared_device : !hal.device
    %semaphore = hal.semaphore.create %dev, initial_value = %c0 : !hal.semaphore
    %0 = call @predict_ex_dispatch_1$async(%semaphore, %c0, %arg0, %arg1, %semaphore, %c1) : (!hal.semaphore, index, !hal.buffer_view, !hal.buffer_view, !hal.semaphore, index) -> !hal.buffer_view
    %1 = hal.semaphore.await %semaphore, min_value = %c1 : i32
    hal.check_success %1, "semaphore wait failed"
    return %0 : !hal.buffer_view
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::HAL::InlineDeviceSwitchesPass ***
func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c3240000 = constant 3240000 : index
  %c18432 = constant 18432 : index
  %c1 = constant 1 : index
  %c1605632 = constant 1605632 : index
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator %dev : !hal.allocator
  %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
  %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  %0 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
  hal.command_buffer.push_descriptor_set %cmd, %0, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
  %1 = hal.device.match.id %dev, pattern = ["vulkan*"] : (!hal.device) -> i1
  cond_br %1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %2 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%c32]
  %3 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%c112]
  %4 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%c112]
  %5 = hal.command_buffer.device %cmd : !hal.device
  %6 = hal.variable.load @_executable_predict_ex_dispatch_1_dispatch_0 : !hal.executable
  hal.command_buffer.dispatch %cmd, %6, entry_point = 0, workgroup_xyz = [%2, %3, %4]
  br ^bb3
^bb2:  // pred: ^bb0
  iree.unreachable
^bb3:  // pred: ^bb1
  hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
  hal.command_buffer.end %cmd
  hal.ex.submit_and_wait %dev, %cmd
  return %buffer : !hal.buffer
}

// *** IR Dump After mlir::iree_compiler::IREE::HAL::InlineDeviceSwitchesPass ***
func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !hal.executable {
  %dev = hal.ex.shared_device : !hal.device
  %0 = hal.device.match.id %dev, pattern = ["vulkan*"] : (!hal.device) -> i1
  cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %1 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
  %exe = hal.executable.create %dev, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv, layouts = [%1] : !hal.executable
  br ^bb5(%exe : !hal.executable)
^bb2:  // pred: ^bb0
  %true = constant true
  cond_br %true, ^bb3, ^bb4
^bb3:  // pred: ^bb2
  %2 = iree.null : !hal.executable
  br ^bb5(%2 : !hal.executable)
^bb4:  // pred: ^bb2
  iree.unreachable
^bb5(%3: !hal.executable):  // 2 preds: ^bb1, ^bb3
  return %3 : !hal.executable
}

// *** IR Dump After ConvertAffineToStandard ***
module  {
  hal.variable @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !hal.descriptor_set_layout attributes {sym_visibility = "private"}
  func private @_descriptor_set_layout_0_initializer() -> !hal.descriptor_set_layout {
    %dev = hal.ex.shared_device : !hal.device
    %descriptor_set_layout = hal.descriptor_set_layout.create %dev, PushOnly, bindings = [#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">] : !hal.descriptor_set_layout
    return %descriptor_set_layout : !hal.descriptor_set_layout
  }
  hal.variable @_executable_layout_0 init(@_executable_layout_0_initializer) : !hal.executable_layout attributes {sym_visibility = "private"}
  func private @_executable_layout_0_initializer() -> !hal.executable_layout {
    %0 = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
    %dev = hal.ex.shared_device : !hal.device
    %executable_layout = hal.executable_layout.create %dev, push_constants = 0, set_layouts = [%0] : !hal.executable_layout
    return %executable_layout : !hal.executable_layout
  }
  hal.variable @_executable_predict_ex_dispatch_1_dispatch_0 init(@_executable_predict_ex_dispatch_1_dispatch_0_initializer) : !hal.executable attributes {sym_visibility = "private"}
  func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !hal.executable {
    %dev = hal.ex.shared_device : !hal.device
    %0 = hal.device.match.id %dev, pattern = ["vulkan*"] : (!hal.device) -> i1
    cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %1 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
    %exe = hal.executable.create %dev, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv, layouts = [%1] : !hal.executable
    br ^bb5(%exe : !hal.executable)
  ^bb2:  // pred: ^bb0
    %true = constant true
    cond_br %true, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %2 = iree.null : !hal.executable
    br ^bb5(%2 : !hal.executable)
  ^bb4:  // pred: ^bb2
    iree.unreachable
  ^bb5(%3: !hal.executable):  // 2 preds: ^bb1, ^bb3
    return %3 : !hal.executable
  }
  hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vulkan_spirv, filter="vulkan*" {
      hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index} {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):  // no predecessors
        %c16 = constant 16 : index
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %0 = cmpi sle, %arg0, %c0 : index
        %1 = subi %c0, %arg0 : index
        %2 = subi %arg0, %c1 : index
        %3 = select %0, %1, %2 : index
        %4 = divi_signed %3, %c16 : index
        %5 = subi %c0, %4 : index
        %6 = addi %4, %c1 : index
        %7 = select %0, %5, %6 : index
        %c4 = constant 4 : index
        %c0_0 = constant 0 : index
        %c1_1 = constant 1 : index
        %8 = cmpi sle, %arg1, %c0_0 : index
        %9 = subi %c0_0, %arg1 : index
        %10 = subi %arg1, %c1_1 : index
        %11 = select %8, %9, %10 : index
        %12 = divi_signed %11, %c4 : index
        %13 = subi %c0_0, %12 : index
        %14 = addi %12, %c1_1 : index
        %15 = select %8, %13, %14 : index
        %c4_2 = constant 4 : index
        %c0_3 = constant 0 : index
        %c1_4 = constant 1 : index
        %16 = cmpi sle, %arg2, %c0_3 : index
        %17 = subi %c0_3, %arg2 : index
        %18 = subi %arg2, %c1_4 : index
        %19 = select %16, %17, %18 : index
        %20 = divi_signed %19, %c4_2 : index
        %21 = subi %c0_3, %20 : index
        %22 = addi %20, %c1_4 : index
        %23 = select %16, %21, %22 : index
        hal.return %7, %15, %23 : index, index, index
      }
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
        spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
          spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
            %0 = spv.constant 16 : i32
            %1 = spv.constant 6 : i32
            %2 = spv.constant 1 : i32
            %3 = spv.constant 3 : i32
            %4 = spv.constant 2 : i32
            %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
            %6 = spv.constant 384 : i32
            %7 = spv.constant 128 : i32
            %8 = spv.constant 900 : i32
            %9 = spv.constant 4 : i32
            %10 = spv.constant 0 : i32
            %11 = spv.constant 896 : i32
            %12 = spv.constant 8 : i32
            %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
            %17 = spv.Load "Input" %16 : vector<3xi32>
            %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
            %19 = spv.Load "Input" %16 : vector<3xi32>
            %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
            %21 = spv.Load "Input" %16 : vector<3xi32>
            %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
            %23 = spv.IMul %22, %9 : i32
            %24 = spv.IMul %20, %9 : i32
            %25 = spv.IMul %18, %0 : i32
            %26 = spv.IMul %23, %4 : i32
            %27 = spv.IMul %24, %4 : i32
            %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
            %29 = spv.Load "Input" %28 : vector<3xi32>
            %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
            %31 = spv.Load "Input" %28 : vector<3xi32>
            %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
            %33 = spv.Load "Input" %28 : vector<3xi32>
            %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
            %35 = spv.IMul %34, %9 : i32
            %36 = spv.IMul %30, %9 : i32
            %37 = spv.IMul %34, %12 : i32
            %38 = spv.IMul %32, %4 : i32
            %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            spv.loop {
              spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
              %79 = spv.SLessThan %74, %3 : i32
              spv.BranchConditional %79, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
              %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              spv.loop {
                spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                %94 = spv.SLessThan %89, %3 : i32
                spv.BranchConditional %94, ^bb2, ^bb3
              ^bb2:  // pred: ^bb1
                %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                spv.loop {
                  spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                  %109 = spv.SLessThan %104, %0 : i32
                  spv.BranchConditional %109, ^bb2, ^bb3
                ^bb2:  // pred: ^bb1
                  %110 = spv.IAdd %25, %36 : i32
                  %111 = spv.SDiv %110, %9 : i32
                  %112 = spv.IMul %74, %6 : i32
                  %113 = spv.IMul %89, %7 : i32
                  %114 = spv.IAdd %112, %113 : i32
                  %115 = spv.IMul %104, %12 : i32
                  %116 = spv.IAdd %114, %115 : i32
                  %117 = spv.IAdd %116, %111 : i32
                  %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
                  %120 = spv.IAdd %104, %2 : i32
                  %121 = spv.IMul %120, %12 : i32
                  %122 = spv.IAdd %114, %121 : i32
                  %123 = spv.IAdd %122, %111 : i32
                  %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
                  %126 = spv.IAdd %104, %4 : i32
                  %127 = spv.IMul %126, %12 : i32
                  %128 = spv.IAdd %114, %127 : i32
                  %129 = spv.IAdd %128, %111 : i32
                  %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
                  %132 = spv.IAdd %104, %3 : i32
                  %133 = spv.IMul %132, %12 : i32
                  %134 = spv.IAdd %114, %133 : i32
                  %135 = spv.IAdd %134, %111 : i32
                  %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
                  %138 = spv.IAdd %37, %74 : i32
                  %139 = spv.IAdd %38, %89 : i32
                  %140 = spv.IAdd %26, %138 : i32
                  %141 = spv.IAdd %27, %139 : i32
                  %142 = spv.SDiv %104, %9 : i32
                  %143 = spv.IMul %140, %8 : i32
                  %144 = spv.IMul %141, %9 : i32
                  %145 = spv.IAdd %143, %144 : i32
                  %146 = spv.IAdd %145, %142 : i32
                  %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
                  %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
                  %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
                  %151 = spv.FMul %150, %119 : vector<4xf32>
                  %152 = spv.FAdd %151, %105 : vector<4xf32>
                  %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
                  %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
                  %155 = spv.FMul %154, %125 : vector<4xf32>
                  %156 = spv.FAdd %155, %152 : vector<4xf32>
                  %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
                  %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
                  %159 = spv.FMul %158, %131 : vector<4xf32>
                  %160 = spv.FAdd %159, %156 : vector<4xf32>
                  %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
                  %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
                  %163 = spv.FMul %162, %137 : vector<4xf32>
                  %164 = spv.FAdd %163, %160 : vector<4xf32>
                  %165 = spv.IAdd %74, %4 : i32
                  %166 = spv.IAdd %37, %165 : i32
                  %167 = spv.IAdd %26, %166 : i32
                  %168 = spv.IMul %167, %8 : i32
                  %169 = spv.IAdd %168, %144 : i32
                  %170 = spv.IAdd %169, %142 : i32
                  %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
                  %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
                  %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
                  %175 = spv.FMul %174, %119 : vector<4xf32>
                  %176 = spv.FAdd %175, %106 : vector<4xf32>
                  %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
                  %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
                  %179 = spv.FMul %178, %125 : vector<4xf32>
                  %180 = spv.FAdd %179, %176 : vector<4xf32>
                  %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
                  %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
                  %183 = spv.FMul %182, %131 : vector<4xf32>
                  %184 = spv.FAdd %183, %180 : vector<4xf32>
                  %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
                  %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
                  %187 = spv.FMul %186, %137 : vector<4xf32>
                  %188 = spv.FAdd %187, %184 : vector<4xf32>
                  %189 = spv.IAdd %74, %9 : i32
                  %190 = spv.IAdd %37, %189 : i32
                  %191 = spv.IAdd %26, %190 : i32
                  %192 = spv.IMul %191, %8 : i32
                  %193 = spv.IAdd %192, %144 : i32
                  %194 = spv.IAdd %193, %142 : i32
                  %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
                  %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
                  %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
                  %199 = spv.FMul %198, %119 : vector<4xf32>
                  %200 = spv.FAdd %199, %107 : vector<4xf32>
                  %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
                  %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
                  %203 = spv.FMul %202, %125 : vector<4xf32>
                  %204 = spv.FAdd %203, %200 : vector<4xf32>
                  %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
                  %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
                  %207 = spv.FMul %206, %131 : vector<4xf32>
                  %208 = spv.FAdd %207, %204 : vector<4xf32>
                  %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
                  %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
                  %211 = spv.FMul %210, %137 : vector<4xf32>
                  %212 = spv.FAdd %211, %208 : vector<4xf32>
                  %213 = spv.IAdd %74, %1 : i32
                  %214 = spv.IAdd %37, %213 : i32
                  %215 = spv.IAdd %26, %214 : i32
                  %216 = spv.IMul %215, %8 : i32
                  %217 = spv.IAdd %216, %144 : i32
                  %218 = spv.IAdd %217, %142 : i32
                  %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
                  %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
                  %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
                  %223 = spv.FMul %222, %119 : vector<4xf32>
                  %224 = spv.FAdd %223, %108 : vector<4xf32>
                  %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
                  %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
                  %227 = spv.FMul %226, %125 : vector<4xf32>
                  %228 = spv.FAdd %227, %224 : vector<4xf32>
                  %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
                  %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
                  %231 = spv.FMul %230, %131 : vector<4xf32>
                  %232 = spv.FAdd %231, %228 : vector<4xf32>
                  %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
                  %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
                  %235 = spv.FMul %234, %137 : vector<4xf32>
                  %236 = spv.FAdd %235, %232 : vector<4xf32>
                  spv.Store "Function" %95, %164 : vector<4xf32>
                  spv.Store "Function" %96, %188 : vector<4xf32>
                  spv.Store "Function" %97, %212 : vector<4xf32>
                  spv.Store "Function" %98, %236 : vector<4xf32>
                  %237 = spv.IAdd %104, %9 : i32
                  spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb3:  // pred: ^bb1
                  spv.mlir.merge
                }
                %99 = spv.Load "Function" %98 : vector<4xf32>
                %100 = spv.Load "Function" %97 : vector<4xf32>
                %101 = spv.Load "Function" %96 : vector<4xf32>
                %102 = spv.Load "Function" %95 : vector<4xf32>
                spv.Store "Function" %80, %102 : vector<4xf32>
                spv.Store "Function" %81, %101 : vector<4xf32>
                spv.Store "Function" %82, %100 : vector<4xf32>
                spv.Store "Function" %83, %99 : vector<4xf32>
                %103 = spv.IAdd %89, %2 : i32
                spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb3:  // pred: ^bb1
                spv.mlir.merge
              }
              %84 = spv.Load "Function" %83 : vector<4xf32>
              %85 = spv.Load "Function" %82 : vector<4xf32>
              %86 = spv.Load "Function" %81 : vector<4xf32>
              %87 = spv.Load "Function" %80 : vector<4xf32>
              spv.Store "Function" %39, %87 : vector<4xf32>
              spv.Store "Function" %40, %86 : vector<4xf32>
              spv.Store "Function" %41, %85 : vector<4xf32>
              spv.Store "Function" %42, %84 : vector<4xf32>
              %88 = spv.IAdd %74, %2 : i32
              spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb3:  // pred: ^bb1
              spv.mlir.merge
            }
            %43 = spv.Load "Function" %42 : vector<4xf32>
            %44 = spv.Load "Function" %41 : vector<4xf32>
            %45 = spv.Load "Function" %40 : vector<4xf32>
            %46 = spv.Load "Function" %39 : vector<4xf32>
            %47 = spv.IAdd %35, %3 : i32
            %48 = spv.IAdd %23, %47 : i32
            %49 = spv.IAdd %24, %32 : i32
            %50 = spv.IAdd %25, %36 : i32
            %51 = spv.SDiv %50, %9 : i32
            %52 = spv.IMul %48, %11 : i32
            %53 = spv.IMul %49, %12 : i32
            %54 = spv.IAdd %52, %53 : i32
            %55 = spv.IAdd %54, %51 : i32
            %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
            %57 = spv.IAdd %35, %4 : i32
            %58 = spv.IAdd %23, %57 : i32
            %59 = spv.IMul %58, %11 : i32
            %60 = spv.IAdd %59, %53 : i32
            %61 = spv.IAdd %60, %51 : i32
            %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
            %63 = spv.IAdd %35, %2 : i32
            %64 = spv.IAdd %23, %63 : i32
            %65 = spv.IMul %64, %11 : i32
            %66 = spv.IAdd %65, %53 : i32
            %67 = spv.IAdd %66, %51 : i32
            %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
            %69 = spv.IAdd %23, %35 : i32
            %70 = spv.IMul %69, %11 : i32
            %71 = spv.IAdd %70, %53 : i32
            %72 = spv.IAdd %71, %51 : i32
            %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
            spv.Return
          }
          spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
          spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
        }
        hal.interface @legacy_io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c3240000 = constant 3240000 : index
    %c18432 = constant 18432 : index
    %c1 = constant 1 : index
    %c1605632 = constant 1605632 : index
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %dev = hal.ex.shared_device : !hal.device
    %allocator = hal.device.allocator %dev : !hal.allocator
    %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
    %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
    hal.command_buffer.begin %cmd
    %0 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
    hal.command_buffer.push_descriptor_set %cmd, %0, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
    %1 = hal.device.match.id %dev, pattern = ["vulkan*"] : (!hal.device) -> i1
    cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %c2_0 = constant 2 : index
    %c28 = constant 28 : index
    %c28_1 = constant 28 : index
    %2 = hal.command_buffer.device %cmd : !hal.device
    %3 = hal.variable.load @_executable_predict_ex_dispatch_1_dispatch_0 : !hal.executable
    hal.command_buffer.dispatch %cmd, %3, entry_point = 0, workgroup_xyz = [%c2_0, %c28, %c28_1]
    br ^bb3
  ^bb2:  // pred: ^bb0
    iree.unreachable
  ^bb3:  // pred: ^bb1
    hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
    hal.command_buffer.end %cmd
    hal.ex.submit_and_wait %dev, %cmd
    return %buffer : !hal.buffer
  }
  func @predict_ex_dispatch_1$async(%arg0: !hal.semaphore, %arg1: index, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.semaphore, %arg5: index) -> !hal.buffer_view attributes {iree.module.export = "predict_ex_dispatch_1$async"} {
    %c1 = constant 1 : index
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c50331680_i32 = constant 50331680 : i32
    %0 = hal.semaphore.await %arg0, min_value = %arg1 : i32
    hal.check_success %0, "semaphore wait failed"
    %buffer = hal.buffer_view.buffer %arg2 : !hal.buffer
    %buffer_0 = hal.buffer_view.buffer %arg3 : !hal.buffer
    %1 = call @predict_ex_dispatch_1(%buffer, %buffer_0) : (!hal.buffer, !hal.buffer) -> !hal.buffer
    %view = hal.buffer_view.create %1, element_type = %c50331680_i32, shape = [%c1, %c112, %c112, %c32] : !hal.buffer_view
    hal.semaphore.signal %arg4, value = %arg5
    return %view : !hal.buffer_view
  }
  func @predict_ex_dispatch_1$sync(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export = "predict_ex_dispatch_1", iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dev = hal.ex.shared_device : !hal.device
    %semaphore = hal.semaphore.create %dev, initial_value = %c0 : !hal.semaphore
    %0 = call @predict_ex_dispatch_1$async(%semaphore, %c0, %arg0, %arg1, %semaphore, %c1) : (!hal.semaphore, index, !hal.buffer_view, !hal.buffer_view, !hal.semaphore, index) -> !hal.buffer_view
    %1 = hal.semaphore.await %semaphore, min_value = %c1 : i32
    hal.check_success %1, "semaphore wait failed"
    return %0 : !hal.buffer_view
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::HAL::MemoizeDeviceQueriesPass ***
module  {
  hal.variable @_device_match_id_0 init(@_device_match_id_0_initializer) : i1 attributes {sym_visibility = "private"}
  func private @_device_match_id_0_initializer() -> i1 {
    %dev = hal.ex.shared_device : !hal.device
    %0 = hal.device.match.id %dev, pattern = ["vulkan*"] : (!hal.device) -> i1
    return %0 : i1
  }
  hal.variable @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !hal.descriptor_set_layout attributes {sym_visibility = "private"}
  func private @_descriptor_set_layout_0_initializer() -> !hal.descriptor_set_layout {
    %dev = hal.ex.shared_device : !hal.device
    %descriptor_set_layout = hal.descriptor_set_layout.create %dev, PushOnly, bindings = [#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">] : !hal.descriptor_set_layout
    return %descriptor_set_layout : !hal.descriptor_set_layout
  }
  hal.variable @_executable_layout_0 init(@_executable_layout_0_initializer) : !hal.executable_layout attributes {sym_visibility = "private"}
  func private @_executable_layout_0_initializer() -> !hal.executable_layout {
    %0 = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
    %dev = hal.ex.shared_device : !hal.device
    %executable_layout = hal.executable_layout.create %dev, push_constants = 0, set_layouts = [%0] : !hal.executable_layout
    return %executable_layout : !hal.executable_layout
  }
  hal.variable @_executable_predict_ex_dispatch_1_dispatch_0 init(@_executable_predict_ex_dispatch_1_dispatch_0_initializer) : !hal.executable attributes {sym_visibility = "private"}
  func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !hal.executable {
    %dev = hal.ex.shared_device : !hal.device
    %0 = hal.variable.load @_device_match_id_0 : i1
    cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %1 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
    %exe = hal.executable.create %dev, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv, layouts = [%1] : !hal.executable
    br ^bb5(%exe : !hal.executable)
  ^bb2:  // pred: ^bb0
    %true = constant true
    cond_br %true, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %2 = iree.null : !hal.executable
    br ^bb5(%2 : !hal.executable)
  ^bb4:  // pred: ^bb2
    iree.unreachable
  ^bb5(%3: !hal.executable):  // 2 preds: ^bb1, ^bb3
    return %3 : !hal.executable
  }
  hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vulkan_spirv, filter="vulkan*" {
      hal.executable.entry_point @predict_ex_dispatch_1_dispatch_0 attributes {interface = @legacy_io, num_workgroups = [2, 28, 28, 0], ordinal = 0 : i32, signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> (), workload_rank = 4 : index} {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):  // no predecessors
        %c16 = constant 16 : index
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %0 = cmpi sle, %arg0, %c0 : index
        %1 = subi %c0, %arg0 : index
        %2 = subi %arg0, %c1 : index
        %3 = select %0, %1, %2 : index
        %4 = divi_signed %3, %c16 : index
        %5 = subi %c0, %4 : index
        %6 = addi %4, %c1 : index
        %7 = select %0, %5, %6 : index
        %c4 = constant 4 : index
        %c0_0 = constant 0 : index
        %c1_1 = constant 1 : index
        %8 = cmpi sle, %arg1, %c0_0 : index
        %9 = subi %c0_0, %arg1 : index
        %10 = subi %arg1, %c1_1 : index
        %11 = select %8, %9, %10 : index
        %12 = divi_signed %11, %c4 : index
        %13 = subi %c0_0, %12 : index
        %14 = addi %12, %c1_1 : index
        %15 = select %8, %13, %14 : index
        %c4_2 = constant 4 : index
        %c0_3 = constant 0 : index
        %c1_4 = constant 1 : index
        %16 = cmpi sle, %arg2, %c0_3 : index
        %17 = subi %c0_3, %arg2 : index
        %18 = subi %arg2, %c1_4 : index
        %19 = select %16, %17, %18 : index
        %20 = divi_signed %19, %c4_2 : index
        %21 = subi %c0_3, %20 : index
        %22 = addi %20, %c1_4 : index
        %23 = select %16, %21, %22 : index
        hal.return %7, %15, %23 : index, index, index
      }
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
        spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
          spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
          spv.globalVariable @__resource_var_24889313392296__ bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313408872__ bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.globalVariable @__resource_var_24889313392104__ bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spv.func @predict_ex_dispatch_1_dispatch_0() "None" {
            %0 = spv.constant 16 : i32
            %1 = spv.constant 6 : i32
            %2 = spv.constant 1 : i32
            %3 = spv.constant 3 : i32
            %4 = spv.constant 2 : i32
            %5 = spv.constant dense<0.000000e+00> : vector<4xf32>
            %6 = spv.constant 384 : i32
            %7 = spv.constant 128 : i32
            %8 = spv.constant 900 : i32
            %9 = spv.constant 4 : i32
            %10 = spv.constant 0 : i32
            %11 = spv.constant 896 : i32
            %12 = spv.constant 8 : i32
            %13 = spv.mlir.addressof @__resource_var_24889313392104__ : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %14 = spv.mlir.addressof @__resource_var_24889313408872__ : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %15 = spv.mlir.addressof @__resource_var_24889313392296__ : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %16 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
            %17 = spv.Load "Input" %16 : vector<3xi32>
            %18 = spv.CompositeExtract %17[0 : i32] : vector<3xi32>
            %19 = spv.Load "Input" %16 : vector<3xi32>
            %20 = spv.CompositeExtract %19[1 : i32] : vector<3xi32>
            %21 = spv.Load "Input" %16 : vector<3xi32>
            %22 = spv.CompositeExtract %21[2 : i32] : vector<3xi32>
            %23 = spv.IMul %22, %9 : i32
            %24 = spv.IMul %20, %9 : i32
            %25 = spv.IMul %18, %0 : i32
            %26 = spv.IMul %23, %4 : i32
            %27 = spv.IMul %24, %4 : i32
            %28 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
            %29 = spv.Load "Input" %28 : vector<3xi32>
            %30 = spv.CompositeExtract %29[0 : i32] : vector<3xi32>
            %31 = spv.Load "Input" %28 : vector<3xi32>
            %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
            %33 = spv.Load "Input" %28 : vector<3xi32>
            %34 = spv.CompositeExtract %33[2 : i32] : vector<3xi32>
            %35 = spv.IMul %34, %9 : i32
            %36 = spv.IMul %30, %9 : i32
            %37 = spv.IMul %34, %12 : i32
            %38 = spv.IMul %32, %4 : i32
            %39 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %40 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %41 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            %42 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
            spv.loop {
              spv.Branch ^bb1(%10, %5, %5, %5, %5 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb1(%74: i32, %75: vector<4xf32>, %76: vector<4xf32>, %77: vector<4xf32>, %78: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
              %79 = spv.SLessThan %74, %3 : i32
              spv.BranchConditional %79, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
              %80 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %81 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %82 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              %83 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
              spv.loop {
                spv.Branch ^bb1(%10, %75, %76, %77, %78 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb1(%89: i32, %90: vector<4xf32>, %91: vector<4xf32>, %92: vector<4xf32>, %93: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                %94 = spv.SLessThan %89, %3 : i32
                spv.BranchConditional %94, ^bb2, ^bb3
              ^bb2:  // pred: ^bb1
                %95 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %96 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %97 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                %98 = spv.Variable : !spv.ptr<vector<4xf32>, Function>
                spv.loop {
                  spv.Branch ^bb1(%10, %90, %91, %92, %93 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb1(%104: i32, %105: vector<4xf32>, %106: vector<4xf32>, %107: vector<4xf32>, %108: vector<4xf32>):  // 2 preds: ^bb0, ^bb2
                  %109 = spv.SLessThan %104, %0 : i32
                  spv.BranchConditional %109, ^bb2, ^bb3
                ^bb2:  // pred: ^bb1
                  %110 = spv.IAdd %25, %36 : i32
                  %111 = spv.SDiv %110, %9 : i32
                  %112 = spv.IMul %74, %6 : i32
                  %113 = spv.IMul %89, %7 : i32
                  %114 = spv.IAdd %112, %113 : i32
                  %115 = spv.IMul %104, %12 : i32
                  %116 = spv.IAdd %114, %115 : i32
                  %117 = spv.IAdd %116, %111 : i32
                  %118 = spv.AccessChain %14[%10, %117] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %119 = spv.Load "StorageBuffer" %118 : vector<4xf32>
                  %120 = spv.IAdd %104, %2 : i32
                  %121 = spv.IMul %120, %12 : i32
                  %122 = spv.IAdd %114, %121 : i32
                  %123 = spv.IAdd %122, %111 : i32
                  %124 = spv.AccessChain %14[%10, %123] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %125 = spv.Load "StorageBuffer" %124 : vector<4xf32>
                  %126 = spv.IAdd %104, %4 : i32
                  %127 = spv.IMul %126, %12 : i32
                  %128 = spv.IAdd %114, %127 : i32
                  %129 = spv.IAdd %128, %111 : i32
                  %130 = spv.AccessChain %14[%10, %129] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %131 = spv.Load "StorageBuffer" %130 : vector<4xf32>
                  %132 = spv.IAdd %104, %3 : i32
                  %133 = spv.IMul %132, %12 : i32
                  %134 = spv.IAdd %114, %133 : i32
                  %135 = spv.IAdd %134, %111 : i32
                  %136 = spv.AccessChain %14[%10, %135] : !spv.ptr<!spv.struct<(!spv.array<1152 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %137 = spv.Load "StorageBuffer" %136 : vector<4xf32>
                  %138 = spv.IAdd %37, %74 : i32
                  %139 = spv.IAdd %38, %89 : i32
                  %140 = spv.IAdd %26, %138 : i32
                  %141 = spv.IAdd %27, %139 : i32
                  %142 = spv.SDiv %104, %9 : i32
                  %143 = spv.IMul %140, %8 : i32
                  %144 = spv.IMul %141, %9 : i32
                  %145 = spv.IAdd %143, %144 : i32
                  %146 = spv.IAdd %145, %142 : i32
                  %147 = spv.AccessChain %13[%10, %146] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %148 = spv.Load "StorageBuffer" %147 : vector<4xf32>
                  %149 = spv.CompositeExtract %148[0 : i32] : vector<4xf32>
                  %150 = spv.CompositeConstruct %149, %149, %149, %149 : vector<4xf32>
                  %151 = spv.FMul %150, %119 : vector<4xf32>
                  %152 = spv.FAdd %151, %105 : vector<4xf32>
                  %153 = spv.CompositeExtract %148[1 : i32] : vector<4xf32>
                  %154 = spv.CompositeConstruct %153, %153, %153, %153 : vector<4xf32>
                  %155 = spv.FMul %154, %125 : vector<4xf32>
                  %156 = spv.FAdd %155, %152 : vector<4xf32>
                  %157 = spv.CompositeExtract %148[2 : i32] : vector<4xf32>
                  %158 = spv.CompositeConstruct %157, %157, %157, %157 : vector<4xf32>
                  %159 = spv.FMul %158, %131 : vector<4xf32>
                  %160 = spv.FAdd %159, %156 : vector<4xf32>
                  %161 = spv.CompositeExtract %148[3 : i32] : vector<4xf32>
                  %162 = spv.CompositeConstruct %161, %161, %161, %161 : vector<4xf32>
                  %163 = spv.FMul %162, %137 : vector<4xf32>
                  %164 = spv.FAdd %163, %160 : vector<4xf32>
                  %165 = spv.IAdd %74, %4 : i32
                  %166 = spv.IAdd %37, %165 : i32
                  %167 = spv.IAdd %26, %166 : i32
                  %168 = spv.IMul %167, %8 : i32
                  %169 = spv.IAdd %168, %144 : i32
                  %170 = spv.IAdd %169, %142 : i32
                  %171 = spv.AccessChain %13[%10, %170] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %172 = spv.Load "StorageBuffer" %171 : vector<4xf32>
                  %173 = spv.CompositeExtract %172[0 : i32] : vector<4xf32>
                  %174 = spv.CompositeConstruct %173, %173, %173, %173 : vector<4xf32>
                  %175 = spv.FMul %174, %119 : vector<4xf32>
                  %176 = spv.FAdd %175, %106 : vector<4xf32>
                  %177 = spv.CompositeExtract %172[1 : i32] : vector<4xf32>
                  %178 = spv.CompositeConstruct %177, %177, %177, %177 : vector<4xf32>
                  %179 = spv.FMul %178, %125 : vector<4xf32>
                  %180 = spv.FAdd %179, %176 : vector<4xf32>
                  %181 = spv.CompositeExtract %172[2 : i32] : vector<4xf32>
                  %182 = spv.CompositeConstruct %181, %181, %181, %181 : vector<4xf32>
                  %183 = spv.FMul %182, %131 : vector<4xf32>
                  %184 = spv.FAdd %183, %180 : vector<4xf32>
                  %185 = spv.CompositeExtract %172[3 : i32] : vector<4xf32>
                  %186 = spv.CompositeConstruct %185, %185, %185, %185 : vector<4xf32>
                  %187 = spv.FMul %186, %137 : vector<4xf32>
                  %188 = spv.FAdd %187, %184 : vector<4xf32>
                  %189 = spv.IAdd %74, %9 : i32
                  %190 = spv.IAdd %37, %189 : i32
                  %191 = spv.IAdd %26, %190 : i32
                  %192 = spv.IMul %191, %8 : i32
                  %193 = spv.IAdd %192, %144 : i32
                  %194 = spv.IAdd %193, %142 : i32
                  %195 = spv.AccessChain %13[%10, %194] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %196 = spv.Load "StorageBuffer" %195 : vector<4xf32>
                  %197 = spv.CompositeExtract %196[0 : i32] : vector<4xf32>
                  %198 = spv.CompositeConstruct %197, %197, %197, %197 : vector<4xf32>
                  %199 = spv.FMul %198, %119 : vector<4xf32>
                  %200 = spv.FAdd %199, %107 : vector<4xf32>
                  %201 = spv.CompositeExtract %196[1 : i32] : vector<4xf32>
                  %202 = spv.CompositeConstruct %201, %201, %201, %201 : vector<4xf32>
                  %203 = spv.FMul %202, %125 : vector<4xf32>
                  %204 = spv.FAdd %203, %200 : vector<4xf32>
                  %205 = spv.CompositeExtract %196[2 : i32] : vector<4xf32>
                  %206 = spv.CompositeConstruct %205, %205, %205, %205 : vector<4xf32>
                  %207 = spv.FMul %206, %131 : vector<4xf32>
                  %208 = spv.FAdd %207, %204 : vector<4xf32>
                  %209 = spv.CompositeExtract %196[3 : i32] : vector<4xf32>
                  %210 = spv.CompositeConstruct %209, %209, %209, %209 : vector<4xf32>
                  %211 = spv.FMul %210, %137 : vector<4xf32>
                  %212 = spv.FAdd %211, %208 : vector<4xf32>
                  %213 = spv.IAdd %74, %1 : i32
                  %214 = spv.IAdd %37, %213 : i32
                  %215 = spv.IAdd %26, %214 : i32
                  %216 = spv.IMul %215, %8 : i32
                  %217 = spv.IAdd %216, %144 : i32
                  %218 = spv.IAdd %217, %142 : i32
                  %219 = spv.AccessChain %13[%10, %218] : !spv.ptr<!spv.struct<(!spv.array<202500 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
                  %220 = spv.Load "StorageBuffer" %219 : vector<4xf32>
                  %221 = spv.CompositeExtract %220[0 : i32] : vector<4xf32>
                  %222 = spv.CompositeConstruct %221, %221, %221, %221 : vector<4xf32>
                  %223 = spv.FMul %222, %119 : vector<4xf32>
                  %224 = spv.FAdd %223, %108 : vector<4xf32>
                  %225 = spv.CompositeExtract %220[1 : i32] : vector<4xf32>
                  %226 = spv.CompositeConstruct %225, %225, %225, %225 : vector<4xf32>
                  %227 = spv.FMul %226, %125 : vector<4xf32>
                  %228 = spv.FAdd %227, %224 : vector<4xf32>
                  %229 = spv.CompositeExtract %220[2 : i32] : vector<4xf32>
                  %230 = spv.CompositeConstruct %229, %229, %229, %229 : vector<4xf32>
                  %231 = spv.FMul %230, %131 : vector<4xf32>
                  %232 = spv.FAdd %231, %228 : vector<4xf32>
                  %233 = spv.CompositeExtract %220[3 : i32] : vector<4xf32>
                  %234 = spv.CompositeConstruct %233, %233, %233, %233 : vector<4xf32>
                  %235 = spv.FMul %234, %137 : vector<4xf32>
                  %236 = spv.FAdd %235, %232 : vector<4xf32>
                  spv.Store "Function" %95, %164 : vector<4xf32>
                  spv.Store "Function" %96, %188 : vector<4xf32>
                  spv.Store "Function" %97, %212 : vector<4xf32>
                  spv.Store "Function" %98, %236 : vector<4xf32>
                  %237 = spv.IAdd %104, %9 : i32
                  spv.Branch ^bb1(%237, %164, %188, %212, %236 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
                ^bb3:  // pred: ^bb1
                  spv.mlir.merge
                }
                %99 = spv.Load "Function" %98 : vector<4xf32>
                %100 = spv.Load "Function" %97 : vector<4xf32>
                %101 = spv.Load "Function" %96 : vector<4xf32>
                %102 = spv.Load "Function" %95 : vector<4xf32>
                spv.Store "Function" %80, %102 : vector<4xf32>
                spv.Store "Function" %81, %101 : vector<4xf32>
                spv.Store "Function" %82, %100 : vector<4xf32>
                spv.Store "Function" %83, %99 : vector<4xf32>
                %103 = spv.IAdd %89, %2 : i32
                spv.Branch ^bb1(%103, %102, %101, %100, %99 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
              ^bb3:  // pred: ^bb1
                spv.mlir.merge
              }
              %84 = spv.Load "Function" %83 : vector<4xf32>
              %85 = spv.Load "Function" %82 : vector<4xf32>
              %86 = spv.Load "Function" %81 : vector<4xf32>
              %87 = spv.Load "Function" %80 : vector<4xf32>
              spv.Store "Function" %39, %87 : vector<4xf32>
              spv.Store "Function" %40, %86 : vector<4xf32>
              spv.Store "Function" %41, %85 : vector<4xf32>
              spv.Store "Function" %42, %84 : vector<4xf32>
              %88 = spv.IAdd %74, %2 : i32
              spv.Branch ^bb1(%88, %87, %86, %85, %84 : i32, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
            ^bb3:  // pred: ^bb1
              spv.mlir.merge
            }
            %43 = spv.Load "Function" %42 : vector<4xf32>
            %44 = spv.Load "Function" %41 : vector<4xf32>
            %45 = spv.Load "Function" %40 : vector<4xf32>
            %46 = spv.Load "Function" %39 : vector<4xf32>
            %47 = spv.IAdd %35, %3 : i32
            %48 = spv.IAdd %23, %47 : i32
            %49 = spv.IAdd %24, %32 : i32
            %50 = spv.IAdd %25, %36 : i32
            %51 = spv.SDiv %50, %9 : i32
            %52 = spv.IMul %48, %11 : i32
            %53 = spv.IMul %49, %12 : i32
            %54 = spv.IAdd %52, %53 : i32
            %55 = spv.IAdd %54, %51 : i32
            %56 = spv.AccessChain %15[%10, %55] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %56, %43 : vector<4xf32>
            %57 = spv.IAdd %35, %4 : i32
            %58 = spv.IAdd %23, %57 : i32
            %59 = spv.IMul %58, %11 : i32
            %60 = spv.IAdd %59, %53 : i32
            %61 = spv.IAdd %60, %51 : i32
            %62 = spv.AccessChain %15[%10, %61] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %62, %44 : vector<4xf32>
            %63 = spv.IAdd %35, %2 : i32
            %64 = spv.IAdd %23, %63 : i32
            %65 = spv.IMul %64, %11 : i32
            %66 = spv.IAdd %65, %53 : i32
            %67 = spv.IAdd %66, %51 : i32
            %68 = spv.AccessChain %15[%10, %67] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %68, %45 : vector<4xf32>
            %69 = spv.IAdd %23, %35 : i32
            %70 = spv.IMul %69, %11 : i32
            %71 = spv.IAdd %70, %53 : i32
            %72 = spv.IAdd %71, %51 : i32
            %73 = spv.AccessChain %15[%10, %72] : !spv.ptr<!spv.struct<(!spv.array<100352 x vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
            spv.Store "StorageBuffer" %73, %46 : vector<4xf32>
            spv.Return
          }
          spv.EntryPoint "GLCompute" @predict_ex_dispatch_1_dispatch_0, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
          spv.ExecutionMode @predict_ex_dispatch_1_dispatch_0 "LocalSize", 4, 4, 1
        }
        hal.interface @legacy_io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
  func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c3240000 = constant 3240000 : index
    %c18432 = constant 18432 : index
    %c1 = constant 1 : index
    %c1605632 = constant 1605632 : index
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %dev = hal.ex.shared_device : !hal.device
    %allocator = hal.device.allocator %dev : !hal.allocator
    %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
    %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
    hal.command_buffer.begin %cmd
    %0 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
    hal.command_buffer.push_descriptor_set %cmd, %0, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
    %1 = hal.variable.load @_device_match_id_0 : i1
    cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %c2_0 = constant 2 : index
    %c28 = constant 28 : index
    %c28_1 = constant 28 : index
    %2 = hal.command_buffer.device %cmd : !hal.device
    %3 = hal.variable.load @_executable_predict_ex_dispatch_1_dispatch_0 : !hal.executable
    hal.command_buffer.dispatch %cmd, %3, entry_point = 0, workgroup_xyz = [%c2_0, %c28, %c28_1]
    br ^bb3
  ^bb2:  // pred: ^bb0
    iree.unreachable
  ^bb3:  // pred: ^bb1
    hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
    hal.command_buffer.end %cmd
    hal.ex.submit_and_wait %dev, %cmd
    return %buffer : !hal.buffer
  }
  func @predict_ex_dispatch_1$async(%arg0: !hal.semaphore, %arg1: index, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.semaphore, %arg5: index) -> !hal.buffer_view attributes {iree.module.export = "predict_ex_dispatch_1$async"} {
    %c1 = constant 1 : index
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c50331680_i32 = constant 50331680 : i32
    %0 = hal.semaphore.await %arg0, min_value = %arg1 : i32
    hal.check_success %0, "semaphore wait failed"
    %buffer = hal.buffer_view.buffer %arg2 : !hal.buffer
    %buffer_0 = hal.buffer_view.buffer %arg3 : !hal.buffer
    %1 = call @predict_ex_dispatch_1(%buffer, %buffer_0) : (!hal.buffer, !hal.buffer) -> !hal.buffer
    %view = hal.buffer_view.create %1, element_type = %c50331680_i32, shape = [%c1, %c112, %c112, %c32] : !hal.buffer_view
    hal.semaphore.signal %arg4, value = %arg5
    return %view : !hal.buffer_view
  }
  func @predict_ex_dispatch_1$sync(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export = "predict_ex_dispatch_1", iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dev = hal.ex.shared_device : !hal.device
    %semaphore = hal.semaphore.create %dev, initial_value = %c0 : !hal.semaphore
    %0 = call @predict_ex_dispatch_1$async(%semaphore, %c0, %arg0, %arg1, %semaphore, %c1) : (!hal.semaphore, index, !hal.buffer_view, !hal.buffer_view, !hal.semaphore, index) -> !hal.buffer_view
    %1 = hal.semaphore.await %semaphore, min_value = %c1 : i32
    hal.check_success %1, "semaphore wait failed"
    return %0 : !hal.buffer_view
  }
}


// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
  %c3240000 = constant 3240000 : index
  %c18432 = constant 18432 : index
  %c1 = constant 1 : index
  %c1605632 = constant 1605632 : index
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c28 = constant 28 : index
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator %dev : !hal.allocator
  %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
  %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  %0 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
  hal.command_buffer.push_descriptor_set %cmd, %0, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
  %1 = hal.variable.load @_device_match_id_0 : i1
  cond_br %1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %2 = hal.variable.load @_executable_predict_ex_dispatch_1_dispatch_0 : !hal.executable
  hal.command_buffer.dispatch %cmd, %2, entry_point = 0, workgroup_xyz = [%c2, %c28, %c28]
  hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
  hal.command_buffer.end %cmd
  hal.ex.submit_and_wait %dev, %cmd
  return %buffer : !hal.buffer
^bb2:  // pred: ^bb0
  iree.unreachable
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1$sync(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export = "predict_ex_dispatch_1", iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %dev = hal.ex.shared_device : !hal.device
  %semaphore = hal.semaphore.create %dev, initial_value = %c0 : !hal.semaphore
  %0 = call @predict_ex_dispatch_1$async(%semaphore, %c0, %arg0, %arg1, %semaphore, %c1) : (!hal.semaphore, index, !hal.buffer_view, !hal.buffer_view, !hal.semaphore, index) -> !hal.buffer_view
  %1 = hal.semaphore.await %semaphore, min_value = %c1 : i32
  hal.check_success %1, "semaphore wait failed"
  return %0 : !hal.buffer_view
}

// *** IR Dump After Canonicalizer ***
func @predict_ex_dispatch_1$async(%arg0: !hal.semaphore, %arg1: index, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.semaphore, %arg5: index) -> !hal.buffer_view attributes {iree.module.export = "predict_ex_dispatch_1$async"} {
  %c1 = constant 1 : index
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c50331680_i32 = constant 50331680 : i32
  %0 = hal.semaphore.await %arg0, min_value = %arg1 : i32
  hal.check_success %0, "semaphore wait failed"
  %buffer = hal.buffer_view.buffer %arg2 : !hal.buffer
  %buffer_0 = hal.buffer_view.buffer %arg3 : !hal.buffer
  %1 = call @predict_ex_dispatch_1(%buffer, %buffer_0) : (!hal.buffer, !hal.buffer) -> !hal.buffer
  %view = hal.buffer_view.create %1, element_type = %c50331680_i32, shape = [%c1, %c112, %c112, %c32] : !hal.buffer_view
  hal.semaphore.signal %arg4, value = %arg5
  return %view : !hal.buffer_view
}

// *** IR Dump After mlir::iree_compiler::IREE::HAL::SerializeExecutablesPass ***
hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.binary @vulkan_spirv attributes {data = dense<"0x080000005350564524E7FFFF0800000034000000010000000400000020000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000027060000030223070000010016000000110100000000000011000200010000000A000B005350565F4B48525F73746F726167655F6275666665725F73746F726167655F636C617373000000000E00030000000000010000000F000E000500000019000000707265646963745F65785F64697370617463685F315F64697370617463685F3000000000050000000400000010000600190000001100000004000000040000000100000005000B00040000005F5F6275696C74696E5F7661725F4C6F63616C496E766F636174696F6E49645F5F00000005000900050000005F5F6275696C74696E5F7661725F576F726B67726F757049645F5F0005000A000C0000005F5F7265736F757263655F7661725F32343838393331333339323239365F5F0005000A00110000005F5F7265736F757263655F7661725F32343838393331333430383837325F5F0005000A00160000005F5F7265736F757263655F7661725F32343838393331333339323130345F5F0005000B0019000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000047000400040000000B0000001B00000047000400050000000B0000001A000000470004000800000006000000100000004800050007000000000000002300000000000000470003000700000002000000470004000C0000002100000002000000470004000C0000002200000000000000470004000F0000000600000010000000480005000E000000000000002300000000000000470003000E00000002000000470004001100000021000000010000004700040011000000220000000000000047000400140000000600000010000000480005001300000000000000230000000000000047000300130000000200000047000400160000002100000000000000470004001600000022000000000000001500040003000000200000000000000017000400020000000300000003000000200004000100000001000000020000003B0004000100000004000000010000003B000400010000000500000001000000160003000A0000002000000017000400090000000A000000040000002B000400030000000B000000008801001C00040008000000090000000B0000001E000300070000000800000020000400060000000C000000070000003B000400060000000C0000000C0000002B0004000300000010000000800400001C0004000F00000009000000100000001E0003000E0000000F000000200004000D0000000C0000000E0000003B0004000D000000110000000C0000002B0004000300000015000000041703001C0004001400000009000000150000001E000300130000001400000020000400120000000C000000130000003B00040012000000160000000C00000013000200180000002100030017000000180000002B000400030000001B000000100000002B000400030000001C000000060000002B000400030000001D000000010000002B000400030000001E000000030000002B000400030000001F000000020000002B0004000A00000021000000000000002C0007000900000020000000210000002100000021000000210000002B0004000300000022000000800100002B0004000300000023000000800000002B0004000300000024000000840300002B0004000300000025000000040000002B0004000300000026000000000000002B0004000300000027000000800300002B000400030000002800000008000000200004003E0000000700000009000000140002004C000000200004006F0000000C000000090000003600050018000000190000000000000017000000F80002001A0000003B0004003E0000003F000000070000003B0004003E00000040000000070000003B0004003E00000041000000070000003B0004003E00000042000000070000003B0004003E0000004E000000070000003B0004003E0000004F000000070000003B0004003E00000050000000070000003B0004003E00000051000000070000003B0004003E0000005B000000070000003B0004003E0000005C000000070000003B0004003E0000005D000000070000003B0004003E0000005E000000070000003D00040002000000290000000500000051000500030000002A00000029000000000000003D000400020000002B0000000500000051000500030000002C0000002B000000010000003D000400020000002D0000000500000051000500030000002E0000002D0000000200000084000500030000002F0000002E000000250000008400050003000000300000002C000000250000008400050003000000310000002A0000001B0000008400050003000000320000002F0000001F000000840005000300000033000000300000001F0000003D00040002000000340000000400000051000500030000003500000034000000000000003D00040002000000360000000400000051000500030000003700000036000000010000003D000400020000003800000004000000510005000300000039000000380000000200000084000500030000003A000000390000002500000084000500030000003B000000350000002500000084000500030000003C000000390000002800000084000500030000003D000000370000001F000000F900020043000000F800020043000000F50007000300000046000000F100000047000000260000001A000000F50007000900000048000000F000000047000000200000001A000000F50007000900000049000000EF00000047000000200000001A000000F5000700090000004A000000EE00000047000000200000001A000000F5000700090000004B000000ED00000047000000200000001A000000B10005004C0000004D000000460000001E000000F6000400450000004400000000000000FA0004004D0000004400000045000000F800020044000000F900020052000000F800020052000000F50007000300000054000000EC000000550000002600000044000000F50007000900000056000000EB000000550000004800000044000000F50007000900000057000000EA000000550000004900000044000000F50007000900000058000000E9000000550000004A00000044000000F50007000900000059000000E8000000550000004B00000044000000B10005004C0000005A000000540000001E000000F6000400470000005300000000000000FA0004005A0000005300000047000000F800020053000000F90002005F000000F80002005F000000F50007000300000061000000E7000000600000002600000053000000F500070009000000620000009E000000600000005600000053000000F50007000900000063000000B6000000600000005700000053000000F50007000900000064000000CE000000600000005800000053000000F50007000900000065000000E6000000600000005900000053000000B10005004C00000066000000610000001B000000F6000400550000006000000000000000FA000400660000006000000055000000F800020060000000800005000300000067000000310000003B0000008700050003000000680000006700000025000000840005000300000069000000460000002200000084000500030000006A000000540000002300000080000500030000006B000000690000006A00000084000500030000006C000000610000002800000080000500030000006D0000006B0000006C00000080000500030000006E0000006D00000068000000410006006F0000007000000011000000260000006E0000003D000400090000007100000070000000800005000300000072000000610000001D00000084000500030000007300000072000000280000008000050003000000740000006B000000730000008000050003000000750000007400000068000000410006006F000000760000001100000026000000750000003D000400090000007700000076000000800005000300000078000000610000001F000000840005000300000079000000780000002800000080000500030000007A0000006B0000007900000080000500030000007B0000007A00000068000000410006006F0000007C00000011000000260000007B0000003D000400090000007D0000007C00000080000500030000007E000000610000001E00000084000500030000007F0000007E000000280000008000050003000000800000006B0000007F0000008000050003000000810000008000000068000000410006006F000000820000001100000026000000810000003D0004000900000083000000820000008000050003000000840000003C000000460000008000050003000000850000003D00000054000000800005000300000086000000320000008400000080000500030000008700000033000000850000008700050003000000880000006100000025000000840005000300000089000000860000002400000084000500030000008A000000870000002500000080000500030000008B000000890000008A00000080000500030000008C0000008B00000088000000410006006F0000008D00000016000000260000008C0000003D000400090000008E0000008D000000510005000A0000008F0000008E000000000000005000070009000000900000008F0000008F0000008F0000008F00000085000500090000009100000090000000710000008100050009000000920000009100000062000000510005000A000000930000008E000000010000005000070009000000940000009300000093000000930000009300000085000500090000009500000094000000770000008100050009000000960000009500000092000000510005000A000000970000008E0000000200000050000700090000009800000097000000970000009700000097000000850005000900000099000000980000007D00000081000500090000009A0000009900000096000000510005000A0000009B0000008E0000000300000050000700090000009C0000009B0000009B0000009B0000009B00000085000500090000009D0000009C0000008300000081000500090000009E0000009D0000009A00000080000500030000009F000000460000001F0000008000050003000000A00000003C0000009F0000008000050003000000A100000032000000A00000008400050003000000A2000000A1000000240000008000050003000000A3000000A20000008A0000008000050003000000A4000000A300000088000000410006006F000000A50000001600000026000000A40000003D00040009000000A6000000A5000000510005000A000000A7000000A6000000000000005000070009000000A8000000A7000000A7000000A7000000A70000008500050009000000A9000000A8000000710000008100050009000000AA000000A900000063000000510005000A000000AB000000A6000000010000005000070009000000AC000000AB000000AB000000AB000000AB0000008500050009000000AD000000AC000000770000008100050009000000AE000000AD000000AA000000510005000A000000AF000000A6000000020000005000070009000000B0000000AF000000AF000000AF000000AF0000008500050009000000B1000000B00000007D0000008100050009000000B2000000B1000000AE000000510005000A000000B3000000A6000000030000005000070009000000B4000000B3000000B3000000B3000000B30000008500050009000000B5000000B4000000830000008100050009000000B6000000B5000000B20000008000050003000000B700000046000000250000008000050003000000B80000003C000000B70000008000050003000000B900000032000000B80000008400050003000000BA000000B9000000240000008000050003000000BB000000BA0000008A0000008000050003000000BC000000BB00000088000000410006006F000000BD0000001600000026000000BC0000003D00040009000000BE000000BD000000510005000A000000BF000000BE000000000000005000070009000000C0000000BF000000BF000000BF000000BF0000008500050009000000C1000000C0000000710000008100050009000000C2000000C100000064000000510005000A000000C3000000BE000000010000005000070009000000C4000000C3000000C3000000C3000000C30000008500050009000000C5000000C4000000770000008100050009000000C6000000C5000000C2000000510005000A000000C7000000BE000000020000005000070009000000C8000000C7000000C7000000C7000000C70000008500050009000000C9000000C80000007D0000008100050009000000CA000000C9000000C6000000510005000A000000CB000000BE000000030000005000070009000000CC000000CB000000CB000000CB000000CB0000008500050009000000CD000000CC000000830000008100050009000000CE000000CD000000CA0000008000050003000000CF000000460000001C0000008000050003000000D00000003C000000CF0000008000050003000000D100000032000000D00000008400050003000000D2000000D1000000240000008000050003000000D3000000D20000008A0000008000050003000000D4000000D300000088000000410006006F000000D50000001600000026000000D40000003D00040009000000D6000000D5000000510005000A000000D7000000D6000000000000005000070009000000D8000000D7000000D7000000D7000000D70000008500050009000000D9000000D8000000710000008100050009000000DA000000D900000065000000510005000A000000DB000000D6000000010000005000070009000000DC000000DB000000DB000000DB000000DB0000008500050009000000DD000000DC000000770000008100050009000000DE000000DD000000DA000000510005000A000000DF000000D6000000020000005000070009000000E0000000DF000000DF000000DF000000DF0000008500050009000000E1000000E00000007D0000008100050009000000E2000000E1000000DE000000510005000A000000E3000000D6000000030000005000070009000000E4000000E3000000E3000000E3000000E30000008500050009000000E5000000E4000000830000008100050009000000E6000000E5000000E20000003E0003005B0000009E0000003E0003005C000000B60000003E0003005D000000CE0000003E0003005E000000E60000008000050003000000E70000006100000025000000F90002005F000000F8000200550000003D00040009000000E80000005E0000003D00040009000000E90000005D0000003D00040009000000EA0000005C0000003D00040009000000EB0000005B0000003E0003004E000000EB0000003E0003004F000000EA0000003E00030050000000E90000003E00030051000000E80000008000050003000000EC000000540000001D000000F900020052000000F8000200470000003D00040009000000ED000000510000003D00040009000000EE000000500000003D00040009000000EF0000004F0000003D00040009000000F00000004E0000003E0003003F000000F00000003E00030040000000EF0000003E00030041000000EE0000003E00030042000000ED0000008000050003000000F1000000460000001D000000F900020043000000F8000200450000003D00040009000000F2000000420000003D00040009000000F3000000410000003D00040009000000F4000000400000003D00040009000000F50000003F0000008000050003000000F60000003A0000001E0000008000050003000000F70000002F000000F60000008000050003000000F800000030000000370000008000050003000000F9000000310000003B0000008700050003000000FA000000F9000000250000008400050003000000FB000000F7000000270000008400050003000000FC000000F8000000280000008000050003000000FD000000FB000000FC0000008000050003000000FE000000FD000000FA000000410006006F000000FF0000000C00000026000000FE0000003E000300FF000000F20000008000050003000000000100003A0000001F0000008000050003000000010100002F00000000010000840005000300000002010000010100002700000080000500030000000301000002010000FC00000080000500030000000401000003010000FA000000410006006F000000050100000C00000026000000040100003E00030005010000F30000008000050003000000060100003A0000001D0000008000050003000000070100002F00000006010000840005000300000008010000070100002700000080000500030000000901000008010000FC00000080000500030000000A01000009010000FA000000410006006F0000000B0100000C000000260000000A0100003E0003000B010000F400000080000500030000000C0100002F0000003A00000084000500030000000D0100000C0100002700000080000500030000000E0100000D010000FC00000080000500030000000F0100000E010000FA000000410006006F000000100100000C000000260000000F0100003E00030010010000F5000000FD0001003800010008000C0004000800"> : vector<6380xi8>, format = 1397773893 : i32} {
  }
}

// *** IR Dump After Canonicalizer ***
func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !hal.executable {
  %dev = hal.ex.shared_device : !hal.device
  %0 = hal.variable.load @_device_match_id_0 : i1
  cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %1 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
  %exe = hal.executable.create %dev, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv, layouts = [%1] : !hal.executable
  br ^bb3(%exe : !hal.executable)
^bb2:  // pred: ^bb0
  %2 = iree.null : !hal.executable
  br ^bb3(%2 : !hal.executable)
^bb3(%3: !hal.executable):  // 2 preds: ^bb1, ^bb2
  return %3 : !hal.executable
}

// *** IR Dump After Canonicalizer ***
module  {
  hal.variable @_device_match_id_0 init(@_device_match_id_0_initializer) : i1 attributes {sym_visibility = "private"}
  func private @_device_match_id_0_initializer() -> i1 {
    %dev = hal.ex.shared_device : !hal.device
    %0 = hal.device.match.id %dev, pattern = ["vulkan*"] : (!hal.device) -> i1
    return %0 : i1
  }
  hal.variable @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !hal.descriptor_set_layout attributes {sym_visibility = "private"}
  func private @_descriptor_set_layout_0_initializer() -> !hal.descriptor_set_layout {
    %dev = hal.ex.shared_device : !hal.device
    %descriptor_set_layout = hal.descriptor_set_layout.create %dev, PushOnly, bindings = [#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<2, "StorageBuffer", "Write|Discard">] : !hal.descriptor_set_layout
    return %descriptor_set_layout : !hal.descriptor_set_layout
  }
  hal.variable @_executable_layout_0 init(@_executable_layout_0_initializer) : !hal.executable_layout attributes {sym_visibility = "private"}
  func private @_executable_layout_0_initializer() -> !hal.executable_layout {
    %0 = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
    %dev = hal.ex.shared_device : !hal.device
    %executable_layout = hal.executable_layout.create %dev, push_constants = 0, set_layouts = [%0] : !hal.executable_layout
    return %executable_layout : !hal.executable_layout
  }
  hal.variable @_executable_predict_ex_dispatch_1_dispatch_0 init(@_executable_predict_ex_dispatch_1_dispatch_0_initializer) : !hal.executable attributes {sym_visibility = "private"}
  func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !hal.executable {
    %dev = hal.ex.shared_device : !hal.device
    %0 = hal.variable.load @_device_match_id_0 : i1
    cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %1 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
    %exe = hal.executable.create %dev, @predict_ex_dispatch_1_dispatch_0::@vulkan_spirv, layouts = [%1] : !hal.executable
    br ^bb3(%exe : !hal.executable)
  ^bb2:  // pred: ^bb0
    %2 = iree.null : !hal.executable
    br ^bb3(%2 : !hal.executable)
  ^bb3(%3: !hal.executable):  // 2 preds: ^bb1, ^bb2
    return %3 : !hal.executable
  }
  hal.executable @predict_ex_dispatch_1_dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.binary @vulkan_spirv attributes {data = dense<"0x080000005350564524E7FFFF0800000034000000010000000400000020000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000027060000030223070000010016000000110100000000000011000200010000000A000B005350565F4B48525F73746F726167655F6275666665725F73746F726167655F636C617373000000000E00030000000000010000000F000E000500000019000000707265646963745F65785F64697370617463685F315F64697370617463685F3000000000050000000400000010000600190000001100000004000000040000000100000005000B00040000005F5F6275696C74696E5F7661725F4C6F63616C496E766F636174696F6E49645F5F00000005000900050000005F5F6275696C74696E5F7661725F576F726B67726F757049645F5F0005000A000C0000005F5F7265736F757263655F7661725F32343838393331333339323239365F5F0005000A00110000005F5F7265736F757263655F7661725F32343838393331333430383837325F5F0005000A00160000005F5F7265736F757263655F7661725F32343838393331333339323130345F5F0005000B0019000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000047000400040000000B0000001B00000047000400050000000B0000001A000000470004000800000006000000100000004800050007000000000000002300000000000000470003000700000002000000470004000C0000002100000002000000470004000C0000002200000000000000470004000F0000000600000010000000480005000E000000000000002300000000000000470003000E00000002000000470004001100000021000000010000004700040011000000220000000000000047000400140000000600000010000000480005001300000000000000230000000000000047000300130000000200000047000400160000002100000000000000470004001600000022000000000000001500040003000000200000000000000017000400020000000300000003000000200004000100000001000000020000003B0004000100000004000000010000003B000400010000000500000001000000160003000A0000002000000017000400090000000A000000040000002B000400030000000B000000008801001C00040008000000090000000B0000001E000300070000000800000020000400060000000C000000070000003B000400060000000C0000000C0000002B0004000300000010000000800400001C0004000F00000009000000100000001E0003000E0000000F000000200004000D0000000C0000000E0000003B0004000D000000110000000C0000002B0004000300000015000000041703001C0004001400000009000000150000001E000300130000001400000020000400120000000C000000130000003B00040012000000160000000C00000013000200180000002100030017000000180000002B000400030000001B000000100000002B000400030000001C000000060000002B000400030000001D000000010000002B000400030000001E000000030000002B000400030000001F000000020000002B0004000A00000021000000000000002C0007000900000020000000210000002100000021000000210000002B0004000300000022000000800100002B0004000300000023000000800000002B0004000300000024000000840300002B0004000300000025000000040000002B0004000300000026000000000000002B0004000300000027000000800300002B000400030000002800000008000000200004003E0000000700000009000000140002004C000000200004006F0000000C000000090000003600050018000000190000000000000017000000F80002001A0000003B0004003E0000003F000000070000003B0004003E00000040000000070000003B0004003E00000041000000070000003B0004003E00000042000000070000003B0004003E0000004E000000070000003B0004003E0000004F000000070000003B0004003E00000050000000070000003B0004003E00000051000000070000003B0004003E0000005B000000070000003B0004003E0000005C000000070000003B0004003E0000005D000000070000003B0004003E0000005E000000070000003D00040002000000290000000500000051000500030000002A00000029000000000000003D000400020000002B0000000500000051000500030000002C0000002B000000010000003D000400020000002D0000000500000051000500030000002E0000002D0000000200000084000500030000002F0000002E000000250000008400050003000000300000002C000000250000008400050003000000310000002A0000001B0000008400050003000000320000002F0000001F000000840005000300000033000000300000001F0000003D00040002000000340000000400000051000500030000003500000034000000000000003D00040002000000360000000400000051000500030000003700000036000000010000003D000400020000003800000004000000510005000300000039000000380000000200000084000500030000003A000000390000002500000084000500030000003B000000350000002500000084000500030000003C000000390000002800000084000500030000003D000000370000001F000000F900020043000000F800020043000000F50007000300000046000000F100000047000000260000001A000000F50007000900000048000000F000000047000000200000001A000000F50007000900000049000000EF00000047000000200000001A000000F5000700090000004A000000EE00000047000000200000001A000000F5000700090000004B000000ED00000047000000200000001A000000B10005004C0000004D000000460000001E000000F6000400450000004400000000000000FA0004004D0000004400000045000000F800020044000000F900020052000000F800020052000000F50007000300000054000000EC000000550000002600000044000000F50007000900000056000000EB000000550000004800000044000000F50007000900000057000000EA000000550000004900000044000000F50007000900000058000000E9000000550000004A00000044000000F50007000900000059000000E8000000550000004B00000044000000B10005004C0000005A000000540000001E000000F6000400470000005300000000000000FA0004005A0000005300000047000000F800020053000000F90002005F000000F80002005F000000F50007000300000061000000E7000000600000002600000053000000F500070009000000620000009E000000600000005600000053000000F50007000900000063000000B6000000600000005700000053000000F50007000900000064000000CE000000600000005800000053000000F50007000900000065000000E6000000600000005900000053000000B10005004C00000066000000610000001B000000F6000400550000006000000000000000FA000400660000006000000055000000F800020060000000800005000300000067000000310000003B0000008700050003000000680000006700000025000000840005000300000069000000460000002200000084000500030000006A000000540000002300000080000500030000006B000000690000006A00000084000500030000006C000000610000002800000080000500030000006D0000006B0000006C00000080000500030000006E0000006D00000068000000410006006F0000007000000011000000260000006E0000003D000400090000007100000070000000800005000300000072000000610000001D00000084000500030000007300000072000000280000008000050003000000740000006B000000730000008000050003000000750000007400000068000000410006006F000000760000001100000026000000750000003D000400090000007700000076000000800005000300000078000000610000001F000000840005000300000079000000780000002800000080000500030000007A0000006B0000007900000080000500030000007B0000007A00000068000000410006006F0000007C00000011000000260000007B0000003D000400090000007D0000007C00000080000500030000007E000000610000001E00000084000500030000007F0000007E000000280000008000050003000000800000006B0000007F0000008000050003000000810000008000000068000000410006006F000000820000001100000026000000810000003D0004000900000083000000820000008000050003000000840000003C000000460000008000050003000000850000003D00000054000000800005000300000086000000320000008400000080000500030000008700000033000000850000008700050003000000880000006100000025000000840005000300000089000000860000002400000084000500030000008A000000870000002500000080000500030000008B000000890000008A00000080000500030000008C0000008B00000088000000410006006F0000008D00000016000000260000008C0000003D000400090000008E0000008D000000510005000A0000008F0000008E000000000000005000070009000000900000008F0000008F0000008F0000008F00000085000500090000009100000090000000710000008100050009000000920000009100000062000000510005000A000000930000008E000000010000005000070009000000940000009300000093000000930000009300000085000500090000009500000094000000770000008100050009000000960000009500000092000000510005000A000000970000008E0000000200000050000700090000009800000097000000970000009700000097000000850005000900000099000000980000007D00000081000500090000009A0000009900000096000000510005000A0000009B0000008E0000000300000050000700090000009C0000009B0000009B0000009B0000009B00000085000500090000009D0000009C0000008300000081000500090000009E0000009D0000009A00000080000500030000009F000000460000001F0000008000050003000000A00000003C0000009F0000008000050003000000A100000032000000A00000008400050003000000A2000000A1000000240000008000050003000000A3000000A20000008A0000008000050003000000A4000000A300000088000000410006006F000000A50000001600000026000000A40000003D00040009000000A6000000A5000000510005000A000000A7000000A6000000000000005000070009000000A8000000A7000000A7000000A7000000A70000008500050009000000A9000000A8000000710000008100050009000000AA000000A900000063000000510005000A000000AB000000A6000000010000005000070009000000AC000000AB000000AB000000AB000000AB0000008500050009000000AD000000AC000000770000008100050009000000AE000000AD000000AA000000510005000A000000AF000000A6000000020000005000070009000000B0000000AF000000AF000000AF000000AF0000008500050009000000B1000000B00000007D0000008100050009000000B2000000B1000000AE000000510005000A000000B3000000A6000000030000005000070009000000B4000000B3000000B3000000B3000000B30000008500050009000000B5000000B4000000830000008100050009000000B6000000B5000000B20000008000050003000000B700000046000000250000008000050003000000B80000003C000000B70000008000050003000000B900000032000000B80000008400050003000000BA000000B9000000240000008000050003000000BB000000BA0000008A0000008000050003000000BC000000BB00000088000000410006006F000000BD0000001600000026000000BC0000003D00040009000000BE000000BD000000510005000A000000BF000000BE000000000000005000070009000000C0000000BF000000BF000000BF000000BF0000008500050009000000C1000000C0000000710000008100050009000000C2000000C100000064000000510005000A000000C3000000BE000000010000005000070009000000C4000000C3000000C3000000C3000000C30000008500050009000000C5000000C4000000770000008100050009000000C6000000C5000000C2000000510005000A000000C7000000BE000000020000005000070009000000C8000000C7000000C7000000C7000000C70000008500050009000000C9000000C80000007D0000008100050009000000CA000000C9000000C6000000510005000A000000CB000000BE000000030000005000070009000000CC000000CB000000CB000000CB000000CB0000008500050009000000CD000000CC000000830000008100050009000000CE000000CD000000CA0000008000050003000000CF000000460000001C0000008000050003000000D00000003C000000CF0000008000050003000000D100000032000000D00000008400050003000000D2000000D1000000240000008000050003000000D3000000D20000008A0000008000050003000000D4000000D300000088000000410006006F000000D50000001600000026000000D40000003D00040009000000D6000000D5000000510005000A000000D7000000D6000000000000005000070009000000D8000000D7000000D7000000D7000000D70000008500050009000000D9000000D8000000710000008100050009000000DA000000D900000065000000510005000A000000DB000000D6000000010000005000070009000000DC000000DB000000DB000000DB000000DB0000008500050009000000DD000000DC000000770000008100050009000000DE000000DD000000DA000000510005000A000000DF000000D6000000020000005000070009000000E0000000DF000000DF000000DF000000DF0000008500050009000000E1000000E00000007D0000008100050009000000E2000000E1000000DE000000510005000A000000E3000000D6000000030000005000070009000000E4000000E3000000E3000000E3000000E30000008500050009000000E5000000E4000000830000008100050009000000E6000000E5000000E20000003E0003005B0000009E0000003E0003005C000000B60000003E0003005D000000CE0000003E0003005E000000E60000008000050003000000E70000006100000025000000F90002005F000000F8000200550000003D00040009000000E80000005E0000003D00040009000000E90000005D0000003D00040009000000EA0000005C0000003D00040009000000EB0000005B0000003E0003004E000000EB0000003E0003004F000000EA0000003E00030050000000E90000003E00030051000000E80000008000050003000000EC000000540000001D000000F900020052000000F8000200470000003D00040009000000ED000000510000003D00040009000000EE000000500000003D00040009000000EF0000004F0000003D00040009000000F00000004E0000003E0003003F000000F00000003E00030040000000EF0000003E00030041000000EE0000003E00030042000000ED0000008000050003000000F1000000460000001D000000F900020043000000F8000200450000003D00040009000000F2000000420000003D00040009000000F3000000410000003D00040009000000F4000000400000003D00040009000000F50000003F0000008000050003000000F60000003A0000001E0000008000050003000000F70000002F000000F60000008000050003000000F800000030000000370000008000050003000000F9000000310000003B0000008700050003000000FA000000F9000000250000008400050003000000FB000000F7000000270000008400050003000000FC000000F8000000280000008000050003000000FD000000FB000000FC0000008000050003000000FE000000FD000000FA000000410006006F000000FF0000000C00000026000000FE0000003E000300FF000000F20000008000050003000000000100003A0000001F0000008000050003000000010100002F00000000010000840005000300000002010000010100002700000080000500030000000301000002010000FC00000080000500030000000401000003010000FA000000410006006F000000050100000C00000026000000040100003E00030005010000F30000008000050003000000060100003A0000001D0000008000050003000000070100002F00000006010000840005000300000008010000070100002700000080000500030000000901000008010000FC00000080000500030000000A01000009010000FA000000410006006F0000000B0100000C000000260000000A0100003E0003000B010000F400000080000500030000000C0100002F0000003A00000084000500030000000D0100000C0100002700000080000500030000000E0100000D010000FC00000080000500030000000F0100000E010000FA000000410006006F000000100100000C000000260000000F0100003E00030010010000F5000000FD0001003800010008000C0004000800"> : vector<6380xi8>, format = 1397773893 : i32} {
    }
  }
  func @predict_ex_dispatch_1(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !hal.buffer attributes {iree.module.export = "predict_ex_dispatch_1$raw", noinline} {
    %c3240000 = constant 3240000 : index
    %c18432 = constant 18432 : index
    %c1 = constant 1 : index
    %c1605632 = constant 1605632 : index
    %c0 = constant 0 : index
    %c2 = constant 2 : index
    %c28 = constant 28 : index
    %dev = hal.ex.shared_device : !hal.device
    %allocator = hal.device.allocator %dev : !hal.allocator
    %buffer = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c1605632 : !hal.buffer
    %cmd = hal.command_buffer.create %dev, OneShot, "Transfer|Dispatch" : !hal.command_buffer
    hal.command_buffer.begin %cmd
    %0 = hal.variable.load @_executable_layout_0 : !hal.executable_layout
    hal.command_buffer.push_descriptor_set %cmd, %0, set = %c0, bindings = [%c0 = (%arg0, %c0, %c3240000), %c1 = (%arg1, %c0, %c18432), %c2 = (%buffer, %c0, %c1605632)]
    %1 = hal.variable.load @_device_match_id_0 : i1
    cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = hal.variable.load @_executable_predict_ex_dispatch_1_dispatch_0 : !hal.executable
    hal.command_buffer.dispatch %cmd, %2, entry_point = 0, workgroup_xyz = [%c2, %c28, %c28]
    hal.command_buffer.execution_barrier %cmd, "Dispatch|CommandRetire", "CommandIssue|Dispatch", "None"
    hal.command_buffer.end %cmd
    hal.ex.submit_and_wait %dev, %cmd
    return %buffer : !hal.buffer
  ^bb2:  // pred: ^bb0
    iree.unreachable
  }
  func @predict_ex_dispatch_1$async(%arg0: !hal.semaphore, %arg1: index, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.semaphore, %arg5: index) -> !hal.buffer_view attributes {iree.module.export = "predict_ex_dispatch_1$async"} {
    %c1 = constant 1 : index
    %c112 = constant 112 : index
    %c32 = constant 32 : index
    %c50331680_i32 = constant 50331680 : i32
    %0 = hal.semaphore.await %arg0, min_value = %arg1 : i32
    hal.check_success %0, "semaphore wait failed"
    %buffer = hal.buffer_view.buffer %arg2 : !hal.buffer
    %buffer_0 = hal.buffer_view.buffer %arg3 : !hal.buffer
    %1 = call @predict_ex_dispatch_1(%buffer, %buffer_0) : (!hal.buffer, !hal.buffer) -> !hal.buffer
    %view = hal.buffer_view.create %1, element_type = %c50331680_i32, shape = [%c1, %c112, %c112, %c32] : !hal.buffer_view
    hal.semaphore.signal %arg4, value = %arg5
    return %view : !hal.buffer_view
  }
  func @predict_ex_dispatch_1$sync(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export = "predict_ex_dispatch_1", iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dev = hal.ex.shared_device : !hal.device
    %semaphore = hal.semaphore.create %dev, initial_value = %c0 : !hal.semaphore
    %0 = call @predict_ex_dispatch_1$async(%semaphore, %c0, %arg0, %arg1, %semaphore, %c1) : (!hal.semaphore, index, !hal.buffer_view, !hal.buffer_view, !hal.semaphore, index) -> !hal.buffer_view
    %1 = hal.semaphore.await %semaphore, min_value = %c1 : i32
    hal.check_success %1, "semaphore wait failed"
    return %0 : !hal.buffer_view
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::VM::ConversionPass ***
module  {
  vm.module @module {
    vm.global.i32 @_device_match_id_0 init(@_device_match_id_0_initializer) : i32
    vm.rodata @_utf8_vulkan_7197BF52A22CAFD7 dense<[118, 117, 108, 107, 97, 110, 42]> : vector<7xi8>
    vm.func private @_device_match_id_0_initializer() -> i32 {
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %_utf8_vulkan_7197BF52A22CAFD7 = vm.const.ref.rodata @_utf8_vulkan_7197BF52A22CAFD7 : !vm.ref<!iree.byte_buffer>
      %0 = vm.call @hal.device.match.id(%ref, %_utf8_vulkan_7197BF52A22CAFD7) : (!vm.ref<!hal.device>, !vm.ref<!iree.byte_buffer>) -> i32
      vm.return %0 : i32
    }
    vm.global.ref @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !vm.ref<!hal.descriptor_set_layout>
    vm.func private @_descriptor_set_layout_0_initializer() -> !vm.ref<!hal.descriptor_set_layout> {
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %c1 = vm.const.i32 1 : i32
      %zero = vm.const.i32.zero : i32
      %c7 = vm.const.i32 7 : i32
      %c1_0 = vm.const.i32 1 : i32
      %c1_1 = vm.const.i32 1 : i32
      %c7_2 = vm.const.i32 7 : i32
      %c1_3 = vm.const.i32 1 : i32
      %c2 = vm.const.i32 2 : i32
      %c7_4 = vm.const.i32 7 : i32
      %c6 = vm.const.i32 6 : i32
      %ref_5 = vm.call.variadic @hal.descriptor_set_layout.create(%ref, %c1, [(%zero, %c7, %c1_0), (%c1_1, %c7_2, %c1_3), (%c2, %c7_4, %c6)]) : (!vm.ref<!hal.device>, i32, tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout>
      vm.return %ref_5 : !vm.ref<!hal.descriptor_set_layout>
    }
    vm.global.ref @_executable_layout_0 init(@_executable_layout_0_initializer) : !vm.ref<!hal.executable_layout>
    vm.func private @_executable_layout_0_initializer() -> !vm.ref<!hal.executable_layout> {
      %_descriptor_set_layout_0 = vm.global.load.ref @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %zero = vm.const.i32.zero : i32
      %ref_0 = vm.call.variadic @hal.executable_layout.create(%ref, %zero, [%_descriptor_set_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout>
      vm.return %ref_0 : !vm.ref<!hal.executable_layout>
    }
    vm.global.ref @_executable_predict_ex_dispatch_1_dispatch_0 init(@_executable_predict_ex_dispatch_1_dispatch_0_initializer) : !vm.ref<!hal.executable>
    vm.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv dense<"0x080000005350564524E7FFFF0800000034000000010000000400000020000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000027060000030223070000010016000000110100000000000011000200010000000A000B005350565F4B48525F73746F726167655F6275666665725F73746F726167655F636C617373000000000E00030000000000010000000F000E000500000019000000707265646963745F65785F64697370617463685F315F64697370617463685F3000000000050000000400000010000600190000001100000004000000040000000100000005000B00040000005F5F6275696C74696E5F7661725F4C6F63616C496E766F636174696F6E49645F5F00000005000900050000005F5F6275696C74696E5F7661725F576F726B67726F757049645F5F0005000A000C0000005F5F7265736F757263655F7661725F32343838393331333339323239365F5F0005000A00110000005F5F7265736F757263655F7661725F32343838393331333430383837325F5F0005000A00160000005F5F7265736F757263655F7661725F32343838393331333339323130345F5F0005000B0019000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000047000400040000000B0000001B00000047000400050000000B0000001A000000470004000800000006000000100000004800050007000000000000002300000000000000470003000700000002000000470004000C0000002100000002000000470004000C0000002200000000000000470004000F0000000600000010000000480005000E000000000000002300000000000000470003000E00000002000000470004001100000021000000010000004700040011000000220000000000000047000400140000000600000010000000480005001300000000000000230000000000000047000300130000000200000047000400160000002100000000000000470004001600000022000000000000001500040003000000200000000000000017000400020000000300000003000000200004000100000001000000020000003B0004000100000004000000010000003B000400010000000500000001000000160003000A0000002000000017000400090000000A000000040000002B000400030000000B000000008801001C00040008000000090000000B0000001E000300070000000800000020000400060000000C000000070000003B000400060000000C0000000C0000002B0004000300000010000000800400001C0004000F00000009000000100000001E0003000E0000000F000000200004000D0000000C0000000E0000003B0004000D000000110000000C0000002B0004000300000015000000041703001C0004001400000009000000150000001E000300130000001400000020000400120000000C000000130000003B00040012000000160000000C00000013000200180000002100030017000000180000002B000400030000001B000000100000002B000400030000001C000000060000002B000400030000001D000000010000002B000400030000001E000000030000002B000400030000001F000000020000002B0004000A00000021000000000000002C0007000900000020000000210000002100000021000000210000002B0004000300000022000000800100002B0004000300000023000000800000002B0004000300000024000000840300002B0004000300000025000000040000002B0004000300000026000000000000002B0004000300000027000000800300002B000400030000002800000008000000200004003E0000000700000009000000140002004C000000200004006F0000000C000000090000003600050018000000190000000000000017000000F80002001A0000003B0004003E0000003F000000070000003B0004003E00000040000000070000003B0004003E00000041000000070000003B0004003E00000042000000070000003B0004003E0000004E000000070000003B0004003E0000004F000000070000003B0004003E00000050000000070000003B0004003E00000051000000070000003B0004003E0000005B000000070000003B0004003E0000005C000000070000003B0004003E0000005D000000070000003B0004003E0000005E000000070000003D00040002000000290000000500000051000500030000002A00000029000000000000003D000400020000002B0000000500000051000500030000002C0000002B000000010000003D000400020000002D0000000500000051000500030000002E0000002D0000000200000084000500030000002F0000002E000000250000008400050003000000300000002C000000250000008400050003000000310000002A0000001B0000008400050003000000320000002F0000001F000000840005000300000033000000300000001F0000003D00040002000000340000000400000051000500030000003500000034000000000000003D00040002000000360000000400000051000500030000003700000036000000010000003D000400020000003800000004000000510005000300000039000000380000000200000084000500030000003A000000390000002500000084000500030000003B000000350000002500000084000500030000003C000000390000002800000084000500030000003D000000370000001F000000F900020043000000F800020043000000F50007000300000046000000F100000047000000260000001A000000F50007000900000048000000F000000047000000200000001A000000F50007000900000049000000EF00000047000000200000001A000000F5000700090000004A000000EE00000047000000200000001A000000F5000700090000004B000000ED00000047000000200000001A000000B10005004C0000004D000000460000001E000000F6000400450000004400000000000000FA0004004D0000004400000045000000F800020044000000F900020052000000F800020052000000F50007000300000054000000EC000000550000002600000044000000F50007000900000056000000EB000000550000004800000044000000F50007000900000057000000EA000000550000004900000044000000F50007000900000058000000E9000000550000004A00000044000000F50007000900000059000000E8000000550000004B00000044000000B10005004C0000005A000000540000001E000000F6000400470000005300000000000000FA0004005A0000005300000047000000F800020053000000F90002005F000000F80002005F000000F50007000300000061000000E7000000600000002600000053000000F500070009000000620000009E000000600000005600000053000000F50007000900000063000000B6000000600000005700000053000000F50007000900000064000000CE000000600000005800000053000000F50007000900000065000000E6000000600000005900000053000000B10005004C00000066000000610000001B000000F6000400550000006000000000000000FA000400660000006000000055000000F800020060000000800005000300000067000000310000003B0000008700050003000000680000006700000025000000840005000300000069000000460000002200000084000500030000006A000000540000002300000080000500030000006B000000690000006A00000084000500030000006C000000610000002800000080000500030000006D0000006B0000006C00000080000500030000006E0000006D00000068000000410006006F0000007000000011000000260000006E0000003D000400090000007100000070000000800005000300000072000000610000001D00000084000500030000007300000072000000280000008000050003000000740000006B000000730000008000050003000000750000007400000068000000410006006F000000760000001100000026000000750000003D000400090000007700000076000000800005000300000078000000610000001F000000840005000300000079000000780000002800000080000500030000007A0000006B0000007900000080000500030000007B0000007A00000068000000410006006F0000007C00000011000000260000007B0000003D000400090000007D0000007C00000080000500030000007E000000610000001E00000084000500030000007F0000007E000000280000008000050003000000800000006B0000007F0000008000050003000000810000008000000068000000410006006F000000820000001100000026000000810000003D0004000900000083000000820000008000050003000000840000003C000000460000008000050003000000850000003D00000054000000800005000300000086000000320000008400000080000500030000008700000033000000850000008700050003000000880000006100000025000000840005000300000089000000860000002400000084000500030000008A000000870000002500000080000500030000008B000000890000008A00000080000500030000008C0000008B00000088000000410006006F0000008D00000016000000260000008C0000003D000400090000008E0000008D000000510005000A0000008F0000008E000000000000005000070009000000900000008F0000008F0000008F0000008F00000085000500090000009100000090000000710000008100050009000000920000009100000062000000510005000A000000930000008E000000010000005000070009000000940000009300000093000000930000009300000085000500090000009500000094000000770000008100050009000000960000009500000092000000510005000A000000970000008E0000000200000050000700090000009800000097000000970000009700000097000000850005000900000099000000980000007D00000081000500090000009A0000009900000096000000510005000A0000009B0000008E0000000300000050000700090000009C0000009B0000009B0000009B0000009B00000085000500090000009D0000009C0000008300000081000500090000009E0000009D0000009A00000080000500030000009F000000460000001F0000008000050003000000A00000003C0000009F0000008000050003000000A100000032000000A00000008400050003000000A2000000A1000000240000008000050003000000A3000000A20000008A0000008000050003000000A4000000A300000088000000410006006F000000A50000001600000026000000A40000003D00040009000000A6000000A5000000510005000A000000A7000000A6000000000000005000070009000000A8000000A7000000A7000000A7000000A70000008500050009000000A9000000A8000000710000008100050009000000AA000000A900000063000000510005000A000000AB000000A6000000010000005000070009000000AC000000AB000000AB000000AB000000AB0000008500050009000000AD000000AC000000770000008100050009000000AE000000AD000000AA000000510005000A000000AF000000A6000000020000005000070009000000B0000000AF000000AF000000AF000000AF0000008500050009000000B1000000B00000007D0000008100050009000000B2000000B1000000AE000000510005000A000000B3000000A6000000030000005000070009000000B4000000B3000000B3000000B3000000B30000008500050009000000B5000000B4000000830000008100050009000000B6000000B5000000B20000008000050003000000B700000046000000250000008000050003000000B80000003C000000B70000008000050003000000B900000032000000B80000008400050003000000BA000000B9000000240000008000050003000000BB000000BA0000008A0000008000050003000000BC000000BB00000088000000410006006F000000BD0000001600000026000000BC0000003D00040009000000BE000000BD000000510005000A000000BF000000BE000000000000005000070009000000C0000000BF000000BF000000BF000000BF0000008500050009000000C1000000C0000000710000008100050009000000C2000000C100000064000000510005000A000000C3000000BE000000010000005000070009000000C4000000C3000000C3000000C3000000C30000008500050009000000C5000000C4000000770000008100050009000000C6000000C5000000C2000000510005000A000000C7000000BE000000020000005000070009000000C8000000C7000000C7000000C7000000C70000008500050009000000C9000000C80000007D0000008100050009000000CA000000C9000000C6000000510005000A000000CB000000BE000000030000005000070009000000CC000000CB000000CB000000CB000000CB0000008500050009000000CD000000CC000000830000008100050009000000CE000000CD000000CA0000008000050003000000CF000000460000001C0000008000050003000000D00000003C000000CF0000008000050003000000D100000032000000D00000008400050003000000D2000000D1000000240000008000050003000000D3000000D20000008A0000008000050003000000D4000000D300000088000000410006006F000000D50000001600000026000000D40000003D00040009000000D6000000D5000000510005000A000000D7000000D6000000000000005000070009000000D8000000D7000000D7000000D7000000D70000008500050009000000D9000000D8000000710000008100050009000000DA000000D900000065000000510005000A000000DB000000D6000000010000005000070009000000DC000000DB000000DB000000DB000000DB0000008500050009000000DD000000DC000000770000008100050009000000DE000000DD000000DA000000510005000A000000DF000000D6000000020000005000070009000000E0000000DF000000DF000000DF000000DF0000008500050009000000E1000000E00000007D0000008100050009000000E2000000E1000000DE000000510005000A000000E3000000D6000000030000005000070009000000E4000000E3000000E3000000E3000000E30000008500050009000000E5000000E4000000830000008100050009000000E6000000E5000000E20000003E0003005B0000009E0000003E0003005C000000B60000003E0003005D000000CE0000003E0003005E000000E60000008000050003000000E70000006100000025000000F90002005F000000F8000200550000003D00040009000000E80000005E0000003D00040009000000E90000005D0000003D00040009000000EA0000005C0000003D00040009000000EB0000005B0000003E0003004E000000EB0000003E0003004F000000EA0000003E00030050000000E90000003E00030051000000E80000008000050003000000EC000000540000001D000000F900020052000000F8000200470000003D00040009000000ED000000510000003D00040009000000EE000000500000003D00040009000000EF0000004F0000003D00040009000000F00000004E0000003E0003003F000000F00000003E00030040000000EF0000003E00030041000000EE0000003E00030042000000ED0000008000050003000000F1000000460000001D000000F900020043000000F8000200450000003D00040009000000F2000000420000003D00040009000000F3000000410000003D00040009000000F4000000400000003D00040009000000F50000003F0000008000050003000000F60000003A0000001E0000008000050003000000F70000002F000000F60000008000050003000000F800000030000000370000008000050003000000F9000000310000003B0000008700050003000000FA000000F9000000250000008400050003000000FB000000F7000000270000008400050003000000FC000000F8000000280000008000050003000000FD000000FB000000FC0000008000050003000000FE000000FD000000FA000000410006006F000000FF0000000C00000026000000FE0000003E000300FF000000F20000008000050003000000000100003A0000001F0000008000050003000000010100002F00000000010000840005000300000002010000010100002700000080000500030000000301000002010000FC00000080000500030000000401000003010000FA000000410006006F000000050100000C00000026000000040100003E00030005010000F30000008000050003000000060100003A0000001D0000008000050003000000070100002F00000006010000840005000300000008010000070100002700000080000500030000000901000008010000FC00000080000500030000000A01000009010000FA000000410006006F0000000B0100000C000000260000000A0100003E0003000B010000F400000080000500030000000C0100002F0000003A00000084000500030000000D0100000C0100002700000080000500030000000E0100000D010000FC00000080000500030000000F0100000E010000FA000000410006006F000000100100000C000000260000000F0100003E00030010010000F5000000FD0001003800010008000C0004000800"> : vector<6380xi8>
    vm.func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !vm.ref<!hal.executable> {
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
      vm.cond_br %_device_match_id_0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      %c1397773893 = vm.const.i32 1397773893 : i32
      %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv = vm.const.ref.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv : !vm.ref<!iree.byte_buffer>
      %ref_0 = vm.call.variadic @hal.executable.create(%ref, %c1397773893, %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv, [%_executable_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
      vm.br ^bb3(%ref_0 : !vm.ref<!hal.executable>)
    ^bb2:  // pred: ^bb0
      %null = vm.const.ref.zero : !vm.ref<!hal.executable>
      vm.br ^bb3(%null : !vm.ref<!hal.executable>)
    ^bb3(%0: !vm.ref<!hal.executable>):  // 2 preds: ^bb1, ^bb2
      vm.return %0 : !vm.ref<!hal.executable>
    }
    vm.func @predict_ex_dispatch_1(%arg0: !vm.ref<!hal.buffer>, %arg1: !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer> attributes {noinline} {
      %c3240000 = vm.const.i32 3240000 : i32
      %c18432 = vm.const.i32 18432 : i32
      %c1 = vm.const.i32 1 : i32
      %c1605632 = vm.const.i32 1605632 : i32
      %zero = vm.const.i32.zero : i32
      %c2 = vm.const.i32 2 : i32
      %c28 = vm.const.i32 28 : i32
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_0 = vm.call @hal.device.allocator(%ref) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
      %c50 = vm.const.i32 50 : i32
      %c15 = vm.const.i32 15 : i32
      %ref_1 = vm.call @hal.allocator.allocate(%ref_0, %c50, %c15, %c1605632) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
      %c1_2 = vm.const.i32 1 : i32
      %c3 = vm.const.i32 3 : i32
      %ref_3 = vm.call @hal.command_buffer.create(%ref, %c1_2, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
      vm.call @hal.command_buffer.begin(%ref_3) : (!vm.ref<!hal.command_buffer>) -> ()
      %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      vm.call.variadic @hal.command_buffer.push_descriptor_set(%ref_3, %_executable_layout_0, %zero, [(%zero, %arg0, %zero, %c3240000), (%c1, %arg1, %zero, %c18432), (%c2, %ref_1, %zero, %c1605632)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...)
      %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
      vm.cond_br %_device_match_id_0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %_executable_predict_ex_dispatch_1_dispatch_0 = vm.global.load.ref @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
      %zero_4 = vm.const.i32.zero : i32
      vm.call @hal.command_buffer.dispatch(%ref_3, %_executable_predict_ex_dispatch_1_dispatch_0, %zero_4, %c2, %c28, %c28) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
      %c20 = vm.const.i32 20 : i32
      %c5 = vm.const.i32 5 : i32
      %zero_5 = vm.const.i32.zero : i32
      vm.call @hal.command_buffer.execution_barrier(%ref_3, %c20, %c5, %zero_5) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
      vm.call @hal.command_buffer.end(%ref_3) : (!vm.ref<!hal.command_buffer>) -> ()
      vm.call @hal.ex.submit_and_wait(%ref, %ref_3) : (!vm.ref<!hal.device>, !vm.ref<!hal.command_buffer>) -> ()
      vm.return %ref_1 : !vm.ref<!hal.buffer>
    ^bb2:  // pred: ^bb0
      %c2_6 = vm.const.i32 2 : i32
      vm.fail %c2_6, "unreachable location reached"
    }
    vm.export @predict_ex_dispatch_1 as("predict_ex_dispatch_1$raw")
    vm.func @predict_ex_dispatch_1$async(%arg0: !vm.ref<!hal.semaphore>, %arg1: i32, %arg2: !vm.ref<!hal.buffer_view>, %arg3: !vm.ref<!hal.buffer_view>, %arg4: !vm.ref<!hal.semaphore>, %arg5: i32) -> !vm.ref<!hal.buffer_view> {
      %c1 = vm.const.i32 1 : i32
      %c112 = vm.const.i32 112 : i32
      %c32 = vm.const.i32 32 : i32
      %c50331680 = vm.const.i32 50331680 : i32
      %0 = vm.call @hal.semaphore.await(%arg0, %arg1) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_fail %0, "semaphore wait failed"
      %ref = vm.call @hal.buffer_view.buffer(%arg2) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_0 = vm.call @hal.buffer_view.buffer(%arg3) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_1 = vm.call @predict_ex_dispatch_1(%ref, %ref_0) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
      %ref_2 = vm.call.variadic @hal.buffer_view.create(%ref_1, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
      vm.call @hal.semaphore.signal(%arg4, %arg5) : (!vm.ref<!hal.semaphore>, i32) -> ()
      vm.return %ref_2 : !vm.ref<!hal.buffer_view>
    }
    vm.export @predict_ex_dispatch_1$async
    vm.func @predict_ex_dispatch_1$sync(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
      %zero = vm.const.i32.zero : i32
      %c1 = vm.const.i32 1 : i32
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_0 = vm.call @hal.semaphore.create(%ref, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.semaphore>
      %ref_1 = vm.call @predict_ex_dispatch_1$async(%ref_0, %zero, %arg0, %arg1, %ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32, !vm.ref<!hal.buffer_view>, !vm.ref<!hal.buffer_view>, !vm.ref<!hal.semaphore>, i32) -> !vm.ref<!hal.buffer_view>
      %0 = vm.call @hal.semaphore.await(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_fail %0, "semaphore wait failed"
      vm.return %ref_1 : !vm.ref<!hal.buffer_view>
    }
    vm.export @predict_ex_dispatch_1$sync as("predict_ex_dispatch_1")
    vm.import @hal.ex.shared_device() -> !vm.ref<!hal.device> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.ex.submit_and_wait(%device : !vm.ref<!hal.device>, %command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.allocator.allocate(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %allocation_size : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
    vm.import @hal.allocator.wrap.byte_buffer(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %source : !vm.ref<!iree.byte_buffer>, %offset : i32, %length : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
    vm.import @hal.buffer.allocator(%buffer : !vm.ref<!hal.buffer>) -> !vm.ref<!hal.allocator> attributes {sym_visibility = "private"}
    vm.import @hal.buffer.subspan(%source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %length : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
    vm.import @hal.buffer.fill(%target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32, %pattern : i32) attributes {sym_visibility = "private"}
    vm.import @hal.buffer.load(%source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %length : i32) -> i32 attributes {sym_visibility = "private"}
    vm.import @hal.buffer.store(%value : i32, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32) attributes {sym_visibility = "private"}
    vm.import @hal.buffer_view.create(%buffer : !vm.ref<!hal.buffer>, %element_type : i32, %shape : i32 ...) -> !vm.ref<!hal.buffer_view> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.buffer(%buffer_view : !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.byte_length(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.element_type(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.rank(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.dim(%buffer_view : !vm.ref<!hal.buffer_view>, %index : i32) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.trace(%key : !vm.ref<!iree.byte_buffer>, %operands : !vm.ref<!hal.buffer_view> ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.create(%device : !vm.ref<!hal.device>, %modes : i32, %command_categories : i32) -> !vm.ref<!hal.command_buffer> attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.begin(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.end(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.execution_barrier(%command_buffer : !vm.ref<!hal.command_buffer>, %source_stage_mask : i32, %target_stage_mask : i32, %flags : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.fill_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32, %pattern : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.copy_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.push_constants(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %offset : i32, %values : i32 ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.push_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.bind_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %descriptor_set : !vm.ref<!hal.descriptor_set>, %dynamic_offsets : i32 ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.dispatch(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroup_x : i32, %workgroup_y : i32, %workgroup_z : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.dispatch.indirect(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroups_buffer : !vm.ref<!hal.buffer>, %workgroups_offset : i32) attributes {sym_visibility = "private"}
    vm.import @hal.descriptor_set.create(%device : !vm.ref<!hal.device>, %set_layout : !vm.ref<!hal.descriptor_set_layout>, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) -> !vm.ref<!hal.descriptor_set> attributes {sym_visibility = "private"}
    vm.import @hal.descriptor_set_layout.create(%device : !vm.ref<!hal.device>, %usage_type : i32, %bindings : tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.device.allocator(%device : !vm.ref<!hal.device>) -> !vm.ref<!hal.allocator> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.device.match.id(%device : !vm.ref<!hal.device>, %pattern : !vm.ref<!iree.byte_buffer>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.executable.create(%device : !vm.ref<!hal.device>, %executable_format : i32, %executable_data : !vm.ref<!iree.byte_buffer>, %executable_layouts : !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.executable_layout.create(%device : !vm.ref<!hal.device>, %push_constants : i32, %set_layouts : !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.semaphore.create(%device : !vm.ref<!hal.device>, %initial_value : i32) -> !vm.ref<!hal.semaphore> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.semaphore.query(%semaphore : !vm.ref<!hal.semaphore>) -> (i32, i32) attributes {sym_visibility = "private"}
    vm.import @hal.semaphore.signal(%semaphore : !vm.ref<!hal.semaphore>, %new_value : i32) attributes {sym_visibility = "private"}
    vm.import @hal.semaphore.fail(%semaphore : !vm.ref<!hal.semaphore>, %status : i32) attributes {sym_visibility = "private"}
    vm.import @hal.semaphore.await(%semaphore : !vm.ref<!hal.semaphore>, %min_value : i32) -> i32 attributes {sym_visibility = "private"}
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::VM::GlobalInitializationPass ***
vm.module @module {
  vm.global.i32 @_device_match_id_0 mutable : i32
  vm.rodata @_utf8_vulkan_7197BF52A22CAFD7 dense<[118, 117, 108, 107, 97, 110, 42]> : vector<7xi8>
  vm.func private @_device_match_id_0_initializer() -> i32 {
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %_utf8_vulkan_7197BF52A22CAFD7 = vm.const.ref.rodata @_utf8_vulkan_7197BF52A22CAFD7 : !vm.ref<!iree.byte_buffer>
    %0 = vm.call @hal.device.match.id(%ref, %_utf8_vulkan_7197BF52A22CAFD7) : (!vm.ref<!hal.device>, !vm.ref<!iree.byte_buffer>) -> i32
    vm.return %0 : i32
  }
  vm.global.ref @_descriptor_set_layout_0 mutable : !vm.ref<!hal.descriptor_set_layout>
  vm.func private @_descriptor_set_layout_0_initializer() -> !vm.ref<!hal.descriptor_set_layout> {
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %c1 = vm.const.i32 1 : i32
    %zero = vm.const.i32.zero : i32
    %c7 = vm.const.i32 7 : i32
    %c1_0 = vm.const.i32 1 : i32
    %c1_1 = vm.const.i32 1 : i32
    %c7_2 = vm.const.i32 7 : i32
    %c1_3 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    %c7_4 = vm.const.i32 7 : i32
    %c6 = vm.const.i32 6 : i32
    %ref_5 = vm.call.variadic @hal.descriptor_set_layout.create(%ref, %c1, [(%zero, %c7, %c1_0), (%c1_1, %c7_2, %c1_3), (%c2, %c7_4, %c6)]) : (!vm.ref<!hal.device>, i32, tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout>
    vm.return %ref_5 : !vm.ref<!hal.descriptor_set_layout>
  }
  vm.global.ref @_executable_layout_0 mutable : !vm.ref<!hal.executable_layout>
  vm.func private @_executable_layout_0_initializer() -> !vm.ref<!hal.executable_layout> {
    %_descriptor_set_layout_0 = vm.global.load.ref @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %zero = vm.const.i32.zero : i32
    %ref_0 = vm.call.variadic @hal.executable_layout.create(%ref, %zero, [%_descriptor_set_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout>
    vm.return %ref_0 : !vm.ref<!hal.executable_layout>
  }
  vm.global.ref @_executable_predict_ex_dispatch_1_dispatch_0 mutable : !vm.ref<!hal.executable>
  vm.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv dense<"0x080000005350564524E7FFFF0800000034000000010000000400000020000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000027060000030223070000010016000000110100000000000011000200010000000A000B005350565F4B48525F73746F726167655F6275666665725F73746F726167655F636C617373000000000E00030000000000010000000F000E000500000019000000707265646963745F65785F64697370617463685F315F64697370617463685F3000000000050000000400000010000600190000001100000004000000040000000100000005000B00040000005F5F6275696C74696E5F7661725F4C6F63616C496E766F636174696F6E49645F5F00000005000900050000005F5F6275696C74696E5F7661725F576F726B67726F757049645F5F0005000A000C0000005F5F7265736F757263655F7661725F32343838393331333339323239365F5F0005000A00110000005F5F7265736F757263655F7661725F32343838393331333430383837325F5F0005000A00160000005F5F7265736F757263655F7661725F32343838393331333339323130345F5F0005000B0019000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000047000400040000000B0000001B00000047000400050000000B0000001A000000470004000800000006000000100000004800050007000000000000002300000000000000470003000700000002000000470004000C0000002100000002000000470004000C0000002200000000000000470004000F0000000600000010000000480005000E000000000000002300000000000000470003000E00000002000000470004001100000021000000010000004700040011000000220000000000000047000400140000000600000010000000480005001300000000000000230000000000000047000300130000000200000047000400160000002100000000000000470004001600000022000000000000001500040003000000200000000000000017000400020000000300000003000000200004000100000001000000020000003B0004000100000004000000010000003B000400010000000500000001000000160003000A0000002000000017000400090000000A000000040000002B000400030000000B000000008801001C00040008000000090000000B0000001E000300070000000800000020000400060000000C000000070000003B000400060000000C0000000C0000002B0004000300000010000000800400001C0004000F00000009000000100000001E0003000E0000000F000000200004000D0000000C0000000E0000003B0004000D000000110000000C0000002B0004000300000015000000041703001C0004001400000009000000150000001E000300130000001400000020000400120000000C000000130000003B00040012000000160000000C00000013000200180000002100030017000000180000002B000400030000001B000000100000002B000400030000001C000000060000002B000400030000001D000000010000002B000400030000001E000000030000002B000400030000001F000000020000002B0004000A00000021000000000000002C0007000900000020000000210000002100000021000000210000002B0004000300000022000000800100002B0004000300000023000000800000002B0004000300000024000000840300002B0004000300000025000000040000002B0004000300000026000000000000002B0004000300000027000000800300002B000400030000002800000008000000200004003E0000000700000009000000140002004C000000200004006F0000000C000000090000003600050018000000190000000000000017000000F80002001A0000003B0004003E0000003F000000070000003B0004003E00000040000000070000003B0004003E00000041000000070000003B0004003E00000042000000070000003B0004003E0000004E000000070000003B0004003E0000004F000000070000003B0004003E00000050000000070000003B0004003E00000051000000070000003B0004003E0000005B000000070000003B0004003E0000005C000000070000003B0004003E0000005D000000070000003B0004003E0000005E000000070000003D00040002000000290000000500000051000500030000002A00000029000000000000003D000400020000002B0000000500000051000500030000002C0000002B000000010000003D000400020000002D0000000500000051000500030000002E0000002D0000000200000084000500030000002F0000002E000000250000008400050003000000300000002C000000250000008400050003000000310000002A0000001B0000008400050003000000320000002F0000001F000000840005000300000033000000300000001F0000003D00040002000000340000000400000051000500030000003500000034000000000000003D00040002000000360000000400000051000500030000003700000036000000010000003D000400020000003800000004000000510005000300000039000000380000000200000084000500030000003A000000390000002500000084000500030000003B000000350000002500000084000500030000003C000000390000002800000084000500030000003D000000370000001F000000F900020043000000F800020043000000F50007000300000046000000F100000047000000260000001A000000F50007000900000048000000F000000047000000200000001A000000F50007000900000049000000EF00000047000000200000001A000000F5000700090000004A000000EE00000047000000200000001A000000F5000700090000004B000000ED00000047000000200000001A000000B10005004C0000004D000000460000001E000000F6000400450000004400000000000000FA0004004D0000004400000045000000F800020044000000F900020052000000F800020052000000F50007000300000054000000EC000000550000002600000044000000F50007000900000056000000EB000000550000004800000044000000F50007000900000057000000EA000000550000004900000044000000F50007000900000058000000E9000000550000004A00000044000000F50007000900000059000000E8000000550000004B00000044000000B10005004C0000005A000000540000001E000000F6000400470000005300000000000000FA0004005A0000005300000047000000F800020053000000F90002005F000000F80002005F000000F50007000300000061000000E7000000600000002600000053000000F500070009000000620000009E000000600000005600000053000000F50007000900000063000000B6000000600000005700000053000000F50007000900000064000000CE000000600000005800000053000000F50007000900000065000000E6000000600000005900000053000000B10005004C00000066000000610000001B000000F6000400550000006000000000000000FA000400660000006000000055000000F800020060000000800005000300000067000000310000003B0000008700050003000000680000006700000025000000840005000300000069000000460000002200000084000500030000006A000000540000002300000080000500030000006B000000690000006A00000084000500030000006C000000610000002800000080000500030000006D0000006B0000006C00000080000500030000006E0000006D00000068000000410006006F0000007000000011000000260000006E0000003D000400090000007100000070000000800005000300000072000000610000001D00000084000500030000007300000072000000280000008000050003000000740000006B000000730000008000050003000000750000007400000068000000410006006F000000760000001100000026000000750000003D000400090000007700000076000000800005000300000078000000610000001F000000840005000300000079000000780000002800000080000500030000007A0000006B0000007900000080000500030000007B0000007A00000068000000410006006F0000007C00000011000000260000007B0000003D000400090000007D0000007C00000080000500030000007E000000610000001E00000084000500030000007F0000007E000000280000008000050003000000800000006B0000007F0000008000050003000000810000008000000068000000410006006F000000820000001100000026000000810000003D0004000900000083000000820000008000050003000000840000003C000000460000008000050003000000850000003D00000054000000800005000300000086000000320000008400000080000500030000008700000033000000850000008700050003000000880000006100000025000000840005000300000089000000860000002400000084000500030000008A000000870000002500000080000500030000008B000000890000008A00000080000500030000008C0000008B00000088000000410006006F0000008D00000016000000260000008C0000003D000400090000008E0000008D000000510005000A0000008F0000008E000000000000005000070009000000900000008F0000008F0000008F0000008F00000085000500090000009100000090000000710000008100050009000000920000009100000062000000510005000A000000930000008E000000010000005000070009000000940000009300000093000000930000009300000085000500090000009500000094000000770000008100050009000000960000009500000092000000510005000A000000970000008E0000000200000050000700090000009800000097000000970000009700000097000000850005000900000099000000980000007D00000081000500090000009A0000009900000096000000510005000A0000009B0000008E0000000300000050000700090000009C0000009B0000009B0000009B0000009B00000085000500090000009D0000009C0000008300000081000500090000009E0000009D0000009A00000080000500030000009F000000460000001F0000008000050003000000A00000003C0000009F0000008000050003000000A100000032000000A00000008400050003000000A2000000A1000000240000008000050003000000A3000000A20000008A0000008000050003000000A4000000A300000088000000410006006F000000A50000001600000026000000A40000003D00040009000000A6000000A5000000510005000A000000A7000000A6000000000000005000070009000000A8000000A7000000A7000000A7000000A70000008500050009000000A9000000A8000000710000008100050009000000AA000000A900000063000000510005000A000000AB000000A6000000010000005000070009000000AC000000AB000000AB000000AB000000AB0000008500050009000000AD000000AC000000770000008100050009000000AE000000AD000000AA000000510005000A000000AF000000A6000000020000005000070009000000B0000000AF000000AF000000AF000000AF0000008500050009000000B1000000B00000007D0000008100050009000000B2000000B1000000AE000000510005000A000000B3000000A6000000030000005000070009000000B4000000B3000000B3000000B3000000B30000008500050009000000B5000000B4000000830000008100050009000000B6000000B5000000B20000008000050003000000B700000046000000250000008000050003000000B80000003C000000B70000008000050003000000B900000032000000B80000008400050003000000BA000000B9000000240000008000050003000000BB000000BA0000008A0000008000050003000000BC000000BB00000088000000410006006F000000BD0000001600000026000000BC0000003D00040009000000BE000000BD000000510005000A000000BF000000BE000000000000005000070009000000C0000000BF000000BF000000BF000000BF0000008500050009000000C1000000C0000000710000008100050009000000C2000000C100000064000000510005000A000000C3000000BE000000010000005000070009000000C4000000C3000000C3000000C3000000C30000008500050009000000C5000000C4000000770000008100050009000000C6000000C5000000C2000000510005000A000000C7000000BE000000020000005000070009000000C8000000C7000000C7000000C7000000C70000008500050009000000C9000000C80000007D0000008100050009000000CA000000C9000000C6000000510005000A000000CB000000BE000000030000005000070009000000CC000000CB000000CB000000CB000000CB0000008500050009000000CD000000CC000000830000008100050009000000CE000000CD000000CA0000008000050003000000CF000000460000001C0000008000050003000000D00000003C000000CF0000008000050003000000D100000032000000D00000008400050003000000D2000000D1000000240000008000050003000000D3000000D20000008A0000008000050003000000D4000000D300000088000000410006006F000000D50000001600000026000000D40000003D00040009000000D6000000D5000000510005000A000000D7000000D6000000000000005000070009000000D8000000D7000000D7000000D7000000D70000008500050009000000D9000000D8000000710000008100050009000000DA000000D900000065000000510005000A000000DB000000D6000000010000005000070009000000DC000000DB000000DB000000DB000000DB0000008500050009000000DD000000DC000000770000008100050009000000DE000000DD000000DA000000510005000A000000DF000000D6000000020000005000070009000000E0000000DF000000DF000000DF000000DF0000008500050009000000E1000000E00000007D0000008100050009000000E2000000E1000000DE000000510005000A000000E3000000D6000000030000005000070009000000E4000000E3000000E3000000E3000000E30000008500050009000000E5000000E4000000830000008100050009000000E6000000E5000000E20000003E0003005B0000009E0000003E0003005C000000B60000003E0003005D000000CE0000003E0003005E000000E60000008000050003000000E70000006100000025000000F90002005F000000F8000200550000003D00040009000000E80000005E0000003D00040009000000E90000005D0000003D00040009000000EA0000005C0000003D00040009000000EB0000005B0000003E0003004E000000EB0000003E0003004F000000EA0000003E00030050000000E90000003E00030051000000E80000008000050003000000EC000000540000001D000000F900020052000000F8000200470000003D00040009000000ED000000510000003D00040009000000EE000000500000003D00040009000000EF0000004F0000003D00040009000000F00000004E0000003E0003003F000000F00000003E00030040000000EF0000003E00030041000000EE0000003E00030042000000ED0000008000050003000000F1000000460000001D000000F900020043000000F8000200450000003D00040009000000F2000000420000003D00040009000000F3000000410000003D00040009000000F4000000400000003D00040009000000F50000003F0000008000050003000000F60000003A0000001E0000008000050003000000F70000002F000000F60000008000050003000000F800000030000000370000008000050003000000F9000000310000003B0000008700050003000000FA000000F9000000250000008400050003000000FB000000F7000000270000008400050003000000FC000000F8000000280000008000050003000000FD000000FB000000FC0000008000050003000000FE000000FD000000FA000000410006006F000000FF0000000C00000026000000FE0000003E000300FF000000F20000008000050003000000000100003A0000001F0000008000050003000000010100002F00000000010000840005000300000002010000010100002700000080000500030000000301000002010000FC00000080000500030000000401000003010000FA000000410006006F000000050100000C00000026000000040100003E00030005010000F30000008000050003000000060100003A0000001D0000008000050003000000070100002F00000006010000840005000300000008010000070100002700000080000500030000000901000008010000FC00000080000500030000000A01000009010000FA000000410006006F0000000B0100000C000000260000000A0100003E0003000B010000F400000080000500030000000C0100002F0000003A00000084000500030000000D0100000C0100002700000080000500030000000E0100000D010000FC00000080000500030000000F0100000E010000FA000000410006006F000000100100000C000000260000000F0100003E00030010010000F5000000FD0001003800010008000C0004000800"> : vector<6380xi8>
  vm.func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !vm.ref<!hal.executable> {
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
    vm.cond_br %_device_match_id_0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
    %c1397773893 = vm.const.i32 1397773893 : i32
    %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv = vm.const.ref.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv : !vm.ref<!iree.byte_buffer>
    %ref_0 = vm.call.variadic @hal.executable.create(%ref, %c1397773893, %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv, [%_executable_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
    vm.br ^bb3(%ref_0 : !vm.ref<!hal.executable>)
  ^bb2:  // pred: ^bb0
    %null = vm.const.ref.zero : !vm.ref<!hal.executable>
    vm.br ^bb3(%null : !vm.ref<!hal.executable>)
  ^bb3(%0: !vm.ref<!hal.executable>):  // 2 preds: ^bb1, ^bb2
    vm.return %0 : !vm.ref<!hal.executable>
  }
  vm.func @predict_ex_dispatch_1(%arg0: !vm.ref<!hal.buffer>, %arg1: !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer> attributes {noinline} {
    %c3240000 = vm.const.i32 3240000 : i32
    %c18432 = vm.const.i32 18432 : i32
    %c1 = vm.const.i32 1 : i32
    %c1605632 = vm.const.i32 1605632 : i32
    %zero = vm.const.i32.zero : i32
    %c2 = vm.const.i32 2 : i32
    %c28 = vm.const.i32 28 : i32
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %ref_0 = vm.call @hal.device.allocator(%ref) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
    %c50 = vm.const.i32 50 : i32
    %c15 = vm.const.i32 15 : i32
    %ref_1 = vm.call @hal.allocator.allocate(%ref_0, %c50, %c15, %c1605632) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
    %c1_2 = vm.const.i32 1 : i32
    %c3 = vm.const.i32 3 : i32
    %ref_3 = vm.call @hal.command_buffer.create(%ref, %c1_2, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
    vm.call @hal.command_buffer.begin(%ref_3) : (!vm.ref<!hal.command_buffer>) -> ()
    %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
    vm.call.variadic @hal.command_buffer.push_descriptor_set(%ref_3, %_executable_layout_0, %zero, [(%zero, %arg0, %zero, %c3240000), (%c1, %arg1, %zero, %c18432), (%c2, %ref_1, %zero, %c1605632)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...)
    %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
    vm.cond_br %_device_match_id_0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %_executable_predict_ex_dispatch_1_dispatch_0 = vm.global.load.ref @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
    %zero_4 = vm.const.i32.zero : i32
    vm.call @hal.command_buffer.dispatch(%ref_3, %_executable_predict_ex_dispatch_1_dispatch_0, %zero_4, %c2, %c28, %c28) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
    %c20 = vm.const.i32 20 : i32
    %c5 = vm.const.i32 5 : i32
    %zero_5 = vm.const.i32.zero : i32
    vm.call @hal.command_buffer.execution_barrier(%ref_3, %c20, %c5, %zero_5) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
    vm.call @hal.command_buffer.end(%ref_3) : (!vm.ref<!hal.command_buffer>) -> ()
    vm.call @hal.ex.submit_and_wait(%ref, %ref_3) : (!vm.ref<!hal.device>, !vm.ref<!hal.command_buffer>) -> ()
    vm.return %ref_1 : !vm.ref<!hal.buffer>
  ^bb2:  // pred: ^bb0
    %c2_6 = vm.const.i32 2 : i32
    vm.fail %c2_6, "unreachable location reached"
  }
  vm.export @predict_ex_dispatch_1 as("predict_ex_dispatch_1$raw")
  vm.func @predict_ex_dispatch_1$async(%arg0: !vm.ref<!hal.semaphore>, %arg1: i32, %arg2: !vm.ref<!hal.buffer_view>, %arg3: !vm.ref<!hal.buffer_view>, %arg4: !vm.ref<!hal.semaphore>, %arg5: i32) -> !vm.ref<!hal.buffer_view> {
    %c1 = vm.const.i32 1 : i32
    %c112 = vm.const.i32 112 : i32
    %c32 = vm.const.i32 32 : i32
    %c50331680 = vm.const.i32 50331680 : i32
    %0 = vm.call @hal.semaphore.await(%arg0, %arg1) : (!vm.ref<!hal.semaphore>, i32) -> i32
    vm.cond_fail %0, "semaphore wait failed"
    %ref = vm.call @hal.buffer_view.buffer(%arg2) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
    %ref_0 = vm.call @hal.buffer_view.buffer(%arg3) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
    %ref_1 = vm.call @predict_ex_dispatch_1(%ref, %ref_0) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
    %ref_2 = vm.call.variadic @hal.buffer_view.create(%ref_1, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
    vm.call @hal.semaphore.signal(%arg4, %arg5) : (!vm.ref<!hal.semaphore>, i32) -> ()
    vm.return %ref_2 : !vm.ref<!hal.buffer_view>
  }
  vm.export @predict_ex_dispatch_1$async
  vm.func @predict_ex_dispatch_1$sync(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %zero = vm.const.i32.zero : i32
    %c1 = vm.const.i32 1 : i32
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %ref_0 = vm.call @hal.semaphore.create(%ref, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.semaphore>
    %ref_1 = vm.call @predict_ex_dispatch_1$async(%ref_0, %zero, %arg0, %arg1, %ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32, !vm.ref<!hal.buffer_view>, !vm.ref<!hal.buffer_view>, !vm.ref<!hal.semaphore>, i32) -> !vm.ref<!hal.buffer_view>
    %0 = vm.call @hal.semaphore.await(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> i32
    vm.cond_fail %0, "semaphore wait failed"
    vm.return %ref_1 : !vm.ref<!hal.buffer_view>
  }
  vm.export @predict_ex_dispatch_1$sync as("predict_ex_dispatch_1")
  vm.import @hal.ex.shared_device() -> !vm.ref<!hal.device> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.ex.submit_and_wait(%device : !vm.ref<!hal.device>, %command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
  vm.import @hal.allocator.allocate(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %allocation_size : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
  vm.import @hal.allocator.wrap.byte_buffer(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %source : !vm.ref<!iree.byte_buffer>, %offset : i32, %length : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
  vm.import @hal.buffer.allocator(%buffer : !vm.ref<!hal.buffer>) -> !vm.ref<!hal.allocator> attributes {sym_visibility = "private"}
  vm.import @hal.buffer.subspan(%source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %length : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
  vm.import @hal.buffer.fill(%target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32, %pattern : i32) attributes {sym_visibility = "private"}
  vm.import @hal.buffer.load(%source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %length : i32) -> i32 attributes {sym_visibility = "private"}
  vm.import @hal.buffer.store(%value : i32, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32) attributes {sym_visibility = "private"}
  vm.import @hal.buffer_view.create(%buffer : !vm.ref<!hal.buffer>, %element_type : i32, %shape : i32 ...) -> !vm.ref<!hal.buffer_view> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.buffer_view.buffer(%buffer_view : !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.buffer_view.byte_length(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.buffer_view.element_type(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.buffer_view.rank(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.buffer_view.dim(%buffer_view : !vm.ref<!hal.buffer_view>, %index : i32) -> i32 attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.buffer_view.trace(%key : !vm.ref<!iree.byte_buffer>, %operands : !vm.ref<!hal.buffer_view> ...) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.create(%device : !vm.ref<!hal.device>, %modes : i32, %command_categories : i32) -> !vm.ref<!hal.command_buffer> attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.begin(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.end(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.execution_barrier(%command_buffer : !vm.ref<!hal.command_buffer>, %source_stage_mask : i32, %target_stage_mask : i32, %flags : i32) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.fill_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32, %pattern : i32) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.copy_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.push_constants(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %offset : i32, %values : i32 ...) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.push_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.bind_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %descriptor_set : !vm.ref<!hal.descriptor_set>, %dynamic_offsets : i32 ...) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.dispatch(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroup_x : i32, %workgroup_y : i32, %workgroup_z : i32) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.dispatch.indirect(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroups_buffer : !vm.ref<!hal.buffer>, %workgroups_offset : i32) attributes {sym_visibility = "private"}
  vm.import @hal.descriptor_set.create(%device : !vm.ref<!hal.device>, %set_layout : !vm.ref<!hal.descriptor_set_layout>, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) -> !vm.ref<!hal.descriptor_set> attributes {sym_visibility = "private"}
  vm.import @hal.descriptor_set_layout.create(%device : !vm.ref<!hal.device>, %usage_type : i32, %bindings : tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.device.allocator(%device : !vm.ref<!hal.device>) -> !vm.ref<!hal.allocator> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.device.match.id(%device : !vm.ref<!hal.device>, %pattern : !vm.ref<!iree.byte_buffer>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.executable.create(%device : !vm.ref<!hal.device>, %executable_format : i32, %executable_data : !vm.ref<!iree.byte_buffer>, %executable_layouts : !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.executable_layout.create(%device : !vm.ref<!hal.device>, %push_constants : i32, %set_layouts : !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.semaphore.create(%device : !vm.ref<!hal.device>, %initial_value : i32) -> !vm.ref<!hal.semaphore> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.semaphore.query(%semaphore : !vm.ref<!hal.semaphore>) -> (i32, i32) attributes {sym_visibility = "private"}
  vm.import @hal.semaphore.signal(%semaphore : !vm.ref<!hal.semaphore>, %new_value : i32) attributes {sym_visibility = "private"}
  vm.import @hal.semaphore.fail(%semaphore : !vm.ref<!hal.semaphore>, %status : i32) attributes {sym_visibility = "private"}
  vm.import @hal.semaphore.await(%semaphore : !vm.ref<!hal.semaphore>, %min_value : i32) -> i32 attributes {sym_visibility = "private"}
  vm.func @__init() {
    %0 = vm.call @_device_match_id_0_initializer() : () -> i32
    vm.global.store.i32 %0, @_device_match_id_0 : i32
    %ref = vm.call @_descriptor_set_layout_0_initializer() : () -> !vm.ref<!hal.descriptor_set_layout>
    vm.global.store.ref %ref, @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
    %ref_0 = vm.call @_executable_layout_0_initializer() : () -> !vm.ref<!hal.executable_layout>
    vm.global.store.ref %ref_0, @_executable_layout_0 : !vm.ref<!hal.executable_layout>
    %ref_1 = vm.call @_executable_predict_ex_dispatch_1_dispatch_0_initializer() : () -> !vm.ref<!hal.executable>
    vm.global.store.ref %ref_1, @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
    vm.return
  }
  vm.export @__init
}

// *** IR Dump After Canonicalizer ***
vm.func private @_executable_predict_ex_dispatch_1_dispatch_0_initializer() -> !vm.ref<!hal.executable> {
  %c1397773893 = vm.const.i32 1397773893 : i32
  %null = vm.const.ref.zero : !vm.ref<!hal.executable>
  %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
  vm.cond_br %_device_match_id_0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
  %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv = vm.const.ref.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv : !vm.ref<!iree.byte_buffer>
  %ref_0 = vm.call.variadic @hal.executable.create(%ref, %c1397773893, %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv, [%_executable_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
  vm.br ^bb3(%ref_0 : !vm.ref<!hal.executable>)
^bb2:  // pred: ^bb0
  vm.br ^bb3(%null : !vm.ref<!hal.executable>)
^bb3(%0: !vm.ref<!hal.executable>):  // 2 preds: ^bb1, ^bb2
  vm.return %0 : !vm.ref<!hal.executable>
}

// *** IR Dump After Canonicalizer ***
vm.func @predict_ex_dispatch_1$sync(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %zero = vm.const.i32.zero : i32
  %c1 = vm.const.i32 1 : i32
  %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %ref_0 = vm.call @hal.semaphore.create(%ref, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.semaphore>
  %ref_1 = vm.call @predict_ex_dispatch_1$async(%ref_0, %zero, %arg0, %arg1, %ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32, !vm.ref<!hal.buffer_view>, !vm.ref<!hal.buffer_view>, !vm.ref<!hal.semaphore>, i32) -> !vm.ref<!hal.buffer_view>
  %0 = vm.call @hal.semaphore.await(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> i32
  vm.cond_br %0, ^bb2(%0 : i32), ^bb1
^bb1:  // pred: ^bb0
  vm.return %ref_1 : !vm.ref<!hal.buffer_view>
^bb2(%1: i32):  // pred: ^bb0
  vm.fail %1, "semaphore wait failed"
}

// *** IR Dump After Canonicalizer ***
vm.func private @_executable_layout_0_initializer() -> !vm.ref<!hal.executable_layout> {
  %zero = vm.const.i32.zero : i32
  %_descriptor_set_layout_0 = vm.global.load.ref @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
  %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %ref_0 = vm.call.variadic @hal.executable_layout.create(%ref, %zero, [%_descriptor_set_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout>
  vm.return %ref_0 : !vm.ref<!hal.executable_layout>
}

// *** IR Dump After Canonicalizer ***
vm.func private @_descriptor_set_layout_0_initializer() -> !vm.ref<!hal.descriptor_set_layout> {
  %zero = vm.const.i32.zero : i32
  %c1 = vm.const.i32 1 : i32
  %c2 = vm.const.i32 2 : i32
  %c7 = vm.const.i32 7 : i32
  %c6 = vm.const.i32 6 : i32
  %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %ref_0 = vm.call.variadic @hal.descriptor_set_layout.create(%ref, %c1, [(%zero, %c7, %c1), (%c1, %c7, %c1), (%c2, %c7, %c6)]) : (!vm.ref<!hal.device>, i32, tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout>
  vm.return %ref_0 : !vm.ref<!hal.descriptor_set_layout>
}

// *** IR Dump After Canonicalizer ***
vm.func @predict_ex_dispatch_1$async(%arg0: !vm.ref<!hal.semaphore>, %arg1: i32, %arg2: !vm.ref<!hal.buffer_view>, %arg3: !vm.ref<!hal.buffer_view>, %arg4: !vm.ref<!hal.semaphore>, %arg5: i32) -> !vm.ref<!hal.buffer_view> {
  %c1 = vm.const.i32 1 : i32
  %c112 = vm.const.i32 112 : i32
  %c32 = vm.const.i32 32 : i32
  %c50331680 = vm.const.i32 50331680 : i32
  %0 = vm.call @hal.semaphore.await(%arg0, %arg1) : (!vm.ref<!hal.semaphore>, i32) -> i32
  vm.cond_br %0, ^bb2(%0 : i32), ^bb1
^bb1:  // pred: ^bb0
  %ref = vm.call @hal.buffer_view.buffer(%arg2) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  %ref_0 = vm.call @hal.buffer_view.buffer(%arg3) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  %ref_1 = vm.call @predict_ex_dispatch_1(%ref, %ref_0) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
  %ref_2 = vm.call.variadic @hal.buffer_view.create(%ref_1, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
  vm.call @hal.semaphore.signal(%arg4, %arg5) : (!vm.ref<!hal.semaphore>, i32) -> ()
  vm.return %ref_2 : !vm.ref<!hal.buffer_view>
^bb2(%1: i32):  // pred: ^bb0
  vm.fail %1, "semaphore wait failed"
}

// *** IR Dump After Canonicalizer ***
vm.func @predict_ex_dispatch_1(%arg0: !vm.ref<!hal.buffer>, %arg1: !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer> attributes {noinline} {
  %c3240000 = vm.const.i32 3240000 : i32
  %c18432 = vm.const.i32 18432 : i32
  %c1605632 = vm.const.i32 1605632 : i32
  %c28 = vm.const.i32 28 : i32
  %c50 = vm.const.i32 50 : i32
  %c15 = vm.const.i32 15 : i32
  %c1 = vm.const.i32 1 : i32
  %c3 = vm.const.i32 3 : i32
  %c20 = vm.const.i32 20 : i32
  %c5 = vm.const.i32 5 : i32
  %zero = vm.const.i32.zero : i32
  %c2 = vm.const.i32 2 : i32
  %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %ref_0 = vm.call @hal.device.allocator(%ref) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
  %ref_1 = vm.call @hal.allocator.allocate(%ref_0, %c50, %c15, %c1605632) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
  %ref_2 = vm.call @hal.command_buffer.create(%ref, %c1, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
  vm.call @hal.command_buffer.begin(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
  %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
  vm.call.variadic @hal.command_buffer.push_descriptor_set(%ref_2, %_executable_layout_0, %zero, [(%zero, %arg0, %zero, %c3240000), (%c1, %arg1, %zero, %c18432), (%c2, %ref_1, %zero, %c1605632)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...)
  %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
  vm.cond_br %_device_match_id_0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %_executable_predict_ex_dispatch_1_dispatch_0 = vm.global.load.ref @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
  vm.call @hal.command_buffer.dispatch(%ref_2, %_executable_predict_ex_dispatch_1_dispatch_0, %zero, %c2, %c28, %c28) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
  vm.call @hal.command_buffer.execution_barrier(%ref_2, %c20, %c5, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
  vm.call @hal.command_buffer.end(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
  vm.call @hal.ex.submit_and_wait(%ref, %ref_2) : (!vm.ref<!hal.device>, !vm.ref<!hal.command_buffer>) -> ()
  vm.return %ref_1 : !vm.ref<!hal.buffer>
^bb2:  // pred: ^bb0
  vm.fail %c2, "unreachable location reached"
}

// *** IR Dump After Canonicalizer ***
vm.func @predict_ex_dispatch_1$async(%arg0: !vm.ref<!hal.semaphore>, %arg1: i32, %arg2: !vm.ref<!hal.buffer_view>, %arg3: !vm.ref<!hal.buffer_view>, %arg4: !vm.ref<!hal.semaphore>, %arg5: i32) -> !vm.ref<!hal.buffer_view> {
  %c1 = vm.const.i32 1 : i32
  %c112 = vm.const.i32 112 : i32
  %c32 = vm.const.i32 32 : i32
  %c50331680 = vm.const.i32 50331680 : i32
  %0 = vm.call @hal.semaphore.await(%arg0, %arg1) : (!vm.ref<!hal.semaphore>, i32) -> i32
  vm.cond_br %0, ^bb2(%0 : i32), ^bb1
^bb1:  // pred: ^bb0
  %ref = vm.call @hal.buffer_view.buffer(%arg2) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  %ref_0 = vm.call @hal.buffer_view.buffer(%arg3) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  %ref_1 = vm.call @predict_ex_dispatch_1(%ref, %ref_0) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
  %ref_2 = vm.call.variadic @hal.buffer_view.create(%ref_1, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
  vm.call @hal.semaphore.signal(%arg4, %arg5) : (!vm.ref<!hal.semaphore>, i32) -> ()
  vm.return %ref_2 : !vm.ref<!hal.buffer_view>
^bb2(%1: i32):  // pred: ^bb0
  vm.fail %1, "semaphore wait failed"
}

// *** IR Dump After Canonicalizer ***
vm.func @__init() {
  %c1 = vm.const.i32 1 : i32
  %c2 = vm.const.i32 2 : i32
  %c7 = vm.const.i32 7 : i32
  %c6 = vm.const.i32 6 : i32
  %zero = vm.const.i32.zero : i32
  %c1397773893 = vm.const.i32 1397773893 : i32
  %null = vm.const.ref.zero : !vm.ref<!hal.executable>
  %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %_utf8_vulkan_7197BF52A22CAFD7 = vm.const.ref.rodata @_utf8_vulkan_7197BF52A22CAFD7 : !vm.ref<!iree.byte_buffer>
  %0 = vm.call @hal.device.match.id(%ref, %_utf8_vulkan_7197BF52A22CAFD7) : (!vm.ref<!hal.device>, !vm.ref<!iree.byte_buffer>) -> i32
  vm.global.store.i32 %0, @_device_match_id_0 : i32
  %ref_0 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %ref_1 = vm.call.variadic @hal.descriptor_set_layout.create(%ref_0, %c1, [(%zero, %c7, %c1), (%c1, %c7, %c1), (%c2, %c7, %c6)]) : (!vm.ref<!hal.device>, i32, tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout>
  vm.global.store.ref %ref_1, @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
  %_descriptor_set_layout_0 = vm.global.load.ref @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
  %ref_2 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %ref_3 = vm.call.variadic @hal.executable_layout.create(%ref_2, %zero, [%_descriptor_set_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout>
  vm.global.store.ref %ref_3, @_executable_layout_0 : !vm.ref<!hal.executable_layout>
  %ref_4 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
  vm.cond_br %_device_match_id_0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
  %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv = vm.const.ref.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv : !vm.ref<!iree.byte_buffer>
  %ref_5 = vm.call.variadic @hal.executable.create(%ref_4, %c1397773893, %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv, [%_executable_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
  vm.br ^bb3(%ref_5 : !vm.ref<!hal.executable>)
^bb2:  // pred: ^bb0
  vm.br ^bb3(%null : !vm.ref<!hal.executable>)
^bb3(%1: !vm.ref<!hal.executable>):  // 2 preds: ^bb1, ^bb2
  vm.global.store.ref %1, @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
  vm.return
}

// *** IR Dump After Canonicalizer ***
vm.func @predict_ex_dispatch_1$sync(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
  %zero = vm.const.i32.zero : i32
  %c1 = vm.const.i32 1 : i32
  %c112 = vm.const.i32 112 : i32
  %c32 = vm.const.i32 32 : i32
  %c50331680 = vm.const.i32 50331680 : i32
  %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %ref_0 = vm.call @hal.semaphore.create(%ref, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.semaphore>
  %0 = vm.call @hal.semaphore.await(%ref_0, %zero) : (!vm.ref<!hal.semaphore>, i32) -> i32
  vm.cond_br %0, ^bb2(%0 : i32), ^bb1
^bb1:  // pred: ^bb0
  %ref_1 = vm.call @hal.buffer_view.buffer(%arg0) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  %ref_2 = vm.call @hal.buffer_view.buffer(%arg1) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  %ref_3 = vm.call @predict_ex_dispatch_1(%ref_1, %ref_2) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
  %ref_4 = vm.call.variadic @hal.buffer_view.create(%ref_3, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
  vm.call @hal.semaphore.signal(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> ()
  %1 = vm.call @hal.semaphore.await(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> i32
  vm.cond_br %1, ^bb2(%1 : i32), ^bb3
^bb2(%2: i32):  // 2 preds: ^bb0, ^bb1
  vm.fail %2, "semaphore wait failed"
^bb3:  // pred: ^bb1
  vm.return %ref_4 : !vm.ref<!hal.buffer_view>
}

// *** IR Dump After Canonicalizer ***
vm.func @predict_ex_dispatch_1(%arg0: !vm.ref<!hal.buffer>, %arg1: !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer> attributes {noinline} {
  %c3240000 = vm.const.i32 3240000 : i32
  %c18432 = vm.const.i32 18432 : i32
  %c1605632 = vm.const.i32 1605632 : i32
  %c28 = vm.const.i32 28 : i32
  %c50 = vm.const.i32 50 : i32
  %c15 = vm.const.i32 15 : i32
  %c1 = vm.const.i32 1 : i32
  %c3 = vm.const.i32 3 : i32
  %c20 = vm.const.i32 20 : i32
  %c5 = vm.const.i32 5 : i32
  %zero = vm.const.i32.zero : i32
  %c2 = vm.const.i32 2 : i32
  %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
  %ref_0 = vm.call @hal.device.allocator(%ref) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
  %ref_1 = vm.call @hal.allocator.allocate(%ref_0, %c50, %c15, %c1605632) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
  %ref_2 = vm.call @hal.command_buffer.create(%ref, %c1, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
  vm.call @hal.command_buffer.begin(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
  %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
  vm.call.variadic @hal.command_buffer.push_descriptor_set(%ref_2, %_executable_layout_0, %zero, [(%zero, %arg0, %zero, %c3240000), (%c1, %arg1, %zero, %c18432), (%c2, %ref_1, %zero, %c1605632)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...)
  %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
  vm.cond_br %_device_match_id_0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %_executable_predict_ex_dispatch_1_dispatch_0 = vm.global.load.ref @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
  vm.call @hal.command_buffer.dispatch(%ref_2, %_executable_predict_ex_dispatch_1_dispatch_0, %zero, %c2, %c28, %c28) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
  vm.call @hal.command_buffer.execution_barrier(%ref_2, %c20, %c5, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
  vm.call @hal.command_buffer.end(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
  vm.call @hal.ex.submit_and_wait(%ref, %ref_2) : (!vm.ref<!hal.device>, !vm.ref<!hal.command_buffer>) -> ()
  vm.return %ref_1 : !vm.ref<!hal.buffer>
^bb2:  // pred: ^bb0
  vm.fail %c2, "unreachable location reached"
}

// *** IR Dump After Inliner ***
module  {
  vm.module @module {
    vm.global.i32 @_device_match_id_0 mutable : i32
    vm.rodata @_utf8_vulkan_7197BF52A22CAFD7 dense<[118, 117, 108, 107, 97, 110, 42]> : vector<7xi8>
    vm.global.ref @_descriptor_set_layout_0 mutable : !vm.ref<!hal.descriptor_set_layout>
    vm.global.ref @_executable_layout_0 mutable : !vm.ref<!hal.executable_layout>
    vm.global.ref @_executable_predict_ex_dispatch_1_dispatch_0 mutable : !vm.ref<!hal.executable>
    vm.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv dense<"0x080000005350564524E7FFFF0800000034000000010000000400000020000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000027060000030223070000010016000000110100000000000011000200010000000A000B005350565F4B48525F73746F726167655F6275666665725F73746F726167655F636C617373000000000E00030000000000010000000F000E000500000019000000707265646963745F65785F64697370617463685F315F64697370617463685F3000000000050000000400000010000600190000001100000004000000040000000100000005000B00040000005F5F6275696C74696E5F7661725F4C6F63616C496E766F636174696F6E49645F5F00000005000900050000005F5F6275696C74696E5F7661725F576F726B67726F757049645F5F0005000A000C0000005F5F7265736F757263655F7661725F32343838393331333339323239365F5F0005000A00110000005F5F7265736F757263655F7661725F32343838393331333430383837325F5F0005000A00160000005F5F7265736F757263655F7661725F32343838393331333339323130345F5F0005000B0019000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000047000400040000000B0000001B00000047000400050000000B0000001A000000470004000800000006000000100000004800050007000000000000002300000000000000470003000700000002000000470004000C0000002100000002000000470004000C0000002200000000000000470004000F0000000600000010000000480005000E000000000000002300000000000000470003000E00000002000000470004001100000021000000010000004700040011000000220000000000000047000400140000000600000010000000480005001300000000000000230000000000000047000300130000000200000047000400160000002100000000000000470004001600000022000000000000001500040003000000200000000000000017000400020000000300000003000000200004000100000001000000020000003B0004000100000004000000010000003B000400010000000500000001000000160003000A0000002000000017000400090000000A000000040000002B000400030000000B000000008801001C00040008000000090000000B0000001E000300070000000800000020000400060000000C000000070000003B000400060000000C0000000C0000002B0004000300000010000000800400001C0004000F00000009000000100000001E0003000E0000000F000000200004000D0000000C0000000E0000003B0004000D000000110000000C0000002B0004000300000015000000041703001C0004001400000009000000150000001E000300130000001400000020000400120000000C000000130000003B00040012000000160000000C00000013000200180000002100030017000000180000002B000400030000001B000000100000002B000400030000001C000000060000002B000400030000001D000000010000002B000400030000001E000000030000002B000400030000001F000000020000002B0004000A00000021000000000000002C0007000900000020000000210000002100000021000000210000002B0004000300000022000000800100002B0004000300000023000000800000002B0004000300000024000000840300002B0004000300000025000000040000002B0004000300000026000000000000002B0004000300000027000000800300002B000400030000002800000008000000200004003E0000000700000009000000140002004C000000200004006F0000000C000000090000003600050018000000190000000000000017000000F80002001A0000003B0004003E0000003F000000070000003B0004003E00000040000000070000003B0004003E00000041000000070000003B0004003E00000042000000070000003B0004003E0000004E000000070000003B0004003E0000004F000000070000003B0004003E00000050000000070000003B0004003E00000051000000070000003B0004003E0000005B000000070000003B0004003E0000005C000000070000003B0004003E0000005D000000070000003B0004003E0000005E000000070000003D00040002000000290000000500000051000500030000002A00000029000000000000003D000400020000002B0000000500000051000500030000002C0000002B000000010000003D000400020000002D0000000500000051000500030000002E0000002D0000000200000084000500030000002F0000002E000000250000008400050003000000300000002C000000250000008400050003000000310000002A0000001B0000008400050003000000320000002F0000001F000000840005000300000033000000300000001F0000003D00040002000000340000000400000051000500030000003500000034000000000000003D00040002000000360000000400000051000500030000003700000036000000010000003D000400020000003800000004000000510005000300000039000000380000000200000084000500030000003A000000390000002500000084000500030000003B000000350000002500000084000500030000003C000000390000002800000084000500030000003D000000370000001F000000F900020043000000F800020043000000F50007000300000046000000F100000047000000260000001A000000F50007000900000048000000F000000047000000200000001A000000F50007000900000049000000EF00000047000000200000001A000000F5000700090000004A000000EE00000047000000200000001A000000F5000700090000004B000000ED00000047000000200000001A000000B10005004C0000004D000000460000001E000000F6000400450000004400000000000000FA0004004D0000004400000045000000F800020044000000F900020052000000F800020052000000F50007000300000054000000EC000000550000002600000044000000F50007000900000056000000EB000000550000004800000044000000F50007000900000057000000EA000000550000004900000044000000F50007000900000058000000E9000000550000004A00000044000000F50007000900000059000000E8000000550000004B00000044000000B10005004C0000005A000000540000001E000000F6000400470000005300000000000000FA0004005A0000005300000047000000F800020053000000F90002005F000000F80002005F000000F50007000300000061000000E7000000600000002600000053000000F500070009000000620000009E000000600000005600000053000000F50007000900000063000000B6000000600000005700000053000000F50007000900000064000000CE000000600000005800000053000000F50007000900000065000000E6000000600000005900000053000000B10005004C00000066000000610000001B000000F6000400550000006000000000000000FA000400660000006000000055000000F800020060000000800005000300000067000000310000003B0000008700050003000000680000006700000025000000840005000300000069000000460000002200000084000500030000006A000000540000002300000080000500030000006B000000690000006A00000084000500030000006C000000610000002800000080000500030000006D0000006B0000006C00000080000500030000006E0000006D00000068000000410006006F0000007000000011000000260000006E0000003D000400090000007100000070000000800005000300000072000000610000001D00000084000500030000007300000072000000280000008000050003000000740000006B000000730000008000050003000000750000007400000068000000410006006F000000760000001100000026000000750000003D000400090000007700000076000000800005000300000078000000610000001F000000840005000300000079000000780000002800000080000500030000007A0000006B0000007900000080000500030000007B0000007A00000068000000410006006F0000007C00000011000000260000007B0000003D000400090000007D0000007C00000080000500030000007E000000610000001E00000084000500030000007F0000007E000000280000008000050003000000800000006B0000007F0000008000050003000000810000008000000068000000410006006F000000820000001100000026000000810000003D0004000900000083000000820000008000050003000000840000003C000000460000008000050003000000850000003D00000054000000800005000300000086000000320000008400000080000500030000008700000033000000850000008700050003000000880000006100000025000000840005000300000089000000860000002400000084000500030000008A000000870000002500000080000500030000008B000000890000008A00000080000500030000008C0000008B00000088000000410006006F0000008D00000016000000260000008C0000003D000400090000008E0000008D000000510005000A0000008F0000008E000000000000005000070009000000900000008F0000008F0000008F0000008F00000085000500090000009100000090000000710000008100050009000000920000009100000062000000510005000A000000930000008E000000010000005000070009000000940000009300000093000000930000009300000085000500090000009500000094000000770000008100050009000000960000009500000092000000510005000A000000970000008E0000000200000050000700090000009800000097000000970000009700000097000000850005000900000099000000980000007D00000081000500090000009A0000009900000096000000510005000A0000009B0000008E0000000300000050000700090000009C0000009B0000009B0000009B0000009B00000085000500090000009D0000009C0000008300000081000500090000009E0000009D0000009A00000080000500030000009F000000460000001F0000008000050003000000A00000003C0000009F0000008000050003000000A100000032000000A00000008400050003000000A2000000A1000000240000008000050003000000A3000000A20000008A0000008000050003000000A4000000A300000088000000410006006F000000A50000001600000026000000A40000003D00040009000000A6000000A5000000510005000A000000A7000000A6000000000000005000070009000000A8000000A7000000A7000000A7000000A70000008500050009000000A9000000A8000000710000008100050009000000AA000000A900000063000000510005000A000000AB000000A6000000010000005000070009000000AC000000AB000000AB000000AB000000AB0000008500050009000000AD000000AC000000770000008100050009000000AE000000AD000000AA000000510005000A000000AF000000A6000000020000005000070009000000B0000000AF000000AF000000AF000000AF0000008500050009000000B1000000B00000007D0000008100050009000000B2000000B1000000AE000000510005000A000000B3000000A6000000030000005000070009000000B4000000B3000000B3000000B3000000B30000008500050009000000B5000000B4000000830000008100050009000000B6000000B5000000B20000008000050003000000B700000046000000250000008000050003000000B80000003C000000B70000008000050003000000B900000032000000B80000008400050003000000BA000000B9000000240000008000050003000000BB000000BA0000008A0000008000050003000000BC000000BB00000088000000410006006F000000BD0000001600000026000000BC0000003D00040009000000BE000000BD000000510005000A000000BF000000BE000000000000005000070009000000C0000000BF000000BF000000BF000000BF0000008500050009000000C1000000C0000000710000008100050009000000C2000000C100000064000000510005000A000000C3000000BE000000010000005000070009000000C4000000C3000000C3000000C3000000C30000008500050009000000C5000000C4000000770000008100050009000000C6000000C5000000C2000000510005000A000000C7000000BE000000020000005000070009000000C8000000C7000000C7000000C7000000C70000008500050009000000C9000000C80000007D0000008100050009000000CA000000C9000000C6000000510005000A000000CB000000BE000000030000005000070009000000CC000000CB000000CB000000CB000000CB0000008500050009000000CD000000CC000000830000008100050009000000CE000000CD000000CA0000008000050003000000CF000000460000001C0000008000050003000000D00000003C000000CF0000008000050003000000D100000032000000D00000008400050003000000D2000000D1000000240000008000050003000000D3000000D20000008A0000008000050003000000D4000000D300000088000000410006006F000000D50000001600000026000000D40000003D00040009000000D6000000D5000000510005000A000000D7000000D6000000000000005000070009000000D8000000D7000000D7000000D7000000D70000008500050009000000D9000000D8000000710000008100050009000000DA000000D900000065000000510005000A000000DB000000D6000000010000005000070009000000DC000000DB000000DB000000DB000000DB0000008500050009000000DD000000DC000000770000008100050009000000DE000000DD000000DA000000510005000A000000DF000000D6000000020000005000070009000000E0000000DF000000DF000000DF000000DF0000008500050009000000E1000000E00000007D0000008100050009000000E2000000E1000000DE000000510005000A000000E3000000D6000000030000005000070009000000E4000000E3000000E3000000E3000000E30000008500050009000000E5000000E4000000830000008100050009000000E6000000E5000000E20000003E0003005B0000009E0000003E0003005C000000B60000003E0003005D000000CE0000003E0003005E000000E60000008000050003000000E70000006100000025000000F90002005F000000F8000200550000003D00040009000000E80000005E0000003D00040009000000E90000005D0000003D00040009000000EA0000005C0000003D00040009000000EB0000005B0000003E0003004E000000EB0000003E0003004F000000EA0000003E00030050000000E90000003E00030051000000E80000008000050003000000EC000000540000001D000000F900020052000000F8000200470000003D00040009000000ED000000510000003D00040009000000EE000000500000003D00040009000000EF0000004F0000003D00040009000000F00000004E0000003E0003003F000000F00000003E00030040000000EF0000003E00030041000000EE0000003E00030042000000ED0000008000050003000000F1000000460000001D000000F900020043000000F8000200450000003D00040009000000F2000000420000003D00040009000000F3000000410000003D00040009000000F4000000400000003D00040009000000F50000003F0000008000050003000000F60000003A0000001E0000008000050003000000F70000002F000000F60000008000050003000000F800000030000000370000008000050003000000F9000000310000003B0000008700050003000000FA000000F9000000250000008400050003000000FB000000F7000000270000008400050003000000FC000000F8000000280000008000050003000000FD000000FB000000FC0000008000050003000000FE000000FD000000FA000000410006006F000000FF0000000C00000026000000FE0000003E000300FF000000F20000008000050003000000000100003A0000001F0000008000050003000000010100002F00000000010000840005000300000002010000010100002700000080000500030000000301000002010000FC00000080000500030000000401000003010000FA000000410006006F000000050100000C00000026000000040100003E00030005010000F30000008000050003000000060100003A0000001D0000008000050003000000070100002F00000006010000840005000300000008010000070100002700000080000500030000000901000008010000FC00000080000500030000000A01000009010000FA000000410006006F0000000B0100000C000000260000000A0100003E0003000B010000F400000080000500030000000C0100002F0000003A00000084000500030000000D0100000C0100002700000080000500030000000E0100000D010000FC00000080000500030000000F0100000E010000FA000000410006006F000000100100000C000000260000000F0100003E00030010010000F5000000FD0001003800010008000C0004000800"> : vector<6380xi8>
    vm.func @predict_ex_dispatch_1(%arg0: !vm.ref<!hal.buffer>, %arg1: !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer> attributes {noinline} {
      %c3240000 = vm.const.i32 3240000 : i32
      %c18432 = vm.const.i32 18432 : i32
      %c1605632 = vm.const.i32 1605632 : i32
      %c28 = vm.const.i32 28 : i32
      %c50 = vm.const.i32 50 : i32
      %c15 = vm.const.i32 15 : i32
      %c1 = vm.const.i32 1 : i32
      %c3 = vm.const.i32 3 : i32
      %c20 = vm.const.i32 20 : i32
      %c5 = vm.const.i32 5 : i32
      %zero = vm.const.i32.zero : i32
      %c2 = vm.const.i32 2 : i32
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_0 = vm.call @hal.device.allocator(%ref) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
      %ref_1 = vm.call @hal.allocator.allocate(%ref_0, %c50, %c15, %c1605632) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
      %ref_2 = vm.call @hal.command_buffer.create(%ref, %c1, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
      vm.call @hal.command_buffer.begin(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
      %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      vm.call.variadic @hal.command_buffer.push_descriptor_set(%ref_2, %_executable_layout_0, %zero, [(%zero, %arg0, %zero, %c3240000), (%c1, %arg1, %zero, %c18432), (%c2, %ref_1, %zero, %c1605632)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...)
      %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
      vm.cond_br %_device_match_id_0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %_executable_predict_ex_dispatch_1_dispatch_0 = vm.global.load.ref @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
      vm.call @hal.command_buffer.dispatch(%ref_2, %_executable_predict_ex_dispatch_1_dispatch_0, %zero, %c2, %c28, %c28) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
      vm.call @hal.command_buffer.execution_barrier(%ref_2, %c20, %c5, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
      vm.call @hal.command_buffer.end(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
      vm.call @hal.ex.submit_and_wait(%ref, %ref_2) : (!vm.ref<!hal.device>, !vm.ref<!hal.command_buffer>) -> ()
      vm.return %ref_1 : !vm.ref<!hal.buffer>
    ^bb2:  // pred: ^bb0
      vm.fail %c2, "unreachable location reached"
    }
    vm.export @predict_ex_dispatch_1 as("predict_ex_dispatch_1$raw")
    vm.func @predict_ex_dispatch_1$async(%arg0: !vm.ref<!hal.semaphore>, %arg1: i32, %arg2: !vm.ref<!hal.buffer_view>, %arg3: !vm.ref<!hal.buffer_view>, %arg4: !vm.ref<!hal.semaphore>, %arg5: i32) -> !vm.ref<!hal.buffer_view> {
      %c1 = vm.const.i32 1 : i32
      %c112 = vm.const.i32 112 : i32
      %c32 = vm.const.i32 32 : i32
      %c50331680 = vm.const.i32 50331680 : i32
      %0 = vm.call @hal.semaphore.await(%arg0, %arg1) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %0, ^bb2(%0 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %ref = vm.call @hal.buffer_view.buffer(%arg2) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_0 = vm.call @hal.buffer_view.buffer(%arg3) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_1 = vm.call @predict_ex_dispatch_1(%ref, %ref_0) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
      %ref_2 = vm.call.variadic @hal.buffer_view.create(%ref_1, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
      vm.call @hal.semaphore.signal(%arg4, %arg5) : (!vm.ref<!hal.semaphore>, i32) -> ()
      vm.return %ref_2 : !vm.ref<!hal.buffer_view>
    ^bb2(%1: i32):  // pred: ^bb0
      vm.fail %1, "semaphore wait failed"
    }
    vm.export @predict_ex_dispatch_1$async
    vm.func @predict_ex_dispatch_1$sync(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
      %zero = vm.const.i32.zero : i32
      %c1 = vm.const.i32 1 : i32
      %c112 = vm.const.i32 112 : i32
      %c32 = vm.const.i32 32 : i32
      %c50331680 = vm.const.i32 50331680 : i32
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_0 = vm.call @hal.semaphore.create(%ref, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.semaphore>
      %0 = vm.call @hal.semaphore.await(%ref_0, %zero) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %0, ^bb2(%0 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %ref_1 = vm.call @hal.buffer_view.buffer(%arg0) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_2 = vm.call @hal.buffer_view.buffer(%arg1) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_3 = vm.call @predict_ex_dispatch_1(%ref_1, %ref_2) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
      %ref_4 = vm.call.variadic @hal.buffer_view.create(%ref_3, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
      vm.call @hal.semaphore.signal(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> ()
      %1 = vm.call @hal.semaphore.await(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %1, ^bb2(%1 : i32), ^bb3
    ^bb2(%2: i32):  // 2 preds: ^bb0, ^bb1
      vm.fail %2, "semaphore wait failed"
    ^bb3:  // pred: ^bb1
      vm.return %ref_4 : !vm.ref<!hal.buffer_view>
    }
    vm.export @predict_ex_dispatch_1$sync as("predict_ex_dispatch_1")
    vm.import @hal.ex.shared_device() -> !vm.ref<!hal.device> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.ex.submit_and_wait(%device : !vm.ref<!hal.device>, %command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.allocator.allocate(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %allocation_size : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
    vm.import @hal.allocator.wrap.byte_buffer(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %source : !vm.ref<!iree.byte_buffer>, %offset : i32, %length : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
    vm.import @hal.buffer.allocator(%buffer : !vm.ref<!hal.buffer>) -> !vm.ref<!hal.allocator> attributes {sym_visibility = "private"}
    vm.import @hal.buffer.subspan(%source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %length : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
    vm.import @hal.buffer.fill(%target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32, %pattern : i32) attributes {sym_visibility = "private"}
    vm.import @hal.buffer.load(%source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %length : i32) -> i32 attributes {sym_visibility = "private"}
    vm.import @hal.buffer.store(%value : i32, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32) attributes {sym_visibility = "private"}
    vm.import @hal.buffer_view.create(%buffer : !vm.ref<!hal.buffer>, %element_type : i32, %shape : i32 ...) -> !vm.ref<!hal.buffer_view> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.buffer(%buffer_view : !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.byte_length(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.element_type(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.rank(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.dim(%buffer_view : !vm.ref<!hal.buffer_view>, %index : i32) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.trace(%key : !vm.ref<!iree.byte_buffer>, %operands : !vm.ref<!hal.buffer_view> ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.create(%device : !vm.ref<!hal.device>, %modes : i32, %command_categories : i32) -> !vm.ref<!hal.command_buffer> attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.begin(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.end(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.execution_barrier(%command_buffer : !vm.ref<!hal.command_buffer>, %source_stage_mask : i32, %target_stage_mask : i32, %flags : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.fill_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32, %pattern : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.copy_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %source_buffer : !vm.ref<!hal.buffer>, %source_offset : i32, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i32, %length : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.push_constants(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %offset : i32, %values : i32 ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.push_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.bind_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %descriptor_set : !vm.ref<!hal.descriptor_set>, %dynamic_offsets : i32 ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.dispatch(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroup_x : i32, %workgroup_y : i32, %workgroup_z : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.dispatch.indirect(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroups_buffer : !vm.ref<!hal.buffer>, %workgroups_offset : i32) attributes {sym_visibility = "private"}
    vm.import @hal.descriptor_set.create(%device : !vm.ref<!hal.device>, %set_layout : !vm.ref<!hal.descriptor_set_layout>, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) -> !vm.ref<!hal.descriptor_set> attributes {sym_visibility = "private"}
    vm.import @hal.descriptor_set_layout.create(%device : !vm.ref<!hal.device>, %usage_type : i32, %bindings : tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.device.allocator(%device : !vm.ref<!hal.device>) -> !vm.ref<!hal.allocator> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.device.match.id(%device : !vm.ref<!hal.device>, %pattern : !vm.ref<!iree.byte_buffer>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.executable.create(%device : !vm.ref<!hal.device>, %executable_format : i32, %executable_data : !vm.ref<!iree.byte_buffer>, %executable_layouts : !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.executable_layout.create(%device : !vm.ref<!hal.device>, %push_constants : i32, %set_layouts : !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.semaphore.create(%device : !vm.ref<!hal.device>, %initial_value : i32) -> !vm.ref<!hal.semaphore> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.semaphore.query(%semaphore : !vm.ref<!hal.semaphore>) -> (i32, i32) attributes {sym_visibility = "private"}
    vm.import @hal.semaphore.signal(%semaphore : !vm.ref<!hal.semaphore>, %new_value : i32) attributes {sym_visibility = "private"}
    vm.import @hal.semaphore.fail(%semaphore : !vm.ref<!hal.semaphore>, %status : i32) attributes {sym_visibility = "private"}
    vm.import @hal.semaphore.await(%semaphore : !vm.ref<!hal.semaphore>, %min_value : i32) -> i32 attributes {sym_visibility = "private"}
    vm.func @__init() {
      %c1 = vm.const.i32 1 : i32
      %c2 = vm.const.i32 2 : i32
      %c7 = vm.const.i32 7 : i32
      %c6 = vm.const.i32 6 : i32
      %zero = vm.const.i32.zero : i32
      %c1397773893 = vm.const.i32 1397773893 : i32
      %null = vm.const.ref.zero : !vm.ref<!hal.executable>
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %_utf8_vulkan_7197BF52A22CAFD7 = vm.const.ref.rodata @_utf8_vulkan_7197BF52A22CAFD7 : !vm.ref<!iree.byte_buffer>
      %0 = vm.call @hal.device.match.id(%ref, %_utf8_vulkan_7197BF52A22CAFD7) : (!vm.ref<!hal.device>, !vm.ref<!iree.byte_buffer>) -> i32
      vm.global.store.i32 %0, @_device_match_id_0 : i32
      %ref_0 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_1 = vm.call.variadic @hal.descriptor_set_layout.create(%ref_0, %c1, [(%zero, %c7, %c1), (%c1, %c7, %c1), (%c2, %c7, %c6)]) : (!vm.ref<!hal.device>, i32, tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout>
      vm.global.store.ref %ref_1, @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
      %_descriptor_set_layout_0 = vm.global.load.ref @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
      %ref_2 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_3 = vm.call.variadic @hal.executable_layout.create(%ref_2, %zero, [%_descriptor_set_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout>
      vm.global.store.ref %ref_3, @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      %ref_4 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
      vm.cond_br %_device_match_id_0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv = vm.const.ref.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv : !vm.ref<!iree.byte_buffer>
      %ref_5 = vm.call.variadic @hal.executable.create(%ref_4, %c1397773893, %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv, [%_executable_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
      vm.br ^bb3(%ref_5 : !vm.ref<!hal.executable>)
    ^bb2:  // pred: ^bb0
      vm.br ^bb3(%null : !vm.ref<!hal.executable>)
    ^bb3(%1: !vm.ref<!hal.executable>):  // 2 preds: ^bb1, ^bb2
      vm.global.store.ref %1, @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
      vm.return
    }
    vm.export @__init
  }
}


// *** IR Dump After SymbolDCE ***
module  {
  vm.module @module {
    vm.global.i32 @_device_match_id_0 mutable : i32
    vm.rodata @_utf8_vulkan_7197BF52A22CAFD7 dense<[118, 117, 108, 107, 97, 110, 42]> : vector<7xi8>
    vm.global.ref @_descriptor_set_layout_0 mutable : !vm.ref<!hal.descriptor_set_layout>
    vm.global.ref @_executable_layout_0 mutable : !vm.ref<!hal.executable_layout>
    vm.global.ref @_executable_predict_ex_dispatch_1_dispatch_0 mutable : !vm.ref<!hal.executable>
    vm.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv dense<"0x080000005350564524E7FFFF0800000034000000010000000400000020000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000027060000030223070000010016000000110100000000000011000200010000000A000B005350565F4B48525F73746F726167655F6275666665725F73746F726167655F636C617373000000000E00030000000000010000000F000E000500000019000000707265646963745F65785F64697370617463685F315F64697370617463685F3000000000050000000400000010000600190000001100000004000000040000000100000005000B00040000005F5F6275696C74696E5F7661725F4C6F63616C496E766F636174696F6E49645F5F00000005000900050000005F5F6275696C74696E5F7661725F576F726B67726F757049645F5F0005000A000C0000005F5F7265736F757263655F7661725F32343838393331333339323239365F5F0005000A00110000005F5F7265736F757263655F7661725F32343838393331333430383837325F5F0005000A00160000005F5F7265736F757263655F7661725F32343838393331333339323130345F5F0005000B0019000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000047000400040000000B0000001B00000047000400050000000B0000001A000000470004000800000006000000100000004800050007000000000000002300000000000000470003000700000002000000470004000C0000002100000002000000470004000C0000002200000000000000470004000F0000000600000010000000480005000E000000000000002300000000000000470003000E00000002000000470004001100000021000000010000004700040011000000220000000000000047000400140000000600000010000000480005001300000000000000230000000000000047000300130000000200000047000400160000002100000000000000470004001600000022000000000000001500040003000000200000000000000017000400020000000300000003000000200004000100000001000000020000003B0004000100000004000000010000003B000400010000000500000001000000160003000A0000002000000017000400090000000A000000040000002B000400030000000B000000008801001C00040008000000090000000B0000001E000300070000000800000020000400060000000C000000070000003B000400060000000C0000000C0000002B0004000300000010000000800400001C0004000F00000009000000100000001E0003000E0000000F000000200004000D0000000C0000000E0000003B0004000D000000110000000C0000002B0004000300000015000000041703001C0004001400000009000000150000001E000300130000001400000020000400120000000C000000130000003B00040012000000160000000C00000013000200180000002100030017000000180000002B000400030000001B000000100000002B000400030000001C000000060000002B000400030000001D000000010000002B000400030000001E000000030000002B000400030000001F000000020000002B0004000A00000021000000000000002C0007000900000020000000210000002100000021000000210000002B0004000300000022000000800100002B0004000300000023000000800000002B0004000300000024000000840300002B0004000300000025000000040000002B0004000300000026000000000000002B0004000300000027000000800300002B000400030000002800000008000000200004003E0000000700000009000000140002004C000000200004006F0000000C000000090000003600050018000000190000000000000017000000F80002001A0000003B0004003E0000003F000000070000003B0004003E00000040000000070000003B0004003E00000041000000070000003B0004003E00000042000000070000003B0004003E0000004E000000070000003B0004003E0000004F000000070000003B0004003E00000050000000070000003B0004003E00000051000000070000003B0004003E0000005B000000070000003B0004003E0000005C000000070000003B0004003E0000005D000000070000003B0004003E0000005E000000070000003D00040002000000290000000500000051000500030000002A00000029000000000000003D000400020000002B0000000500000051000500030000002C0000002B000000010000003D000400020000002D0000000500000051000500030000002E0000002D0000000200000084000500030000002F0000002E000000250000008400050003000000300000002C000000250000008400050003000000310000002A0000001B0000008400050003000000320000002F0000001F000000840005000300000033000000300000001F0000003D00040002000000340000000400000051000500030000003500000034000000000000003D00040002000000360000000400000051000500030000003700000036000000010000003D000400020000003800000004000000510005000300000039000000380000000200000084000500030000003A000000390000002500000084000500030000003B000000350000002500000084000500030000003C000000390000002800000084000500030000003D000000370000001F000000F900020043000000F800020043000000F50007000300000046000000F100000047000000260000001A000000F50007000900000048000000F000000047000000200000001A000000F50007000900000049000000EF00000047000000200000001A000000F5000700090000004A000000EE00000047000000200000001A000000F5000700090000004B000000ED00000047000000200000001A000000B10005004C0000004D000000460000001E000000F6000400450000004400000000000000FA0004004D0000004400000045000000F800020044000000F900020052000000F800020052000000F50007000300000054000000EC000000550000002600000044000000F50007000900000056000000EB000000550000004800000044000000F50007000900000057000000EA000000550000004900000044000000F50007000900000058000000E9000000550000004A00000044000000F50007000900000059000000E8000000550000004B00000044000000B10005004C0000005A000000540000001E000000F6000400470000005300000000000000FA0004005A0000005300000047000000F800020053000000F90002005F000000F80002005F000000F50007000300000061000000E7000000600000002600000053000000F500070009000000620000009E000000600000005600000053000000F50007000900000063000000B6000000600000005700000053000000F50007000900000064000000CE000000600000005800000053000000F50007000900000065000000E6000000600000005900000053000000B10005004C00000066000000610000001B000000F6000400550000006000000000000000FA000400660000006000000055000000F800020060000000800005000300000067000000310000003B0000008700050003000000680000006700000025000000840005000300000069000000460000002200000084000500030000006A000000540000002300000080000500030000006B000000690000006A00000084000500030000006C000000610000002800000080000500030000006D0000006B0000006C00000080000500030000006E0000006D00000068000000410006006F0000007000000011000000260000006E0000003D000400090000007100000070000000800005000300000072000000610000001D00000084000500030000007300000072000000280000008000050003000000740000006B000000730000008000050003000000750000007400000068000000410006006F000000760000001100000026000000750000003D000400090000007700000076000000800005000300000078000000610000001F000000840005000300000079000000780000002800000080000500030000007A0000006B0000007900000080000500030000007B0000007A00000068000000410006006F0000007C00000011000000260000007B0000003D000400090000007D0000007C00000080000500030000007E000000610000001E00000084000500030000007F0000007E000000280000008000050003000000800000006B0000007F0000008000050003000000810000008000000068000000410006006F000000820000001100000026000000810000003D0004000900000083000000820000008000050003000000840000003C000000460000008000050003000000850000003D00000054000000800005000300000086000000320000008400000080000500030000008700000033000000850000008700050003000000880000006100000025000000840005000300000089000000860000002400000084000500030000008A000000870000002500000080000500030000008B000000890000008A00000080000500030000008C0000008B00000088000000410006006F0000008D00000016000000260000008C0000003D000400090000008E0000008D000000510005000A0000008F0000008E000000000000005000070009000000900000008F0000008F0000008F0000008F00000085000500090000009100000090000000710000008100050009000000920000009100000062000000510005000A000000930000008E000000010000005000070009000000940000009300000093000000930000009300000085000500090000009500000094000000770000008100050009000000960000009500000092000000510005000A000000970000008E0000000200000050000700090000009800000097000000970000009700000097000000850005000900000099000000980000007D00000081000500090000009A0000009900000096000000510005000A0000009B0000008E0000000300000050000700090000009C0000009B0000009B0000009B0000009B00000085000500090000009D0000009C0000008300000081000500090000009E0000009D0000009A00000080000500030000009F000000460000001F0000008000050003000000A00000003C0000009F0000008000050003000000A100000032000000A00000008400050003000000A2000000A1000000240000008000050003000000A3000000A20000008A0000008000050003000000A4000000A300000088000000410006006F000000A50000001600000026000000A40000003D00040009000000A6000000A5000000510005000A000000A7000000A6000000000000005000070009000000A8000000A7000000A7000000A7000000A70000008500050009000000A9000000A8000000710000008100050009000000AA000000A900000063000000510005000A000000AB000000A6000000010000005000070009000000AC000000AB000000AB000000AB000000AB0000008500050009000000AD000000AC000000770000008100050009000000AE000000AD000000AA000000510005000A000000AF000000A6000000020000005000070009000000B0000000AF000000AF000000AF000000AF0000008500050009000000B1000000B00000007D0000008100050009000000B2000000B1000000AE000000510005000A000000B3000000A6000000030000005000070009000000B4000000B3000000B3000000B3000000B30000008500050009000000B5000000B4000000830000008100050009000000B6000000B5000000B20000008000050003000000B700000046000000250000008000050003000000B80000003C000000B70000008000050003000000B900000032000000B80000008400050003000000BA000000B9000000240000008000050003000000BB000000BA0000008A0000008000050003000000BC000000BB00000088000000410006006F000000BD0000001600000026000000BC0000003D00040009000000BE000000BD000000510005000A000000BF000000BE000000000000005000070009000000C0000000BF000000BF000000BF000000BF0000008500050009000000C1000000C0000000710000008100050009000000C2000000C100000064000000510005000A000000C3000000BE000000010000005000070009000000C4000000C3000000C3000000C3000000C30000008500050009000000C5000000C4000000770000008100050009000000C6000000C5000000C2000000510005000A000000C7000000BE000000020000005000070009000000C8000000C7000000C7000000C7000000C70000008500050009000000C9000000C80000007D0000008100050009000000CA000000C9000000C6000000510005000A000000CB000000BE000000030000005000070009000000CC000000CB000000CB000000CB000000CB0000008500050009000000CD000000CC000000830000008100050009000000CE000000CD000000CA0000008000050003000000CF000000460000001C0000008000050003000000D00000003C000000CF0000008000050003000000D100000032000000D00000008400050003000000D2000000D1000000240000008000050003000000D3000000D20000008A0000008000050003000000D4000000D300000088000000410006006F000000D50000001600000026000000D40000003D00040009000000D6000000D5000000510005000A000000D7000000D6000000000000005000070009000000D8000000D7000000D7000000D7000000D70000008500050009000000D9000000D8000000710000008100050009000000DA000000D900000065000000510005000A000000DB000000D6000000010000005000070009000000DC000000DB000000DB000000DB000000DB0000008500050009000000DD000000DC000000770000008100050009000000DE000000DD000000DA000000510005000A000000DF000000D6000000020000005000070009000000E0000000DF000000DF000000DF000000DF0000008500050009000000E1000000E00000007D0000008100050009000000E2000000E1000000DE000000510005000A000000E3000000D6000000030000005000070009000000E4000000E3000000E3000000E3000000E30000008500050009000000E5000000E4000000830000008100050009000000E6000000E5000000E20000003E0003005B0000009E0000003E0003005C000000B60000003E0003005D000000CE0000003E0003005E000000E60000008000050003000000E70000006100000025000000F90002005F000000F8000200550000003D00040009000000E80000005E0000003D00040009000000E90000005D0000003D00040009000000EA0000005C0000003D00040009000000EB0000005B0000003E0003004E000000EB0000003E0003004F000000EA0000003E00030050000000E90000003E00030051000000E80000008000050003000000EC000000540000001D000000F900020052000000F8000200470000003D00040009000000ED000000510000003D00040009000000EE000000500000003D00040009000000EF0000004F0000003D00040009000000F00000004E0000003E0003003F000000F00000003E00030040000000EF0000003E00030041000000EE0000003E00030042000000ED0000008000050003000000F1000000460000001D000000F900020043000000F8000200450000003D00040009000000F2000000420000003D00040009000000F3000000410000003D00040009000000F4000000400000003D00040009000000F50000003F0000008000050003000000F60000003A0000001E0000008000050003000000F70000002F000000F60000008000050003000000F800000030000000370000008000050003000000F9000000310000003B0000008700050003000000FA000000F9000000250000008400050003000000FB000000F7000000270000008400050003000000FC000000F8000000280000008000050003000000FD000000FB000000FC0000008000050003000000FE000000FD000000FA000000410006006F000000FF0000000C00000026000000FE0000003E000300FF000000F20000008000050003000000000100003A0000001F0000008000050003000000010100002F00000000010000840005000300000002010000010100002700000080000500030000000301000002010000FC00000080000500030000000401000003010000FA000000410006006F000000050100000C00000026000000040100003E00030005010000F30000008000050003000000060100003A0000001D0000008000050003000000070100002F00000006010000840005000300000008010000070100002700000080000500030000000901000008010000FC00000080000500030000000A01000009010000FA000000410006006F0000000B0100000C000000260000000A0100003E0003000B010000F400000080000500030000000C0100002F0000003A00000084000500030000000D0100000C0100002700000080000500030000000E0100000D010000FC00000080000500030000000F0100000E010000FA000000410006006F000000100100000C000000260000000F0100003E00030010010000F5000000FD0001003800010008000C0004000800"> : vector<6380xi8>
    vm.func @predict_ex_dispatch_1(%arg0: !vm.ref<!hal.buffer>, %arg1: !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer> attributes {noinline} {
      %c3240000 = vm.const.i32 3240000 : i32
      %c18432 = vm.const.i32 18432 : i32
      %c1605632 = vm.const.i32 1605632 : i32
      %c28 = vm.const.i32 28 : i32
      %c50 = vm.const.i32 50 : i32
      %c15 = vm.const.i32 15 : i32
      %c1 = vm.const.i32 1 : i32
      %c3 = vm.const.i32 3 : i32
      %c20 = vm.const.i32 20 : i32
      %c5 = vm.const.i32 5 : i32
      %zero = vm.const.i32.zero : i32
      %c2 = vm.const.i32 2 : i32
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_0 = vm.call @hal.device.allocator(%ref) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
      %ref_1 = vm.call @hal.allocator.allocate(%ref_0, %c50, %c15, %c1605632) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
      %ref_2 = vm.call @hal.command_buffer.create(%ref, %c1, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
      vm.call @hal.command_buffer.begin(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
      %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      vm.call.variadic @hal.command_buffer.push_descriptor_set(%ref_2, %_executable_layout_0, %zero, [(%zero, %arg0, %zero, %c3240000), (%c1, %arg1, %zero, %c18432), (%c2, %ref_1, %zero, %c1605632)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...)
      %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
      vm.cond_br %_device_match_id_0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %_executable_predict_ex_dispatch_1_dispatch_0 = vm.global.load.ref @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
      vm.call @hal.command_buffer.dispatch(%ref_2, %_executable_predict_ex_dispatch_1_dispatch_0, %zero, %c2, %c28, %c28) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
      vm.call @hal.command_buffer.execution_barrier(%ref_2, %c20, %c5, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
      vm.call @hal.command_buffer.end(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
      vm.call @hal.ex.submit_and_wait(%ref, %ref_2) : (!vm.ref<!hal.device>, !vm.ref<!hal.command_buffer>) -> ()
      vm.return %ref_1 : !vm.ref<!hal.buffer>
    ^bb2:  // pred: ^bb0
      vm.fail %c2, "unreachable location reached"
    }
    vm.export @predict_ex_dispatch_1 as("predict_ex_dispatch_1$raw")
    vm.func @predict_ex_dispatch_1$async(%arg0: !vm.ref<!hal.semaphore>, %arg1: i32, %arg2: !vm.ref<!hal.buffer_view>, %arg3: !vm.ref<!hal.buffer_view>, %arg4: !vm.ref<!hal.semaphore>, %arg5: i32) -> !vm.ref<!hal.buffer_view> {
      %c1 = vm.const.i32 1 : i32
      %c112 = vm.const.i32 112 : i32
      %c32 = vm.const.i32 32 : i32
      %c50331680 = vm.const.i32 50331680 : i32
      %0 = vm.call @hal.semaphore.await(%arg0, %arg1) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %0, ^bb2(%0 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %ref = vm.call @hal.buffer_view.buffer(%arg2) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_0 = vm.call @hal.buffer_view.buffer(%arg3) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_1 = vm.call @predict_ex_dispatch_1(%ref, %ref_0) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
      %ref_2 = vm.call.variadic @hal.buffer_view.create(%ref_1, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
      vm.call @hal.semaphore.signal(%arg4, %arg5) : (!vm.ref<!hal.semaphore>, i32) -> ()
      vm.return %ref_2 : !vm.ref<!hal.buffer_view>
    ^bb2(%1: i32):  // pred: ^bb0
      vm.fail %1, "semaphore wait failed"
    }
    vm.export @predict_ex_dispatch_1$async
    vm.func @predict_ex_dispatch_1$sync(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
      %zero = vm.const.i32.zero : i32
      %c1 = vm.const.i32 1 : i32
      %c112 = vm.const.i32 112 : i32
      %c32 = vm.const.i32 32 : i32
      %c50331680 = vm.const.i32 50331680 : i32
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_0 = vm.call @hal.semaphore.create(%ref, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.semaphore>
      %0 = vm.call @hal.semaphore.await(%ref_0, %zero) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %0, ^bb2(%0 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %ref_1 = vm.call @hal.buffer_view.buffer(%arg0) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_2 = vm.call @hal.buffer_view.buffer(%arg1) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_3 = vm.call @predict_ex_dispatch_1(%ref_1, %ref_2) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
      %ref_4 = vm.call.variadic @hal.buffer_view.create(%ref_3, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
      vm.call @hal.semaphore.signal(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> ()
      %1 = vm.call @hal.semaphore.await(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %1, ^bb2(%1 : i32), ^bb3
    ^bb2(%2: i32):  // 2 preds: ^bb0, ^bb1
      vm.fail %2, "semaphore wait failed"
    ^bb3:  // pred: ^bb1
      vm.return %ref_4 : !vm.ref<!hal.buffer_view>
    }
    vm.export @predict_ex_dispatch_1$sync as("predict_ex_dispatch_1")
    vm.import @hal.ex.shared_device() -> !vm.ref<!hal.device> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.ex.submit_and_wait(%device : !vm.ref<!hal.device>, %command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.allocator.allocate(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %allocation_size : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
    vm.import @hal.buffer_view.create(%buffer : !vm.ref<!hal.buffer>, %element_type : i32, %shape : i32 ...) -> !vm.ref<!hal.buffer_view> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.buffer(%buffer_view : !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.command_buffer.create(%device : !vm.ref<!hal.device>, %modes : i32, %command_categories : i32) -> !vm.ref<!hal.command_buffer> attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.begin(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.end(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.execution_barrier(%command_buffer : !vm.ref<!hal.command_buffer>, %source_stage_mask : i32, %target_stage_mask : i32, %flags : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.push_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.dispatch(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroup_x : i32, %workgroup_y : i32, %workgroup_z : i32) attributes {sym_visibility = "private"}
    vm.import @hal.descriptor_set_layout.create(%device : !vm.ref<!hal.device>, %usage_type : i32, %bindings : tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.device.allocator(%device : !vm.ref<!hal.device>) -> !vm.ref<!hal.allocator> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.device.match.id(%device : !vm.ref<!hal.device>, %pattern : !vm.ref<!iree.byte_buffer>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.executable.create(%device : !vm.ref<!hal.device>, %executable_format : i32, %executable_data : !vm.ref<!iree.byte_buffer>, %executable_layouts : !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.executable_layout.create(%device : !vm.ref<!hal.device>, %push_constants : i32, %set_layouts : !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.semaphore.create(%device : !vm.ref<!hal.device>, %initial_value : i32) -> !vm.ref<!hal.semaphore> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.semaphore.signal(%semaphore : !vm.ref<!hal.semaphore>, %new_value : i32) attributes {sym_visibility = "private"}
    vm.import @hal.semaphore.await(%semaphore : !vm.ref<!hal.semaphore>, %min_value : i32) -> i32 attributes {sym_visibility = "private"}
    vm.func @__init() {
      %c1 = vm.const.i32 1 : i32
      %c2 = vm.const.i32 2 : i32
      %c7 = vm.const.i32 7 : i32
      %c6 = vm.const.i32 6 : i32
      %zero = vm.const.i32.zero : i32
      %c1397773893 = vm.const.i32 1397773893 : i32
      %null = vm.const.ref.zero : !vm.ref<!hal.executable>
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %_utf8_vulkan_7197BF52A22CAFD7 = vm.const.ref.rodata @_utf8_vulkan_7197BF52A22CAFD7 : !vm.ref<!iree.byte_buffer>
      %0 = vm.call @hal.device.match.id(%ref, %_utf8_vulkan_7197BF52A22CAFD7) : (!vm.ref<!hal.device>, !vm.ref<!iree.byte_buffer>) -> i32
      vm.global.store.i32 %0, @_device_match_id_0 : i32
      %ref_0 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_1 = vm.call.variadic @hal.descriptor_set_layout.create(%ref_0, %c1, [(%zero, %c7, %c1), (%c1, %c7, %c1), (%c2, %c7, %c6)]) : (!vm.ref<!hal.device>, i32, tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout>
      vm.global.store.ref %ref_1, @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
      %_descriptor_set_layout_0 = vm.global.load.ref @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
      %ref_2 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_3 = vm.call.variadic @hal.executable_layout.create(%ref_2, %zero, [%_descriptor_set_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout>
      vm.global.store.ref %ref_3, @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      %ref_4 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
      vm.cond_br %_device_match_id_0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv = vm.const.ref.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv : !vm.ref<!iree.byte_buffer>
      %ref_5 = vm.call.variadic @hal.executable.create(%ref_4, %c1397773893, %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv, [%_executable_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
      vm.br ^bb3(%ref_5 : !vm.ref<!hal.executable>)
    ^bb2:  // pred: ^bb0
      vm.br ^bb3(%null : !vm.ref<!hal.executable>)
    ^bb3(%1: !vm.ref<!hal.executable>):  // 2 preds: ^bb1, ^bb2
      vm.global.store.ref %1, @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
      vm.return
    }
    vm.export @__init
  }
}


// *** IR Dump After mlir::iree_compiler::IREE::VM::SinkDefiningOpsPass ***
vm.module @module {
  vm.global.i32 @_device_match_id_0 mutable : i32
  vm.rodata @_utf8_vulkan_7197BF52A22CAFD7 dense<[118, 117, 108, 107, 97, 110, 42]> : vector<7xi8>
  vm.global.ref @_descriptor_set_layout_0 mutable : !vm.ref<!hal.descriptor_set_layout>
  vm.global.ref @_executable_layout_0 mutable : !vm.ref<!hal.executable_layout>
  vm.global.ref @_executable_predict_ex_dispatch_1_dispatch_0 mutable : !vm.ref<!hal.executable>
  vm.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv dense<"0x080000005350564524E7FFFF0800000034000000010000000400000020000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000027060000030223070000010016000000110100000000000011000200010000000A000B005350565F4B48525F73746F726167655F6275666665725F73746F726167655F636C617373000000000E00030000000000010000000F000E000500000019000000707265646963745F65785F64697370617463685F315F64697370617463685F3000000000050000000400000010000600190000001100000004000000040000000100000005000B00040000005F5F6275696C74696E5F7661725F4C6F63616C496E766F636174696F6E49645F5F00000005000900050000005F5F6275696C74696E5F7661725F576F726B67726F757049645F5F0005000A000C0000005F5F7265736F757263655F7661725F32343838393331333339323239365F5F0005000A00110000005F5F7265736F757263655F7661725F32343838393331333430383837325F5F0005000A00160000005F5F7265736F757263655F7661725F32343838393331333339323130345F5F0005000B0019000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000047000400040000000B0000001B00000047000400050000000B0000001A000000470004000800000006000000100000004800050007000000000000002300000000000000470003000700000002000000470004000C0000002100000002000000470004000C0000002200000000000000470004000F0000000600000010000000480005000E000000000000002300000000000000470003000E00000002000000470004001100000021000000010000004700040011000000220000000000000047000400140000000600000010000000480005001300000000000000230000000000000047000300130000000200000047000400160000002100000000000000470004001600000022000000000000001500040003000000200000000000000017000400020000000300000003000000200004000100000001000000020000003B0004000100000004000000010000003B000400010000000500000001000000160003000A0000002000000017000400090000000A000000040000002B000400030000000B000000008801001C00040008000000090000000B0000001E000300070000000800000020000400060000000C000000070000003B000400060000000C0000000C0000002B0004000300000010000000800400001C0004000F00000009000000100000001E0003000E0000000F000000200004000D0000000C0000000E0000003B0004000D000000110000000C0000002B0004000300000015000000041703001C0004001400000009000000150000001E000300130000001400000020000400120000000C000000130000003B00040012000000160000000C00000013000200180000002100030017000000180000002B000400030000001B000000100000002B000400030000001C000000060000002B000400030000001D000000010000002B000400030000001E000000030000002B000400030000001F000000020000002B0004000A00000021000000000000002C0007000900000020000000210000002100000021000000210000002B0004000300000022000000800100002B0004000300000023000000800000002B0004000300000024000000840300002B0004000300000025000000040000002B0004000300000026000000000000002B0004000300000027000000800300002B000400030000002800000008000000200004003E0000000700000009000000140002004C000000200004006F0000000C000000090000003600050018000000190000000000000017000000F80002001A0000003B0004003E0000003F000000070000003B0004003E00000040000000070000003B0004003E00000041000000070000003B0004003E00000042000000070000003B0004003E0000004E000000070000003B0004003E0000004F000000070000003B0004003E00000050000000070000003B0004003E00000051000000070000003B0004003E0000005B000000070000003B0004003E0000005C000000070000003B0004003E0000005D000000070000003B0004003E0000005E000000070000003D00040002000000290000000500000051000500030000002A00000029000000000000003D000400020000002B0000000500000051000500030000002C0000002B000000010000003D000400020000002D0000000500000051000500030000002E0000002D0000000200000084000500030000002F0000002E000000250000008400050003000000300000002C000000250000008400050003000000310000002A0000001B0000008400050003000000320000002F0000001F000000840005000300000033000000300000001F0000003D00040002000000340000000400000051000500030000003500000034000000000000003D00040002000000360000000400000051000500030000003700000036000000010000003D000400020000003800000004000000510005000300000039000000380000000200000084000500030000003A000000390000002500000084000500030000003B000000350000002500000084000500030000003C000000390000002800000084000500030000003D000000370000001F000000F900020043000000F800020043000000F50007000300000046000000F100000047000000260000001A000000F50007000900000048000000F000000047000000200000001A000000F50007000900000049000000EF00000047000000200000001A000000F5000700090000004A000000EE00000047000000200000001A000000F5000700090000004B000000ED00000047000000200000001A000000B10005004C0000004D000000460000001E000000F6000400450000004400000000000000FA0004004D0000004400000045000000F800020044000000F900020052000000F800020052000000F50007000300000054000000EC000000550000002600000044000000F50007000900000056000000EB000000550000004800000044000000F50007000900000057000000EA000000550000004900000044000000F50007000900000058000000E9000000550000004A00000044000000F50007000900000059000000E8000000550000004B00000044000000B10005004C0000005A000000540000001E000000F6000400470000005300000000000000FA0004005A0000005300000047000000F800020053000000F90002005F000000F80002005F000000F50007000300000061000000E7000000600000002600000053000000F500070009000000620000009E000000600000005600000053000000F50007000900000063000000B6000000600000005700000053000000F50007000900000064000000CE000000600000005800000053000000F50007000900000065000000E6000000600000005900000053000000B10005004C00000066000000610000001B000000F6000400550000006000000000000000FA000400660000006000000055000000F800020060000000800005000300000067000000310000003B0000008700050003000000680000006700000025000000840005000300000069000000460000002200000084000500030000006A000000540000002300000080000500030000006B000000690000006A00000084000500030000006C000000610000002800000080000500030000006D0000006B0000006C00000080000500030000006E0000006D00000068000000410006006F0000007000000011000000260000006E0000003D000400090000007100000070000000800005000300000072000000610000001D00000084000500030000007300000072000000280000008000050003000000740000006B000000730000008000050003000000750000007400000068000000410006006F000000760000001100000026000000750000003D000400090000007700000076000000800005000300000078000000610000001F000000840005000300000079000000780000002800000080000500030000007A0000006B0000007900000080000500030000007B0000007A00000068000000410006006F0000007C00000011000000260000007B0000003D000400090000007D0000007C00000080000500030000007E000000610000001E00000084000500030000007F0000007E000000280000008000050003000000800000006B0000007F0000008000050003000000810000008000000068000000410006006F000000820000001100000026000000810000003D0004000900000083000000820000008000050003000000840000003C000000460000008000050003000000850000003D00000054000000800005000300000086000000320000008400000080000500030000008700000033000000850000008700050003000000880000006100000025000000840005000300000089000000860000002400000084000500030000008A000000870000002500000080000500030000008B000000890000008A00000080000500030000008C0000008B00000088000000410006006F0000008D00000016000000260000008C0000003D000400090000008E0000008D000000510005000A0000008F0000008E000000000000005000070009000000900000008F0000008F0000008F0000008F00000085000500090000009100000090000000710000008100050009000000920000009100000062000000510005000A000000930000008E000000010000005000070009000000940000009300000093000000930000009300000085000500090000009500000094000000770000008100050009000000960000009500000092000000510005000A000000970000008E0000000200000050000700090000009800000097000000970000009700000097000000850005000900000099000000980000007D00000081000500090000009A0000009900000096000000510005000A0000009B0000008E0000000300000050000700090000009C0000009B0000009B0000009B0000009B00000085000500090000009D0000009C0000008300000081000500090000009E0000009D0000009A00000080000500030000009F000000460000001F0000008000050003000000A00000003C0000009F0000008000050003000000A100000032000000A00000008400050003000000A2000000A1000000240000008000050003000000A3000000A20000008A0000008000050003000000A4000000A300000088000000410006006F000000A50000001600000026000000A40000003D00040009000000A6000000A5000000510005000A000000A7000000A6000000000000005000070009000000A8000000A7000000A7000000A7000000A70000008500050009000000A9000000A8000000710000008100050009000000AA000000A900000063000000510005000A000000AB000000A6000000010000005000070009000000AC000000AB000000AB000000AB000000AB0000008500050009000000AD000000AC000000770000008100050009000000AE000000AD000000AA000000510005000A000000AF000000A6000000020000005000070009000000B0000000AF000000AF000000AF000000AF0000008500050009000000B1000000B00000007D0000008100050009000000B2000000B1000000AE000000510005000A000000B3000000A6000000030000005000070009000000B4000000B3000000B3000000B3000000B30000008500050009000000B5000000B4000000830000008100050009000000B6000000B5000000B20000008000050003000000B700000046000000250000008000050003000000B80000003C000000B70000008000050003000000B900000032000000B80000008400050003000000BA000000B9000000240000008000050003000000BB000000BA0000008A0000008000050003000000BC000000BB00000088000000410006006F000000BD0000001600000026000000BC0000003D00040009000000BE000000BD000000510005000A000000BF000000BE000000000000005000070009000000C0000000BF000000BF000000BF000000BF0000008500050009000000C1000000C0000000710000008100050009000000C2000000C100000064000000510005000A000000C3000000BE000000010000005000070009000000C4000000C3000000C3000000C3000000C30000008500050009000000C5000000C4000000770000008100050009000000C6000000C5000000C2000000510005000A000000C7000000BE000000020000005000070009000000C8000000C7000000C7000000C7000000C70000008500050009000000C9000000C80000007D0000008100050009000000CA000000C9000000C6000000510005000A000000CB000000BE000000030000005000070009000000CC000000CB000000CB000000CB000000CB0000008500050009000000CD000000CC000000830000008100050009000000CE000000CD000000CA0000008000050003000000CF000000460000001C0000008000050003000000D00000003C000000CF0000008000050003000000D100000032000000D00000008400050003000000D2000000D1000000240000008000050003000000D3000000D20000008A0000008000050003000000D4000000D300000088000000410006006F000000D50000001600000026000000D40000003D00040009000000D6000000D5000000510005000A000000D7000000D6000000000000005000070009000000D8000000D7000000D7000000D7000000D70000008500050009000000D9000000D8000000710000008100050009000000DA000000D900000065000000510005000A000000DB000000D6000000010000005000070009000000DC000000DB000000DB000000DB000000DB0000008500050009000000DD000000DC000000770000008100050009000000DE000000DD000000DA000000510005000A000000DF000000D6000000020000005000070009000000E0000000DF000000DF000000DF000000DF0000008500050009000000E1000000E00000007D0000008100050009000000E2000000E1000000DE000000510005000A000000E3000000D6000000030000005000070009000000E4000000E3000000E3000000E3000000E30000008500050009000000E5000000E4000000830000008100050009000000E6000000E5000000E20000003E0003005B0000009E0000003E0003005C000000B60000003E0003005D000000CE0000003E0003005E000000E60000008000050003000000E70000006100000025000000F90002005F000000F8000200550000003D00040009000000E80000005E0000003D00040009000000E90000005D0000003D00040009000000EA0000005C0000003D00040009000000EB0000005B0000003E0003004E000000EB0000003E0003004F000000EA0000003E00030050000000E90000003E00030051000000E80000008000050003000000EC000000540000001D000000F900020052000000F8000200470000003D00040009000000ED000000510000003D00040009000000EE000000500000003D00040009000000EF0000004F0000003D00040009000000F00000004E0000003E0003003F000000F00000003E00030040000000EF0000003E00030041000000EE0000003E00030042000000ED0000008000050003000000F1000000460000001D000000F900020043000000F8000200450000003D00040009000000F2000000420000003D00040009000000F3000000410000003D00040009000000F4000000400000003D00040009000000F50000003F0000008000050003000000F60000003A0000001E0000008000050003000000F70000002F000000F60000008000050003000000F800000030000000370000008000050003000000F9000000310000003B0000008700050003000000FA000000F9000000250000008400050003000000FB000000F7000000270000008400050003000000FC000000F8000000280000008000050003000000FD000000FB000000FC0000008000050003000000FE000000FD000000FA000000410006006F000000FF0000000C00000026000000FE0000003E000300FF000000F20000008000050003000000000100003A0000001F0000008000050003000000010100002F00000000010000840005000300000002010000010100002700000080000500030000000301000002010000FC00000080000500030000000401000003010000FA000000410006006F000000050100000C00000026000000040100003E00030005010000F30000008000050003000000060100003A0000001D0000008000050003000000070100002F00000006010000840005000300000008010000070100002700000080000500030000000901000008010000FC00000080000500030000000A01000009010000FA000000410006006F0000000B0100000C000000260000000A0100003E0003000B010000F400000080000500030000000C0100002F0000003A00000084000500030000000D0100000C0100002700000080000500030000000E0100000D010000FC00000080000500030000000F0100000E010000FA000000410006006F000000100100000C000000260000000F0100003E00030010010000F5000000FD0001003800010008000C0004000800"> : vector<6380xi8>
  vm.func @predict_ex_dispatch_1(%arg0: !vm.ref<!hal.buffer>, %arg1: !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer> attributes {noinline} {
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %ref_0 = vm.call @hal.device.allocator(%ref) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
    %c1605632 = vm.const.i32 1605632 : i32
    %c50 = vm.const.i32 50 : i32
    %c15 = vm.const.i32 15 : i32
    %ref_1 = vm.call @hal.allocator.allocate(%ref_0, %c50, %c15, %c1605632) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
    %c1 = vm.const.i32 1 : i32
    %c3 = vm.const.i32 3 : i32
    %ref_2 = vm.call @hal.command_buffer.create(%ref, %c1, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
    vm.call @hal.command_buffer.begin(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
    %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
    %c3240000 = vm.const.i32 3240000 : i32
    %c18432 = vm.const.i32 18432 : i32
    %zero = vm.const.i32.zero : i32
    %c2 = vm.const.i32 2 : i32
    vm.call.variadic @hal.command_buffer.push_descriptor_set(%ref_2, %_executable_layout_0, %zero, [(%zero, %arg0, %zero, %c3240000), (%c1, %arg1, %zero, %c18432), (%c2, %ref_1, %zero, %c1605632)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...)
    %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
    vm.cond_br %_device_match_id_0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %_executable_predict_ex_dispatch_1_dispatch_0 = vm.global.load.ref @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
    %c28 = vm.const.i32 28 : i32
    vm.call @hal.command_buffer.dispatch(%ref_2, %_executable_predict_ex_dispatch_1_dispatch_0, %zero, %c2, %c28, %c28) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
    %c20 = vm.const.i32 20 : i32
    %c5 = vm.const.i32 5 : i32
    vm.call @hal.command_buffer.execution_barrier(%ref_2, %c20, %c5, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
    vm.call @hal.command_buffer.end(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
    vm.call @hal.ex.submit_and_wait(%ref, %ref_2) : (!vm.ref<!hal.device>, !vm.ref<!hal.command_buffer>) -> ()
    vm.return %ref_1 : !vm.ref<!hal.buffer>
  ^bb2:  // pred: ^bb0
    vm.fail %c2, "unreachable location reached"
  }
  vm.export @predict_ex_dispatch_1 as("predict_ex_dispatch_1$raw")
  vm.func @predict_ex_dispatch_1$async(%arg0: !vm.ref<!hal.semaphore>, %arg1: i32, %arg2: !vm.ref<!hal.buffer_view>, %arg3: !vm.ref<!hal.buffer_view>, %arg4: !vm.ref<!hal.semaphore>, %arg5: i32) -> !vm.ref<!hal.buffer_view> {
    %0 = vm.call @hal.semaphore.await(%arg0, %arg1) : (!vm.ref<!hal.semaphore>, i32) -> i32
    vm.cond_br %0, ^bb2(%0 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    %ref = vm.call @hal.buffer_view.buffer(%arg2) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
    %ref_0 = vm.call @hal.buffer_view.buffer(%arg3) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
    %ref_1 = vm.call @predict_ex_dispatch_1(%ref, %ref_0) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
    %c1 = vm.const.i32 1 : i32
    %c112 = vm.const.i32 112 : i32
    %c32 = vm.const.i32 32 : i32
    %c50331680 = vm.const.i32 50331680 : i32
    %ref_2 = vm.call.variadic @hal.buffer_view.create(%ref_1, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
    vm.call @hal.semaphore.signal(%arg4, %arg5) : (!vm.ref<!hal.semaphore>, i32) -> ()
    vm.return %ref_2 : !vm.ref<!hal.buffer_view>
  ^bb2(%1: i32):  // pred: ^bb0
    vm.fail %1, "semaphore wait failed"
  }
  vm.export @predict_ex_dispatch_1$async
  vm.func @predict_ex_dispatch_1$sync(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %zero = vm.const.i32.zero : i32
    %ref_0 = vm.call @hal.semaphore.create(%ref, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.semaphore>
    %0 = vm.call @hal.semaphore.await(%ref_0, %zero) : (!vm.ref<!hal.semaphore>, i32) -> i32
    vm.cond_br %0, ^bb2(%0 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    %ref_1 = vm.call @hal.buffer_view.buffer(%arg0) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
    %ref_2 = vm.call @hal.buffer_view.buffer(%arg1) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
    %ref_3 = vm.call @predict_ex_dispatch_1(%ref_1, %ref_2) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
    %c1 = vm.const.i32 1 : i32
    %c112 = vm.const.i32 112 : i32
    %c32 = vm.const.i32 32 : i32
    %c50331680 = vm.const.i32 50331680 : i32
    %ref_4 = vm.call.variadic @hal.buffer_view.create(%ref_3, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
    vm.call @hal.semaphore.signal(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> ()
    %1 = vm.call @hal.semaphore.await(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> i32
    vm.cond_br %1, ^bb2(%1 : i32), ^bb3
  ^bb2(%2: i32):  // 2 preds: ^bb0, ^bb1
    vm.fail %2, "semaphore wait failed"
  ^bb3:  // pred: ^bb1
    vm.return %ref_4 : !vm.ref<!hal.buffer_view>
  }
  vm.export @predict_ex_dispatch_1$sync as("predict_ex_dispatch_1")
  vm.import @hal.ex.shared_device() -> !vm.ref<!hal.device> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.ex.submit_and_wait(%device : !vm.ref<!hal.device>, %command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
  vm.import @hal.allocator.allocate(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %allocation_size : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
  vm.import @hal.buffer_view.create(%buffer : !vm.ref<!hal.buffer>, %element_type : i32, %shape : i32 ...) -> !vm.ref<!hal.buffer_view> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.buffer_view.buffer(%buffer_view : !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.command_buffer.create(%device : !vm.ref<!hal.device>, %modes : i32, %command_categories : i32) -> !vm.ref<!hal.command_buffer> attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.begin(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.end(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.execution_barrier(%command_buffer : !vm.ref<!hal.command_buffer>, %source_stage_mask : i32, %target_stage_mask : i32, %flags : i32) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.push_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) attributes {sym_visibility = "private"}
  vm.import @hal.command_buffer.dispatch(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroup_x : i32, %workgroup_y : i32, %workgroup_z : i32) attributes {sym_visibility = "private"}
  vm.import @hal.descriptor_set_layout.create(%device : !vm.ref<!hal.device>, %usage_type : i32, %bindings : tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.device.allocator(%device : !vm.ref<!hal.device>) -> !vm.ref<!hal.allocator> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.device.match.id(%device : !vm.ref<!hal.device>, %pattern : !vm.ref<!iree.byte_buffer>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.executable.create(%device : !vm.ref<!hal.device>, %executable_format : i32, %executable_data : !vm.ref<!iree.byte_buffer>, %executable_layouts : !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.executable_layout.create(%device : !vm.ref<!hal.device>, %push_constants : i32, %set_layouts : !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.semaphore.create(%device : !vm.ref<!hal.device>, %initial_value : i32) -> !vm.ref<!hal.semaphore> attributes {nosideeffects, sym_visibility = "private"}
  vm.import @hal.semaphore.signal(%semaphore : !vm.ref<!hal.semaphore>, %new_value : i32) attributes {sym_visibility = "private"}
  vm.import @hal.semaphore.await(%semaphore : !vm.ref<!hal.semaphore>, %min_value : i32) -> i32 attributes {sym_visibility = "private"}
  vm.func @__init() {
    %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %_utf8_vulkan_7197BF52A22CAFD7 = vm.const.ref.rodata @_utf8_vulkan_7197BF52A22CAFD7 : !vm.ref<!iree.byte_buffer>
    %0 = vm.call @hal.device.match.id(%ref, %_utf8_vulkan_7197BF52A22CAFD7) : (!vm.ref<!hal.device>, !vm.ref<!iree.byte_buffer>) -> i32
    vm.global.store.i32 %0, @_device_match_id_0 : i32
    %ref_0 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %c1 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    %c7 = vm.const.i32 7 : i32
    %c6 = vm.const.i32 6 : i32
    %zero = vm.const.i32.zero : i32
    %ref_1 = vm.call.variadic @hal.descriptor_set_layout.create(%ref_0, %c1, [(%zero, %c7, %c1), (%c1, %c7, %c1), (%c2, %c7, %c6)]) : (!vm.ref<!hal.device>, i32, tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout>
    vm.global.store.ref %ref_1, @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
    %_descriptor_set_layout_0 = vm.global.load.ref @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
    %ref_2 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %ref_3 = vm.call.variadic @hal.executable_layout.create(%ref_2, %zero, [%_descriptor_set_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout>
    vm.global.store.ref %ref_3, @_executable_layout_0 : !vm.ref<!hal.executable_layout>
    %ref_4 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
    %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
    vm.cond_br %_device_match_id_0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
    %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv = vm.const.ref.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv : !vm.ref<!iree.byte_buffer>
    %c1397773893 = vm.const.i32 1397773893 : i32
    %ref_5 = vm.call.variadic @hal.executable.create(%ref_4, %c1397773893, %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv, [%_executable_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
    vm.br ^bb3(%ref_5 : !vm.ref<!hal.executable>)
  ^bb2:  // pred: ^bb0
    %null = vm.const.ref.zero : !vm.ref<!hal.executable>
    vm.br ^bb3(%null : !vm.ref<!hal.executable>)
  ^bb3(%1: !vm.ref<!hal.executable>):  // 2 preds: ^bb1, ^bb2
    vm.global.store.ref %1, @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
    vm.return
  }
  vm.export @__init
}

module  {
  vm.module @module {
    vm.global.i32 @_device_match_id_0 mutable : i32
    vm.rodata @_utf8_vulkan_7197BF52A22CAFD7 dense<[118, 117, 108, 107, 97, 110, 42]> : vector<7xi8>
    vm.global.ref @_descriptor_set_layout_0 mutable : !vm.ref<!hal.descriptor_set_layout>
    vm.global.ref @_executable_layout_0 mutable : !vm.ref<!hal.executable_layout>
    vm.global.ref @_executable_predict_ex_dispatch_1_dispatch_0 mutable : !vm.ref<!hal.executable>
    vm.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv dense<"0x080000005350564524E7FFFF0800000034000000010000000400000020000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000027060000030223070000010016000000110100000000000011000200010000000A000B005350565F4B48525F73746F726167655F6275666665725F73746F726167655F636C617373000000000E00030000000000010000000F000E000500000019000000707265646963745F65785F64697370617463685F315F64697370617463685F3000000000050000000400000010000600190000001100000004000000040000000100000005000B00040000005F5F6275696C74696E5F7661725F4C6F63616C496E766F636174696F6E49645F5F00000005000900050000005F5F6275696C74696E5F7661725F576F726B67726F757049645F5F0005000A000C0000005F5F7265736F757263655F7661725F32343838393331333339323239365F5F0005000A00110000005F5F7265736F757263655F7661725F32343838393331333430383837325F5F0005000A00160000005F5F7265736F757263655F7661725F32343838393331333339323130345F5F0005000B0019000000707265646963745F65785F64697370617463685F315F64697370617463685F300000000047000400040000000B0000001B00000047000400050000000B0000001A000000470004000800000006000000100000004800050007000000000000002300000000000000470003000700000002000000470004000C0000002100000002000000470004000C0000002200000000000000470004000F0000000600000010000000480005000E000000000000002300000000000000470003000E00000002000000470004001100000021000000010000004700040011000000220000000000000047000400140000000600000010000000480005001300000000000000230000000000000047000300130000000200000047000400160000002100000000000000470004001600000022000000000000001500040003000000200000000000000017000400020000000300000003000000200004000100000001000000020000003B0004000100000004000000010000003B000400010000000500000001000000160003000A0000002000000017000400090000000A000000040000002B000400030000000B000000008801001C00040008000000090000000B0000001E000300070000000800000020000400060000000C000000070000003B000400060000000C0000000C0000002B0004000300000010000000800400001C0004000F00000009000000100000001E0003000E0000000F000000200004000D0000000C0000000E0000003B0004000D000000110000000C0000002B0004000300000015000000041703001C0004001400000009000000150000001E000300130000001400000020000400120000000C000000130000003B00040012000000160000000C00000013000200180000002100030017000000180000002B000400030000001B000000100000002B000400030000001C000000060000002B000400030000001D000000010000002B000400030000001E000000030000002B000400030000001F000000020000002B0004000A00000021000000000000002C0007000900000020000000210000002100000021000000210000002B0004000300000022000000800100002B0004000300000023000000800000002B0004000300000024000000840300002B0004000300000025000000040000002B0004000300000026000000000000002B0004000300000027000000800300002B000400030000002800000008000000200004003E0000000700000009000000140002004C000000200004006F0000000C000000090000003600050018000000190000000000000017000000F80002001A0000003B0004003E0000003F000000070000003B0004003E00000040000000070000003B0004003E00000041000000070000003B0004003E00000042000000070000003B0004003E0000004E000000070000003B0004003E0000004F000000070000003B0004003E00000050000000070000003B0004003E00000051000000070000003B0004003E0000005B000000070000003B0004003E0000005C000000070000003B0004003E0000005D000000070000003B0004003E0000005E000000070000003D00040002000000290000000500000051000500030000002A00000029000000000000003D000400020000002B0000000500000051000500030000002C0000002B000000010000003D000400020000002D0000000500000051000500030000002E0000002D0000000200000084000500030000002F0000002E000000250000008400050003000000300000002C000000250000008400050003000000310000002A0000001B0000008400050003000000320000002F0000001F000000840005000300000033000000300000001F0000003D00040002000000340000000400000051000500030000003500000034000000000000003D00040002000000360000000400000051000500030000003700000036000000010000003D000400020000003800000004000000510005000300000039000000380000000200000084000500030000003A000000390000002500000084000500030000003B000000350000002500000084000500030000003C000000390000002800000084000500030000003D000000370000001F000000F900020043000000F800020043000000F50007000300000046000000F100000047000000260000001A000000F50007000900000048000000F000000047000000200000001A000000F50007000900000049000000EF00000047000000200000001A000000F5000700090000004A000000EE00000047000000200000001A000000F5000700090000004B000000ED00000047000000200000001A000000B10005004C0000004D000000460000001E000000F6000400450000004400000000000000FA0004004D0000004400000045000000F800020044000000F900020052000000F800020052000000F50007000300000054000000EC000000550000002600000044000000F50007000900000056000000EB000000550000004800000044000000F50007000900000057000000EA000000550000004900000044000000F50007000900000058000000E9000000550000004A00000044000000F50007000900000059000000E8000000550000004B00000044000000B10005004C0000005A000000540000001E000000F6000400470000005300000000000000FA0004005A0000005300000047000000F800020053000000F90002005F000000F80002005F000000F50007000300000061000000E7000000600000002600000053000000F500070009000000620000009E000000600000005600000053000000F50007000900000063000000B6000000600000005700000053000000F50007000900000064000000CE000000600000005800000053000000F50007000900000065000000E6000000600000005900000053000000B10005004C00000066000000610000001B000000F6000400550000006000000000000000FA000400660000006000000055000000F800020060000000800005000300000067000000310000003B0000008700050003000000680000006700000025000000840005000300000069000000460000002200000084000500030000006A000000540000002300000080000500030000006B000000690000006A00000084000500030000006C000000610000002800000080000500030000006D0000006B0000006C00000080000500030000006E0000006D00000068000000410006006F0000007000000011000000260000006E0000003D000400090000007100000070000000800005000300000072000000610000001D00000084000500030000007300000072000000280000008000050003000000740000006B000000730000008000050003000000750000007400000068000000410006006F000000760000001100000026000000750000003D000400090000007700000076000000800005000300000078000000610000001F000000840005000300000079000000780000002800000080000500030000007A0000006B0000007900000080000500030000007B0000007A00000068000000410006006F0000007C00000011000000260000007B0000003D000400090000007D0000007C00000080000500030000007E000000610000001E00000084000500030000007F0000007E000000280000008000050003000000800000006B0000007F0000008000050003000000810000008000000068000000410006006F000000820000001100000026000000810000003D0004000900000083000000820000008000050003000000840000003C000000460000008000050003000000850000003D00000054000000800005000300000086000000320000008400000080000500030000008700000033000000850000008700050003000000880000006100000025000000840005000300000089000000860000002400000084000500030000008A000000870000002500000080000500030000008B000000890000008A00000080000500030000008C0000008B00000088000000410006006F0000008D00000016000000260000008C0000003D000400090000008E0000008D000000510005000A0000008F0000008E000000000000005000070009000000900000008F0000008F0000008F0000008F00000085000500090000009100000090000000710000008100050009000000920000009100000062000000510005000A000000930000008E000000010000005000070009000000940000009300000093000000930000009300000085000500090000009500000094000000770000008100050009000000960000009500000092000000510005000A000000970000008E0000000200000050000700090000009800000097000000970000009700000097000000850005000900000099000000980000007D00000081000500090000009A0000009900000096000000510005000A0000009B0000008E0000000300000050000700090000009C0000009B0000009B0000009B0000009B00000085000500090000009D0000009C0000008300000081000500090000009E0000009D0000009A00000080000500030000009F000000460000001F0000008000050003000000A00000003C0000009F0000008000050003000000A100000032000000A00000008400050003000000A2000000A1000000240000008000050003000000A3000000A20000008A0000008000050003000000A4000000A300000088000000410006006F000000A50000001600000026000000A40000003D00040009000000A6000000A5000000510005000A000000A7000000A6000000000000005000070009000000A8000000A7000000A7000000A7000000A70000008500050009000000A9000000A8000000710000008100050009000000AA000000A900000063000000510005000A000000AB000000A6000000010000005000070009000000AC000000AB000000AB000000AB000000AB0000008500050009000000AD000000AC000000770000008100050009000000AE000000AD000000AA000000510005000A000000AF000000A6000000020000005000070009000000B0000000AF000000AF000000AF000000AF0000008500050009000000B1000000B00000007D0000008100050009000000B2000000B1000000AE000000510005000A000000B3000000A6000000030000005000070009000000B4000000B3000000B3000000B3000000B30000008500050009000000B5000000B4000000830000008100050009000000B6000000B5000000B20000008000050003000000B700000046000000250000008000050003000000B80000003C000000B70000008000050003000000B900000032000000B80000008400050003000000BA000000B9000000240000008000050003000000BB000000BA0000008A0000008000050003000000BC000000BB00000088000000410006006F000000BD0000001600000026000000BC0000003D00040009000000BE000000BD000000510005000A000000BF000000BE000000000000005000070009000000C0000000BF000000BF000000BF000000BF0000008500050009000000C1000000C0000000710000008100050009000000C2000000C100000064000000510005000A000000C3000000BE000000010000005000070009000000C4000000C3000000C3000000C3000000C30000008500050009000000C5000000C4000000770000008100050009000000C6000000C5000000C2000000510005000A000000C7000000BE000000020000005000070009000000C8000000C7000000C7000000C7000000C70000008500050009000000C9000000C80000007D0000008100050009000000CA000000C9000000C6000000510005000A000000CB000000BE000000030000005000070009000000CC000000CB000000CB000000CB000000CB0000008500050009000000CD000000CC000000830000008100050009000000CE000000CD000000CA0000008000050003000000CF000000460000001C0000008000050003000000D00000003C000000CF0000008000050003000000D100000032000000D00000008400050003000000D2000000D1000000240000008000050003000000D3000000D20000008A0000008000050003000000D4000000D300000088000000410006006F000000D50000001600000026000000D40000003D00040009000000D6000000D5000000510005000A000000D7000000D6000000000000005000070009000000D8000000D7000000D7000000D7000000D70000008500050009000000D9000000D8000000710000008100050009000000DA000000D900000065000000510005000A000000DB000000D6000000010000005000070009000000DC000000DB000000DB000000DB000000DB0000008500050009000000DD000000DC000000770000008100050009000000DE000000DD000000DA000000510005000A000000DF000000D6000000020000005000070009000000E0000000DF000000DF000000DF000000DF0000008500050009000000E1000000E00000007D0000008100050009000000E2000000E1000000DE000000510005000A000000E3000000D6000000030000005000070009000000E4000000E3000000E3000000E3000000E30000008500050009000000E5000000E4000000830000008100050009000000E6000000E5000000E20000003E0003005B0000009E0000003E0003005C000000B60000003E0003005D000000CE0000003E0003005E000000E60000008000050003000000E70000006100000025000000F90002005F000000F8000200550000003D00040009000000E80000005E0000003D00040009000000E90000005D0000003D00040009000000EA0000005C0000003D00040009000000EB0000005B0000003E0003004E000000EB0000003E0003004F000000EA0000003E00030050000000E90000003E00030051000000E80000008000050003000000EC000000540000001D000000F900020052000000F8000200470000003D00040009000000ED000000510000003D00040009000000EE000000500000003D00040009000000EF0000004F0000003D00040009000000F00000004E0000003E0003003F000000F00000003E00030040000000EF0000003E00030041000000EE0000003E00030042000000ED0000008000050003000000F1000000460000001D000000F900020043000000F8000200450000003D00040009000000F2000000420000003D00040009000000F3000000410000003D00040009000000F4000000400000003D00040009000000F50000003F0000008000050003000000F60000003A0000001E0000008000050003000000F70000002F000000F60000008000050003000000F800000030000000370000008000050003000000F9000000310000003B0000008700050003000000FA000000F9000000250000008400050003000000FB000000F7000000270000008400050003000000FC000000F8000000280000008000050003000000FD000000FB000000FC0000008000050003000000FE000000FD000000FA000000410006006F000000FF0000000C00000026000000FE0000003E000300FF000000F20000008000050003000000000100003A0000001F0000008000050003000000010100002F00000000010000840005000300000002010000010100002700000080000500030000000301000002010000FC00000080000500030000000401000003010000FA000000410006006F000000050100000C00000026000000040100003E00030005010000F30000008000050003000000060100003A0000001D0000008000050003000000070100002F00000006010000840005000300000008010000070100002700000080000500030000000901000008010000FC00000080000500030000000A01000009010000FA000000410006006F0000000B0100000C000000260000000A0100003E0003000B010000F400000080000500030000000C0100002F0000003A00000084000500030000000D0100000C0100002700000080000500030000000E0100000D010000FC00000080000500030000000F0100000E010000FA000000410006006F000000100100000C000000260000000F0100003E00030010010000F5000000FD0001003800010008000C0004000800"> : vector<6380xi8>
    vm.func @predict_ex_dispatch_1(%arg0: !vm.ref<!hal.buffer>, %arg1: !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer> attributes {noinline} {
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_0 = vm.call @hal.device.allocator(%ref) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
      %c1605632 = vm.const.i32 1605632 : i32
      %c50 = vm.const.i32 50 : i32
      %c15 = vm.const.i32 15 : i32
      %ref_1 = vm.call @hal.allocator.allocate(%ref_0, %c50, %c15, %c1605632) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
      %c1 = vm.const.i32 1 : i32
      %c3 = vm.const.i32 3 : i32
      %ref_2 = vm.call @hal.command_buffer.create(%ref, %c1, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
      vm.call @hal.command_buffer.begin(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
      %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      %c3240000 = vm.const.i32 3240000 : i32
      %c18432 = vm.const.i32 18432 : i32
      %zero = vm.const.i32.zero : i32
      %c2 = vm.const.i32 2 : i32
      vm.call.variadic @hal.command_buffer.push_descriptor_set(%ref_2, %_executable_layout_0, %zero, [(%zero, %arg0, %zero, %c3240000), (%c1, %arg1, %zero, %c18432), (%c2, %ref_1, %zero, %c1605632)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...)
      %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
      vm.cond_br %_device_match_id_0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %_executable_predict_ex_dispatch_1_dispatch_0 = vm.global.load.ref @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
      %c28 = vm.const.i32 28 : i32
      vm.call @hal.command_buffer.dispatch(%ref_2, %_executable_predict_ex_dispatch_1_dispatch_0, %zero, %c2, %c28, %c28) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
      %c20 = vm.const.i32 20 : i32
      %c5 = vm.const.i32 5 : i32
      vm.call @hal.command_buffer.execution_barrier(%ref_2, %c20, %c5, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
      vm.call @hal.command_buffer.end(%ref_2) : (!vm.ref<!hal.command_buffer>) -> ()
      vm.call @hal.ex.submit_and_wait(%ref, %ref_2) : (!vm.ref<!hal.device>, !vm.ref<!hal.command_buffer>) -> ()
      vm.return %ref_1 : !vm.ref<!hal.buffer>
    ^bb2:  // pred: ^bb0
      vm.fail %c2, "unreachable location reached"
    }
    vm.export @predict_ex_dispatch_1 as("predict_ex_dispatch_1$raw")
    vm.func @predict_ex_dispatch_1$async(%arg0: !vm.ref<!hal.semaphore>, %arg1: i32, %arg2: !vm.ref<!hal.buffer_view>, %arg3: !vm.ref<!hal.buffer_view>, %arg4: !vm.ref<!hal.semaphore>, %arg5: i32) -> !vm.ref<!hal.buffer_view> {
      %0 = vm.call @hal.semaphore.await(%arg0, %arg1) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %0, ^bb2(%0 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %ref = vm.call @hal.buffer_view.buffer(%arg2) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_0 = vm.call @hal.buffer_view.buffer(%arg3) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_1 = vm.call @predict_ex_dispatch_1(%ref, %ref_0) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
      %c1 = vm.const.i32 1 : i32
      %c112 = vm.const.i32 112 : i32
      %c32 = vm.const.i32 32 : i32
      %c50331680 = vm.const.i32 50331680 : i32
      %ref_2 = vm.call.variadic @hal.buffer_view.create(%ref_1, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
      vm.call @hal.semaphore.signal(%arg4, %arg5) : (!vm.ref<!hal.semaphore>, i32) -> ()
      vm.return %ref_2 : !vm.ref<!hal.buffer_view>
    ^bb2(%1: i32):  // pred: ^bb0
      vm.fail %1, "semaphore wait failed"
    }
    vm.export @predict_ex_dispatch_1$async
    vm.func @predict_ex_dispatch_1$sync(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {f = "I32!B14!d1d225d225d16B11!d3d3d16d32R18!B14!d1d112d112d32", fv = "1"}} {
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %zero = vm.const.i32.zero : i32
      %ref_0 = vm.call @hal.semaphore.create(%ref, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.semaphore>
      %0 = vm.call @hal.semaphore.await(%ref_0, %zero) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %0, ^bb2(%0 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %ref_1 = vm.call @hal.buffer_view.buffer(%arg0) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_2 = vm.call @hal.buffer_view.buffer(%arg1) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
      %ref_3 = vm.call @predict_ex_dispatch_1(%ref_1, %ref_2) : (!vm.ref<!hal.buffer>, !vm.ref<!hal.buffer>) -> !vm.ref<!hal.buffer>
      %c1 = vm.const.i32 1 : i32
      %c112 = vm.const.i32 112 : i32
      %c32 = vm.const.i32 32 : i32
      %c50331680 = vm.const.i32 50331680 : i32
      %ref_4 = vm.call.variadic @hal.buffer_view.create(%ref_3, %c50331680, [%c1, %c112, %c112, %c32]) : (!vm.ref<!hal.buffer>, i32, i32 ...) -> !vm.ref<!hal.buffer_view>
      vm.call @hal.semaphore.signal(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> ()
      %1 = vm.call @hal.semaphore.await(%ref_0, %c1) : (!vm.ref<!hal.semaphore>, i32) -> i32
      vm.cond_br %1, ^bb2(%1 : i32), ^bb3
    ^bb2(%2: i32):  // 2 preds: ^bb0, ^bb1
      vm.fail %2, "semaphore wait failed"
    ^bb3:  // pred: ^bb1
      vm.return %ref_4 : !vm.ref<!hal.buffer_view>
    }
    vm.export @predict_ex_dispatch_1$sync as("predict_ex_dispatch_1")
    vm.import @hal.ex.shared_device() -> !vm.ref<!hal.device> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.ex.submit_and_wait(%device : !vm.ref<!hal.device>, %command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.allocator.allocate(%allocator : !vm.ref<!hal.allocator>, %memory_types : i32, %buffer_usage : i32, %allocation_size : i32) -> !vm.ref<!hal.buffer> attributes {sym_visibility = "private"}
    vm.import @hal.buffer_view.create(%buffer : !vm.ref<!hal.buffer>, %element_type : i32, %shape : i32 ...) -> !vm.ref<!hal.buffer_view> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.buffer_view.buffer(%buffer_view : !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.command_buffer.create(%device : !vm.ref<!hal.device>, %modes : i32, %command_categories : i32) -> !vm.ref<!hal.command_buffer> attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.begin(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.end(%command_buffer : !vm.ref<!hal.command_buffer>) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.execution_barrier(%command_buffer : !vm.ref<!hal.command_buffer>, %source_stage_mask : i32, %target_stage_mask : i32, %flags : i32) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.push_descriptor_set(%command_buffer : !vm.ref<!hal.command_buffer>, %executable_layout : !vm.ref<!hal.executable_layout>, %set : i32, %bindings : tuple<i32, !vm.ref<!hal.buffer>, i32, i32> ...) attributes {sym_visibility = "private"}
    vm.import @hal.command_buffer.dispatch(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroup_x : i32, %workgroup_y : i32, %workgroup_z : i32) attributes {sym_visibility = "private"}
    vm.import @hal.descriptor_set_layout.create(%device : !vm.ref<!hal.device>, %usage_type : i32, %bindings : tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.device.allocator(%device : !vm.ref<!hal.device>) -> !vm.ref<!hal.allocator> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.device.match.id(%device : !vm.ref<!hal.device>, %pattern : !vm.ref<!iree.byte_buffer>) -> i32 attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.executable.create(%device : !vm.ref<!hal.device>, %executable_format : i32, %executable_data : !vm.ref<!iree.byte_buffer>, %executable_layouts : !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.executable_layout.create(%device : !vm.ref<!hal.device>, %push_constants : i32, %set_layouts : !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.semaphore.create(%device : !vm.ref<!hal.device>, %initial_value : i32) -> !vm.ref<!hal.semaphore> attributes {nosideeffects, sym_visibility = "private"}
    vm.import @hal.semaphore.signal(%semaphore : !vm.ref<!hal.semaphore>, %new_value : i32) attributes {sym_visibility = "private"}
    vm.import @hal.semaphore.await(%semaphore : !vm.ref<!hal.semaphore>, %min_value : i32) -> i32 attributes {sym_visibility = "private"}
    vm.func @__init() {
      %ref = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %_utf8_vulkan_7197BF52A22CAFD7 = vm.const.ref.rodata @_utf8_vulkan_7197BF52A22CAFD7 : !vm.ref<!iree.byte_buffer>
      %0 = vm.call @hal.device.match.id(%ref, %_utf8_vulkan_7197BF52A22CAFD7) : (!vm.ref<!hal.device>, !vm.ref<!iree.byte_buffer>) -> i32
      vm.global.store.i32 %0, @_device_match_id_0 : i32
      %ref_0 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %c1 = vm.const.i32 1 : i32
      %c2 = vm.const.i32 2 : i32
      %c7 = vm.const.i32 7 : i32
      %c6 = vm.const.i32 6 : i32
      %zero = vm.const.i32.zero : i32
      %ref_1 = vm.call.variadic @hal.descriptor_set_layout.create(%ref_0, %c1, [(%zero, %c7, %c1), (%c1, %c7, %c1), (%c2, %c7, %c6)]) : (!vm.ref<!hal.device>, i32, tuple<i32, i32, i32> ...) -> !vm.ref<!hal.descriptor_set_layout>
      vm.global.store.ref %ref_1, @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
      %_descriptor_set_layout_0 = vm.global.load.ref @_descriptor_set_layout_0 : !vm.ref<!hal.descriptor_set_layout>
      %ref_2 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %ref_3 = vm.call.variadic @hal.executable_layout.create(%ref_2, %zero, [%_descriptor_set_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!hal.descriptor_set_layout> ...) -> !vm.ref<!hal.executable_layout>
      vm.global.store.ref %ref_3, @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      %ref_4 = vm.call @hal.ex.shared_device() : () -> !vm.ref<!hal.device>
      %_device_match_id_0 = vm.global.load.i32 @_device_match_id_0 : i32
      vm.cond_br %_device_match_id_0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %_executable_layout_0 = vm.global.load.ref @_executable_layout_0 : !vm.ref<!hal.executable_layout>
      %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv = vm.const.ref.rodata @_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv : !vm.ref<!iree.byte_buffer>
      %c1397773893 = vm.const.i32 1397773893 : i32
      %ref_5 = vm.call.variadic @hal.executable.create(%ref_4, %c1397773893, %_predict_ex_dispatch_1_dispatch_0_vulkan_spirv_binary_spirv, [%_executable_layout_0]) : (!vm.ref<!hal.device>, i32, !vm.ref<!iree.byte_buffer>, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
      vm.br ^bb3(%ref_5 : !vm.ref<!hal.executable>)
    ^bb2:  // pred: ^bb0
      %null = vm.const.ref.zero : !vm.ref<!hal.executable>
      vm.br ^bb3(%null : !vm.ref<!hal.executable>)
    ^bb3(%1: !vm.ref<!hal.executable>):  // 2 preds: ^bb1, ^bb2
      vm.global.store.ref %1, @_executable_predict_ex_dispatch_1_dispatch_0 : !vm.ref<!hal.executable>
      vm.return
    }
    vm.export @__init
  }
}

