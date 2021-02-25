// RUN: iree-opt -split-input-file -iree-codegen-annotate-codegen-action %s | IreeFileCheck %s

module attributes {
  iree.codegen.target.policy = #iree.codegen.target.policy<"TestAccelerator", FirstMatch, [
    #iree.codegen.target.choice<
      {"test" = "dont-use-this" } -> [
        #iree.codegen.op.policy<"linalg.matmul", FirstMatch, [
          #iree.codegen.op.choice<
            #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<8x64xf32>>]> ->
            [#iree.codegen.action.tile<along_dimensions = [0, 1], with_tile_size = [8, 64]>]>
        ]>
      ]
    >,
    #iree.codegen.target.choice<
      {"test" = "use-this" } -> [
        #iree.codegen.op.policy<"linalg.conv_2d_input_nhwc_filter_hwcf", FirstMatch, [
          #iree.codegen.op.choice<
            #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<?x2x2x64xf32>>]> ->
            [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_size = [2, 2, 64]>]>,
          #iree.codegen.op.choice<
            #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<?x4x4x16xf32>>]> ->
            [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_size = [4, 4, 16]>]>
        ]>
      ]
    >,
    #iree.codegen.target.choice<
      {"test" = "use-this" } -> [
        #iree.codegen.op.policy<"linalg.conv_2d_input_nhwc_filter_hwcf", FirstMatch, [
          #iree.codegen.op.choice<
            #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<?x2x2x8xf32>>]> ->
            [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_size = [2, 2, 8]>]>
        ]>
      ]
    >
  ]>
} {

// Can match both choice#1 and choice#2
// CHECK-LABEL: func @op_choice_match_first_choice
func @op_choice_match_first_choice(
    %input: tensor<1x225x225x16xf32>,
    %filter: tensor<3x3x16x64xf32>,
    %init : tensor<1x112x112x64xf32>)
-> tensor<1x112x112x64xf32> {
  //      CHECK: linalg.conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME: iree.codegen.actions = [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_size = [2, 2, 64]>]
  %0 = linalg.conv_2d_input_nhwc_filter_hwcf
       {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
       ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x64xf32>)
       outs(%init : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  return %0: tensor<1x112x112x64xf32>
}

// CHECK-LABEL: func @op_choice_match_second_choice
func @op_choice_match_second_choice(
    %input: tensor<1x225x225x16xf32>,
    %filter: tensor<3x3x16x32xf32>,
    %init : tensor<1x112x112x32xf32>)
-> tensor<1x112x112x32xf32> {
  //      CHECK: linalg.conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME: iree.codegen.actions = [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_size = [4, 4, 16]>]
  %0 = linalg.conv_2d_input_nhwc_filter_hwcf
       {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
       ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
       outs(%init : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  return %0: tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func @op_choice_failed_match_dim
func @op_choice_failed_match_dim(
    %input: tensor<1x225x225x16xf32>,
    %filter: tensor<3x3x16x8xf32>,
    %init : tensor<1x112x112x8xf32>)
-> tensor<1x112x112x8xf32> {
  // CHECK-NOT: iree.codegen.actions
  %0 = linalg.conv_2d_input_nhwc_filter_hwcf
       {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
       ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x8xf32>)
       outs(%init : tensor<1x112x112x8xf32>) -> tensor<1x112x112x8xf32>
  return %0: tensor<1x112x112x8xf32>
}

// CHECK-LABEL: func @op_choice_failed_match_element_type
func @op_choice_failed_match_element_type(
    %input: tensor<1x225x225x16xf16>,
    %filter: tensor<3x3x16x32xf16>,
    %init : tensor<1x112x112x32xf16>)
-> tensor<1x112x112x32xf16> {
  // CHECK-NOT: iree.codegen.actions
  %0 = linalg.conv_2d_input_nhwc_filter_hwcf
       {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
       ins(%input, %filter : tensor<1x225x225x16xf16>, tensor<3x3x16x32xf16>)
       outs(%init : tensor<1x112x112x32xf16>) -> tensor<1x112x112x32xf16>
  return %0: tensor<1x112x112x32xf16>
}

// Both the #1 (due to failed filter) and #3 (due to #2 was chosen) target choice are not chosen.
// CHECK-LABEL: func @target_choice_failed_match_dict_filter
func @target_choice_failed_match_dict_filter(
    %a: tensor<8x128xf32>, %b: tensor<128x64xf32>, %c: tensor<8x64xf32>)
-> tensor<8x64xf32> {
  // CHECK-NOT: iree.codegen.actions
  %0 = linalg.matmul
       ins(%a, %b: tensor<8x128xf32>, tensor<128x64xf32>)
       outs(%c : tensor<8x64xf32>) -> tensor<8x64xf32>
  return %0: tensor<8x64xf32>
}

}
