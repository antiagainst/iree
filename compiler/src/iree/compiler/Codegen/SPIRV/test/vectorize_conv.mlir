// RUN: iree-opt --split-input-file --iree-spirv-vectorize %s | FileCheck %s

func.func @ncw_conv_1d(%input: tensor<2x4x4xf32>, %filter: tensor<4x4x1xf32>, %init: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
         ins(%input, %filter : tensor<2x4x4xf32>, tensor<4x4x1xf32>)
         outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %0: tensor<2x4x4xf32>
}
