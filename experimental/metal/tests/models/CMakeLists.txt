# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_check_single_backend_test_suite(
  NAME
    check_metal-spirv_metal
  SRCS
    "${IREE_SOURCE_DIR}/tests/e2e/models/bert_encoder_unrolled_fake_weights.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/models/collatz.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/models/edge_detection.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/models/fragment_000.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/models/fullyconnected.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/models/mnist_fake_weights.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/models/mobilenetv3_fake_weights.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/models/resnet50_fake_weights.mlir"
    # "${IREE_SOURCE_DIR}/tests/e2e/models/unidirectional_lstm.mlir"
  TARGET_BACKEND
    "metal-spirv"
  DRIVER
    "metal"
  COMPILER_FLAGS
    "--iree-input-type=mhlo"
  TIMEOUT
    900
)

