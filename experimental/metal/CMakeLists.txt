# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    metal
  HDRS
    "api.h"
  SRCS
    "api.h"
    "metal_driver.m"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::tracing
    iree::hal
    iree::schemas::metal_executable_def_c_fbs
    "-framework Foundation"
    "-framework Metal"
  PUBLIC
)
