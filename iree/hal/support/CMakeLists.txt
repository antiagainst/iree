# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of custom configuration vars

set(IREE_HAL_SUPPORT_PROFILER_DEFS)
set(IREE_HAL_SUPPORT_PROFILER_DEPS)

if(IREE_VULKAN_ENABLE_NSIGHT_GRAPHICS)
  iree_cc_library(
    NAME
      nsight_graphics_profiler
    HDRS
      "nsight_graphics_profiler.h"
    SRCS
      "nsight_graphics_profiler.c"
    DEPS
      iree::base
      NVIDIA::NsightGraphics
  )

  list(APPEND IREE_HAL_SUPPORT_PROFILER_DEFS IREE_ENABLE_NSIGHT_GRAPHICS)
  list(APPEND IREE_HAL_SUPPORT_PROFILER_DEPS ::nsight_graphics_profiler)
endif()

iree_cc_library(
  NAME
    profiler
  HDRS
    "profiler.h"
  SRCS
    "profiler.c"
  LOCAL_DEFINES
    ${IREE_HAL_SUPPORT_PROFILER_DEFS}
  DEPS
    iree::base
    ${IREE_HAL_SUPPORT_PROFILER_DEPS}
  PUBLIC
)
