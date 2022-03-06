// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/support/profiler.h"

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"

#ifdef IREE_ENABLE_NSIGHT_GRAPHICS
#include "iree/hal/support/nsight_graphics_profiler.h"

IREE_FLAG(bool, vulkan_enable_nsight_graphics, false,
          "Enable Vulkan profiling via Nsight Graphics.");
#endif

iree_status_t iree_hal_support_profiler_attach() {
#ifdef IREE_ENABLE_NSIGHT_GRAPHICS
  if (FLAG_vulkan_enable_nsight_graphics) {
    return iree_hal_support_nsight_graphics_profiler_attach();
  }
#endif
  return iree_ok_status();
}

void iree_hal_support_profiler_detach() {
#ifdef IREE_ENABLE_NSIGHT_GRAPHICS
  if (FLAG_vulkan_enable_nsight_graphics) {
    iree_hal_support_nsight_graphics_profiler_detach();
  }
#endif
}
