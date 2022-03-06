// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_SUPPORT_NSIGHT_GRAPHICS_PROFILER_H_
#define IREE_HAL_SUPPORT_NSIGHT_GRAPHICS_PROFILER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

IREE_API_EXPORT iree_status_t
iree_hal_support_nsight_graphics_profiler_attach();

IREE_API_EXPORT void iree_hal_support_nsight_graphics_profiler_detach();

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_SUPPORT_NSIGHT_GRAPHICS_PROFILER_H_
