// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_METAL_BUILTIN_EXECUTABLES_H_
#define IREE_EXPERIMENTAL_METAL_BUILTIN_EXECUTABLES_H_

#import <Metal/Metal.h>

#include "experimental/metal/metal_kernel_library.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_metal_builtin_executable_t {
  iree_allocator_t host_allocator;

  iree_host_size_t entry_point_count;
  iree_hal_metal_kernel_params_t entry_points[];
} iree_hal_metal_builtin_executable_t;

// Creates builtin executables for polyfill features not directly supported by
// Metal API.
iree_status_t iree_hal_metal_builtin_executable_create(
    id<MTLDevice> device, iree_allocator_t host_allocator,
    iree_hal_metal_builtin_executable_t** out_executable);

void iree_hal_metal_builtin_executable_destroy(
    iree_hal_metal_builtin_executable_t* executable);

iree_status_t iree_hal_metal_builtin_executable_fill_buffer(
    const iree_hal_metal_builtin_executable_t* executable,
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    uint32_t pattern);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_METAL_BUILTIN_EXECUTABLES_H_
