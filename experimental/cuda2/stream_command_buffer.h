// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_CUDA2_STREAM_COMMAND_BUFFER_H_
#define EXPERIMENTAL_CUDA2_STREAM_COMMAND_BUFFER_H_

#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "experimental/cuda2/cuda_headers.h"
#include "experimental/cuda2/nccl_dynamic_symbols.h"
#include "experimental/cuda2/tracing.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates command buffer that immediately issues commands against the given
// CUDA |stream|. Access to |stream| must be synchronized by the user.
//
// |block_pool| will be used by the stream command buffer to retain copies of
// input data until reset. It must remain live for the lifetime of the command
// buffers that use it.
//
// This command buffer is used to both replay deferred command buffers and
// perform inline execution. When replaying the scratch data required for things
// like buffer updates is retained by the source deferred command buffer and as
// such the |block_pool| and can be NULL to avoid a double copy.
iree_status_t iree_hal_cuda2_stream_command_buffer_create(
    iree_hal_device_t* device,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols,
    const iree_hal_cuda2_nccl_dynamic_symbols_t* nccl_symbols,
    iree_hal_cuda2_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, CUstream stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a CUDA stream-based command buffer.
bool iree_hal_cuda2_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_CUDA2_STREAM_COMMAND_BUFFER_H_
