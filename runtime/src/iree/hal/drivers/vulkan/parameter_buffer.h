// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_PARAMETER_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_PARAMETER_BUFFER_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Size, in bytes, of the shared storage mode parameter buffer.
// The given amount of system memory will be allocated and is accessible to both
// the CPU and the GPU.
//
// Larger values here will use more memory but allow more concurrent/complex
// command buffers. As most models that run in these environments are only a few
// hundred dispatches per command buffer we can approximate an average
// consumption of 500 dispatches x worst-case 256B per dispatch of parameters
// and get 128KB.
#define IREE_HAL_VULKAN_PARAMETER_BUFFER_DEFAULT_CAPACITY (128 * 1024)

// A parameter uniform buffer used for uploading parameters to the device.
// This allows for high-frequency writes of parameters at appropriate alignment.
//
// Intended usage is to retain one of these per device queue and use them during
// command buffer recording targeting that particular queue. This avoids
// allocating a lot of small buffers. The underlying buffer has shared storage
// mode; so it resides in system memory and is accessible to both the CPU and
// the GPU.
//
// Parameters handled by this buffer include:
// * Argument buffers for descriptor sets
// * Source buffer for buffer update commands
//
// Thread safe; multiple threads can reserve spaces concurrently.
typedef struct iree_hal_vulkan_parameter_buffer_t {
  // Maximum number of bytes in the buffer.
  uint32_t capacity;

  iree_hal_buffer_t* device_buffer;
  iree_hal_buffer_mapping_t mapping;
  // Device handle to the buffer.
  VkBuffer vulkan_handle;
  // Host pointer to the buffer.
  uint8_t* host_ptr;

  // Non-recursive mutex guarding access to the offset field.
  iree_slim_mutex_t offset_mutex;

  // Current write offset of the device buffer.
  uint32_t offset IREE_GUARDED_BY(offset_mutex);

  // The number of command buffers that are being recorded or executed on
  // device. If this reaches zero, we know that there are no users of the
  // parameter buffer so we can discard the contents and reset the offset to
  // zero.
  iree_atomic_int32_t pending_command_buffers;
} iree_hal_vulkan_parameter_buffer_t;

// Initializes |out_parameter_buffer| with the given |buffer_capacity|.
iree_status_t iree_hal_vulkan_parameter_buffer_initialize(
    iree_host_size_t buffer_capacity, iree_hal_allocator_t* device_allocator,
    iree_hal_vulkan_parameter_buffer_t* out_parameter_buffer);

void iree_hal_vulkan_parameter_buffer_deinitialize(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer);

// Reserves |length| bytes from the parameter buffer and returns a pointer to it
// in |out_reservation|.
iree_status_t iree_hal_vulkan_parameter_buffer_reserve(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer,
    iree_host_size_t length, iree_host_size_t alignment,
    iree_byte_span_t* out_reservation, uint32_t* out_offset);

// Appends |data| of |length| bytes to the parameter buffer.
iree_status_t iree_hal_vulkan_parameter_buffer_append(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer,
    iree_const_byte_span_t source, iree_host_size_t alignment,
    uint32_t* out_offset);

// Resets the parameter buffer to discard all its contents.
void iree_hal_vulkan_parameter_buffer_reset(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer);

// Increases the count of command buffers using this parameter buffer by one.
void iree_hal_vulkan_parameter_buffer_increase_command_buffer_refcount(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer);

// Decreases the count of command buffers using this parameter buffer by one,
// which may trigger reclaiming of resources.
void iree_hal_vulkan_parameter_buffer_decrease_command_buffer_refcount(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_PARAMETER_BUFFER_H_
