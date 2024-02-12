// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/parameter_buffer.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/drivers/vulkan/base_buffer.h"

iree_status_t iree_hal_vulkan_parameter_buffer_initialize(
    iree_host_size_t buffer_capacity, iree_hal_allocator_t* device_allocator,
    iree_hal_vulkan_parameter_buffer_t* out_parameter_buffer) {
  IREE_ASSERT_ARGUMENT(out_parameter_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_parameter_buffer, 0, sizeof(*out_parameter_buffer));

  // Create a device local and host visible buffer.
  const iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
               IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ |
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
  };
  iree_hal_buffer_t* device_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(device_allocator, buffer_params,
                                             buffer_capacity, &device_buffer));

  // Map to get its host pointer.
  iree_status_t status = iree_hal_buffer_map_range(
      device_buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, /*local_byte_offset=*/0,
      /*local_byte_length=*/VK_WHOLE_SIZE, &out_parameter_buffer->mapping);

  if (iree_status_is_ok(status)) {
    out_parameter_buffer->capacity = (uint32_t)buffer_capacity;
    out_parameter_buffer->device_buffer = device_buffer;
    out_parameter_buffer->vulkan_handle =
        iree_hal_vulkan_buffer_handle(device_buffer);
    out_parameter_buffer->host_ptr =
        out_parameter_buffer->mapping.contents.data;
    iree_slim_mutex_initialize(&out_parameter_buffer->offset_mutex);
    out_parameter_buffer->offset = 0;
    iree_atomic_store_int32(&out_parameter_buffer->pending_command_buffers, 0,
                            iree_memory_order_relaxed);
  } else {
    iree_hal_allocator_deallocate_buffer(device_allocator, device_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_vulkan_parameter_buffer_deinitialize(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer) {
  iree_slim_mutex_deinitialize(&parameter_buffer->offset_mutex);
  iree_hal_buffer_unmap_range(&parameter_buffer->mapping);
  iree_hal_buffer_destroy(parameter_buffer->device_buffer);
}

iree_status_t iree_hal_vulkan_parameter_buffer_reserve(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer,
    iree_host_size_t length, iree_host_size_t alignment,
    iree_byte_span_t* out_reservation, uint32_t* out_offset) {
  if (length > parameter_buffer->capacity) {
    // This will never fit in the staging buffer.
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reservation (%" PRIhsz
                            " bytes) exceeds the maximum capacity of "
                            "the staging buffer (%" PRIu32 " bytes)",
                            length, parameter_buffer->capacity);
  }

  iree_slim_mutex_lock(&parameter_buffer->offset_mutex);
  uint32_t aligned_offset =
      iree_host_align(parameter_buffer->offset, alignment);
  if (aligned_offset + length > parameter_buffer->capacity) {
    iree_slim_mutex_unlock(&parameter_buffer->offset_mutex);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "failed to reserve %" PRIhsz " bytes in staging buffer", length);
  }
  parameter_buffer->offset = aligned_offset + length;
  iree_slim_mutex_unlock(&parameter_buffer->offset_mutex);

  *out_reservation =
      iree_make_byte_span(parameter_buffer->host_ptr + aligned_offset, length);
  *out_offset = aligned_offset;

  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_parameter_buffer_append(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer,
    iree_const_byte_span_t source, iree_host_size_t alignment,
    uint32_t* out_offset) {
  iree_byte_span_t reservation;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_parameter_buffer_reserve(
      parameter_buffer, source.data_length, alignment, &reservation,
      out_offset));
  memcpy(reservation.data, source.data, source.data_length);
  return iree_ok_status();
}

void iree_hal_vulkan_parameter_buffer_reset(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer) {
  iree_slim_mutex_lock(&parameter_buffer->offset_mutex);
  parameter_buffer->offset = 0;
  iree_slim_mutex_unlock(&parameter_buffer->offset_mutex);
}

void iree_hal_vulkan_parameter_buffer_increase_command_buffer_refcount(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer) {
  iree_atomic_fetch_add_int32(&parameter_buffer->pending_command_buffers, 1,
                              iree_memory_order_relaxed);
}

void iree_hal_vulkan_parameter_buffer_decrease_command_buffer_refcount(
    iree_hal_vulkan_parameter_buffer_t* parameter_buffer) {
  iree_slim_mutex_lock(&parameter_buffer->offset_mutex);
  if (iree_atomic_fetch_sub_int32(&parameter_buffer->pending_command_buffers, 1,
                                  iree_memory_order_relaxed) == 1) {
    parameter_buffer->offset = 0;
  }
  iree_slim_mutex_unlock(&parameter_buffer->offset_mutex);
}
