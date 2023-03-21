// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/metal/metal_device.h"

#include "experimental/metal/api.h"
#include "experimental/metal/builtin_executables.h"
#include "experimental/metal/direct_allocator.h"
#include "experimental/metal/direct_command_buffer.h"
#include "experimental/metal/metal_fence.h"
#include "experimental/metal/metal_shared_event.h"
#include "experimental/metal/nop_executable_cache.h"
#include "experimental/metal/pipeline_layout.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/buffer_transfer.h"

typedef struct iree_hal_metal_device_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command buffers can
  // contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Original driver that owns this device.
  iree_hal_driver_t* driver;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  id<MTLDevice> device;
  // We only expose one single command queue for now. This simplifies synchronization.
  // We can relax this to support multiple queues when needed later.
  id<MTLCommandQueue> queue;

  iree_hal_metal_command_buffer_resource_reference_mode_t command_buffer_resource_reference_mode;

  iree_hal_metal_builtin_executable_t* builtin_executable;

  // A dispatch queue and associated event listener for running Objective-C blocks containing tasks
  // like singaling semaphores and walking up threads.
  dispatch_queue_t dispatch_queue;
  MTLSharedEventListener* event_listener;

  MTLCaptureManager* capture_manager;
} iree_hal_metal_device_t;

static const iree_hal_device_vtable_t iree_hal_metal_device_vtable;

static iree_hal_metal_device_t* iree_hal_metal_device_cast(iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_device_vtable);
  return (iree_hal_metal_device_t*)base_value;
}

static const iree_hal_metal_device_t* iree_hal_metal_device_const_cast(
    const iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_device_vtable);
  return (const iree_hal_metal_device_t*)base_value;
}

void iree_hal_metal_device_params_initialize(iree_hal_metal_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->command_dispatch_type = IREE_HAL_METAL_COMMAND_DISPATCH_TYPE_CONCURRENT;
  out_params->command_buffer_resource_reference_mode =
      IREE_HAL_METAL_COMMAND_BUFFER_RESOURCE_REFERENCE_MODE_UNRETAINED;
  out_params->resource_hazard_tracking_mode =
      IREE_HAL_METAL_RESOURCE_HAZARD_TRACKING_MODE_UNTRACKED;
}

const iree_hal_metal_device_params_t* iree_hal_metal_device_params(
    const iree_hal_device_t* base_device) {
  const iree_hal_metal_device_t* device = iree_hal_metal_device_const_cast(base_device);
  return iree_hal_metal_driver_device_params(device->driver);
}

static iree_status_t iree_hal_metal_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_metal_device_params_t* params, id<MTLDevice> metal_device,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_metal_device_t* device = NULL;

  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);

  iree_status_t status = iree_hal_metal_allocator_create((iree_hal_device_t*)device, metal_device,
                                                         params->resource_hazard_tracking_mode,
                                                         host_allocator, &device->device_allocator);
  iree_hal_metal_builtin_executable_t* builtin_executable = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_metal_builtin_executable_create(metal_device, host_allocator, &builtin_executable);
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_metal_device_vtable, &device->resource);
    iree_string_view_append_to_buffer(identifier, &device->identifier,
                                      (char*)device + iree_sizeof_struct(*device));
    iree_arena_block_pool_initialize(params->arena_block_size, host_allocator, &device->block_pool);
    device->driver = driver;
    iree_hal_driver_retain(device->driver);
    device->host_allocator = host_allocator;
    device->device = [metal_device retain];          // +1
    device->queue = [metal_device newCommandQueue];  // +1
    device->command_buffer_resource_reference_mode = params->command_buffer_resource_reference_mode;
    device->builtin_executable = builtin_executable;
    dispatch_queue_attr_t queue_attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, /*relative_priority=*/0);
    device->dispatch_queue = dispatch_queue_create("dev.iree.queue.metal", queue_attr);
    device->event_listener =
        [[MTLSharedEventListener alloc] initWithDispatchQueue:device->dispatch_queue];  // +1
    device->capture_manager = NULL;

    *out_device = (iree_hal_device_t*)device;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_metal_device_create(iree_hal_driver_t* driver, iree_string_view_t identifier,
                                           const iree_hal_metal_device_params_t* params,
                                           id<MTLDevice> device, iree_allocator_t host_allocator,
                                           iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_metal_device_create_internal(driver, identifier, params, device,
                                                               host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  [device->event_listener release];  // -1
  dispatch_release(device->dispatch_queue);

  iree_hal_metal_builtin_executable_destroy(device->builtin_executable);

  iree_hal_allocator_release(device->device_allocator);
  [device->queue release];   // -1
  [device->device release];  // -1

  iree_arena_block_pool_deinitialize(&device->block_pool);
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_metal_device_id(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_metal_device_host_allocator(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_metal_device_allocator(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_metal_replace_device_allocator(iree_hal_device_t* base_device,
                                                    iree_hal_allocator_t* new_allocator) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static iree_status_t iree_hal_metal_device_trim(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_metal_device_query_i64(iree_hal_device_t* base_device,
                                                     iree_string_view_t category,
                                                     iree_string_view_t key, int64_t* out_value) {
  *out_value = 0;

  if (iree_string_view_equal(category, iree_make_cstring_view("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, iree_make_cstring_view("metal-msl-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "unknown device configuration key value '%.*s :: %.*s'",
                          (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_metal_device_create_channel(iree_hal_device_t* base_device,
                                                          iree_hal_queue_affinity_t queue_affinity,
                                                          iree_hal_channel_params_t params,
                                                          iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented channel create");
}

static iree_status_t iree_hal_metal_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t binding_capacity, iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  if (iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_NESTED))
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented nested command buffer");
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT))
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented multi-shot command buffer");
  return iree_hal_metal_direct_command_buffer_create(
      base_device, mode, command_categories, binding_capacity,
      device->command_buffer_resource_reference_mode, device->queue, device->host_allocator,
      &device->block_pool, device->builtin_executable, out_command_buffer);
}

static iree_status_t iree_hal_metal_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device, iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count, const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_descriptor_set_layout_create(device->host_allocator, flags, binding_count,
                                                     bindings, out_descriptor_set_layout);
}

static iree_status_t iree_hal_metal_device_create_event(iree_hal_device_t* base_device,
                                                        iree_hal_event_t** out_event) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_fence_create(device->device, device->host_allocator, out_event);
}

static iree_status_t iree_hal_metal_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier, iree_loop_t loop,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_nop_executable_cache_create(device->device, device->host_allocator,
                                                    identifier, out_executable_cache);
}

static iree_status_t iree_hal_metal_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count, iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_pipeline_layout_create(device->host_allocator, set_layout_count,
                                               set_layouts, push_constants, out_pipeline_layout);
}

static iree_status_t iree_hal_metal_device_create_semaphore(iree_hal_device_t* base_device,
                                                            uint64_t initial_value,
                                                            iree_hal_semaphore_t** out_semaphore) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_shared_event_create(device->device, initial_value, device->event_listener,
                                            device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t iree_hal_metal_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  if (iree_hal_metal_shared_event_isa(semaphore)) {
    // Fast-path for semaphores related to this device.
    // TODO(benvanik): ensure the creating devices are compatible in cases where
    // multiple devices are used.
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  }
  // TODO(benvanik): semaphore APIs for querying allowed export formats. We
  // can check device caps to see what external semaphore types are supported.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_metal_copy_semaphore_list(iree_allocator_t host_allocator,
                                                        const iree_hal_semaphore_list_t source_list,
                                                        iree_hal_semaphore_list_t* target_list) {
  target_list->count = source_list.count;
  target_list->semaphores = NULL;
  target_list->payload_values = NULL;
  if (source_list.count == 0) return iree_ok_status();

  iree_hal_semaphore_t** copied_semaphores;
  iree_host_size_t size = sizeof(iree_hal_semaphore_t*) * source_list.count;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, size, (void**)&copied_semaphores));
  memcpy(copied_semaphores, source_list.semaphores, size);

  uint64_t* copied_values;
  size = sizeof(uint64_t) * source_list.count;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, size, (void**)&copied_values));
  memcpy(copied_values, source_list.payload_values, size);

  target_list->semaphores = copied_semaphores;
  target_list->payload_values = copied_values;
  return iree_ok_status();
}

static void iree_hal_metal_free_semaphore_list(iree_allocator_t host_allocator,
                                               iree_hal_semaphore_list_t list) {
  iree_allocator_free(host_allocator, list.semaphores);
  iree_allocator_free(host_allocator, list.payload_values);
}

static iree_status_t iree_hal_metal_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_allocator_pool_t pool,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  printf("[metal]  entering iree_hal_metal_device_queue_alloca\n");
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Copy the semaphore lists to heap--we will need to access them later.
  iree_hal_semaphore_list_t saved_wait_list;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_copy_semaphore_list(device->host_allocator, wait_semaphore_list,
                                             &saved_wait_list));
  iree_hal_semaphore_list_t saved_signal_list;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_copy_semaphore_list(device->host_allocator, signal_semaphore_list,
                                             &saved_signal_list));
  printf("[metal]  copied wait/signal semaphores\n");

  // IREE will free resources once their refcounts become zero on host. However, there are work
  // happening async still needing access. So make sure we retain all semaphores until later.
  for (iree_host_size_t i = 0; i < saved_wait_list.count; ++i) {
    iree_hal_semaphore_retain(saved_wait_list.semaphores[i]);  // +1
  }
  for (iree_host_size_t i = 0; i < saved_signal_list.count; ++i) {
    iree_hal_semaphore_retain(saved_signal_list.semaphores[i]);  // +1
  }
  printf("[metal]  retained wait/signal semaphores\n");

  // TODO(antiagainst): handle errors from async dispatch.
  void (^async_alloc_buffer)(void) = ^{
    printf("[metal]  entering async_alloc_buffer\n");
    printf("[metal]  allocation_size: %lu\n", allocation_size);
    iree_hal_allocator_allocate_buffer(device->device_allocator, params, allocation_size,
                                       iree_const_byte_span_empty(), out_buffer);
    printf("[metal]  allocated buffer\n");
    printf("[metal]  signal semaphore count = %lu\n", saved_signal_list.count);
    for (int i = 0; i < saved_wait_list.count; ++i)
      printf("[metal]    %p = %llu\n", saved_signal_list.semaphores[i],
             saved_signal_list.payload_values[i]);
    iree_hal_semaphore_list_signal(saved_signal_list);
    printf("[metal]  signaled semaphores\n");

    for (iree_host_size_t i = 0; i < saved_wait_list.count; ++i) {
      iree_hal_semaphore_release(saved_wait_list.semaphores[i]);  // -1
    }
    for (iree_host_size_t i = 0; i < saved_signal_list.count; ++i) {
      iree_hal_semaphore_release(saved_signal_list.semaphores[i]);  // -1
    }
    printf("[metal]  released semaphores\n");
    iree_hal_metal_free_semaphore_list(device->host_allocator, saved_wait_list);
    iree_hal_metal_free_semaphore_list(device->host_allocator, saved_signal_list);
    printf("[metal]  leaving async_alloc_buffer\n");
  };

  if (wait_semaphore_list.count == 0) {
    printf("[metal]  leaving iree_hal_metal_device_queue_alloca - no wait\n");
    dispatch_async(device->dispatch_queue, async_alloc_buffer);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (wait_semaphore_list.count == 1) {
    printf("[metal]  leaving iree_hal_metal_device_queue_alloca - wait 1 semaphore\n");
    id<MTLSharedEvent> handle = iree_hal_metal_shared_event_handle(saved_wait_list.semaphores[0]);
    printf("[metal]    %p (id=%p)\n", saved_wait_list.semaphores[0], handle);
    [handle notifyListener:device->event_listener
                   atValue:saved_wait_list.payload_values[0]
                     block:^(id<MTLSharedEvent> se, uint64_t v) {
                       async_alloc_buffer();
                     }];
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  printf("[metal]  leaving iree_hal_metal_device_queue_alloca - wait >1 semaphores\n");
  // Create an atomic to count how many semaphores have signaled. Mark it as `__block` so different
  // threads are sharing the same data via reference.
  __block iree_atomic_int32_t wait_count;
  iree_atomic_store_int32(&wait_count, 0, iree_memory_order_release);
  // The total count we are expecting to see.
  iree_host_size_t total_count = wait_semaphore_list.count;

  for (iree_host_size_t i = 0; i < saved_wait_list.count; ++i) {
    id<MTLSharedEvent> handle = iree_hal_metal_shared_event_handle(saved_wait_list.semaphores[i]);
    printf("[metal]    %p (id=%p)\n", saved_wait_list.semaphores[i], handle);
    [handle
        notifyListener:device->event_listener
               atValue:saved_wait_list.payload_values[i]
                 block:^(id<MTLSharedEvent> se, uint64_t v) {
                   // The last signaled semaphore send out the notification. Atomic fetch add
                   // returns the old value, so need to +1.
                   if (iree_atomic_fetch_add_int32(&wait_count, 1, iree_memory_order_release) + 1 ==
                       total_count) {
                     async_alloc_buffer();
                   }
                 }];
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* buffer) {
  // TODO(benvanik): queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(base_device, queue_affinity,
                                                     wait_semaphore_list, signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  @autoreleasepool {
    // First create a new command buffer and encode wait commands for all wait semaphores.
    if (wait_semaphore_list.count > 0) {
      // Copy the full semaphore list to heap--we will need to access them in command buffer
      // completion callback.
      iree_hal_semaphore_t** saved_semaphores;
      iree_host_size_t size = sizeof(iree_hal_semaphore_t*) * wait_semaphore_list.count;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_allocator_malloc(device->host_allocator, size, (void**)&saved_semaphores));
      memcpy(saved_semaphores, wait_semaphore_list.semaphores, size);

      // IREE will free resources once their refcounts become zero on host. However, there are work
      // happening on the GPU async still needing access. So make sure we retain all semaphores
      // until command buffer completion.
      for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
        iree_hal_semaphore_retain(saved_semaphores[i]);  // +1
      }

      MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];  // +1
      descriptor.retainedReferences =
          device->command_buffer_resource_reference_mode ==
          IREE_HAL_METAL_COMMAND_BUFFER_RESOURCE_REFERENCE_MODE_RETAINED;
      descriptor.errorOptions = MTLCommandBufferErrorOptionNone;
      id<MTLCommandBuffer> wait_command_buffer =
          [device->queue commandBufferWithDescriptor:descriptor];  // autoreleased
      for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
        id<MTLSharedEvent> handle =
            iree_hal_metal_shared_event_handle(wait_semaphore_list.semaphores[i]);
        [wait_command_buffer encodeWaitForEvent:handle value:wait_semaphore_list.payload_values[i]];
      }
      [wait_command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
          iree_hal_semaphore_release(saved_semaphores[i]);  // -1
        }
        iree_allocator_free(device->host_allocator, saved_semaphores);
      }];
      [wait_command_buffer commit];
      [descriptor release];  // -1
    }

    // Then commit all recorded compute command buffers.
    for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
      iree_hal_command_buffer_t* command_buffer = command_buffers[i];
      iree_hal_command_buffer_retain(command_buffer);  // +1
      id<MTLCommandBuffer> handle = iree_hal_metal_direct_command_buffer_handle(command_buffer);
      [handle addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        iree_hal_command_buffer_release(command_buffer);  // -1
      }];
      [handle commit];
    }

    // Finally create a new command buffer and encode signal commands for all signal semaphores.
    if (signal_semaphore_list.count > 0) {
      // Copy the full semaphore list to heap--we will need to access them in command buffer
      // completion callback.
      iree_hal_semaphore_t** saved_semaphores;
      iree_host_size_t size = sizeof(iree_hal_semaphore_t*) * signal_semaphore_list.count;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_allocator_malloc(device->host_allocator, size, (void**)&saved_semaphores));

      // IREE will free resources once their refcounts become zero on host. However, there are work
      // happening on the GPU async still needing access. So make sure we retain all semaphores
      // until command buffer completion.
      for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
        iree_hal_semaphore_retain(saved_semaphores[i]);  // +1
      }

      MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];  // +1
      descriptor.retainedReferences =
          device->command_buffer_resource_reference_mode ==
          IREE_HAL_METAL_COMMAND_BUFFER_RESOURCE_REFERENCE_MODE_RETAINED;
      descriptor.errorOptions = MTLCommandBufferErrorOptionNone;
      id<MTLCommandBuffer> signal_command_buffer =
          [device->queue commandBufferWithDescriptor:descriptor];  // autoreleased
      for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
        id<MTLSharedEvent> handle =
            iree_hal_metal_shared_event_handle(signal_semaphore_list.semaphores[i]);
        [signal_command_buffer encodeSignalEvent:handle
                                           value:signal_semaphore_list.payload_values[i]];
      }
      [signal_command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
          iree_hal_semaphore_release(saved_semaphores[i]);  // -1
        }
        iree_allocator_free(device->host_allocator, saved_semaphores);
      }];
      [signal_command_buffer commit];
      [descriptor release];  // -1
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_queue_flush(iree_hal_device_t* base_device,
                                                       iree_hal_queue_affinity_t queue_affinity) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented queue flush");
}

static iree_status_t iree_hal_metal_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  return iree_hal_metal_shared_event_multi_wait(wait_mode, &semaphore_list, timeout);
}

static iree_status_t iree_hal_metal_device_profiling_begin(
    iree_hal_device_t* base_device, const iree_hal_device_profiling_options_t* options) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);

  if (device->capture_manager) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "cannot nest profile capture");
  }

  if (iree_all_bits_set(options->mode, IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS)) {
    device->capture_manager = [[MTLCaptureManager sharedCaptureManager] retain];  // +1

    @autoreleasepool {
      NSURL* capture_url = NULL;
      if (strlen(options->file_path) != 0) {
        if (!iree_string_view_ends_with(IREE_SV(options->file_path), IREE_SV(".gputrace"))) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "capture filename must end with .gputrace");
        }
        if (![device->capture_manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported capture to file (if invoking as command-line "
                                  "binary, make sure there is companion Info.plist under the same "
                                  "directory with 'MetalCaptureEnabled' key being true)");
        }

        NSString* ns_string = [NSString stringWithCString:options->file_path
                                                 encoding:[NSString defaultCStringEncoding]];
        NSString* capture_path = ns_string.stringByStandardizingPath;
        capture_url = [NSURL fileURLWithPath:capture_path isDirectory:false];
      }

      MTLCaptureDescriptor* capture_descriptor = [[[MTLCaptureDescriptor alloc] init] autorelease];
      capture_descriptor.captureObject = device->device;
      if (capture_url) {
        capture_descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        capture_descriptor.outputURL = capture_url;
      } else {
        capture_descriptor.destination = MTLCaptureDestinationDeveloperTools;
      }

      NSError* error = NULL;
      if (![device->capture_manager startCaptureWithDescriptor:capture_descriptor error:&error]) {
#ifndef NDEBUG
        NSLog(@"Failed to start capture: %@", error);
#endif
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_profiling_end(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  if (device->capture_manager) {
    [device->capture_manager stopCapture];
    [device->capture_manager release];  // -1
    device->capture_manager = NULL;
  }
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_metal_device_vtable = {
    .destroy = iree_hal_metal_device_destroy,
    .id = iree_hal_metal_device_id,
    .host_allocator = iree_hal_metal_device_host_allocator,
    .device_allocator = iree_hal_metal_device_allocator,
    .replace_device_allocator = iree_hal_metal_replace_device_allocator,
    .trim = iree_hal_metal_device_trim,
    .query_i64 = iree_hal_metal_device_query_i64,
    .create_channel = iree_hal_metal_device_create_channel,
    .create_command_buffer = iree_hal_metal_device_create_command_buffer,
    .create_descriptor_set_layout = iree_hal_metal_device_create_descriptor_set_layout,
    .create_event = iree_hal_metal_device_create_event,
    .create_executable_cache = iree_hal_metal_device_create_executable_cache,
    .create_pipeline_layout = iree_hal_metal_device_create_pipeline_layout,
    .create_semaphore = iree_hal_metal_device_create_semaphore,
    .query_semaphore_compatibility = iree_hal_metal_device_query_semaphore_compatibility,
    .transfer_range = iree_hal_device_submit_transfer_range_and_wait,
    .queue_alloca = iree_hal_metal_device_queue_alloca,
    .queue_dealloca = iree_hal_metal_device_queue_dealloca,
    .queue_execute = iree_hal_metal_device_queue_execute,
    .queue_flush = iree_hal_metal_device_queue_flush,
    .wait_semaphores = iree_hal_metal_device_wait_semaphores,
    .profiling_begin = iree_hal_metal_device_profiling_begin,
    .profiling_end = iree_hal_metal_device_profiling_end,
};
