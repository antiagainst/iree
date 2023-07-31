// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/graph_command_buffer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "experimental/cuda2/cuda_buffer.h"
#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "experimental/cuda2/cuda_status_util.h"
#include "experimental/cuda2/native_executable.h"
#include "experimental/cuda2/pipeline_layout.h"
#include "iree/base/api.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"

// Segmented submission management
//
// In a CUDA graph, buffer management and kernel launches are represented as
// graph nodes. Dependencies are represented by graph edges. IREE's HAL follows
// the Vulkan command buffer recording model, which "linearizes" the original
// graph. So we have a mismatch here. Implementing IREE's HAL using CUDA graph
// would require rediscover the graph node dependencies from the linear chain of
// command buffer commands; it means looking at both previous and next commands
// sometimes.
//
// Due to these reasons, it's beneficial to have a complete view of the full
// command buffer and extra flexibility during recording, in order to fixup past
// commands, or inspect past/future commands.
//
// Therefore, to implement IREE HAL command buffers using CUDA graphs, we
// perform two steps using a linked list of command segments. First we create
// segments (iree_hal_cuda2_command_buffer_prepare_*) to keep track of all IREE
// HAL commands and the associated data, and then, when finalizing the command
// buffer, we iterate through all the segments and record their contents
// (iree_hal_cuda2_command_segment_record_*) into a proper CUDA graph command
// buffer. A linked list gives us the flexibility to organize command sequence
// in low overhead; and a deferred recording gives us the complete picture of
// the command buffer when really started recording.

//===----------------------------------------------------------------------===//
// Command segment
//===----------------------------------------------------------------------===//

// Command action kind of a command segment.
typedef enum iree_hal_cuda2_command_segment_action_e {
  IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_BARRIER,
  IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_DISPATCH,
  IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_FILL_BUFFER,
  IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_COPY_BUFFER,
  IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_UPDATE_BUFFER,
} iree_hal_cuda2_command_segment_action_t;

// API data for execution/memory barrier command segments.
typedef struct iree_hal_cuda2_barrier_segment_t {
  // Total number of memory barriers.
  iree_host_size_t memory_barrier_count;
  // Total number of buffer barriers.
  iree_host_size_t buffer_barrier_count;
  // The list of memory barriers, pointing to the end of the segment allocation.
  const iree_hal_memory_barrier_t* memory_barriers;
  // The list of buffer barriers, pointing to the end of the segment allocation.
  const iree_hal_buffer_barrier_t* buffer_barriers;
} iree_hal_cuda2_barrier_segment_t;
// + Additional inline allocation for holding all memory barriers.
// + Additional inline allocation for holding all buffer barriers.

// API data for dispatch command segments.
typedef struct iree_hal_cuda2_dispatch_segment_t {
  // Compute kernel information--kernel function, pipeline layout, block
  // dimensions, and so on.
  iree_hal_cuda2_kernel_info_t kernel_info;

  // Workgroup count information.
  uint32_t workgroup_count[3];

  // The list of kernel parameters, pointing to the end of the current segment
  // allocation.
  //
  // This holds a flattened list of all bound descriptor set bindings, with push
  // constants appended at the end.
  //
  // Also, per the CUDA API requirements, we need two levels of indirection
  // for passing kernel arguments in--"If the kernel has N parameters, then
  // kernelParams needs to be an array of N pointers. Each pointer, from
  // kernelParams[0] to kernelParams[N-1], points to the region of memory from
  // which the actual parameter will be copied." It means each kernel_params[i]
  // is itself a pointer to the corresponding element at the *second* inline
  // allocation at the end of the current segment.
  void** kernel_params;
} iree_hal_cuda2_dispatch_segment_t;
// + Additional inline allocation for holding kernel arguments.
// + Additional inline allocation for holding kernel argument payloads.

// API data for fill buffer command segments.
typedef struct iree_hal_cuda2_fill_buffer_segment_t {
  CUdeviceptr target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint32_t pattern;
  iree_host_size_t pattern_length;
} iree_hal_cuda2_fill_buffer_segment_t;

// API data for copy buffer command segments.
typedef struct iree_hal_cuda2_copy_buffer_segment_t {
  CUdeviceptr source_buffer;
  iree_device_size_t source_offset;
  CUdeviceptr target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
} iree_hal_cuda2_copy_buffer_segment_t;

// API data for update buffer command segments.
typedef struct iree_hal_cuda2_update_buffer_segment_t {
  const void* source_buffer;
  CUdeviceptr target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
} iree_hal_cuda2_update_buffer_segment_t;

// A command segment.
typedef struct iree_hal_cuda2_command_segment_t {
  struct iree_hal_cuda2_command_segment_t* next;
  struct iree_hal_cuda2_command_segment_t* prev;
  iree_hal_cuda2_command_segment_action_t action;
  CUgraphNode cu_graph_node;
  union {
    iree_hal_cuda2_barrier_segment_t barrier;
    iree_hal_cuda2_dispatch_segment_t dispatch;
    iree_hal_cuda2_fill_buffer_segment_t fill_buffer;
    iree_hal_cuda2_copy_buffer_segment_t copy_buffer;
    iree_hal_cuda2_update_buffer_segment_t update_buffer;
  };
} iree_hal_cuda2_command_segment_t;

//===----------------------------------------------------------------------===//
// Command segment list
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda2_command_segment_list_t {
  iree_hal_cuda2_command_segment_t* head;
  iree_hal_cuda2_command_segment_t* tail;
} iree_hal_cuda2_command_segment_list_t;

static void iree_hal_cuda2_command_segment_list_reset(
    iree_hal_cuda2_command_segment_list_t* list) {
  memset(list, 0, sizeof(*list));
}

static void iree_hal_cuda2_command_segment_list_push_back(
    iree_hal_cuda2_command_segment_list_t* list,
    iree_hal_cuda2_command_segment_t* segment) {
  if (list->tail) {
    list->tail->next = segment;
  } else {
    list->head = segment;
  }
  segment->next = NULL;
  segment->prev = list->tail;
  list->tail = segment;
}

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_command_buffer_t
//===----------------------------------------------------------------------===//

// Command buffer implementation that directly records into CUDA graphs.
// The command buffer records the commands on the calling thread without
// additional threading indirection.
typedef struct iree_hal_cuda2_graph_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  const iree_hal_cuda2_dynamic_symbols_t* symbols;

  // A resource set to maintain references to all resources used within the
  // command buffer.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // This is used for when we need CUDA to be able to reference memory as it
  // performs asynchronous operations.
  iree_arena_allocator_t arena;

  // Linked list of command segments to be recorded into a command buffer.
  iree_hal_cuda2_command_segment_list_t segments;

  CUcontext cu_context;
  // The CUDA graph under construction.
  CUgraph cu_graph;
  CUgraphExec cu_graph_exec;

  // The previous and current batches of nodes.
  //
  // For now we just synchronize all nodes before and after a given barrier,
  // regardless of the access scope. This is correct but not totally performant.
  // It works for the current compiler though.
  // TODO: Perform analysis to differentiate different access scopes and conduct
  // more fine-grained barriers, when the compiler emits them.
  //
  // These are scratch fields used by barriers for dependent analysis. The
  // previous batch includes the nodes before the last (but after the second to
  // last) barrier command. The current batch includes the nodes after the last
  // barrier command.
  iree_host_size_t previous_batch_count;
  iree_host_size_t current_batch_count;
  CUgraphNode* previous_batch;

  // Iteratively constructed batch of collective operations.
  iree_hal_collective_batch_t collective_batch;

  // The current active push constants.
  int32_t push_constants[IREE_HAL_CUDA_MAX_PUSH_CONSTANT_COUNT];

  // The current bound descriptor sets.
  struct {
    CUdeviceptr bindings[IREE_HAL_CUDA_MAX_DESCRIPTOR_SET_BINDING_COUNT];
  } descriptor_sets[IREE_HAL_CUDA_MAX_DESCRIPTOR_SET_COUNT];
} iree_hal_cuda2_graph_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda2_graph_command_buffer_vtable;

static iree_hal_cuda2_graph_command_buffer_t*
iree_hal_cuda2_graph_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_graph_command_buffer_vtable);
  return (iree_hal_cuda2_graph_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda2_graph_command_buffer_create(
    iree_hal_device_t* device,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols, CUcontext context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_graph_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*command_buffer),
                                (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device, mode, command_categories, queue_affinity, binding_capacity,
      &iree_hal_cuda2_graph_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->symbols = cuda_symbols;
  iree_arena_initialize(block_pool, &command_buffer->arena);
  iree_hal_cuda2_command_segment_list_reset(&command_buffer->segments);
  command_buffer->cu_context = context;
  command_buffer->cu_graph = NULL;
  command_buffer->cu_graph_exec = NULL;

  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);

  if (iree_status_is_ok(status)) {
    iree_hal_collective_batch_initialize(&command_buffer->arena,
                                         command_buffer->resource_set,
                                         &command_buffer->collective_batch);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda2_graph_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Drop any pending collective batches before we tear things down.
  iree_hal_collective_batch_clear(&command_buffer->collective_batch);

  if (command_buffer->cu_graph != NULL) {
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphDestroy(command_buffer->cu_graph));
    command_buffer->cu_graph = NULL;
  }
  if (command_buffer->cu_graph_exec != NULL) {
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphExecDestroy(command_buffer->cu_graph_exec));
    command_buffer->cu_graph_exec = NULL;
  }
  command_buffer->previous_batch_count = 0;
  command_buffer->current_batch_count = 0;
  command_buffer->previous_batch = NULL;

  iree_hal_collective_batch_deinitialize(&command_buffer->collective_batch);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_cuda2_graph_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_cuda2_graph_command_buffer_vtable);
}

CUgraphExec iree_hal_cuda2_graph_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  return command_buffer->cu_graph_exec;
}

static void iree_hal_cuda2_graph_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // TODO(benvanik): tracy event stack.
}

static void iree_hal_cuda2_graph_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(benvanik): tracy event stack.
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_prepare_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }

  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_cuda2_command_segment_t* segment = NULL;
  iree_host_size_t memory_barrier_length =
      memory_barrier_count * sizeof(iree_hal_memory_barrier_t);
  iree_host_size_t buffer_barrier_length =
      buffer_barrier_count * sizeof(iree_hal_buffer_barrier_t);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(
              &command_buffer->arena,
              sizeof(*segment) + memory_barrier_length + buffer_barrier_length,
              (void**)&storage_base));

  // Copy the barriers to the end of the current segments for later access.
  uint8_t* memory_barrier_ptr = storage_base + sizeof(*segment);
  memcpy(memory_barrier_ptr, (const uint8_t*)memory_barriers,
         memory_barrier_length);
  uint8_t* buffer_barrier_ptr = memory_barrier_ptr + memory_barrier_length;
  memcpy(buffer_barrier_ptr, (const uint8_t*)buffer_barriers,
         buffer_barrier_length);

  // Compose and push the barrier segment.
  segment = (iree_hal_cuda2_command_segment_t*)storage_base;
  segment->action = IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_BARRIER;
  iree_hal_cuda2_command_segment_list_push_back(&command_buffer->segments,
                                                segment);

  segment->barrier.memory_barrier_count = memory_barrier_count;
  segment->barrier.buffer_barrier_count = buffer_barrier_count;
  segment->barrier.memory_barriers =
      (const iree_hal_memory_barrier_t*)memory_barrier_ptr;
  segment->barrier.buffer_barriers =
      (const iree_hal_buffer_barrier_t*)buffer_barrier_ptr;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_command_segment_record_barrier(
    iree_hal_cuda2_graph_command_buffer_t* command_buffer,
    iree_hal_cuda2_command_segment_t* base_segment) {
  // We don't need to do anything if the barrier is the last node.
  if (base_segment->next == NULL) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  CUgraphNode* nodes = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(
              &command_buffer->arena,
              command_buffer->current_batch_count * sizeof(CUgraphNode),
              (void**)&nodes));

  command_buffer->previous_batch_count = command_buffer->current_batch_count;
  command_buffer->previous_batch = nodes;

  // Scan and collect all previous segments' graph nodes until we hit another
  // barrier segment.
  for (iree_hal_cuda2_command_segment_t* segment = base_segment->prev; segment;
       segment = segment->prev) {
    if (segment->action == IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_BARRIER) break;
    nodes[--command_buffer->current_batch_count] = segment->cu_graph_node;
  }
  IREE_ASSERT(command_buffer->current_batch_count == 0);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // We could mark the memory as invalidated so that if this is a managed buffer
  // CUDA does not try to copy it back to the host.
  return iree_ok_status();
}

// Splats a pattern value of 1/2/4 bytes out to a 4 byte value.
static uint32_t iree_hal_cuda2_splat_pattern(const void* pattern,
                                             size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *(const uint8_t*)(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *(const uint16_t*)(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *(const uint32_t*)(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_prepare_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  CUdeviceptr target_device_buffer = iree_hal_cuda2_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_cuda2_command_segment_t* segment = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment),
                              (void**)&storage_base));

  // Compose and push the fill buffer segment.
  segment = (iree_hal_cuda2_command_segment_t*)storage_base;
  segment->action = IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_FILL_BUFFER;
  iree_hal_cuda2_command_segment_list_push_back(&command_buffer->segments,
                                                segment);

  segment->fill_buffer.target_buffer = target_device_buffer;
  segment->fill_buffer.target_offset = target_offset;
  segment->fill_buffer.length = length;
  segment->fill_buffer.pattern =
      iree_hal_cuda2_splat_pattern(pattern, pattern_length);
  segment->fill_buffer.pattern_length = pattern_length;

  iree_status_t status = iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda2_command_segment_record_fill_buffer(
    iree_hal_cuda2_graph_command_buffer_t* command_buffer,
    iree_hal_cuda2_command_segment_t* base_segment) {
  iree_hal_cuda2_fill_buffer_segment_t* segment = &base_segment->fill_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  CUDA_MEMSET_NODE_PARAMS params = {
      .dst = segment->target_buffer + segment->target_offset,
      .elementSize = segment->pattern_length,
      .pitch = 0,  // Unused if height == 1
      .width = segment->length / segment->pattern_length,  // Element count
      .height = 1,
      .value = segment->pattern,
  };

  // Serialize all the nodes for now.
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemsetNode(
          &base_segment->cu_graph_node, command_buffer->cu_graph,
          command_buffer->previous_batch, command_buffer->previous_batch_count,
          &params, command_buffer->cu_context),
      "cuGraphAddMemsetNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_prepare_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because CUDA memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  uint8_t* src_storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, length,
                              (void**)&src_storage));
  memcpy(src_storage, (const uint8_t*)source_buffer + source_offset, length);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &target_buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda2_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_cuda2_command_segment_t* segment = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment),
                              (void**)&storage_base));

  // Compose and push the barrier segment.
  segment = (iree_hal_cuda2_command_segment_t*)storage_base;
  segment->action = IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_UPDATE_BUFFER;
  iree_hal_cuda2_command_segment_list_push_back(&command_buffer->segments,
                                                segment);

  segment->update_buffer.source_buffer = src_storage;
  segment->update_buffer.target_buffer = target_device_buffer;
  segment->update_buffer.target_offset = target_offset;
  segment->update_buffer.length = length;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_command_segment_record_update_buffer(
    iree_hal_cuda2_graph_command_buffer_t* command_buffer,
    iree_hal_cuda2_command_segment_t* base_segment) {
  iree_hal_cuda2_update_buffer_segment_t* segment =
      &base_segment->update_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  CUDA_MEMCPY3D params = {
      .srcMemoryType = CU_MEMORYTYPE_HOST,
      .srcHost = segment->source_buffer,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstDevice = segment->target_buffer,
      .dstXInBytes = segment->target_offset,
      .WidthInBytes = segment->length,
      .Height = 1,
      .Depth = 1,
  };

  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemcpyNode(
          &base_segment->cu_graph_node, command_buffer->cu_graph,
          command_buffer->previous_batch, command_buffer->previous_batch_count,
          &params, command_buffer->cu_context),
      "cuGraphAddMemcpyNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_prepare_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_buffer_t* buffers[2] = {source_buffer, target_buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_resource_set_insert(command_buffer->resource_set, 2, buffers));

  CUdeviceptr source_device_buffer = iree_hal_cuda2_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_buffer));
  CUdeviceptr target_device_buffer = iree_hal_cuda2_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));

  source_offset += iree_hal_buffer_byte_offset(source_buffer);
  target_offset += iree_hal_buffer_byte_offset(target_buffer);

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_cuda2_command_segment_t* segment = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment),
                              (void**)&storage_base));

  // Compose and push the barrier segment.
  segment = (iree_hal_cuda2_command_segment_t*)storage_base;
  segment->action = IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_COPY_BUFFER;
  iree_hal_cuda2_command_segment_list_push_back(&command_buffer->segments,
                                                segment);

  segment->copy_buffer.source_buffer = source_device_buffer;
  segment->copy_buffer.source_offset = source_offset;
  segment->copy_buffer.target_buffer = target_device_buffer;
  segment->copy_buffer.target_offset = target_offset;
  segment->copy_buffer.length = length;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_command_segment_record_copy_buffer(
    iree_hal_cuda2_graph_command_buffer_t* command_buffer,
    iree_hal_cuda2_command_segment_t* base_segment) {
  iree_hal_cuda2_copy_buffer_segment_t* segment = &base_segment->copy_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  CUDA_MEMCPY3D params = {
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcDevice = segment->source_buffer,
      .srcXInBytes = segment->source_offset,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstDevice = segment->target_buffer,
      .dstXInBytes = segment->target_offset,
      .WidthInBytes = segment->length,
      .Height = 1,
      .Depth = 1,
  };

  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemcpyNode(
          &base_segment->cu_graph_node, command_buffer->cu_graph,
          command_buffer->previous_batch, command_buffer->previous_batch_count,
          &params, command_buffer->cu_context),
      "cuGraphAddMemcpyNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  return iree_hal_collective_batch_append(&command_buffer->collective_batch,
                                          channel, op, param, send_binding,
                                          recv_binding, element_count);
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  iree_host_size_t constant_base_index = offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    command_buffer->push_constants[i + constant_base_index] =
        ((uint32_t*)values)[i];
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  if (binding_count > IREE_HAL_CUDA_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "exceeded available binding slots for push "
                            "descriptor set #%u; requested %lu vs. maximal %d",
                            set, binding_count,
                            IREE_HAL_CUDA_MAX_DESCRIPTOR_SET_BINDING_COUNT);
  }

  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  CUdeviceptr* current_bindings = command_buffer->descriptor_sets[set].bindings;
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    const iree_hal_descriptor_set_binding_t* binding = &bindings[i];
    CUdeviceptr device_ptr = 0;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));

      CUdeviceptr device_buffer = iree_hal_cuda2_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset = iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = device_buffer + offset + binding->offset;
    };
    current_bindings[binding->binding] = device_ptr;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_prepare_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_count_x, uint32_t workgroup_count_y,
    uint32_t workgroup_count_z) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_cuda2_kernel_info_t kernel_info;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda2_native_executable_entry_point_kernel_info(
              executable, entry_point, &kernel_info));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_cuda2_command_segment_t* segment = NULL;
  // The total number of descriptors across all descriptor sets.
  iree_host_size_t descriptor_count =
      iree_hal_cuda2_pipeline_layout_total_binding_count(kernel_info.layout);
  // The total number of push constants.
  iree_host_size_t push_constant_count =
      iree_hal_cuda2_pipeline_layout_push_constant_count(kernel_info.layout);
  // We append push constants to the end of descriptors to form a linear chain
  // of kernel arguments.
  iree_host_size_t kernel_params_count = descriptor_count + push_constant_count;
  iree_host_size_t kernel_params_length = kernel_params_count * sizeof(void*);
  // Per CUDA API requirements, we need two levels of indirection for passing
  // kernel arguments in.
  iree_host_size_t total_size = sizeof(*segment) + kernel_params_length * 2;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size,
                              (void**)&storage_base));

  // Compose and push the dispatch segment.
  segment = (iree_hal_cuda2_command_segment_t*)storage_base;
  memset(segment, 0, sizeof(*segment));
  segment->action = IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_DISPATCH;
  iree_hal_cuda2_command_segment_list_push_back(&command_buffer->segments,
                                                segment);

  segment->dispatch.kernel_info = kernel_info;
  segment->dispatch.workgroup_count[0] = workgroup_count_x;
  segment->dispatch.workgroup_count[1] = workgroup_count_y;
  segment->dispatch.workgroup_count[2] = workgroup_count_z;

  void** params_ptr = (void**)(storage_base + sizeof(*segment));
  segment->dispatch.kernel_params = params_ptr;

  // Set up kernel arguments to point to the payload slots.
  CUdeviceptr* payload_ptr =
      (CUdeviceptr*)((uint8_t*)params_ptr + kernel_params_length);
  for (size_t i = 0; i < kernel_params_count; i++) {
    params_ptr[i] = &payload_ptr[i];
  }

  // Copy descriptors from all sets to the end of the current segment for later
  // access.
  iree_host_size_t set_count =
      iree_hal_cuda2_pipeline_layout_descriptor_set_count(kernel_info.layout);
  for (iree_host_size_t i = 0; i < set_count; ++i) {
    iree_host_size_t binding_count =
        iree_hal_cuda2_descriptor_set_layout_binding_count(
            iree_hal_cuda2_pipeline_layout_descriptor_set_layout(
                kernel_info.layout, i));
    iree_host_size_t index = iree_hal_cuda2_pipeline_layout_base_binding_index(
        kernel_info.layout, i);
    memcpy(payload_ptr + index, command_buffer->descriptor_sets[i].bindings,
           binding_count * sizeof(CUdeviceptr));
  }

  // Append the push constants to the kernel arguments.
  iree_host_size_t base_index =
      iree_hal_cuda2_pipeline_layout_push_constant_index(kernel_info.layout);
  for (iree_host_size_t i = 0; i < push_constant_count; i++) {
    *((uint32_t*)params_ptr[base_index + i]) =
        command_buffer->push_constants[i];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_command_segment_record_dispatch(
    iree_hal_cuda2_graph_command_buffer_t* command_buffer,
    iree_hal_cuda2_command_segment_t* base_segment) {
  iree_hal_cuda2_dispatch_segment_t* segment = &base_segment->dispatch;
  IREE_TRACE_ZONE_BEGIN(z0);

  CUDA_KERNEL_NODE_PARAMS params = {
      .func = segment->kernel_info.function,
      .blockDimX = segment->kernel_info.block_size[0],
      .blockDimY = segment->kernel_info.block_size[1],
      .blockDimZ = segment->kernel_info.block_size[2],
      .gridDimX = segment->workgroup_count[0],
      .gridDimY = segment->workgroup_count[1],
      .gridDimZ = segment->workgroup_count[2],
      .kernelParams = segment->kernel_params,
      .sharedMemBytes = segment->kernel_info.shared_memory_size,
  };

  // Serialize all the nodes for now.
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddKernelNode(&base_segment->cu_graph_node,
                           command_buffer->cu_graph,
                           command_buffer->previous_batch,
                           command_buffer->previous_batch_count, &params),
      "cuGraphAddKernelNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect dispatch not yet implemented");
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  // TODO(#10144): support indirect command buffers by adding subgraph nodes and
  // tracking the binding table for future cuGraphExecKernelNodeSetParams usage.
  // Need to look into how to update the params of the subgraph nodes - is the
  // graph exec the outer one and if so will it allow node handles from the
  // subgraphs?
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
}

static iree_status_t iree_hal_cuda2_command_segment_record(
    iree_hal_cuda2_graph_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_hal_cuda2_command_segment_t* segment =
           command_buffer->segments.head;
       segment; segment = segment->next) {
    switch (segment->action) {
      case IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_BARRIER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_cuda2_command_segment_record_barrier(command_buffer,
                                                              segment));
        command_buffer->current_batch_count = 0;
      } break;
      case IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_DISPATCH: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_cuda2_command_segment_record_dispatch(command_buffer,
                                                               segment));
        ++command_buffer->current_batch_count;
      } break;
      case IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_FILL_BUFFER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_cuda2_command_segment_record_fill_buffer(
                    command_buffer, segment));
        ++command_buffer->current_batch_count;
      } break;
      case IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_COPY_BUFFER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_cuda2_command_segment_record_copy_buffer(
                    command_buffer, segment));
        ++command_buffer->current_batch_count;
      } break;
      case IREE_HAL_cuda2_COMMAND_SEGMENT_ACTION_UPDATE_BUFFER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_cuda2_command_segment_record_update_buffer(
                    command_buffer, segment));
        ++command_buffer->current_batch_count;
      } break;
      default:
        IREE_ASSERT(false, "unhandled command segment kind");
        break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);

  if (command_buffer->cu_graph != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot re-record command buffer");
  }

  iree_hal_cuda2_command_segment_list_reset(&command_buffer->segments);
  iree_arena_reset(&command_buffer->arena);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a new empty graph to record into.
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphCreate(&command_buffer->cu_graph, /*flags=*/0), "cuGraphCreate");

  // Reset state used during recording.
  command_buffer->previous_batch_count = 0;
  command_buffer->current_batch_count = 0;
  command_buffer->previous_batch = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda2_command_segment_record(command_buffer));

  // Compile the graph.
  CUgraphNode error_node = NULL;
  iree_status_t status = IREE_CURESULT_TO_STATUS(
      command_buffer->symbols,
      cuGraphInstantiate(&command_buffer->cu_graph_exec,
                         command_buffer->cu_graph, &error_node,
                         /*logBuffer=*/NULL, /*bufferSize=*/0));
  if (iree_status_is_ok(status)) {
    // No longer need the source graph used for construction.
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphDestroy(command_buffer->cu_graph));
    command_buffer->cu_graph = NULL;
  }

  iree_hal_resource_set_freeze(command_buffer->resource_set);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda2_graph_command_buffer_vtable = {
        .destroy = iree_hal_cuda2_graph_command_buffer_destroy,
        .begin = iree_hal_cuda2_graph_command_buffer_begin,
        .end = iree_hal_cuda2_graph_command_buffer_end,
        .begin_debug_group =
            iree_hal_cuda2_graph_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_cuda2_graph_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_cuda2_graph_command_buffer_prepare_barrier,
        .signal_event = iree_hal_cuda2_graph_command_buffer_signal_event,
        .reset_event = iree_hal_cuda2_graph_command_buffer_reset_event,
        .wait_events = iree_hal_cuda2_graph_command_buffer_wait_events,
        .discard_buffer = iree_hal_cuda2_graph_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_cuda2_graph_command_buffer_prepare_fill_buffer,
        .update_buffer =
            iree_hal_cuda2_graph_command_buffer_prepare_update_buffer,
        .copy_buffer = iree_hal_cuda2_graph_command_buffer_prepare_copy_buffer,
        .collective = iree_hal_cuda2_graph_command_buffer_collective,
        .push_constants = iree_hal_cuda2_graph_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_cuda2_graph_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_cuda2_graph_command_buffer_prepare_dispatch,
        .dispatch_indirect =
            iree_hal_cuda2_graph_command_buffer_dispatch_indirect,
        .execute_commands =
            iree_hal_cuda2_graph_command_buffer_execute_commands,
};
