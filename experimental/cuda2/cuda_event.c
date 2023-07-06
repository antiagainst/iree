// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/cuda_event.h"

#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "experimental/cuda2/cuda_headers.h"
#include "experimental/cuda2/cuda_status_util.h"
#include "iree/hal/utils/semaphore_base.h"

// Sentinel to indicate the semaphore has failed and an error status is set.
#define IREE_HAL_CUDA_SEMAPHORE_FAILURE_VALUE UINT64_MAX

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda2_semaphore_t {
  iree_hal_semaphore_t base;
  iree_allocator_t host_allocator;
  const iree_hal_cuda2_dynamic_symbols_t* symbols;

  // Guards all mutable fields. We expect low contention on semaphores and since
  // iree_slim_mutex_t is (effectively) just a CAS this keeps things simpler
  // than trying to make the entire structure lock-free.
  iree_slim_mutex_t mutex;

  // Current signaled value. May be IREE_HAL_CUDA_SEMAPHORE_FAILURE_VALUE to
  // indicate that the semaphore has been signaled for failure and
  // |failure_status| contains the error.
  uint64_t current_value;

  // OK or the status passed to iree_hal_semaphore_fail. Owned by the semaphore.
  iree_status_t failure_status;
} iree_hal_cuda2_semaphore_t;

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_timepoint_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda2_timepoint_t {
  iree_hal_semaphore_timepoint_t base;
  const iree_hal_cuda2_semaphore_t* semaphore;
  CUstream stream;
  CUevent event;
} iree_hal_cuda2_timepoint_t;

// Handles timepoint callbacks when either the timepoint is reached or it fails.
// We set the event in either case and let the waiters deal with the fallout.
static iree_status_t iree_hal_cuda2_semaphore_timepoint_callback(
    void* user_data, iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_status_code_t status_code) {
  iree_hal_cuda2_timepoint_t* timepoint =
      (iree_hal_cuda2_timepoint_t*)user_data;

  return IREE_CURESULT_TO_STATUS(
      timepoint->semaphore->symbols,
      cuEventRecord(timepoint->event, timepoint->stream), "cuEventRecord");
}

static const iree_hal_semaphore_vtable_t iree_hal_cuda2_semaphore_vtable;

static iree_hal_cuda2_semaphore_t* iree_hal_cuda2_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_semaphore_vtable);
  return (iree_hal_cuda2_semaphore_t*)base_value;
}

iree_status_t iree_hal_cuda2_semaphore_create(
    iree_hal_cuda2_context_wrapper_t* context, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_cuda2_semaphore_vtable,
                                  &semaphore->base);
    semaphore->context = context;
    iree_atomic_store_int64(&semaphore->value, initial_value,
                            iree_memory_order_release);
    *out_semaphore = &semaphore->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda2_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_cuda2_semaphore_t* semaphore =
      iree_hal_cuda2_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda2_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_cuda2_semaphore_t* semaphore =
      iree_hal_cuda2_semaphore_cast(base_semaphore);
  // TODO: Support semaphores completely.
  *out_value =
      iree_atomic_load_int64(&semaphore->value, iree_memory_order_acquire);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_cuda2_semaphore_t* semaphore =
      iree_hal_cuda2_semaphore_cast(base_semaphore);
  // TODO: Support semaphores completely. Return OK currently as everything is
  // synchronized for each submit to allow things to run.
  iree_atomic_store_int64(&semaphore->value, new_value,
                          iree_memory_order_release);
  iree_hal_semaphore_poll(&semaphore->base);
  return iree_ok_status();
}

static void iree_hal_cuda2_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                          iree_status_t status) {
  iree_hal_cuda2_semaphore_t* semaphore =
      iree_hal_cuda2_semaphore_cast(base_semaphore);
  // TODO: save status and mark timepoint as failed.
  iree_status_ignore(status);
  iree_hal_semaphore_poll(&semaphore->base);
}

static iree_status_t iree_hal_cuda2_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_cuda2_semaphore_t* semaphore =
      iree_hal_cuda2_semaphore_cast(base_semaphore);
  // TODO: Support semaphores completely. Return OK currently as everything is
  // synchronized for each submit to allow things to run.
  iree_hal_semaphore_poll(&semaphore->base);
  return iree_ok_status();
}

static const iree_hal_semaphore_vtable_t iree_hal_cuda2_semaphore_vtable = {
    .destroy = iree_hal_cuda2_semaphore_destroy,
    .query = iree_hal_cuda2_semaphore_query,
    .signal = iree_hal_cuda2_semaphore_signal,
    .fail = iree_hal_cuda2_semaphore_fail,
    .wait = iree_hal_cuda2_semaphore_wait,
};
