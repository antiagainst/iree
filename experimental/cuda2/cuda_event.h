// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_CUDA2_CUDA_EVENT_H_
#define EXPERIMENTAL_CUDA2_CUDA_EVENT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an IREE HAL semaphore with the given |initial_value| backed by
// CUevent under the hood.
iree_status_t iree_hal_cuda2_event_semaphore_create(
    uint64_t initial_value, iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_CUDA2_CUDA_EVENT_H_
