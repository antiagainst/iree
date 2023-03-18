// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/metal/metal_kernel_library.h"

#include <stddef.h>

#include "iree/base/api.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/metal_executable_def_reader.h"
#include "iree/schemas/metal_executable_def_verifier.h"

typedef struct iree_hal_metal_kernel_library_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  iree_host_size_t entry_point_count;
  iree_hal_metal_kernel_params_t entry_points[];
} iree_hal_metal_kernel_library_t;

static const iree_hal_executable_vtable_t iree_hal_metal_kernel_library_vtable;

static iree_hal_metal_kernel_library_t* iree_hal_metal_kernel_library_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_kernel_library_vtable);
  return (iree_hal_metal_kernel_library_t*)base_value;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during runtime.

// There are still some conditions we must be aware of (such as omitted names on functions with
// internal linkage), however we shouldn't need to bounds check anything within the flatbuffer
// after this succeeds.
static iree_status_t iree_hal_metal_kernel_library_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present or less than 16 bytes (%zu total)",
                            flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds and that we can
  // safely walk the file, but not that the actual contents of the flatbuffer meet our expectations.
  int verify_ret =
      iree_MetalExecutableDef_verify_as_root(flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_MetalExecutableDef_table_t executable_def =
      iree_MetalExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_MetalExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  iree_MetalThreadgroupSize_vec_t threadgroup_sizes_vec =
      iree_MetalExecutableDef_threadgroup_sizes(executable_def);
  size_t threadgroup_size_count = iree_MetalThreadgroupSize_vec_len(threadgroup_sizes_vec);
  if (!threadgroup_size_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no threadgroup sizes present");
  }

  if (entry_point_count != threadgroup_size_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "the numbers of entry points (%zu) and thread group sizes (%zu) mismatch",
        entry_point_count, threadgroup_size_count);
  }

  flatbuffers_string_vec_t shader_libraries_vec =
      iree_MetalExecutableDef_shader_libraries_get(executable_def);
  size_t shader_library_count = flatbuffers_string_vec_len(shader_libraries_vec);
  if (shader_library_count) {
    for (size_t i = 0; i < shader_library_count; ++i) {
      if (!flatbuffers_string_len(flatbuffers_string_vec_at(shader_libraries_vec, i))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "executable shader library %zu is empty", i);
      }
    }

    if (entry_point_count != shader_library_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "the numbers of entry points (%zu) and source libraries (%zu) mismatch",
          entry_point_count, shader_library_count);
    }
  }

  flatbuffers_string_vec_t shader_sources_vec =
      iree_MetalExecutableDef_shader_sources_get(executable_def);
  size_t shader_source_count = flatbuffers_string_vec_len(shader_sources_vec);
  if (shader_source_count) {
    for (size_t i = 0; i < shader_source_count; ++i) {
      if (!flatbuffers_string_len(flatbuffers_string_vec_at(shader_sources_vec, i))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "executable shader source %zu is empty", i);
      }
    }

    if (entry_point_count != shader_source_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "the numbers of entry points (%zu) and source strings (%zu) mismatch",
                              entry_point_count, shader_source_count);
    }
  }

  if (!shader_library_count && !shader_source_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing shader library or source strings");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_metal_compile_msl(const char* source_code, const char* entry_point,
                                         id<MTLDevice> device, MTLCompileOptions* compile_options,
                                         id<MTLLibrary>* library, id<MTLFunction>* function,
                                         id<MTLComputePipelineState>* pso) {
  @autoreleasepool {
    NSError* error = nil;

    NSString* shader_source =
        [NSString stringWithCString:source_code
                           encoding:[NSString defaultCStringEncoding]];  // autoreleased
    *library = [device newLibraryWithSource:shader_source
                                    options:compile_options
                                      error:&error];  // +1
    if (*library == nil) {
#ifndef NDEBUG
      NSLog(@"Failed to create MTLLibrary: %@", error);
      NSLog(@"For entry point '%s' in MSL source:\n%@", entry_point, shader_source);
#endif
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "failed to create MTLLibrary from shader source");
    }

    NSString* function_name =
        [NSString stringWithCString:entry_point
                           encoding:[NSString defaultCStringEncoding]];  // autoreleased
    *function = [*library newFunctionWithName:function_name];            // +1
    if (*function == nil) {
#ifndef NDEBUG
      NSLog(@"Failed to create MTLFunction '%@': %@", function_name, error);
      NSLog(@"For entry point '%s' in MSL source:\n%@", entry_point, shader_source);
#endif
      [*library release];
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cannot find entry point '%s' in shader source", entry_point);
    }

    *pso = [device newComputePipelineStateWithFunction:*function error:&error];  // +1
    if (*pso == nil) {
#ifndef NDEBUG
      NSLog(@"Failed to create MTLComputePipelineState: %@", error);
      NSLog(@"For entry point '%s' in MSL source:\n%@", entry_point, shader_source);
#endif
      [*function release];
      [*library release];
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid MSL source");
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_metal_load_mtllib(const char* source_lib, size_t length,
                                         const char* entry_point, id<MTLDevice> device,
                                         id<MTLLibrary>* library, id<MTLFunction>* function,
                                         id<MTLComputePipelineState>* pso) {
  @autoreleasepool {
    NSError* error = nil;

    dispatch_data_t data =
        dispatch_data_create(source_lib, length, /*queue=*/NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    *library = [device newLibraryWithData:data error:&error];  // +1
    if (*library == nil) {
#ifndef NDEBUG
      NSLog(@"Failed to create MTLLibrary: %@", error);
      NSLog(@"For entry point '%s' in Metal library\n", entry_point);
#endif
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "failed to create MTLLibrary from shader source");
    }

    NSString* function_name =
        [NSString stringWithCString:entry_point
                           encoding:[NSString defaultCStringEncoding]];  // autoreleased
    *function = [*library newFunctionWithName:function_name];            // +1
    if (*function == nil) {
#ifndef NDEBUG
      NSLog(@"Failed to create MTLFunction '%@': %@", function_name, error);
      NSLog(@"For entry point '%s' in Metal library\n", entry_point);
#endif
      [*library release];
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cannot find entry point '%s' in shader source", entry_point);
    }

    *pso = [device newComputePipelineStateWithFunction:*function error:&error];  // +1
    if (*pso == nil) {
#ifndef NDEBUG
      NSLog(@"Failed to create MTLComputePipelineState: %@", error);
      NSLog(@"For entry point '%s' in Metal library\n", entry_point);
#endif
      [*function release];
      [*library release];
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid MSL source");
    }
  }
  return iree_ok_status();
}

// Creates an argument encoder and its backing argument buffer for the given kernel |function|'s
// |buffer_index|. The argument encoder will be set to encode into the newly created argument
// buffer. Callers are expected to release both the argument encoder and buffer.
static iree_status_t iree_hal_metal_create_argument_encoder(id<MTLDevice> device,
                                                            id<MTLFunction> function,
                                                            uint32_t buffer_index,
                                                            id<MTLArgumentEncoder>* out_encoder,
                                                            id<MTLBuffer>* out_buffer) {
  id<MTLArgumentEncoder> argument_encoder =
      [function newArgumentEncoderWithBufferIndex:buffer_index];  // +1
  if (!argument_encoder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid argument buffer index #%u",
                            buffer_index);
  }

  __block id<MTLBuffer> argument_buffer =
      [device newBufferWithLength:argument_encoder.encodedLength
                          options:MTLResourceStorageModeShared];  // +1
  if (!argument_buffer) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to create argument buffer with size = %ld bytes",
                            argument_encoder.encodedLength);
  }

  [argument_encoder setArgumentBuffer:argument_buffer offset:0];
  *out_encoder = argument_encoder;
  *out_buffer = argument_buffer;
  return iree_ok_status();
}

iree_status_t iree_hal_metal_kernel_library_create(
    iree_allocator_t host_allocator, id<MTLDevice> device,
    const iree_hal_executable_params_t* executable_params, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_kernel_library_t* executable = NULL;

  IREE_RETURN_IF_ERROR(
      iree_hal_metal_kernel_library_flatbuffer_verify(executable_params->executable_data));

  iree_MetalExecutableDef_table_t executable_def =
      iree_MetalExecutableDef_as_root(executable_params->executable_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_MetalExecutableDef_entry_points_get(executable_def);
  iree_MetalThreadgroupSize_vec_t threadgroup_sizes_vec =
      iree_MetalExecutableDef_threadgroup_sizes(executable_def);
  flatbuffers_string_vec_t shader_libraries_vec =
      iree_MetalExecutableDef_shader_libraries_get(executable_def);
  flatbuffers_string_vec_t shader_sources_vec =
      iree_MetalExecutableDef_shader_sources_get(executable_def);
  iree_host_size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);

  // Calculate the total number of characters across all entry point names. This is only required
  // when tracing so that we can store copies of the names as the flatbuffer storing the strings
  // may be released while the executable is still live.
  iree_host_size_t total_entry_point_name_chars = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < entry_point_count; i++) {
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      total_entry_point_name_chars += flatbuffers_string_len(entry_name);
    }
  });

  // Create the kernel library.
  iree_host_size_t total_size = sizeof(*executable) +
                                entry_point_count * sizeof(executable->entry_points[0]) +
                                total_entry_point_name_chars;
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  IREE_TRACE(char* string_table_buffer =
                 (char*)((char*)executable + sizeof(*executable) +
                         entry_point_count * sizeof(executable->entry_points[0])));
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_metal_kernel_library_vtable, &executable->resource);
    executable->host_allocator = host_allocator;
    executable->entry_point_count = entry_point_count;

    size_t shader_library_count = flatbuffers_string_vec_len(shader_libraries_vec);
    size_t shader_source_count = flatbuffers_string_vec_len(shader_sources_vec);

    // Try to load as Metal library first. Otherwise, compile each MSL source string into a
    // MTLLibrary and get the MTLFunction for the entry point to build the pipeline state object.
    // TODO(antiagainst): We are performing synchronous compilation at runtime here. This is good
    // for debugging purposes but bad for performance. Enable offline compilation and make that as
    // the default.

    MTLCompileOptions* compile_options = [MTLCompileOptions new];  // +1
    compile_options.languageVersion = MTLLanguageVersion3_0;

    for (size_t i = 0, e = iree_max(shader_library_count, shader_source_count); i < e; ++i) {
      id<MTLLibrary> library = nil;
      id<MTLFunction> function = nil;
      id<MTLComputePipelineState> pso = nil;
      if (shader_library_count != 0) {
        flatbuffers_string_t source_library = flatbuffers_string_vec_at(shader_libraries_vec, i);
        flatbuffers_string_t entry_point = flatbuffers_string_vec_at(entry_points_vec, i);

        status = iree_hal_metal_load_mtllib(source_library, flatbuffers_string_len(source_library),
                                            entry_point, device, &library, &function, &pso);
      } else {
        flatbuffers_string_t source_code = flatbuffers_string_vec_at(shader_sources_vec, i);
        flatbuffers_string_t entry_point = flatbuffers_string_vec_at(entry_points_vec, i);

        status = iree_hal_metal_compile_msl(source_code, entry_point, device, compile_options,
                                            &library, &function, &pso);
      }
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches for each entry point.
      iree_hal_metal_kernel_params_t* params = &executable->entry_points[i];

      params->library = library;
      params->function = function;
      params->pso = pso;
      params->threadgroup_size[0] = threadgroup_sizes_vec[i].x;
      params->threadgroup_size[1] = threadgroup_sizes_vec[i].y;
      params->threadgroup_size[2] = threadgroup_sizes_vec[i].z;
      params->layout = executable_params->pipeline_layouts[i];
      iree_hal_pipeline_layout_retain(params->layout);

      // Create argument buffers and encoders for later command buffer usage.
      for (int i = 0; i < IREE_HAL_METAL_MAX_ARGUMENT_BUFFER_COUNT; ++i) {
        if (iree_hal_metal_pipeline_layout_descriptor_set_layout(params->layout, i)) {
          status = iree_hal_metal_create_argument_encoder(device, params->function, i,
                                                          &params->argument_encoders[i],
                                                          &params->argument_buffers[i]);
          if (!iree_status_is_ok(status)) break;
        } else {
          params->argument_buffers[i] = nil;
          params->argument_encoders[i] = nil;
        }
      }
      if (!iree_status_is_ok(status)) break;

      // Stash the entry point name in the string table for use when tracing.
      IREE_TRACE({
        iree_host_size_t entry_name_length = flatbuffers_string_len(entry_point);
        memcpy(string_table_buffer, entry_point, entry_name_length);
        params->function_name = iree_make_string_view(string_table_buffer, entry_name_length);
        string_table_buffer += entry_name_length;
      });
    }

    [compile_options release];  // -1
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_kernel_library_destroy(iree_hal_executable_t* base_executable) {
  iree_hal_metal_kernel_library_t* executable = iree_hal_metal_kernel_library_cast(base_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_metal_kernel_params_t* entry_point = &executable->entry_points[i];
    for (int i = 0; i < IREE_HAL_METAL_MAX_ARGUMENT_BUFFER_COUNT; ++i) {
      if (entry_point->argument_encoders[i]) [entry_point->argument_encoders[i] release];  // -1
      if (entry_point->argument_buffers[i]) [entry_point->argument_buffers[i] release];    // -1
    }
    [entry_point->pso release];
    [entry_point->function release];
    [entry_point->library release];
    iree_hal_pipeline_layout_release(entry_point->layout);
  }
  iree_allocator_free(executable->host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_metal_kernel_library_entry_point_kernel_params(
    iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_metal_kernel_params_t* out_params) {
  iree_hal_metal_kernel_library_t* executable = iree_hal_metal_kernel_library_cast(base_executable);
  if (entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "invalid entry point ordinal %d",
                            entry_point);
  }
  memcpy(out_params, &executable->entry_points[entry_point], sizeof(*out_params));
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t iree_hal_metal_kernel_library_vtable = {
    .destroy = iree_hal_metal_kernel_library_destroy,
};
