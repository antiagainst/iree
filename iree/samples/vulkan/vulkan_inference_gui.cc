// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Vulkan Graphics + IREE API Integration Sample.

// IREE's C API:
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

// Other dependencies (helpers, etc.)
#include "absl/base/macros.h"
#include "absl/types/span.h"
#include "iree/base/init.h"
#include "iree/base/logging.h"
#include "iree/base/main.h"
#include "iree/base/status.h"
#include "iree/tools/vulkan_gui_util.h"

// Dear ImGui
#include "third_party/dear_imgui/imgui.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/vulkan/simple_mul_bytecode_module.h"

namespace iree {
namespace {
bool g_ShowDemoWindow = true;

Status ImGuiRender(iree_hal_device_t* iree_vk_device,
                   iree_vm_context_t* iree_context,
                   iree_vm_function_t main_function) {
  // Demo window.
  if (g_ShowDemoWindow) ImGui::ShowDemoWindow(&g_ShowDemoWindow);

  // Custom window.
  ImGui::Begin("IREE Vulkan Integration Demo", /*p_Open=*/nullptr,
               ImGuiWindowFlags_AlwaysAutoResize);

  ImGui::Checkbox("Show ImGui Demo Window", &g_ShowDemoWindow);
  ImGui::Separator();

  // ImGui Inputs for two input tensors.
  // Run computation whenever any of the values changes.
  static bool dirty = true;
  static float input_x[] = {4.0f, 4.0f, 4.0f, 4.0f};
  static float input_y[] = {2.0f, 2.0f, 2.0f, 2.0f};
  static float latest_output[] = {0.0f, 0.0f, 0.0f, 0.0f};
  ImGui::Text("Multiply numbers using IREE");
  ImGui::PushItemWidth(60);
  // clang-format off
  if (ImGui::DragFloat("= x[0]", &input_x[0], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
  if (ImGui::DragFloat("= x[1]", &input_x[1], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
  if (ImGui::DragFloat("= x[2]", &input_x[2], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
  if (ImGui::DragFloat("= x[3]", &input_x[3], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; }                     // NOLINT
  if (ImGui::DragFloat("= y[0]", &input_y[0], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
  if (ImGui::DragFloat("= y[1]", &input_y[1], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
  if (ImGui::DragFloat("= y[2]", &input_y[2], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
  if (ImGui::DragFloat("= y[3]", &input_y[3], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; }                     // NOLINT
  // clang-format on
  ImGui::PopItemWidth();

  if (dirty) {
    // Some input values changed, run the computation.
    // This is synchronous and doesn't reuse buffers for now.

    // Write inputs into mappable buffers.
    DLOG(INFO) << "Creating I/O buffers...";
    constexpr int32_t kElementCount = 4;
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(iree_vk_device);
    iree_hal_buffer_t* input0_buffer = nullptr;
    iree_hal_buffer_t* input1_buffer = nullptr;
    iree_hal_memory_type_t input_memory_type =
        static_cast<iree_hal_memory_type_t>(
            IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
    iree_hal_buffer_usage_t input_buffer_usage =
        static_cast<iree_hal_buffer_usage_t>(IREE_HAL_BUFFER_USAGE_ALL |
                                             IREE_HAL_BUFFER_USAGE_CONSTANT);
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        allocator, input_memory_type, input_buffer_usage,
        sizeof(float) * kElementCount, &input0_buffer));
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        allocator, input_memory_type, input_buffer_usage,
        sizeof(float) * kElementCount, &input1_buffer));
    IREE_CHECK_OK(iree_hal_buffer_write_data(input0_buffer, 0, &input_x,
                                             sizeof(input_x)));
    IREE_CHECK_OK(iree_hal_buffer_write_data(input1_buffer, 0, &input_y,
                                             sizeof(input_y)));
    // Wrap input buffers in buffer views.
    iree_hal_buffer_view_t* input0_buffer_view = nullptr;
    iree_hal_buffer_view_t* input1_buffer_view = nullptr;
    IREE_CHECK_OK(iree_hal_buffer_view_create(
        input0_buffer, /*shape=*/&kElementCount, /*shape_rank=*/1,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, iree_allocator_system(),
        &input0_buffer_view));
    IREE_CHECK_OK(iree_hal_buffer_view_create(
        input1_buffer, /*shape=*/&kElementCount, /*shape_rank=*/1,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, iree_allocator_system(),
        &input1_buffer_view));
    iree_hal_buffer_release(input0_buffer);
    iree_hal_buffer_release(input1_buffer);
    // Marshal input buffer views through a VM variant list.
    vm::ref<iree_vm_list_t> inputs;
    IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, 2,
                                      iree_allocator_system(), &inputs));
    auto input0_buffer_view_ref =
        iree_hal_buffer_view_move_ref(input0_buffer_view);
    auto input1_buffer_view_ref =
        iree_hal_buffer_view_move_ref(input1_buffer_view);
    IREE_CHECK_OK(
        iree_vm_list_push_ref_move(inputs.get(), &input0_buffer_view_ref));
    IREE_CHECK_OK(
        iree_vm_list_push_ref_move(inputs.get(), &input1_buffer_view_ref));

    // Prepare outputs list to accept results from the invocation.
    vm::ref<iree_vm_list_t> outputs;
    IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr,
                                      kElementCount * sizeof(float),
                                      iree_allocator_system(), &outputs));

    // Synchronously invoke the function.
    IREE_CHECK_OK(iree_vm_invoke(iree_context, main_function,
                                 /*policy=*/nullptr, inputs.get(),
                                 outputs.get(), iree_allocator_system()));

    // Read back the results.
    DLOG(INFO) << "Reading back results...";
    auto* output_buffer_view =
        reinterpret_cast<iree_hal_buffer_view_t*>(iree_vm_list_get_ref_deref(
            outputs.get(), 0, iree_hal_buffer_view_get_descriptor()));
    auto* output_buffer = iree_hal_buffer_view_buffer(output_buffer_view);
    iree_hal_mapped_memory_t mapped_memory;
    IREE_CHECK_OK(iree_hal_buffer_map(output_buffer,
                                      IREE_HAL_MEMORY_ACCESS_READ, 0,
                                      IREE_WHOLE_BUFFER, &mapped_memory));
    memcpy(&latest_output, mapped_memory.contents.data,
           mapped_memory.contents.data_length);
    iree_hal_buffer_unmap(output_buffer, &mapped_memory);

    dirty = false;
  }

  // Display the latest computation output.
  ImGui::Text("X * Y = [%f, %f, %f, %f]",
              latest_output[0],  //
              latest_output[1],  //
              latest_output[2],  //
              latest_output[3]);
  ImGui::Separator();

  // Framerate counter.
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
              1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

  ImGui::End();
  return OkStatus();
}
}  // namespace
}  // namespace iree

int iree::IreeMain(int argc, char** argv) {
  iree::InitializeEnvironment(&argc, &argv);

  const auto* module_file_toc =
      iree::samples::vulkan::simple_mul_bytecode_module_create();
  const char entry_function[] = "simple_mul";

  iree::VulkanGuiAppInfo app_info;
  app_info.app_title = "IREE Samples - Vulkan Inference GUI";
  app_info.bytecode_module = {
      reinterpret_cast<const uint8_t*>(module_file_toc->data),
      module_file_toc->size};
  app_info.entry_function = entry_function;
  app_info.imgui_render = iree::ImGuiRender;

  return VulkanGuiMain(argc, argv, app_info);
}
