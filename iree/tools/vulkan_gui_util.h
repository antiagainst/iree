// Copyright 2020 Google LLC
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

#ifndef IREE_TOOLS_VULKAN_GUI_H_
#define IREE_TOOLS_VULKAN_GUI_H_

#include <functional>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/vm/context.h"
#include "iree/vm/module.h"

namespace iree {

struct VulkanGuiAppInfo {
  // Application properties
  const char* app_title = nullptr;
  int app_width = 1280;
  int app_height = 720;

  // IREE VM bytecode module and entry function.
  // The HAL driver will be 'vulkan'.
  absl::Span<const uint8_t> bytecode_module = {};
  absl::string_view entry_function = {};

  // Rendering callback for rendering a custom window for IREE workload given
  // the Vulkan HAL device, VM context and function.
  using RenderFn = std::function<Status(iree_hal_device_t*, iree_vm_context_t*,
                                        iree_vm_function_t)>;
  RenderFn imgui_render = nullptr;

  operator bool() const {
    return app_title && app_height && app_width && !bytecode_module.empty() &&
           !entry_function.empty() && imgui_render != nullptr;
  }
};

int VulkanGuiMain(int argc, char** argv, const VulkanGuiAppInfo& app_info);

}  // namespace iree

#endif  // IREE_TOOLS_VULKAN_GUI_H_
