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

#include "absl/flags/flag.h"
#include "iree/base/file_io.h"
#include "iree/base/init.h"
#include "iree/base/main.h"
#include "iree/base/status.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/vm_util.h"
#include "iree/tools/vulkan_gui_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "third_party/dear_imgui/imgui.h"

ABSL_FLAG(std::string, module_file, "-",
          "File containing the module to load that contains the entry "
          "function. Defaults to stdin.");

ABSL_FLAG(std::string, entry_function, "",
          "Name of a function contained in the module specified by input_file "
          "to run.");

ABSL_FLAG(std::vector<std::string>, inputs, {},
          "A comma-separated list of of input buffers of the format:"
          "[shape]xtype=[value]\n"
          "2x2xi32=1 2 3 4\n"
          "Optionally, brackets may be used to separate the element values. "
          "They are ignored by the parser.\n"
          "2x2xi32=[[1 2][3 4]]\n"
          "Due to the absence of repeated flags in absl, commas should not be "
          "used to separate elements. They are reserved for separating input "
          "values:\n"
          "2x2xi32=[[1 2][3 4]], 1x2xf32=[[1 2]]");

ABSL_FLAG(std::string, inputs_file, "",
          "Provides a file for input shapes and optional values (see "
          "ParseToVariantListFromFile in vm_util.h for details)");

namespace iree {
namespace {
StatusOr<std::string> GetModuleContentsFromFlags() {
  auto input_file = absl::GetFlag(FLAGS_module_file);
  std::string contents;
  if (input_file == "-") {
    contents = std::string{std::istreambuf_iterator<char>(std::cin),
                           std::istreambuf_iterator<char>()};
  } else {
    IREE_ASSIGN_OR_RETURN(contents, file_io::GetFileContents(input_file));
  }
  return contents;
}

Status ImGuiRender(iree_hal_device_t* device, iree_vm_context_t* context,
                   iree_vm_function_t function) {
  ImGui::Begin(absl::GetFlag(FLAGS_module_file).c_str(), /*p_open=*/nullptr,
               ImGuiWindowFlags_AlwaysAutoResize);

  IREE_RETURN_IF_ERROR(ValidateFunctionAbi(function));

  IREE_ASSIGN_OR_RETURN(auto input_descs, ParseInputSignature(function));
  vm::ref<iree_vm_list_t> inputs;
  if (!absl::GetFlag(FLAGS_inputs_file).empty()) {
    if (!absl::GetFlag(FLAGS_inputs).empty()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Expected only one of inputs and inputs_file to be set";
    }
    IREE_ASSIGN_OR_RETURN(
        inputs, ParseToVariantListFromFile(input_descs,
                                           iree_hal_device_allocator(device),
                                           absl::GetFlag(FLAGS_inputs_file)));
  } else {
    IREE_ASSIGN_OR_RETURN(
        inputs,
        ParseToVariantList(input_descs, iree_hal_device_allocator(device),
                           absl::GetFlag(FLAGS_inputs)));
  }

  IREE_ASSIGN_OR_RETURN(auto output_descs, ParseOutputSignature(function));
  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr,
                                           output_descs.size(),
                                           iree_allocator_system(), &outputs));

  std::string function_name = absl::GetFlag(FLAGS_entry_function);
  LOG(INFO) << "EXEC @" << function_name;
  IREE_RETURN_IF_ERROR(iree_vm_invoke(context, function, /*policy=*/nullptr,
                                      inputs.get(), outputs.get(),
                                      iree_allocator_system()))
      << "invoking function " << function_name;

  std::ostringstream oss;
  IREE_RETURN_IF_ERROR(PrintVariantList(output_descs, outputs.get(), &oss))
      << "printing results";

  inputs.reset();
  outputs.reset();

  ImGui::Text("Entry function:");
  ImGui::Text(function_name.c_str());
  ImGui::Separator();

  ImGui::Text("Invocation result:");
  ImGui::Text(oss.str().c_str());
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

  auto module_file_or = iree::GetModuleContentsFromFlags();
  if (!module_file_or) {
    LOG(FATAL) << "Error when reading module file\n";
  }
  auto entry_function = absl::GetFlag(FLAGS_entry_function);

  iree::VulkanGuiAppInfo app_info;
  app_info.app_title = "Run IREE Bytecode Module under Vulkan GUI";
  app_info.app_width = 640;
  app_info.app_height = 360;
  app_info.bytecode_module = {
      reinterpret_cast<const uint8_t*>(module_file_or->data()),
      module_file_or->size()};
  app_info.entry_function = entry_function;
  app_info.imgui_render = iree::ImGuiRender;

  return VulkanGuiMain(argc, argv, app_info);
}
