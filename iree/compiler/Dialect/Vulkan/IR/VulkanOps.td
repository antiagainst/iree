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

#ifndef IREE_DIALECT_VULKAN_VULKANOPS
#define IREE_DIALECT_VULKAN_VULKANOPS

include "iree/compiler/Dialect/Vulkan/IR/VulkanBase.td"

class VK_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<VK_Dialect, mnemonic, traits>;

def VK_InvokeGLSLKernel : VK_Op<"invoke_glsl_kernel"> {
  let summary = "Invokes a GLSL kernel";
  let description = [{
    This op provides a way to utilize GLSL source code to express computation
    in the MLIR-based compilation flow. It aims to enable interoperability
    between various programming models.

    This op takes the GLSL source code as a string attribute. The GLSL code
    is expected to perform computation on a list of input and output resources.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$input_resources,
    StrAttr:$glsl_source_code
  );
  let results = (outs
    Variadic<AnyRankedTensor>:$output_resources
  );
}

#endif  // IREE_DIALECT_VULKAN_VULKANOPS
