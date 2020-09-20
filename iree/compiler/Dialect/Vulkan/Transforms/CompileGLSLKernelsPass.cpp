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

#include "iree/base/status.h"
#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanOps.h"
#include "iree/compiler/Dialect/Vulkan/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "vulkan-compile-glsl"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

namespace {

iree::StatusOr<std::string> CompileGLSLKernel(llvm::StringRef glsl_source) {
  llvm::SmallString<32> glslPath, spirvPath;
  if (std::error_code error =
          llvm::sys::fs::createTemporaryFile("iree_glsl", "comp", glslPath)) {
    return iree::InternalErrorBuilder(IREE_LOC)
           << "Failed to generate temporary file for GLSL source code: "
           << error.message();
  }
  if (std::error_code error =
          llvm::sys::fs::createTemporaryFile("iree_spirv", "spv", spirvPath)) {
    return iree::InternalErrorBuilder(IREE_LOC)
           << "Failed to generate temporary file for SPIR-V code: "
           << error.message();
  }
  std::error_code error;
  auto inputFile = std::make_unique<llvm::ToolOutputFile>(
      glslPath, error, llvm::sys::fs::F_None);
  if (error) {
    return iree::InternalErrorBuilder(IREE_LOC)
           << "Failed to open temporary file '" << glslPath.c_str()
           << "' for GLSL source code: " << error.message();
  }
  inputFile->os() << glsl_source;
  inputFile->os().flush();

  std::string cmd;
  const char *vulkanSDKPath = std::getenv("VULKAN_SDK");
  cmd = std::string(vulkanSDKPath) + "/bin/glslc";
  if (vulkanSDKPath) {
    cmd = "glslc";
  }
  cmd += (" -c " + glslPath + " -o " + spirvPath).str();

  int systemRet = std::system(cmd.c_str());
  if (systemRet != 0) {
    return iree::InternalErrorBuilder(IREE_LOC)
           << "'" << cmd << "' failed with exit code " << systemRet;
  }

  auto spirvData = llvm::MemoryBuffer::getFile(spirvPath);
  if (!spirvData) {
    return iree::InternalErrorBuilder(IREE_LOC)
           << "Failed to read temporary file '" << spirvPath.c_str()
           << "' for SPIR-V code";
  }
  return spirvData.get()->getBuffer().str();
}

}  // namespace

class CompileGLSLKernelsPass
    : public PassWrapper<CompileGLSLKernelsPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VulkanDialect>();
  }
  void runOnOperation() override {
    auto moduleOp = getOperation();

    llvm::SmallVector<FuncOp, 1> functions;
    for (FuncOp fn : moduleOp.getOps<FuncOp>()) {
      if (isEntryPoint(fn)) functions.push_back(fn);
    }
    if (functions.empty()) return;
    if (functions.size() > 1) {
      functions[1].emitError("expect only one entry point");
      return signalPassFailure();
    }

    for (FuncOp fn : functions) {
      auto invokeOps = llvm::to_vector<1>(fn.getOps<InvokeGLSLKernelOp>());
      if (invokeOps.empty()) continue;
      if (invokeOps.size() > 1) {
        invokeOps[1].emitOpError(
            "cannot more than once in the same entry function");
        return signalPassFailure();
      }

      auto invokeOp = invokeOps.front();
      auto *prevOp = invokeOp.getOperation()->getPrevNode();
      for (; prevOp; prevOp = prevOp->getPrevNode()) {
        if (!isa<HAL::InterfaceLoadTensorOp>(prevOp) &&
            !isa<mlir::ConstantOp>(prevOp)) {
          break;
        }
      }
      if (prevOp) {
        prevOp->emitError(
            "expect ops before vk.invoke_glsl_kernel to be "
            "hal.interface.load.tenor ops");
        return signalPassFailure();
      }
      auto *nextOp = invokeOp.getOperation()->getNextNode();
      for (; nextOp; nextOp = nextOp->getNextNode()) {
        if (!isa<HAL::InterfaceStoreTensorOp>(nextOp)) break;
      }
      if (!nextOp->isKnownTerminator()) {
        nextOp->emitError(
            "expect ops after vk.invoke_glsl_kernel to be "
            "hal.interface.store.tenor ops");
        return signalPassFailure();
      }

      llvm::StringRef glsl_source = invokeOp.glsl_source_code();
      auto status = CompileGLSLKernel(glsl_source);
      if (!status.ok()) {
        invokeOp.emitError("failed to compile: ") << status.status().ToString();
        return signalPassFailure();
      }
      {
        auto builder = OpBuilder::atBlockTerminator(moduleOp.getBody());
        const std::string &spirvCodeStr = status.value();
        auto spirvCodeAtr = builder.getI32VectorAttr(llvm::makeArrayRef(
            reinterpret_cast<const int32_t *>(spirvCodeStr.data()),
            spirvCodeStr.size() / sizeof(int32_t)));
        builder.create<SPIRVCodeOp>(invokeOp.getLoc(), spirvCodeAtr,
                                    fn.getName());
        auto *block = invokeOp.getOperation()->getBlock();
        llvm::SmallVector<Operation *, 8> ops;
        for (auto &op : block->without_terminator()) ops.push_back(&op);
        for (auto *op : llvm::reverse(ops)) op->erase();
      }

      auto numWorkgroupFn = getNumWorkgroupsFn(fn);
      auto loc = numWorkgroupFn.getLoc();
      Block *block = numWorkgroupFn.addEntryBlock();
      auto blockBuilder = OpBuilder::atBlockBegin(block);
      auto val1 = blockBuilder.create<ConstantIndexOp>(loc, 1);
      llvm::SmallVector<Value, 3> values{val1, val1, val1};
      blockBuilder.create<mlir::ReturnOp>(loc, values);
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCompileGLSLKernelsPass() {
  return std::make_unique<CompileGLSLKernelsPass>();
}

static PassRegistration<CompileGLSLKernelsPass> pass(
    "iree-vulkan-compile-glsl-kernels",
    "Compiles vulkan.invoke_glsl_kernel ops to hal.executable.binary ops");

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
