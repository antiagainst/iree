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

#ifndef IREE_HAL_VULKAN_NSIGHT_GRAPHICS_MANAGER_H_
#define IREE_HAL_VULKAN_NSIGHT_GRAPHICS_MANAGER_H_

#include "iree/base/status.h"
#include "iree/hal/debug_capture_manager.h"

namespace iree {
namespace hal {
namespace vulkan {

// Capture manager using Nsight Graphics to inspect Vulkan applications.
class NsightGraphicsManager final : public DebugCaptureManager {
 public:
  NsightGraphicsManager();
  ~NsightGraphicsManager() override;

  Status Connect() override;

  void Disconnect() override;

  bool is_connected() const override { return activity_ != nullptr; }

  void StartCapture() override;

  void StopCapture() override;

  bool is_capturing() const override { return is_capturing_; }

 private:
  const void* activity_;
  bool is_capturing_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_NSIGHT_GRAPHICS_MANAGER_H_