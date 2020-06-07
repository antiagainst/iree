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

#include "iree/hal/vulkan/nsight_graphics_manager.h"

#include "NGFX_Injection.h"
#include "iree/base/logging.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace vulkan {

NsightGraphicsManager::NsightGraphicsManager()
    : activity_(nullptr), is_capturing_(false) {}

NsightGraphicsManager::~NsightGraphicsManager() {}

Status NsightGraphicsManager::Connect() {
  IREE_TRACE_SCOPE0("NsightGraphicsManager::Connect");

  if (activity_ != nullptr) return OkStatus();

  // Emulating Nsight Graphics installations on the machine.
  uint32_t num_installs = 0;
  auto result = NGFX_Injection_EnumerateInstallations(&num_installs, nullptr);
  if (num_installs == 0 || result != NGFX_INJECTION_RESULT_OK) {
    // TODO(antiagainst): convert Nsight Graphics errors to IREE counterparts.
    return UnknownErrorBuilder(IREE_LOC)
           << "Cannot count Nsight Graphics installations: " << result;
  }
  std::vector<NGFX_Injection_InstallationInfo> installs(num_installs);
  result =
      NGFX_Injection_EnumerateInstallations(&num_installs, installs.data());
  if (result != NGFX_INJECTION_RESULT_OK) {
    return UnknownErrorBuilder(IREE_LOC)
           << "Cannot enumerate Nsight Graphics installations: " << result;
  }

  // Grab one of the installations.
  const auto& chosen_install = installs.back();

  // Enumerate activities supported by the installation.
  uint32_t num_activities = 0;
  result = NGFX_Injection_EnumerateActivities(&chosen_install, &num_activities,
                                              nullptr);
  if (num_activities == 0 || result != NGFX_INJECTION_RESULT_OK) {
    return UnknownErrorBuilder(IREE_LOC)
           << "Cannot count Nsight Graphics activities: " << result;
  }
  std::vector<NGFX_Injection_Activity> activities(num_activities);
  result = NGFX_Injection_EnumerateActivities(&chosen_install, &num_activities,
                                              activities.data());
  if (result != NGFX_INJECTION_RESULT_OK) {
    return UnknownErrorBuilder(IREE_LOC)
           << "Cannot enumerate Nsight Graphics activities: " << result;
  }

  // Get the "GPU trace" activity.
  // TODO(antiagainst): expose a CLI option to select the activity.
  for (const auto& activity : activities) {
    if (activity.type == NGFX_INJECTION_ACTIVITY_GPU_TRACE) {
      activity_ = &activity;
      break;
    }
  }

  if (!activity_) {
    return UnknownErrorBuilder(IREE_LOC)
           << "GPU trace activity unavailable on Nsight Graphics: " << result;
  }

  // Inject into the process and set up the activity.
  result = NGFX_Injection_InjectToProcess(
      &chosen_install,
      reinterpret_cast<const NGFX_Injection_Activity*>(activity_));
  if (result != NGFX_INJECTION_RESULT_OK) {
    return UnknownErrorBuilder(IREE_LOC)
           << "Nsight Graphics failed to inject to process: " << result;
  }

  return OkStatus();
}

void NsightGraphicsManager::Disconnect() {
  IREE_TRACE_SCOPE0("NsightGraphicsManager::Disconnect");

  if (activity_ == nullptr) return;

  if (is_capturing()) StopCapture();

  activity_ = nullptr;
}

void NsightGraphicsManager::StartCapture() {
  IREE_TRACE_SCOPE0("NsightGraphicsManager::StartCapture");

  CHECK(is_connected()) << "Can't start capture when not connected";
  CHECK(!is_capturing()) << "Capture is already started";

  LOG(INFO) << "Starting Nsight Graphics capture";
  NGFX_Injection_ExecuteActivityCommand();
  is_capturing_ = true;
}

void NsightGraphicsManager::StopCapture() {
  IREE_TRACE_SCOPE0("NsightGraphicsManager::StopCapture");

  CHECK(is_capturing()) << "Can't stop capture when not capturing";

  LOG(INFO) << "Ending Nsight Graphics capture";
  // Note: Nsight Grahpics SDK 0.7.0 does not support explicitly stop capturing
  // yet. Properly handle this when supported.
  is_capturing_ = false;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
