// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/support/nsight_graphics_profiler.h"

#include <NGFX_Injection.h>

iree_status_t iree_hal_support_nsight_graphics_profiler_attach() {
  // Emulating Nsight Graphics installations on the machine.
  uint32_t num_installs = 0;
  NGFX_Injection_Result result =
      NGFX_Injection_EnumerateInstallations(&num_installs, NULL);
  if (num_installs == 0 || result != NGFX_INJECTION_RESULT_OK) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to enumerate installations with error #%d",
                            result);
  }
  NGFX_Injection_InstallationInfo *installs =
      (NGFX_Injection_InstallationInfo *)iree_alloca(
          num_installs * sizeof(NGFX_Injection_InstallationInfo));
  result = NGFX_Injection_EnumerateInstallations(&num_installs, installs);
  if (result != NGFX_INJECTION_RESULT_OK) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to enumerate installations with error #%d",
                            result);
  }

  // Grab the last installations, which should be the newest.
  NGFX_Injection_InstallationInfo *chosen_install = installs + num_installs - 1;

  // Enumerate activities supported by the installation.
  uint32_t num_activities = 0;
  result =
      NGFX_Injection_EnumerateActivities(chosen_install, &num_activities, NULL);
  if (num_activities == 0 || result != NGFX_INJECTION_RESULT_OK) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to enumerate activities with error #%d",
                            result);
  }
  NGFX_Injection_Activity *activities = (NGFX_Injection_Activity *)iree_alloca(
      num_activities * sizeof(NGFX_Injection_Activity));
  result = NGFX_Injection_EnumerateActivities(chosen_install, &num_activities,
                                              activities);
  if (result != NGFX_INJECTION_RESULT_OK) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to enumerate activities with error #%d",
                            result);
  }

  NGFX_Injection_Activity *chosen_activity = NULL;
  // Get the "GPU trace" activity.
  // TODO(antiagainst): expose a CLI option to select the activity.
  for (int i = 0; i < num_activities; ++i) {
    NGFX_Injection_Activity *activity = activities + i;
    if (activity->type == NGFX_INJECTION_ACTIVITY_GPU_TRACE) {
      chosen_activity = activity;
      break;
    }
  }

  if (!chosen_activity) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to find activity with error #%d", result);
  }

  // Inject into the process and set up the activity.
  result = NGFX_Injection_InjectToProcess(chosen_install, chosen_activity);
  if (result != NGFX_INJECTION_RESULT_OK) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to inject to process with error #%d",
                            result);
  }

  return iree_ok_status();
}

void iree_hal_support_nsight_graphics_profiler_detach() {
  // Nothing to do for detaching.
}
