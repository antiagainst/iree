# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Find Nsight Graphics SDK, a programmatical way to interact with Nsight
# Grahpics, which is a developer tool for debugging and profiling Vulkan
# applications with NVIDIA GPUs.
#
# Imported Targets
#
# This module defines `IMPORTED` `NVIDIA::NsightGraphics`, if Nsight Graphics
# has been found.
#
# Result Variables
#
# This module defins the following varaibles:
#
# * NsightGraphics_FOUND        - "True" if Nsight Graphics was found
# * NsightGraphics_INCLUDE_DIRS - Include directories for Nsight Graphics SDK
# * NsightGraphics_LIBRARIES    - Libraries to link to for Nsight Graphics SDK
#
# This module will also define two cache varabiles:
#
# * NsightGraphics_INCLUDE_DIR - Nsight Graphics SDK include directory
# * NsightGraphics_LIBRARY     - Nsight Graphics SDK library
#
# Hints
#
# In addition to standard locations, the `NSIGHT_GRAPHICS_SDK` environment
# variable can be used to specify the location of the Nsight Graphics SDK root
# directory and will be considered when trying to find the Nsight Grahpics SDK.

if(WIN32)
  find_path(NsightGraphics_INCLUDE_DIR
    NAMES NGFX_Injection.h
    HINTS
        "$ENV{NSIGHT_GRAPHICS_SDK}/include"
        "C:/Program Files/NVIDIA Corporation/Nsight Graphics 2020.3.0/SDKs/NsightGraphicsSDK/0.7.0/include"
  )
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(NsightGraphics_LIBRARY
      NAMES NGFX_Injection
      HINTS
        "$ENV{NSIGHT_GRAPHICS_SDK}/lib/x64"
        "C:/Program Files/NVIDIA Corporation/Nsight Graphics 2020.3.0/SDKs/NsightGraphicsSDK/0.7.0/lib/x64"
    )
  endif()
endif(WIN32)

set(NsightGraphics_INCLUDE_DIRS ${NsightGraphics_INCLUDE_DIR})
set(NsightGraphics_LIBRARIES ${NsightGraphics_LIBRARY})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(NsightGraphics DEFAULT_MSG
  NsightGraphics_INCLUDE_DIR NsightGraphics_LIBRARY
)

mark_as_advanced(NsightGraphics_INCLUDE_DIR NsightGraphics_LIBRARY)

if(NsightGraphics_FOUND AND NOT TARGET NVIDIA::NsightGraphics)
  add_library(NVIDIA::NsightGraphics UNKNOWN IMPORTED GLOBAL)
  set_target_properties(NVIDIA::NsightGraphics PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NsightGraphics_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${NsightGraphics_LIBRARIES}")
endif()