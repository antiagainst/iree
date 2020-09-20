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

load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "VulkanAttributes.cpp",
        "VulkanDialect.cpp",
        "VulkanOps.cpp",
        "VulkanTypes.cpp",
    ],
    hdrs = [
        "VulkanAttributes.h",
        "VulkanDialect.h",
        "VulkanOps.h",
        "VulkanTypes.h",
    ],
    textual_hdrs = [
        "VulkanAttributes.cpp.inc",
        "VulkanAttributes.h.inc",
        "VulkanEnums.cpp.inc",
        "VulkanEnums.h.inc",
        "VulkanOps.cpp.inc",
        "VulkanOps.h.inc",
    ],
    deps = [
        ":VulkanAttrsGen",
        ":VulkanEnumsGen",
        ":VulkanOpsGen",
        "//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

gentbl(
    name = "VulkanAttrsGen",
    tbl_outs = [
        ("-gen-struct-attr-decls", "VulkanAttributes.h.inc"),
        ("-gen-struct-attr-defs", "VulkanAttributes.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VulkanAttributes.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "VulkanEnumsGen",
    tbl_outs = [
        ("-gen-enum-decls", "VulkanEnums.h.inc"),
        ("-gen-enum-defs", "VulkanEnums.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VulkanBase.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "VulkanOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "VulkanOps.h.inc"),
        ("-gen-op-defs", "VulkanOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VulkanOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

iree_tablegen_doc(
    name = "VulkanDialectDocGen",
    tbl_outs = [
        ("-gen-dialect-doc", "VulkanDialect.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VulkanOps.td",
    td_srcs = [
        ":td_files",
    ],
)
