// Copyright 2021 Google LLC
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

#ifndef IREE_DIALECT_IREE_IREECODEGENATTRIBUTES
#define IREE_DIALECT_IREE_IREECODEGENATTRIBUTES

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// Utility definitions
//===----------------------------------------------------------------------===//

class IREE_CodeGen_IsKnownStrEnumCaseFor<string name> :
    CPred<"::mlir::iree_compiler::IREE::CodeGen::symbolize" # name # "("
          "$_self.cast<StringAttr>().getValue()).hasValue()">;

// Wrapper over base StrEnumAttr to set common fields.
class IREE_CodeGen_StrEnumAttr<string name, string description,
                      list<StrEnumAttrCase> cases> :
    StrEnumAttr<name, description, cases> {
  let predicate = And<[
    StrAttr.predicate, IREE_CodeGen_IsKnownStrEnumCaseFor<name>]>;
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

//===----------------------------------------------------------------------===//
// Match criterion enums
//===----------------------------------------------------------------------===//

def IREE_TMC_TileableBy  : StrEnumAttrCase<"TileableBy">;

def IREE_CodeGen_TypeMatchCriterionAttr :
  IREE_CodeGen_StrEnumAttr<
    "TypeMatchCriterion", "IREE CodeGen type match criterion",
    [IREE_TMC_TileableBy]
  >;

def IREE_PMC_FirstMatch : StrEnumAttrCase<"FirstMatch">;

def IREE_CodeGen_PolicyMatchCriterionAttr :
  IREE_CodeGen_StrEnumAttr<
    "PolicyMatchCriterion", "IREE CodeGen policy match criterion",
    [IREE_PMC_FirstMatch]
  >;

//===----------------------------------------------------------------------===//
// Compute hierarchy enums
//===----------------------------------------------------------------------===//

def IREE_CHL_Workgroup  : StrEnumAttrCase<"Workgroup">;
def IREE_CHL_Subgroup   : StrEnumAttrCase<"Subgroup">;
def IREE_CHL_Invocation : StrEnumAttrCase<"Invocation">;

def IREE_CodeGen_ComputeHierarchyLevel :
  IREE_CodeGen_StrEnumAttr<
    "ComputeHierarchyLevel", "IREE CodeGen compute hierarchy level",
    [IREE_CHL_Workgroup, IREE_CHL_Subgroup, IREE_CHL_Invocation]
  >;

//===----------------------------------------------------------------------===//
// CodeGen actions
//===----------------------------------------------------------------------===//

def IREE_CodeGen_Action_DistributeAttr :
  IREE_StructAttr<"codegen.action.distribute",
                  "ActionDistributeAttr",
                  IREE_Dialect, [
    IREE_StructFieldAttr<"along_dimensions", I64ArrayAttr>,
    IREE_StructFieldAttr<"with_workgroup_size", I64ArrayAttr>
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_Action_TileAttr :
  IREE_StructAttr<"codegen.action.tile", "ActionTileAttr", IREE_Dialect, [
    IREE_StructFieldAttr<"along_dimensions", I64ArrayAttr>,
    IREE_StructFieldAttr<"with_tile_sizes", I64ArrayAttr>
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_Action_TileAndDistributeAttr : IREE_StructAttr<
    "codegen.action.tile_and_distribute",
    "ActionTileAndDistributeAttr",
    IREE_Dialect,
  [
    IREE_StructFieldAttr<"along_dimensions", I64ArrayAttr>,
    IREE_StructFieldAttr<"with_tile_sizes", I64ArrayAttr>,
    IREE_StructFieldAttr<"to_hierarchy_level",
                         IREE_CodeGen_ComputeHierarchyLevel>
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_Action_VectorizeAttr :
  IREE_StructAttr<"codegen.action.vectorize",
                  "ActionVectorizeAttr",
                  IREE_Dialect, [
    IREE_StructFieldAttr<"with_vector_size", I64Attr>
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_ActionAttr :
  Attr<
    Or<[
      IREE_CodeGen_Action_DistributeAttr.predicate,
      IREE_CodeGen_Action_TileAttr.predicate,
      IREE_CodeGen_Action_VectorizeAttr.predicate
    ]>,
    "IREE CodeGen action attribute"> {
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_ActionArrayAttr :
  TypedArrayAttrBase<
    IREE_CodeGen_ActionAttr, "IREE CodeGen action array attribute">;

//===----------------------------------------------------------------------===//
// CodeGen op filters
//===----------------------------------------------------------------------===//

def IREE_CodeGen_TypeFilterAttr :
  IREE_StructAttr<"codegen.type.filter", "TypeFilterAttr", IREE_Dialect, [
    IREE_StructFieldAttr<"match_criterion",
                         IREE_CodeGen_TypeMatchCriterionAttr>,
    IREE_StructFieldAttr<"type", TypeAttr>
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_TypeFilterArrayAttr :
  TypedArrayAttrBase<
    IREE_CodeGen_TypeFilterAttr, "IREE CodeGen type filter array attribute">;

def IREE_CodeGen_OpFilterAttr :
  IREE_StructAttr<"codegen.op.filter", "OpFilterAttr", IREE_Dialect, [
    IREE_StructFieldAttr<"on_output_types", IREE_CodeGen_TypeFilterArrayAttr>
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

//===----------------------------------------------------------------------===//
// CodeGen op policy
//===----------------------------------------------------------------------===//

def IREE_CodeGen_OpChoiceAttr :
  IREE_StructAttr<"codegen.op.choice", "OpChoiceAttr", IREE_Dialect, [
    IREE_StructFieldAttr<"filter", IREE_CodeGen_OpFilterAttr>,
    IREE_StructFieldAttr<"actions", IREE_CodeGen_ActionArrayAttr>
  ]> {
  let constBuilderCall = "$0";
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_OpChoiceArrayAttr :
  TypedArrayAttrBase<
    IREE_CodeGen_OpChoiceAttr, "IREE CodeGen op choice array attribute">;

def IREE_CodeGen_OpPolicyAttr :
  IREE_StructAttr<"codegen.op.policy", "OpPolicyAttr", IREE_Dialect, [
    IREE_StructFieldAttr<"op", StrAttr>,
    IREE_StructFieldAttr<"match_criterion",
                         IREE_CodeGen_PolicyMatchCriterionAttr>,
    IREE_StructFieldAttr<"choices", IREE_CodeGen_OpChoiceArrayAttr>
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_OpPolicyArrayAttr :
  TypedArrayAttrBase<
    IREE_CodeGen_OpPolicyAttr, "IREE CodeGen op policy array attribute">;

//===----------------------------------------------------------------------===//
// CodeGen target policy
//===----------------------------------------------------------------------===//

def IREE_CodeGen_TargetChoiceAttr :
  IREE_StructAttr<"codegen.target.choice", "TargetChoiceAttr", IREE_Dialect, [
    IREE_StructFieldAttr<"filter", DictionaryAttr>,
    IREE_StructFieldAttr<"ops", IREE_CodeGen_OpPolicyArrayAttr>
  ]> {
  let constBuilderCall = "$0";
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

def IREE_CodeGen_TargetChoiceArrayAttr :
  TypedArrayAttrBase<
    IREE_CodeGen_TargetChoiceAttr, "IREE CodeGen target choice array attribute">;

def IREE_CodeGen_TargetPolicyAttr :
  IREE_StructAttr<"codegen.target.policy", "TargetPolicyAttr", IREE_Dialect, [
    IREE_StructFieldAttr<"target", StrAttr>,
    IREE_StructFieldAttr<"match_criterion",
                         IREE_CodeGen_PolicyMatchCriterionAttr>,
    IREE_StructFieldAttr<"choices", IREE_CodeGen_TargetChoiceArrayAttr>
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CodeGen";
}

#endif  // IREE_DIALECT_IREE_IREECODEGENATTRIBUTES
