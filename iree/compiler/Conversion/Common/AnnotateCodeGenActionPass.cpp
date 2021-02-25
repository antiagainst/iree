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

//===- AnnotateCodeGenActionPass.cpp --------------------------------------===//
//
// Given target CodeGen policies, this pass matches operations in functions
// against filters in those policies and attaches. If a match is found, the
// corresponding CodeGen actions as attributes to those operations. These
// CodeGen actions will be used to drive passes later in the pipeline.
//
//===----------------------------------------------------------------------===//

#include <tuple>
#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEAttributes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-annotate-codegen-policy"

namespace mlir {
namespace iree_compiler {

namespace {

IREE::CodeGen::TargetPolicyAttr lookupTargetPolicy(Operation *op) {
  while (op) {
    op = SymbolTable::getNearestSymbolTable(op);
    if (!op) break;

    if (auto attr = op->getAttrOfType<IREE::CodeGen::TargetPolicyAttr>(
            "iree.codegen.target.policy")) {
      return attr;
    }

    op = op->getParentOp();
  }
  return {};
}

ArrayAttr filterOpPoliciesForTarget(
    IREE::CodeGen::TargetPolicyAttr targetPolicy,
    TargetFilterCallBackFnType targetFilter) {
  auto choices =
      targetPolicy.choices().getAsRange<IREE::CodeGen::TargetChoiceAttr>();
  for (auto choice : choices) {
    // An empty filter dictionary matches any target.
    if (choice.filter().empty()) return choice.ops();

    bool isCurrentChoiceOkay = true;
    // Loop over to make sure all filter items are satisifed.
    for (auto filter : choice.filter()) {
      if (!targetFilter(filter.first, filter.second)) {
        isCurrentChoiceOkay = false;
        break;
      }
    }
    if (!isCurrentChoiceOkay) continue;

    return choice.ops();
  }
  return nullptr;
}

bool filterOnOpOutputType(IREE::CodeGen::TypeFilterAttr filter, Operation *op,
                          unsigned outputIndex) {
  using namespace IREE::CodeGen;

  ShapedType outputType =
      op->getResult(outputIndex).getType().dyn_cast<ShapedType>();
  llvm::Optional<TypeMatchCriterion> criterion =
      symbolizeTypeMatchCriterion(filter.match_criterion());

  LLVM_DEBUG(llvm::dbgs() << "matching filter '" << filter
                          << "' against output type '" << outputType << "' ");

  switch (criterion.getValue()) {
    case TypeMatchCriterion::TileableBy: {
      if (!outputType || !outputType.hasRank()) return false;

      // Make sure the element type is the same.
      auto tileShape = filter.type().cast<ShapedType>();
      if (outputType.getElementType() != tileShape.getElementType()) {
        LLVM_DEBUG(llvm::dbgs() << "failed: different element type\n");
        return false;
      }

      for (auto pair : llvm::zip(outputType.getShape(), tileShape.getShape())) {
        int64_t outputDim = std::get<0>(pair);
        int64_t tileDim = std::get<1>(pair);

        // Dynamic dimensions means we don't care.
        if (tileDim == ShapedType::kDynamicSize) continue;

        if (outputDim == ShapedType::kDynamicSize || outputDim % tileDim != 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "failed: " << outputDim << " (from output type) vs. "
                     << tileDim << " (from rule)\n");
          return false;
        }
      }
    } break;
  }

  LLVM_DEBUG(llvm::dbgs() << "succeeded\n");
  return true;
}

ArrayAttr filterCodeGenActionsForOp(ArrayAttr opPolicies, Operation *op) {
  for (Attribute opPolicyAttr : opPolicies) {
    auto opPolicy = opPolicyAttr.cast<IREE::CodeGen::OpPolicyAttr>();
    if (opPolicy.op() != op->getName().getStringRef()) continue;

    assert(opPolicy.match_criterion() == "FirstMatch");

    for (Attribute opChoiceAttr : opPolicy.choices()) {
      auto opChoice = opChoiceAttr.cast<IREE::CodeGen::OpChoiceAttr>();
      LLVM_DEBUG(llvm::dbgs()
                 << "inspecting op policy choice: " << opChoiceAttr << "\n");

      bool isCurrentChoiceOkay = true;
      for (auto indexedFilter :
           llvm::enumerate(opChoice.filter().on_output_types())) {
        auto index = indexedFilter.index();
        auto filter =
            indexedFilter.value().cast<IREE::CodeGen::TypeFilterAttr>();
        if (!filterOnOpOutputType(filter, op, index)) {
          isCurrentChoiceOkay = false;
          break;
        }
      }
      if (!isCurrentChoiceOkay) continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "matched op policy choice: " << opChoiceAttr << "\n");
      return opChoice.actions();
    }
  }
  return {};
}

class AnnotateCodeGenAction final : public RewritePattern {
 public:
  AnnotateCodeGenAction(ArrayAttr opPolicies, PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()), opPolicies(opPolicies) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    StringRef codeGenActionAttrName = getCodeGenActionAttrName();

    // If this op already has annotation, fail to match to avoid infinite loop.
    if (op->hasAttr(codeGenActionAttrName)) return failure();

    ArrayAttr actions = filterCodeGenActionsForOp(opPolicies, op);
    if (!actions) return failure();

    op->setAttr(codeGenActionAttrName, actions);
    return success();
  }

 private:
  ArrayAttr opPolicies;
};

class AnnotateCodeGenActionPass
    : public PassWrapper<AnnotateCodeGenActionPass, FunctionPass> {
 public:
  AnnotateCodeGenActionPass(IREE::CodeGen::TargetPolicyAttr targetPolicy,
                            TargetFilterCallBackFnType targetFilter)
      : defaultTargetPolicy(targetPolicy),
        targetFilter(std::move(targetFilter)) {}

  void runOnFunction() override {
    FuncOp funcOp = getFunction();

    // Lookup target policy from enclosing ops. This makes testing easy.
    // Production pipelines should set default target policy when constructing
    // this pass.
    IREE::CodeGen::TargetPolicyAttr targetPolicy = lookupTargetPolicy(funcOp);
    if (!targetPolicy) targetPolicy = defaultTargetPolicy;
    if (!targetPolicy) {
      funcOp.emitError(
          "cannot find target policy: it should be specified during pass "
          "construction or via the iree.codegen.target.policy attribute on an "
          "enclosing symbol table op");
      return signalPassFailure();
    }

    assert(targetPolicy.match_criterion() == "FirstMatch");

    ArrayAttr opPolicies =
        filterOpPoliciesForTarget(targetPolicy, targetFilter);
    if (!opPolicies) {
      funcOp.emitError("cannot find matched target policy choice inside ")
          << targetPolicy;
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "chosen op policies: " << opPolicies << "\n");

    OwningRewritePatternList patterns;
    patterns.insert<AnnotateCodeGenAction>(opPolicies);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  IREE::CodeGen::TargetPolicyAttr defaultTargetPolicy;
  TargetFilterCallBackFnType targetFilter;
};

}  // namespace

llvm::StringRef getCodeGenActionAttrName() { return "iree.codegen.actions"; }

std::unique_ptr<FunctionPass> createAnnotateCodeGenActionPass(
    IREE::CodeGen::TargetPolicyAttr targetPolicy,
    TargetFilterCallBackFnType targetFilter) {
  return std::make_unique<AnnotateCodeGenActionPass>(targetPolicy,
                                                     targetFilter);
}

static PassRegistration<AnnotateCodeGenActionPass> pass(
    "iree-codegen-annotate-codegen-action",
    "Annotate operations with CodeGen actions from matched CodeGen policy", [] {
      TargetFilterCallBackFnType callback = [](StringRef, Attribute value) {
        if (auto strAttr = value.dyn_cast<StringAttr>())
          return strAttr.getValue() == "use-this";
        return false;
      };
      return std::make_unique<AnnotateCodeGenActionPass>(
          /*targetPolicy=*/nullptr, callback);
    });

}  // namespace iree_compiler
}  // namespace mlir
