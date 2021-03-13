// Copyright 2019 Google LLC
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

//===-RemoveDeadMemAllocsPass.cpp - Pass to remove dead alloc-like ops ----===//
//
// Pass to remove operations with Allocate MemoryEffects when the allocations
// are dead.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct RemoveDeadMemAllocs : RewritePattern {
  RemoveDeadMemAllocs(PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memEffect || !memEffect.hasEffect<MemoryEffects::Allocate>()) {
      return failure();
    }
    if (!op->use_empty()) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// Elides allocations that are only used for initializing other buffers.
///
/// For example, the following pattern:
///
///   %alloc = alloc() : memref<1x225x225x3xf32>
///   linalg.fill(%alloc, %cst) : memref<1x225x225x3xf32>, f32
///   linalg.copy(%alloc, %buffer)
///
/// Can be turned into:
///
///   linalg.fill(%buffer, %cst) : memref<1x225x225x3xf32>, f32
struct ElideTransientAllocs : OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto allocOp = copyOp.getSource().getDefiningOp<AllocOp>();
    if (!allocOp) return failure();

    auto bufferUsers = allocOp->getUsers();
    auto bufferUser = bufferUsers.begin();
    if (llvm::hasSingleElement(bufferUsers) ||
        std::next(bufferUser, 2) != bufferUsers.end()) {
      return failure();
    }

    auto fillOp = dyn_cast<linalg::FillOp>(*bufferUser);
    if (!fillOp) fillOp = dyn_cast<linalg::FillOp>(*std::next(bufferUser));
    if (!fillOp) return failure();

    Block *fillBlock = fillOp->getBlock();
    Block *copyBlock = copyOp->getBlock();

    if (fillBlock != copyBlock || !fillOp->isBeforeInBlock(copyOp)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<linalg::FillOp>(copyOp, copyOp.getTarget(),
                                                fillOp.value());
    rewriter.eraseOp(fillOp);
    rewriter.eraseOp(allocOp);

    return success();
  }
};

struct RemoveDeadMemAllocsPass
    : public PassWrapper<RemoveDeadMemAllocsPass, OperationPass<>> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    patterns.insert<RemoveDeadMemAllocs>();
    patterns.insert<ElideTransientAllocs>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<>> createRemoveDeadMemAllocsPass() {
  return std::make_unique<RemoveDeadMemAllocsPass>();
}

static PassRegistration<RemoveDeadMemAllocsPass> pass(
    "iree-codegen-remove-dead-mem-allocs",
    "Remove operations with Allocate semantics that have no uses",
    [] { return std::make_unique<RemoveDeadMemAllocsPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
