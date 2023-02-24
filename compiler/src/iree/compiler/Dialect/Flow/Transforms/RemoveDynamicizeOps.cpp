// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-remove-dynamicize-ops"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

struct RemoveDynamicizeOpsPass
    : public RemoveDynamicizeOpsBase<RemoveDynamicizeOpsPass> {
  void runOnOperation() override;
};

struct RemoveDyamicizeDimOp final
    : public OpRewritePattern<DispatchDynamicizeDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DispatchDynamicizeDimOp dimOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(dimOp, dimOp.getType(),
                                                   dimOp.getValueAttr());
    return success();
  }
};

struct RemoveDynamicizeShapeOp final
    : public OpRewritePattern<DispatchDynamicizeShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DispatchDynamicizeShapeOp shapeOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(shapeOp, shapeOp.getSource());
    return success();
  }
};

}  // namespace

void RemoveDynamicizeOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<RemoveDyamicizeDimOp, RemoveDynamicizeShapeOp>(context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createRemoveDynamicizeOpsPass() {
  return std::make_unique<RemoveDynamicizeOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
