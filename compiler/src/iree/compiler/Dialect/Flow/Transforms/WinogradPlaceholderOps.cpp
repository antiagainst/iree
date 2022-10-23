// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static int constexpr outputTileSize = 6;

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

namespace {

class ConvertConv2DNhwcHwcf final
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {

    // Check that kernel size = 3x3
    auto kernelType = convOp.getInputs()[1].getType().cast<ShapedType>();
    auto kernelShape = kernelType.getShape();
    auto elementType = kernelType.getElementType();
    const int kh = kernelShape[0];
    const int kw = kernelShape[1];
    if ((kh != 3) || (kw != 3)) return failure();
    const int k = kh;

    // Check that strides = 1
    if (!hasAllOneValues(convOp.getStrides())) return failure();

    // Check that dilations = 1
    if (!hasAllOneValues(convOp.getDilations())) return failure();

    // Check that this has not been transformed already
    if (convOp->hasAttr("type")) return failure();

    auto loc = convOp.getLoc();
    Value input = convOp.getInputs()[0];
    auto inputType = input.getType().cast<ShapedType>();
    auto inputShape = inputType.getShape();
    const int in = inputShape[0];
    const int ih = inputShape[1];
    const int iw = inputShape[2];
    const int ic = inputShape[3];
    const int inputTileSize = outputTileSize + k - 1;
    const int padH = outputTileSize * std::ceil((float) (ih - inputTileSize) / outputTileSize) 
                   + inputTileSize - ih;
    const int padW = outputTileSize * std::ceil((float) (iw - inputTileSize) / outputTileSize) 
                   + inputTileSize - iw;
    const int ihm = std::ceil((ih + padH - kh + 1) / outputTileSize);
    const int iwm = std::ceil((iw + padW - kw + 1) / outputTileSize);
    SmallVector<int64_t> shape = {inputTileSize, inputTileSize, in, ihm, iwm, ic};
    auto transformedInputType = RankedTensorType::get(shape, elementType);
    auto tInput = rewriter.create<IREE::Flow::WinogradInputTransformOp>(loc, transformedInputType, input);

    Value kernel = convOp.getInputs()[1];
    const int oc = kernelShape[3];
    auto transformedKernelType = RankedTensorType::get({inputTileSize, inputTileSize, oc, ic}, elementType);
    auto tKernel = rewriter.create<IREE::Flow::WinogradFilterTransformOp>(loc, transformedKernelType, kernel);

    auto bmmOutputType = RankedTensorType::get({inputTileSize, inputTileSize, in * ihm * iwm, oc}, elementType);
    auto bmmResult = rewriter.create<IREE::Flow::WinogradBatchMatmulOp>(loc, bmmOutputType, tInput, tKernel);

    Value output = convOp.getOutputs()[0];
    auto tOutput = rewriter.create<IREE::Flow::WinogradOutputTransformOp>(loc, output.getType(), bmmResult);
    Value result = convOp.getResult(0);

    rewriter.setInsertionPointAfter(convOp);
    // Create generic add that will still allow consumer fusion
    auto outputShape = output.getType().cast<ShapedType>().getShape();
    int64_t iterationSpaceDim = 4;
    SmallVector<AffineExpr> idExprs;
    for (auto i = 0; i < iterationSpaceDim; i++) {
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    }
    SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(iterationSpaceDim, 0, idExprs, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, idExprs, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, idExprs, rewriter.getContext()),
    };
    SmallVector<StringRef> iteratorTypes(iterationSpaceDim, getParallelIteratorTypeName());
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputShape, elementType);
    auto endOp = rewriter.create<linalg::GenericOp>(loc, output.getType(), 
      ValueRange({result, tOutput}), emptyTensor,
      indexingMaps, iteratorTypes, 
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::AddFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });

    convOp->setAttr("type", rewriter.getStringAttr("winograd"));
    result.replaceAllUsesExcept(endOp.getResult(0), {endOp});
    
    return success();
  }
};

struct WinogradPlaceholderOpsPass
    : WinogradPlaceholderOpsBase<WinogradPlaceholderOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertConv2DNhwcHwcf>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createWinogradPlaceholderOpsPass() {
  return std::make_unique<WinogradPlaceholderOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
