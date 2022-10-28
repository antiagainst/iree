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

  static Value createCollapseOrExpand(Value tensor, Location loc, PatternRewriter &rewriter,
                                      SmallVectorImpl<int64_t> &outputShape,
                                      SmallVectorImpl<ReassociationIndices> &reassociations,
                                      bool collapse) {
    auto tensorType = tensor.getType().cast<ShapedType>();
    auto elementTy = tensorType.getElementType();
    auto resultType = RankedTensorType::get(outputShape, elementTy);
    if (collapse)
      return rewriter.create<tensor::CollapseShapeOp>(loc, resultType, tensor, reassociations);
    return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, tensor, reassociations);
  }

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
    auto zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));

    //SmallVector<OpFoldResult> loPad{inputShape.size(), rewriter.getIndexAttr(0)};
    //SmallVector<OpFoldResult> hiPad{inputShape.size(), rewriter.getIndexAttr(0)};
    //hiPad[1] = rewriter.getIndexAttr(padH);
    //hiPad[2] = rewriter.getIndexAttr(padW);
    //auto padTensorOp = rewriter.create<tensor::PadOp>(loc, 
    //     RankedTensorType::get(SmallVector<int64_t>{in, ih + padH, iw + padW, ic}, elementType), input, loPad, hiPad);
    //auto &region = padTensorOp.getRegion();
    //int rank = padTensorOp.getResultType().getRank();
    //SmallVector<Type> blockArgTypes(rank, rewriter.getIndexType());
    //SmallVector<Location> blockArgLocs(rank, loc);
    //rewriter.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
    //rewriter.create<tensor::YieldOp>(loc, zero);
    //rewriter.setInsertionPointAfter(padTensorOp);
    //auto paddedInput = padTensorOp.getResult();

    const int ihm = std::ceil((ih + padH - kh + 1) / outputTileSize);
    const int iwm = std::ceil((iw + padW - kw + 1) / outputTileSize);
    SmallVector<int64_t> shape = {inputTileSize, inputTileSize, in, ihm, iwm, ic};
    auto transformedInputType = RankedTensorType::get(shape, elementType);
    auto tInput = rewriter.create<IREE::Flow::WinogradInputTransformOp>(loc, transformedInputType, input);

    Value kernel = convOp.getInputs()[1];
    const int oc = kernelShape[3];
    auto transformedKernelType = RankedTensorType::get({inputTileSize, inputTileSize, oc, ic}, elementType);
    auto tKernel = rewriter.create<IREE::Flow::WinogradFilterTransformOp>(loc, transformedKernelType, kernel);

    // Add collapse shape
    SmallVector<int64_t> collapsedShape{inputTileSize * inputTileSize, in * ihm * iwm, ic};
    SmallVector<ReassociationIndices> reassociations = {{0, 1}, {2, 3, 4}, {5}};
    auto cInput = createCollapseOrExpand(tInput, loc, rewriter, collapsedShape, reassociations, true);

    SmallVector<int64_t> collapsedFilterShape{inputTileSize * inputTileSize, oc, ic};
    SmallVector<ReassociationIndices> filterReassociations = {{0, 1}, {2}, {3}};
    auto cKernel = createCollapseOrExpand(tKernel, loc, rewriter, collapsedFilterShape, filterReassociations, true);

    SmallVector<int64_t> bmmShape{inputTileSize * inputTileSize, in * ihm * iwm, oc};
    auto bmmOutputType = RankedTensorType::get(bmmShape, elementType);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, bmmShape, elementType);
    Value accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();
    auto bmmResult = rewriter.create<linalg::BatchMatmulOp>(loc, bmmOutputType,
      ValueRange({cInput, cKernel}), ValueRange({accumulator})).getResult(0);

    // Add expand shape
    SmallVector<int64_t> expandedShape{inputTileSize, inputTileSize, in, ihm, iwm, oc};
    SmallVector<ReassociationIndices> resultReassociations = {{0, 1}, {2, 3, 4}, {5}};
    auto eResult = createCollapseOrExpand(bmmResult, loc, rewriter, expandedShape, resultReassociations, false);

    // TODO: Remove this when DPS passes
    // Create padded output and then extract slice
    Value output = convOp.getOutputs()[0];
    auto outputType = output.getType().cast<RankedTensorType>();
    auto outputShape = outputType.getShape();
    const int oh = outputShape[1];
    const int ow = outputShape[2];
    auto paddedOutputType = RankedTensorType::get(SmallVector<int64_t>{in, (outputTileSize * (int64_t) std::ceil((float)oh/outputTileSize)), 
                              (outputTileSize * (int64_t) std::ceil((float)ow/outputTileSize)), oc}, elementType);
    auto pOutput = rewriter.create<IREE::Flow::WinogradOutputTransformOp>(loc, paddedOutputType, eResult);

    // Extract slice
    SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;
    for (int i = 0; i < 4; i++) sizes.push_back(rewriter.getIndexAttr(outputShape[i]));
    auto tOutput = rewriter.create<tensor::ExtractSliceOp>(loc, outputType, pOutput,
      offsets, sizes, strides);

    Value result = convOp.getResult(0);
    result.replaceAllUsesWith(tOutput);

    /*
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
    */
    
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
