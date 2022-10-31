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

  static Value createTransposeOp(Value filter,
                                 Type elementType,
                                 Location loc,
                                 PatternRewriter &rewriter) {
    auto filterShape = filter.getType().cast<ShapedType>().getShape();
    int64_t iterationSpaceDim = 4;
    SmallVector<AffineExpr> idExprs;
    for (auto i = 0; i < iterationSpaceDim; i++) {
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    }
    SmallVector<AffineExpr> inputExprs = {idExprs[2], idExprs[3], idExprs[0], idExprs[1]};
    SmallVector<int64_t> shape = {filterShape[2], filterShape[3], filterShape[0], filterShape[1]};
    SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(iterationSpaceDim, 0, inputExprs, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, idExprs, rewriter.getContext()),
    };
    SmallVector<StringRef> iteratorTypes(iterationSpaceDim, getParallelIteratorTypeName());
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    auto outputType = RankedTensorType::get(shape, elementType);
    auto transposeOp = rewriter.create<linalg::GenericOp>(loc, outputType,
      ValueRange({filter}), emptyTensor,
      indexingMaps, iteratorTypes, 
      [&](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]);
      });
    return transposeOp.getResult(0);
  }

  static Value createBroadcastOp(Value GValue,
                                 int ic, int oc, int Grows, int Gcols,
                                 Type elementType,
                                 Location loc,
                                 PatternRewriter &rewriter, 
                                 bool transpose) {
    int64_t iterationSpaceDim = 4;
    SmallVector<AffineExpr> idExprs;
    for (auto i = 0; i < iterationSpaceDim; i++) {
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    }
    SmallVector<AffineExpr> inputExprs;
    SmallVector<int64_t> shape;
    if (transpose) {
      shape = {ic, oc, Gcols, Grows};
      inputExprs = {idExprs[3], idExprs[2]};
    } else {
      shape = {ic, oc, Grows, Gcols};
      inputExprs = {idExprs[2], idExprs[3]};
    }
    SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(iterationSpaceDim, 0, inputExprs, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, idExprs, rewriter.getContext()),
    };
    SmallVector<StringRef> iteratorTypes(iterationSpaceDim, getParallelIteratorTypeName());
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    auto outputType = RankedTensorType::get(shape, elementType);
    auto broadcastOp = rewriter.create<linalg::GenericOp>(loc, outputType,
      ValueRange({GValue}), emptyTensor,
      indexingMaps, iteratorTypes, 
      [&](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]);
      });
    return broadcastOp.getResult(0);
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
  #if 1
    // Original filter transform here
    auto transformedKernelType = RankedTensorType::get({inputTileSize, inputTileSize, oc, ic}, elementType);
    auto tKernel = rewriter.create<IREE::Flow::WinogradFilterTransformOp>(loc, transformedKernelType, kernel);

#else
    /**** Alternative to filter transform ***/
    // Create generic to broadcast the G and GT values
    SmallVector<float> G{
      1, 0, 0,
      -2./9., -2./9., -2./9.,
      -2./9., 2./9., -2./9.,
      1./90, 1./45, 2./45,
      1./90, -1./45, 2./45,
      32./45, 16./45, 8./45,
      32./45, -16./45, 8./45,
      0, 0, 1
    };
    int Grows = 8;
    int Gcols = 3;
    auto GValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({Grows, Gcols}, rewriter.getF32Type()), G));
    SmallVector<int64_t> GbroadcastShape{ic, oc, Grows, Gcols};
    auto Gbroadcast = createBroadcastOp(GValue, ic, oc, Grows, Gcols, elementType, loc, rewriter, false); 
    auto GTbroadcast = createBroadcastOp(GValue, ic, oc, Grows, Gcols, elementType, loc, rewriter, true); 
    auto transposedKernel = createTransposeOp(kernel, elementType, loc, rewriter); 

    SmallVector<int64_t> collapsedShape{ic * oc, Grows, Gcols};
    SmallVector<ReassociationIndices> reassociations = {{0, 1}, {2}, {3}};
    auto gInput = createCollapseOrExpand(Gbroadcast, loc, rewriter, collapsedShape, reassociations, true);

    collapsedShape = {ic * oc, Gcols, Grows};
    reassociations = {{0, 1}, {2}, {3}};
    auto gtInput = createCollapseOrExpand(GTbroadcast, loc, rewriter, collapsedShape, reassociations, true);

    collapsedShape = {ic * oc, Gcols, Gcols};
    reassociations = {{0, 1}, {2}, {3}};
    auto kInput = createCollapseOrExpand(transposedKernel, loc, rewriter, collapsedShape, reassociations, true);

    SmallVector<int64_t> bmmShape{ic * oc, Gcols, Grows};
    auto bmmOutputType = RankedTensorType::get(bmmShape, elementType);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, bmmShape, elementType);
    Value accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();
    auto bmmResult = rewriter.create<linalg::BatchMatmulOp>(loc, bmmOutputType,
      ValueRange({kInput, gtInput}), ValueRange({accumulator})).getResult(0);

    bmmShape = {ic * oc, Grows, Grows};
    bmmOutputType = RankedTensorType::get(bmmShape, elementType);
    emptyTensor = rewriter.create<tensor::EmptyOp>(loc, bmmShape, elementType);
    accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();
    bmmResult = rewriter.create<linalg::BatchMatmulOp>(loc, bmmOutputType,
      ValueRange({gInput, bmmResult}), ValueRange({accumulator})).getResult(0);

    // Create expand shapes
    SmallVector<int64_t> expandedShape = {ic, oc, Grows, Grows};
    SmallVector<ReassociationIndices> resultReassociations = {{0, 1}, {2}, {3}};
    auto expanded = createCollapseOrExpand(bmmResult, loc, rewriter, expandedShape, resultReassociations, false);

    // Transpose expanded shape
    auto tKernel = createTransposeOp(expanded, elementType, loc, rewriter); 
#endif

    // Add collapse shape
    SmallVector<int64_t> collapsedShape = {inputTileSize * inputTileSize, in * ihm * iwm, ic};
    SmallVector<ReassociationIndices> reassociations = {{0, 1}, {2, 3, 4}, {5}};
    auto cInput = createCollapseOrExpand(tInput, loc, rewriter, collapsedShape, reassociations, true);

    SmallVector<int64_t> collapsedFilterShape = {inputTileSize * inputTileSize, oc, ic};
    SmallVector<ReassociationIndices> filterReassociations = {{0, 1}, {2}, {3}};
    auto cKernel = createCollapseOrExpand(tKernel, loc, rewriter, collapsedFilterShape, filterReassociations, true);

    SmallVector<int64_t> bmmShape = {inputTileSize * inputTileSize, in * ihm * iwm, oc};
    auto bmmOutputType = RankedTensorType::get(bmmShape, elementType);
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(loc, bmmShape, elementType);
    auto accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();
    auto bmmResult = rewriter.create<linalg::BatchMatmulOp>(loc, bmmOutputType,
      ValueRange({cInput, cKernel}), ValueRange({accumulator})).getResult(0);

    // Add expand shape
    SmallVector<int64_t> expandedShape = {inputTileSize, inputTileSize, in, ihm, iwm, oc};
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
