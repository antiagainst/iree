// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== TileAndDistributeToWorkgroupsPass.cpp - Tile to workgroups pass ----===//
//
// This pass distributes the operations within the module to workgroups. This
// pass is created to move tile and distribution out of flow level and into
// the backends. For now this is mostly a bridge pass to connect things during
// the transition, and eventually might just be deprecated in favor of a
// utility method.
//
//===---------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-conv2d-to-winograd"

namespace mlir {
namespace iree_compiler {

static int constexpr outputTileSize = 6;

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static void transpose(SmallVectorImpl<float> &inputTensor, SmallVectorImpl<float> &outputTensor, int dim0, int dim1) {
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim0; j++) {
      outputTensor.push_back(inputTensor[j * dim1 + i]);
    }
  }
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
    auto elementType = kernelType.getElementType();
    auto kernelShape = kernelType.getShape();
    const int kh = kernelShape[0];
    const int kw = kernelShape[1];
    if ((kh != 3) || (kw != 3)) return failure();
    const int k = kh;

    // Check that strides = 1
    if (!hasAllOneValues(convOp.getStrides())) return failure();

    // Check that dilations = 1
    if (!hasAllOneValues(convOp.getDilations())) return failure();

    // Create transformation constants
    // These are tile size specific
    SmallVector<float> BT{
      1,     0, -21./4.,        0,  21./4.,       0, -1, 0,
      0,     1,       1,  -17./4., -17./4.,       1,  1, 0,
      0,    -1,       1,   17./4., -17./4.,      -1,  1, 0,
      0,  1./2,   1./4.,   -5./2.,  -5./4.,       2,  1, 0,
      0,  -1./2,  1./4.,    5./2.,  -5./4.,      -2,  1, 0,
      0,      2,      4,   -5./2.,      -5,   1./2.,  1, 0,
      0,     -2,      4,    5./2.,      -5,  -1./2.,  1, 0,
      0,     -1,      0,   21./4.,       0, -21./4.,  0, 1
    };
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
    SmallVector<float> AT{
      1,1, 1, 1,  1,     1,      1,  0,
      0,1,-1, 2, -2,  1./2,  -1./2,  0,
      0,1, 1, 4,  4,  1./4,   1./4,  0,
      0,1,-1, 8, -8,  1./8,  -1./8,  0,
      0,1, 1,16, 16, 1./16,  1./16,  0,
      0,1,-1,32,-32, 1./32, -1./32,  1
    };

    int inputTileSize = outputTileSize + k - 1;
    auto loc = convOp.getLoc();
    SmallVector<float> B, GT, A;
    transpose(BT, B, inputTileSize, inputTileSize);
    transpose(G, GT, inputTileSize, kh);
    transpose(AT, A, outputTileSize, inputTileSize);

    auto funcOp = convOp->getParentOfType<func::FuncOp>();
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    auto BTValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({inputTileSize, inputTileSize}, rewriter.getF32Type()), BT));
    auto BValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({inputTileSize, inputTileSize}, rewriter.getF32Type()), B));
    /*auto GValue = */rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({inputTileSize, k}, rewriter.getF32Type()), G));
    /*auto GTValue = */rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({k, inputTileSize}, rewriter.getF32Type()), GT));
    /*auto ATValue = */rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({outputTileSize, inputTileSize}, rewriter.getF32Type()), AT));
    /*auto AValue = */rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({inputTileSize, outputTileSize}, rewriter.getF32Type()), A));

    // Create for loops for iterating over input tiles
    // Get original input size from flow.tensor.dispatch
    Value input = convOp.getInputs()[0];
    auto tensorLoadOp = input.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!tensorLoadOp) return failure();
    Value originalInput = tensorLoadOp.getSource();
    auto inputType = originalInput.getType().cast<IREE::Flow::DispatchTensorType>();
    if (!inputType.hasStaticShape()) return failure();
    auto inputShape = inputType.getShape();

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto zerof32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto inputBatch = rewriter.create<arith::ConstantIndexOp>(loc, inputShape[0]);
    auto inputHeight = rewriter.create<arith::ConstantIndexOp>(loc, inputShape[1]);
    auto inputWidth = rewriter.create<arith::ConstantIndexOp>(loc, inputShape[2]);
    auto inputChannels = rewriter.create<arith::ConstantIndexOp>(loc, inputShape[3]);
    auto inputTileSz = rewriter.create<arith::ConstantIndexOp>(loc, inputTileSize);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, outputTileSize);

    auto zeroAttr = rewriter.getIndexAttr(0);
    auto oneAttr = rewriter.getIndexAttr(1);

    rewriter.setInsertionPointAfter(convOp);
    SmallVector<OpFoldResult> offsets(4, zeroAttr);
    SmallVector<OpFoldResult> sizes(4, zeroAttr);

    // Add tiling for N
    {
      auto forOp = rewriter.create<scf::ForOp>(loc, zero, inputBatch, one);
      offsets[0] = forOp.getInductionVar();
      sizes[0] = oneAttr;
      rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
    }

    // Add tiling for N
    {
      auto forOp = rewriter.create<scf::ForOp>(loc, zero, inputChannels, one);
      offsets[3] = forOp.getInductionVar();
      sizes[3] = oneAttr;
      rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
    }

    // Add tiling for H and W
    for (int i = 0; i < 2; i++) {
      Value hi = i == 0 ? inputHeight : inputWidth;
      auto forOp = rewriter.create<scf::ForOp>(loc, zero, hi, step);
      rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
      auto iv = forOp.getInductionVar();
      offsets[i + 1] = iv;
      AffineExpr dim0;
      auto t = rewriter.getAffineConstantExpr(inputTileSize);
      auto delta = i == 0 ? rewriter.getAffineConstantExpr(inputShape[1])
                          : rewriter.getAffineConstantExpr(inputShape[2]);
      bindDims(rewriter.getContext(), dim0);
      AffineMap minMap = AffineMap::get(1, 0, {-dim0 + delta, t}, rewriter.getContext());
      auto size = rewriter.createOrFold<AffineMinOp>(loc, minMap, ValueRange{iv});
      sizes[i + 1] = size;
    }

    // Check whether to go with fast/slow path
    SmallVector<Value> eqZeroCmpVals;
    for (int i = 1; i < 3; i++) {
      if (auto sizeVal = sizes[i].dyn_cast<Value>()) {
        eqZeroCmpVals.push_back(rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, sizeVal, inputTileSz));
      } 
    }
    assert(equivalent.size() == 2);
    Value ifCond = rewriter.create<arith::AndIOp>(loc, eqZeroCmpVals[0], eqZeroCmpVals[1]);

    // Load input tensor tile
    SmallVector<OpFoldResult> strides = {oneAttr, oneAttr, oneAttr, oneAttr}; 
    
    auto thenBuilder = [&](OpBuilder &builder, Location loc) {
      SmallVector<int64_t> inputTileShape(4, 1);
      inputTileShape[1] = inputTileShape[2] = inputTileSize;
      auto tile = builder.getIndexAttr(inputTileSize);
      SmallVector<OpFoldResult> staticSizes = { oneAttr, tile, tile, oneAttr };
      staticSizes[1] = staticSizes[2] = builder.getIndexAttr(inputTileSize);
      auto tensorType = RankedTensorType::get(inputTileShape, elementType);
      auto res = builder.create<IREE::Flow::DispatchTensorLoadOp>(loc, tensorType,
        tensorLoadOp.getSource(), ValueRange({}), offsets, staticSizes, strides).getResult();
      builder.create<scf::YieldOp>(loc, res);
    };
    auto elseBuilder = [&](OpBuilder &builder, Location loc) {
      SmallVector<int64_t> inputTileShape(4, 1);
      inputTileShape[1] = inputTileShape[2] = ShapedType::kDynamicSize;
      auto tensorType = RankedTensorType::get(inputTileShape, elementType);
      auto tensor = builder.create<IREE::Flow::DispatchTensorLoadOp>(loc, tensorType,
        tensorLoadOp.getSource(), ValueRange({}), offsets, sizes, strides).getResult();
      Value paddedInput;
      {
        Value padH = builder.create<arith::SubIOp>(loc, inputTileSz, sizes[1].dyn_cast<Value>());
        Value padW = builder.create<arith::SubIOp>(loc, inputTileSz, sizes[2].dyn_cast<Value>());
        SmallVector<int64_t> inputTileShape(4, 1);
        inputTileShape[1] = inputTileShape[2] = inputTileSize;
        SmallVector<OpFoldResult> lowPad(4, zeroAttr);
        SmallVector<OpFoldResult> highPad = {zeroAttr, padH, padW, zeroAttr};
        auto paddedTensorType = RankedTensorType::get(inputTileShape, elementType);
        auto padTensorOp = builder.create<tensor::PadOp>(loc, paddedTensorType, tensor, lowPad, highPad);
        auto &region = padTensorOp.getRegion();
        int rank = padTensorOp.getResultType().getRank();
        SmallVector<Type> blockArgTypes(rank, builder.getIndexType());
        SmallVector<Location> blockArgLocs(rank, loc);
        builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
        builder.create<tensor::YieldOp>(loc, zerof32);
        builder.setInsertionPointAfter(padTensorOp);
        paddedInput = padTensorOp.getResult();
      }
      builder.create<scf::YieldOp>(loc, paddedInput);
    };

    SmallVector<int64_t> inputTileShape(4, 1);
    inputTileShape[1] = inputTileShape[2] = inputTileSize;
    auto resultType = RankedTensorType::get(inputTileShape, elementType);
    auto paddedInput = rewriter.create<scf::IfOp>(loc, resultType, ifCond, thenBuilder, elseBuilder).getResult(0);
    SmallVector<int64_t> desiredOutputShape = {1, inputTileSize, inputTileSize, 1};
    auto outputType = RankedTensorType::get({inputTileSize, inputTileSize}, elementType);
    auto inputMatrix = rewriter.create<tensor::ExtractSliceOp>(loc, outputType, paddedInput,
      ValueRange({}), ValueRange({}), ValueRange({}), 
        rewriter.getI64ArrayAttr({0, 0, 0, 0}),
        rewriter.getI64ArrayAttr(desiredOutputShape),
        rewriter.getI64ArrayAttr({1, 1, 1, 1}));
    Value interim;
    {
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), elementType);
      Value accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zerof32}, ValueRange{emptyTensor}).result();
      interim = rewriter.create<linalg::MatmulOp>(loc, outputType, ValueRange{inputMatrix, BValue}, accumulator).getResult(0);
    }
    {
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), elementType);
      Value accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zerof32}, ValueRange{emptyTensor}).result();
      rewriter.create<linalg::MatmulOp>(loc, outputType, ValueRange{BTValue, interim}, accumulator);
    }

    //convOp->getParentOfType<ModuleOp>().dump();
    
    return failure();
  }
};

}


namespace {
struct ConvertConv2dToWinogradPass
    : public ConvertConv2dToWinogradBase<
          ConvertConv2dToWinogradPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void ConvertConv2dToWinogradPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertConv2DNhwcHwcf>(
      context);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertConv2dToWinogradPass() {
  return std::make_unique<ConvertConv2dToWinogradPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
