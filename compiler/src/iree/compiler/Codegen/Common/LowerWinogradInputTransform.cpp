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

class ConvertWinogradInputTransform final
    : public OpRewritePattern<IREE::Flow::WinogradInputTransformOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult applyTiling(IREE::Flow::WinogradInputTransformOp inputOp, std::string &tensorFormat, 
                                   std::string &outputTensorFormat, std::string &tilingOrder,
                                   std::vector<int> &tileSizes, std::vector<int> &stepSizes, int numWorkgroups,
                                   PatternRewriter &rewriter) {
      // Define constants (on top of func)
      auto loc = inputOp.getLoc();
      auto funcOp = inputOp->getParentOfType<func::FuncOp>();
      rewriter.setInsertionPointToStart(&funcOp.getBody().front());
      Value input = inputOp.getInput();
      auto inputType = input.getType().cast<ShapedType>();
      auto inputRank = inputType.getRank();
      auto inputShape = inputType.getShape();
      auto elementType = inputType.getElementType();

      Value output = inputOp.getResult();
      auto outputType = output.getType().cast<ShapedType>();
      auto outputRank = outputType.getRank();
      auto outputShape = outputType.getShape();

      std::unordered_map<std::string, std::vector<int64_t>> loopState;
      std::unordered_map<std::string, std::vector<Value>> values;
      std::unordered_map<std::string, std::vector<Attribute>> attrs;
      auto gid = [](const int i, const char c) {
        std::string id{1, c};
        return (c + std::to_string(i));
      };
      // Set default values for loop state
      auto zerof32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
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
      int Bdim = std::sqrt(BT.size());
      SmallVector<float> B;
      transpose(BT, B, Bdim, Bdim);
      auto BTValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
        RankedTensorType::get({Bdim, Bdim}, rewriter.getF32Type()), BT));
      auto BValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
        RankedTensorType::get({Bdim, Bdim}, rewriter.getF32Type()), B));
      for (int i = 0; i < tilingOrder.size(); i++) {
        char tilingDim = tilingOrder[i];
        size_t pos = tensorFormat.find(tilingDim);
        loopState[gid(i, tilingOrder[i])] = {0, inputShape[pos], stepSizes[i], tileSizes[i]};
        values[gid(i, tilingOrder[i])] = {
          rewriter.create<arith::ConstantIndexOp>(loc, 0),
          rewriter.create<arith::ConstantIndexOp>(loc, inputShape[pos]),
          rewriter.create<arith::ConstantIndexOp>(loc, stepSizes[i]),
          rewriter.create<arith::ConstantIndexOp>(loc, tileSizes[i])
        };
        attrs[gid(i, tilingOrder[i])] = {
          rewriter.getIndexAttr(tileSizes[i])
        };
      }

      // Generate loops
      rewriter.setInsertionPoint(inputOp);
      std::unordered_map<char, Value> ids, counts;    
      // Workgroup ids are reversed 
      for (int i = 0; i < numWorkgroups; i++) {
        ids[tilingOrder[i]] = rewriter.create<IREE::HAL::InterfaceWorkgroupIDOp>(loc, numWorkgroups - i - 1);
        counts[tilingOrder[i]] = rewriter.create<IREE::HAL::InterfaceWorkgroupCountOp>(loc, numWorkgroups - i - 1);
        values[gid(i, tilingOrder[i])][0] = ids[tilingOrder[i]];
        values[gid(i, tilingOrder[i])][2] = counts[tilingOrder[i]];
      }

      auto asIndexAttr = [&](int64_t i) {
        return rewriter.getIndexAttr(i);
      };

      SmallVector<OpFoldResult> offsets(inputRank, asIndexAttr(0));
      SmallVector<int64_t> currentSize;
      SmallVector<OpFoldResult> sizes; 
      for (int i = 0; i < inputRank; i++) {
        sizes.push_back(asIndexAttr(inputShape[i]));
        currentSize.push_back(inputShape[i]);
      }
      SmallVector<OpFoldResult> strides(inputRank, asIndexAttr(1));

      SmallVector<OpFoldResult> outputOffsets(outputRank, asIndexAttr(0));
      SmallVector<int64_t> currentOutputSize(outputRank, 0);
      SmallVector<OpFoldResult> outputSizes(outputRank, asIndexAttr(1)); 
      for (int i = 0; i < outputRank; i++) {
        outputSizes[i] = asIndexAttr(outputShape[i]);
        currentOutputSize[i] = outputShape[i];
      }
      SmallVector<OpFoldResult> outputStrides(outputRank, asIndexAttr(1));

      // First build workgroup loop nests
      Value loadedSlice, loadedOutputSlice;
      Value targetOutputTensor;
      SmallVector<OpFoldResult> dispatchOffsets, dispatchSizes;
      for (auto dim : llvm::enumerate(tilingOrder)) {
        if (dim.index() >= numWorkgroups) break;
        auto d = gid(dim.index(), dim.value());
        size_t pos = tensorFormat.find(dim.value());
        size_t opos = outputTensorFormat.find(dim.value());
        Value lo = values[d][0];
        Value hi = values[d][1];
        Value step = values[d][2];
        auto tile = attrs[d][0];
        if ((dim.index() > 0) && (dim.index() < numWorkgroups)) {
          // Emit affine.apply ops
          AffineExpr s0;
          bindSymbols(rewriter.getContext(), s0);
          AffineMap map = AffineMap::get(0, 1, {s0 * rewriter.getAffineConstantExpr(tileSizes[dim.index()])}, rewriter.getContext());
          lo = rewriter.createOrFold<AffineApplyOp>(loc, map, ValueRange{lo});
          step = rewriter.createOrFold<AffineApplyOp>(loc, map, ValueRange{step});
        }
        auto forOp = rewriter.create<scf::ForOp>(loc, lo, hi, step);
        offsets[pos] = forOp.getInductionVar();
        sizes[pos] = tile;
        currentSize[pos] = tile.cast<IntegerAttr>().getValue().getSExtValue();
        if (opos != std::string::npos) {
          outputOffsets[opos] = forOp.getInductionVar();
          outputSizes[opos] = tile;
          currentOutputSize[opos] = tile.cast<IntegerAttr>().getValue().getSExtValue();
        }
        rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
        if (dim.index() == numWorkgroups - 1) {
          // Emit flow.dispatch ops for input
          auto tensorLoadOp = input.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
          if (!tensorLoadOp) return failure();
          auto tensorType = RankedTensorType::get(currentSize, elementType);
          loadedSlice = rewriter.create<IREE::Flow::DispatchTensorLoadOp>(loc, tensorType,
            tensorLoadOp.getSource(), ValueRange({}), offsets, sizes, strides).getResult();
          // Emit flow.dispatch ops for output
          for (auto user : inputOp.getResult().getUsers()) {
            if (auto tensorStoreOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(user)) {
              auto outputTensorType = RankedTensorType::get(currentOutputSize, elementType);
              targetOutputTensor = tensorStoreOp.getTarget();
              loadedOutputSlice = rewriter.create<IREE::Flow::DispatchTensorLoadOp>(loc, outputTensorType,
                tensorStoreOp.getTarget(), ValueRange({}), outputOffsets, outputSizes, outputStrides).getResult();
              dispatchSizes = outputSizes;
              dispatchOffsets = outputOffsets;
              break;
            }
          }
        }
        // Update total size for subsequent tilings
        for (int j = 0; j < tilingOrder.size(); j++) {
          if ((dim.index() != j)  && (tilingOrder[j] == dim.value())) {
            auto dnew = gid(j, tilingOrder[j]);
            values[dnew][1] = values[d][3];
          }
        }
      }

      auto bodyBuilder = [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs, ValueRange iterArgs) {
        SmallVector<Value> eqZeroCmpVals;
        SmallVector<Value> spatialDims;
        int c{0};
        for (auto dim : llvm::enumerate(tilingOrder)) {
          if (dim.index() < numWorkgroups) continue;
          auto d = gid(dim.index(), dim.value());
          size_t pos = tensorFormat.find(dim.value());
          size_t opos = outputTensorFormat.find(dim.value());
          auto tile = attrs[d][0];
          if ((dim.value() == 'h') || (dim.value() == 'w')) {
            // Emit affine min ops for sliding window
            AffineExpr dim0;
            auto t = rewriter.getAffineConstantExpr(tileSizes[dim.index()]);
            auto delta = rewriter.getAffineConstantExpr(inputShape[pos]);
            bindDims(rewriter.getContext(), dim0);
            AffineMap minMap = AffineMap::get(1, 0, {-dim0 + delta, t}, rewriter.getContext());
            auto size = rewriter.createOrFold<AffineMinOp>(loc, minMap, ValueRange{outputIvs[c]});
            eqZeroCmpVals.push_back(rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq, size, values[d][3]));
            spatialDims.push_back(size);
            // Emit affine ops for output dims
            auto s = rewriter.getAffineConstantExpr(stepSizes[dim.index()]);
            AffineMap outputMap = AffineMap::get(1, 0, {dim0.floorDiv(s)}, rewriter.getContext());
            size_t offpos = dim.value() == 'h' ? outputTensorFormat.find('H') : outputTensorFormat.find('W');
            outputOffsets[offpos] = rewriter.createOrFold<AffineApplyOp>(loc, outputMap, ValueRange{outputIvs[c]});
          }
          // Update offsets and size
          offsets[pos] = outputIvs[c];
          sizes[pos] = tile;
          currentSize[pos] = tile.cast<IntegerAttr>().getValue().getSExtValue();
          if (opos != std::string::npos) {
            outputOffsets[opos] = outputIvs[c];
            outputSizes[opos] = tile;
            currentOutputSize[opos] = tile.cast<IntegerAttr>().getValue().getSExtValue();
          }
          // Update total size for subsequent tilings
          for (int j = 0; j < tilingOrder.size(); j++) {
            if ((dim.index() != j)  && (tilingOrder[j] == dim.value())) {
              auto dnew = gid(j, tilingOrder[j]);
              values[dnew][1] = values[d][3];
            }
          }
          c++;
        }

        assert(eqZeroCmpVals.size() == 2);
        Value ifCond = rewriter.create<arith::AndIOp>(loc, eqZeroCmpVals[0], eqZeroCmpVals[1]);

        SmallVector<int64_t> rankReducedSize;
        for (auto size : currentSize) {
          if (size == 1) continue;
          rankReducedSize.push_back(size);
        }

        auto thenBuilder = [&](OpBuilder &builder, Location loc) {
          // Remove unit dims from size during extraction
          auto tensorType = RankedTensorType::get(rankReducedSize, elementType);
          auto res = builder.create<tensor::ExtractSliceOp>(loc, tensorType,
            loadedSlice, offsets, sizes, strides).getResult();
          builder.create<scf::YieldOp>(loc, res);
        };

        auto elseBuilder = [&](OpBuilder &builder, Location loc) {
          // Remove unit dims from size during extraction
          SmallVector<int64_t> rankReducedDynamicSize;
          int k{0};
          for (auto size : llvm::enumerate(currentSize)) {
            if ((tensorFormat[size.index()] == 'h') || (tensorFormat[size.index()] == 'w')) {
              rankReducedDynamicSize.push_back(ShapedType::kDynamicSize);
              sizes[size.index()] = spatialDims[k++];
              continue;
            }
            if (size.value() == 1) continue;
            rankReducedDynamicSize.push_back(size.value());
          }
          auto tensorType = RankedTensorType::get(rankReducedDynamicSize, elementType);
          auto slice = builder.create<tensor::ExtractSliceOp>(loc, tensorType,
            loadedSlice, offsets, sizes, strides).getResult();
          SmallVector<OpFoldResult> hiPad;
          k = 0;
          for (auto dim : llvm::enumerate(tilingOrder)) {
            auto d = gid(dim.index(), dim.value());
            if ((dim.value() == 'h') || (dim.value() == 'w')) {
              hiPad.push_back(builder.create<arith::SubIOp>(loc, values[d][3], spatialDims[k++]).getResult());
            }
          }
          SmallVector<OpFoldResult> lowPad{rankReducedDynamicSize.size(), builder.getIndexAttr(0)};
          auto padTensorOp = builder.create<tensor::PadOp>(loc, 
             RankedTensorType::get(rankReducedSize, elementType), slice, lowPad, hiPad);
          auto &region = padTensorOp.getRegion();
          int rank = padTensorOp.getResultType().getRank();
          SmallVector<Type> blockArgTypes(rank, rewriter.getIndexType());
          SmallVector<Location> blockArgLocs(rank, loc);
          builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
          builder.create<tensor::YieldOp>(loc, zerof32);
          builder.setInsertionPointAfter(padTensorOp);
          builder.create<scf::YieldOp>(loc, padTensorOp.getResult());
        };

        // Compute quadratic form
        auto inputTileTxT = rewriter.create<scf::IfOp>(loc, 
           RankedTensorType::get(rankReducedSize, elementType), 
           ifCond, thenBuilder, elseBuilder).getResult(0);

        // Extract output slice
        outputSizes[3] = outputSizes[4] = rewriter.getIndexAttr(1);
        auto outputSlice = rewriter.create<tensor::ExtractSliceOp>(loc, 
              RankedTensorType::get(rankReducedSize, elementType), iterArgs[iterArgs.size() - 1],
              outputOffsets, outputSizes, outputStrides);

        Value interim, accumulator;
        auto matmulType = RankedTensorType::get({Bdim, Bdim}, elementType);
        accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zerof32}, ValueRange{outputSlice}).result();
        interim = rewriter.create<linalg::MatmulOp>(loc, matmulType, ValueRange{inputTileTxT, BValue}, accumulator).getResult(0);
        accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zerof32}, ValueRange{outputSlice}).result();
        auto transformed = rewriter.create<linalg::MatmulOp>(loc, matmulType, ValueRange{BTValue, interim}, accumulator).getResult(0);

        auto resultSlice = rewriter.create<tensor::InsertSliceOp>(loc, transformed, loadedOutputSlice,
            outputOffsets, outputSizes, outputStrides).getResult();

        return resultSlice;
      };

      // Next build loops with carry
      SmallVector<Value> lbs, ubs, steps;
      for (auto dim : llvm::enumerate(tilingOrder)) {
        if (dim.index() < numWorkgroups) continue;
        auto d = gid(dim.index(), dim.value());
        lbs.push_back(values[d][0]);
        ubs.push_back(values[d][1]);
        steps.push_back(values[d][2]);
      }
      auto loopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, ValueRange({loadedOutputSlice}),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs, ValueRange iterArgs) -> scf::ValueVector {
          return {bodyBuilder(nestedBuilder, loc, outputIvs, iterArgs)};
      });

      // Store result
      rewriter.create<IREE::Flow::DispatchTensorStoreOp>(loc, loopNest.getResults()[0],
         targetOutputTensor, ValueRange({}), dispatchOffsets, dispatchSizes, outputStrides);

      // Remove input op and related load/store
      for (auto user : inputOp.getResult().getUsers()) {
        if (auto tensorStoreOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(user)) {
          rewriter.eraseOp(tensorStoreOp);
          break;
        }
      }
      auto tensorLoadOp = input.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
      if (tensorLoadOp) {
        rewriter.eraseOp(inputOp);
        rewriter.eraseOp(tensorLoadOp);
      }
      return success();
  }

  LogicalResult matchAndRewrite(IREE::Flow::WinogradInputTransformOp inputOp,
                                PatternRewriter &rewriter) const override {

    std::string inputTensorFormat{"nhwc"};
    std::string outputTensorFormat{"ttnHWc"};
    std::string tilingOrder{"nchwc"};
    std::vector<int> tileSizes{1,32,8,8,1};
    std::vector<int> stepSizes{1,32,6,6,1};
    int numWorkgroups = 2; // Implies top 2 will be used for workgroups
    if (failed(applyTiling(inputOp, inputTensorFormat, outputTensorFormat, tilingOrder, tileSizes, stepSizes, numWorkgroups, rewriter)))
      return failure();
  
    return success();
  }
};

}


namespace {
struct LowerWinogradInputTransformPass
    : public LowerWinogradInputTransformBase<
          LowerWinogradInputTransformPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void LowerWinogradInputTransformPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertWinogradInputTransform>(
      context);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLowerWinogradInputTransformPass() {
  return std::make_unique<LowerWinogradInputTransformPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
