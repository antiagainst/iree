// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndDistribute.cpp -----------------------------------------===//
//
// This pass tiles and distributes Linalg ops with buffer semantics to
// invocations.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

#define DEBUG_TYPE "iree-spirv-tile-and-distribute"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Invocation tiling patterns
//===----------------------------------------------------------------------===//

/// Patterns for third level tiling to target invocations.
static void populateTilingToInvocationPatterns(RewritePatternSet &patterns) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&](OpBuilder &builder, Operation *op) {
        return getTileSizes(builder, op, 1);
      };

  auto getThreadProcInfoFn = [](OpBuilder &builder, Location loc,
                                ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  linalg::LinalgLoopDistributionOptions distributionOptions;
  distributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getInnerTileSizeFn)
                           .setDistributionOptions(distributionOptions);

  MLIRContext *context = patterns.getContext();
  auto marker = StringAttr::get(context, getTileReductionMarker());
  auto filter = IREE::LinalgExt::LinalgTransformationFilter(
                    ArrayRef<StringAttr>(), marker)
                    .setMatchByDefault();

  patterns.add<IREE::LinalgExt::LinalgTilingPattern>(context, tilingOptions,
                                                     filter);
  patterns.add<IREE::LinalgExt::TilingInterfaceTilingPattern>(
      context, tilingOptions, filter);
}

static LogicalResult tileParallelDimsToThreads(
    func::FuncOp funcOp, ArrayRef<int64_t> workgroupSize) {
  SmallVector<TilingInterface> computeOps;
  funcOp.walk([&](TilingInterface op) { computeOps.push_back(op); });

  for (TilingInterface tilingOp : computeOps) {
    auto partitionableOp = cast<PartitionableLoopsInterface>(*tilingOp);
    SmallVector<unsigned> partitionedLoops =
        partitionableOp.getPartitionableLoops(kNumMaxParallelDims);
    // If there are no partitioned dimensions, skip the tiling.
    if (partitionedLoops.empty()) continue;

    IRRewriter rewriter(tilingOp->getContext());
    rewriter.setInsertionPoint(tilingOp);

    size_t numLoops = llvm::count_if(
        tilingOp.getLoopIteratorTypes(), [](utils::IteratorType iterator) {
          return iterator == utils::IteratorType::parallel;
        });
    SmallVector<OpFoldResult> numThreads(numLoops, rewriter.getIndexAttr(0));

    int64_t threadId = 0;
    SmallVector<int64_t> threadIds;
    // We map partitioned loops to GPU thread IDs in the reverse order.
    for (auto loop : llvm::enumerate(llvm::reverse(partitionedLoops))) {
      const int64_t dimSize = workgroupSize[loop.index()];
      if (dimSize > 1) {
        numThreads[loop.value()] = rewriter.getIndexAttr(dimSize);
        threadIds.push_back(threadId++);
      }
    }
    std::reverse(threadIds.begin(), threadIds.end());
    for (int64_t i = threadId; i < kNumGPUDims; i++) threadIds.push_back(i);

    auto tilingResult = linalg::tileToForeachThreadOp(rewriter, tilingOp,
                                                      numThreads, threadIds);
    rewriter.replaceOp(tilingOp, tilingResult->tileOp->getResults());
  }
  return success();
}

//====---------------------------------------------------------------------===//
// Reduction tiling patterns
//====---------------------------------------------------------------------===//

static void populateTilingReductionPatterns(RewritePatternSet &patterns) {
  auto getTileSizeFn = [&](OpBuilder &builder, Operation *op) {
    return getTileSizes(builder, op, 2);
  };

  auto filter = IREE::LinalgExt::LinalgTransformationFilter(
      StringAttr::get(patterns.getContext(), getTileReductionMarker()),
      llvm::None);

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getTileSizeFn);

  TilingPatterns<linalg::BatchMatmulOp, linalg::Conv2DNchwFchwOp,
                 linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp,
                 linalg::MatmulOp>::insert(patterns, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and distributing Linalg ops with
/// buffer semantics.
class SPIRVTileAndDistributePass
    : public SPIRVTileAndDistributeBase<SPIRVTileAndDistributePass> {
 public:
  SPIRVTileAndDistributePass() = default;
  SPIRVTileAndDistributePass(const SPIRVTileAndDistributePass &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

//====---------------------------------------------------------------------===//
// Main pass implementation
//====---------------------------------------------------------------------===//

void SPIRVTileAndDistributePass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  if (!isEntryPoint(funcOp)) return;

  {  // Tile and distribute to invocations.
    RewritePatternSet invocationTilingPatterns(context);
    populateTilingToInvocationPatterns(invocationTilingPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(invocationTilingPatterns)))) {
      funcOp.emitOpError() << "failure in tiling";
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);

    populateFoldAffineMinInDistributedLoopsPatterns(canonicalizationPatterns);

    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      // TODO(#4759): This does not converge after the max number of iterations.
      // It indicates that some pattern upstream is generating ops even when the
      // pattern failed to match. Not related to correctness, but would be good
      // to figure out and fix.
      // return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling canonicalization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {  // Tile reduction dimensions.
    RewritePatternSet reductionTilingPatterns(context);
    populateTilingReductionPatterns(reductionTilingPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(reductionTilingPatterns)))) {
      funcOp.emitOpError() << "failing in tile reduction";
      return signalPassFailure();
    }

    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      funcOp.emitOpError() << "failing canonicalizing after tile reduction";
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling reduction dimensions  ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVTileAndDistributePass() {
  return std::make_unique<SPIRVTileAndDistributePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
