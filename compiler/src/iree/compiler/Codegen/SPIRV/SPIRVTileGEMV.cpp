// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileGEMV.cpp --------------------------------------------------===//
//
// This pass tiles GEMV-like Linalg ops with tensor semantics to subgroups.
//
//===----------------------------------------------------------------------===//

#include <iree/compiler/Codegen/SPIRV/Utils.h>
#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-gemv"

namespace mlir {
namespace iree_compiler {
namespace {

void debugPrint(func::FuncOp funcOp, const char *step) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- After " << step << " ---//\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

linalg::LinalgOp findGEMVOp(func::FuncOp funcOp) {
  linalg::LinalgOp gemvOp;
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (linalg::isaContractionOpInterface(linalgOp) &&
        getLoweringConfig(linalgOp)) {
      gemvOp = linalgOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return gemvOp;
}

linalg::LinalgOp findReductionOp(func::FuncOp funcOp) {
  linalg::LinalgOp reductionOp;
  auto isReduction = [](utils::IteratorType type) {
    return type == utils::IteratorType::reduction;
  };
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (llvm::any_of(linalgOp.getIteratorTypesArray(), isReduction)) {
      reductionOp = linalgOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return reductionOp;
}

FailureOr<linalg::LinalgOp>
tileParallelDimsToSubgroups(linalg::LinalgOp gemvOp,
                            ArrayRef<int64_t> blockTileSizes,
                            unsigned numSubgroups) {
  IRRewriter rewriter(gemvOp->getContext());
  rewriter.setInsertionPoint(gemvOp);

  // We only tile one dimension along thread Y. Set all others to have tile size
  // 0 and find the dimension to tile; it's guaranteed to exist due to previous
  // kernel configuration steps.
  SmallVector<OpFoldResult> numTiles(blockTileSizes.size(),
                                     rewriter.getIndexAttr(0));
  auto dimIt = llvm::find_if(blockTileSizes, [&](int64_t s) {
    return s != 0 && s % numSubgroups == 0;
  });
  if (dimIt == blockTileSizes.end()) {
    gemvOp.emitError("cannot find dimension to tile along thread Y");
    return failure();
  }

  unsigned dimIndex = std::distance(blockTileSizes.begin(), dimIt);
  numTiles[dimIndex] = rewriter.getIndexAttr(numSubgroups);
  auto tileId = gpu::GPUThreadMappingAttr::get(gemvOp->getContext(),
                                               gpu::MappingId::DimY);
  ArrayAttr mapping = rewriter.getArrayAttr({tileId});

  auto tilingResult = linalg::tileToForallOp(
      rewriter, cast<TilingInterface>(*gemvOp), numTiles, mapping);
  rewriter.replaceOp(gemvOp, tilingResult->tileOp->getResults());
  return cast<linalg::LinalgOp>(tilingResult->tiledOp);
}

FailureOr<linalg::LinalgOp>
tileReductionDims(linalg::LinalgOp gemvOp,
                  ArrayRef<int64_t> reductionTileSizes) {
  SmallVector<unsigned> dims;
  gemvOp.getReductionDims(dims);
  // Make sure reduction dimensions are the innermost ones.
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[dims.size() - 1 - i] != gemvOp.getNumLoops() - 1 - i)
      return success();
  }

  IRRewriter rewriter(gemvOp.getContext());
  rewriter.setInsertionPoint(gemvOp);

  SmallVector<OpFoldResult> sizes;
  for (int64_t size : reductionTileSizes) {
    sizes.push_back(rewriter.getIndexAttr(size));
  }

  FailureOr<scf::SCFReductionTilingResult> result = scf::tileReductionUsingScf(
      rewriter, cast<PartialReductionOpInterface>(*gemvOp), sizes);
  if (failed(result))
    return failure();
  // After tiling reduction dimensions, we two Linalg ops--the first one is
  // fully parallel and we can tile it to threads in this warp.
  const char *attrName = getSPIRVDistributeDelinearizeXAttrName();
  for (auto [loop, dim] : zip(llvm::reverse(result->loops), llvm::seq(0, 3))) {
    loop->setAttr(attrName, rewriter.getIndexAttr(dim));
  }
  return cast<linalg::LinalgOp>(result->mergeOp);
}

LogicalResult fuseParallelProducer(linalg::LinalgOp consumerOp) {
  SmallVector<scf::ForOp> loops;
  for (Operation *op = consumerOp.getOperation(); !isa<func::FuncOp>(op);
       op = op->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      loops.push_back(forOp);
  }
  llvm::reverse(loops);

  LLVM_DEBUG(llvm::dbgs() << "looking at consumer op: " << consumerOp << "\n");
  // Collect immediate input operands that are fusable into the tiled loop.
  // We have tensor extract slice ops taking slices of the untiled op.
  SmallVector<tensor::ExtractSliceOp> candidates;
  for (OpOperand *operand : consumerOp.getDpsInputOperands()) {
    LLVM_DEBUG(llvm::dbgs() << "looking at operand: " << operand->get() << "\n");
    auto sliceOp = operand->get().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      continue;
    auto linalgOp = sliceOp.getSource().getDefiningOp<linalg::LinalgOp>();
    if (!linalgOp)
      continue;
    // Restrict to fully parallel linalg ops for now for simplicity.
    auto isParallel = [](utils::IteratorType it) {
      return linalg::isParallelIterator(it);
    };
    if (llvm::all_of(linalgOp.getIteratorTypesArray(), isParallel)) {
      LLVM_DEBUG(llvm::dbgs() << "yes\n");
      candidates.push_back(sliceOp);
    }
  }

  // Fuse the candidate immeidate operands into the tiled loop.
  IRRewriter rewriter(consumerOp.getContext());
  rewriter.setInsertionPoint(consumerOp);
  while (!candidates.empty()) {
    tensor::ExtractSliceOp sliceOp = candidates.back();
    candidates.pop_back();
    LLVM_DEBUG(llvm::dbgs() << "processing slice: " << sliceOp << "\n");
    if (!tileAndFuseProducerOfSlice(rewriter, sliceOp, loops))
      return failure();
  }
  return success();
}

/*
void populateVectorizationPatterns(RewritePatternSet &patterns) {
  IREE::LinalgExt::LinalgTransformationFilter filter;
  IREE::LinalgExt::LinalgVectorizationOptions options;
  // Enable vectorizing tensor.extract in Linalg ops.
  options.vectorizeGatherAccesses = true;
  VectorizationPatterns<linalg::FillOp, linalg::GenericOp>::insert(
      patterns, options, filter);
  linalg::populateConvolutionVectorizationPatterns(patterns);
  patterns.add<LinalgVectorizationPattern>(
      patterns.getContext(), options,
      filter.addOpFilter<linalg::ContractionOpInterface>());
}

LogicalResult wrapReductionInWarpOp(linalg::LinalgOp reductionOp, unsigned subgroupSize) {

  IRRewriter rewriter(reductionOp->getContext());
  rewriter.setInsertionPoint(reductionOp);

  auto shape = 
  linalg::vectorize

  Location loc = reductionOp.getLoc();
    auto threadX = rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(),
                                                   gpu::Dimension::x);
    auto warpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
        loc, TypeRange(), threadX.getResult(), subgroupSize);
    rewriter.setInsertionPointToEnd(warpOp.getBody());
    Operation *clonedOp = rewriter.clone(*reductionOp);
    rewriter.create<vector::YieldOp>(loc);
}
*/

LogicalResult canonicalize(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  context->getOrLoadDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return funcOp.emitOpError("failed to appy canonicalization patterns");
  }
  debugPrint(funcOp, "canonicalization");
  return success();
}

class SPIRVTileGEMVPass final : public SPIRVTileGEMVBase<SPIRVTileGEMVPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    linalg::LinalgOp gemvOp = findGEMVOp(funcOp);
    if (!gemvOp) {
      funcOp.emitError("missing matvec-like linalg op");
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "initial GEMV op: " << gemvOp << "\n");

    std::optional<ArrayAttr> attr = getEntryPoint(funcOp)->getWorkgroupSize();
    if (!attr) {
      funcOp.emitError("missing workgroup size attribute");
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "initial workgroup size: " << *attr << "\n");

    auto workgroupSize = llvm::map_to_vector(attr.value(), [&](Attribute attr) {
      return cast<IntegerAttr>(attr).getInt();
    });
    assert(workgroupSize.size() == 3 && workgroupSize[2] == 1);

    unsigned numSubgroups = workgroupSize[1];
    SmallVector<int64_t> blockTileSizes = getTileSizes(gemvOp, 0);
    SmallVector<int64_t> reductionTileSizes = getTileSizes(gemvOp, 1);

    {
      if (failed(tileParallelDimsToSubgroups(gemvOp, blockTileSizes,
                                             numSubgroups))) {
        funcOp.emitError("failed to tile parallel dimensions to subgroups");
        return signalPassFailure();
      }
      debugPrint(funcOp, "tiling parallel dimensions");
    }

    if (failed(canonicalize(funcOp)))
      return signalPassFailure();

    {
      gemvOp = findGEMVOp(funcOp);
      assert(gemvOp);
      auto result = tileReductionDims(gemvOp, reductionTileSizes);
      if (succeeded(result)) {
        gemvOp = *result;
      } else {
        funcOp.emitError("failed to tile reduction dimensions");
        return signalPassFailure();
      }
      debugPrint(funcOp, "tiling reduction dimensions");
    }

    if (failed(canonicalize(funcOp)))
      return signalPassFailure();

    {
      linalg::LinalgOp reductionOp = findReductionOp(funcOp);
      if (!reductionOp || reductionOp.getNumDpsInputs() != 1) {
        funcOp.emitError("expected single-input final reduction op");
        return signalPassFailure();
      }
      Value input = reductionOp.getDpsInputOperand(0)->get();
      while (auto forOp = input.getDefiningOp<scf::ForOp>()) {
        Operation *terminator = forOp.getBody()->getTerminator();
        input = cast<scf::YieldOp>(terminator).getOperand(0);
      }
      gemvOp = input.getDefiningOp<linalg::LinalgOp>();
      LLVM_DEBUG(llvm::dbgs() << "current gemv op: " << gemvOp << "\n");
      if (!gemvOp || failed(fuseParallelProducer(gemvOp))) {
        funcOp.emitError("failed to fuse producer ops");
        return signalPassFailure();
      }
      debugPrint(funcOp, "fusing producer ops");
    }

    if (failed(canonicalize(funcOp)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTileGEMVPass() {
  return std::make_unique<SPIRVTileGEMVPass>();
}

} // namespace iree_compiler
} // namespace mlir
