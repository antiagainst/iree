// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndPromote.cpp --------------------------------------------===//
//
// This pass tiles promote Linalg ops with buffer semantics to use workgroup
// memory and then tiles to invocations.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-and-promote"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

constexpr int kMaxVectorNumBits = 128;

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Reduction tiling patterns
//====---------------------------------------------------------------------===//

static void populateTilingReductionPatterns(
    RewritePatternSet &patterns,
    IREE::LinalgExt::LinalgTransformationFilter filter) {
  auto getTileSizeFn = [&](OpBuilder &builder, Operation *op) {
    return getTileSizes(builder, op, 2);
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getTileSizeFn);
  TilingPatterns<linalg::BatchMatmulOp, linalg::MatmulOp>::insert(
      patterns, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Invocation tiling patterns
//===----------------------------------------------------------------------===//

static void populateTilingToInvocationPatterns(
    RewritePatternSet &patterns,
    IREE::LinalgExt::LinalgTransformationFilter filter) {
  linalg::TileSizeComputationFunction getTileSizeFn = [&](OpBuilder &builder,
                                                          Operation *op) {
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
                           .setTileSizeComputationFunction(getTileSizeFn)
                           .setDistributionOptions(distributionOptions);

  TilingPatterns<linalg::BatchMatmulOp, linalg::FillOp, linalg::GenericOp,
                 linalg::MatmulOp>::insert(patterns, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Promotion patterns
//===----------------------------------------------------------------------===//

static const char promoteLHSMarker[] = "promote_lhs";
static const char promoteRHSMarker[] = "promote_rhs";
static const char promoteBothMarker[] = "promote_lhs_and_rhs";

template <typename OpTy>
struct CreateSubviewForPromotion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto markerName = IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker;
    auto marker = op->template getAttrOfType<StringAttr>(markerName);
    if (!marker || !marker.getValue().startswith("promote_")) return failure();

    auto linalgOp = cast<linalg::LinalgOp>(*op);
    Value lhs = linalgOp.getDpsInputOperand(0)->get();
    Value rhs = linalgOp.getDpsInputOperand(1)->get();
    auto lhsSubview = createSubviewFromSubspan(lhs, rewriter);
    auto rhsSubview = createSubviewFromSubspan(rhs, rewriter);
    if (!lhsSubview && !rhsSubview) return failure();
    if (lhsSubview) lhs = lhsSubview;
    if (rhsSubview) rhs = rhsSubview;

    auto newOp = rewriter.create<OpTy>(
        op.getLoc(), TypeRange{op.getType(0)}, ValueRange{lhs, rhs},
        ValueRange{linalgOp.getDpsInitOperand(0)->get()});
    newOp->setAttr(markerName, marker);
    rewriter.eraseOp(op);
    return success();
  }

  memref::SubViewOp createSubviewFromSubspan(Value subview,
                                         PatternRewriter &rewriter) const {
    auto subspan =
        subview.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspan) return {};

    auto memrefType = subspan.getType().dyn_cast<MemRefType>();
    if (!memrefType || !memrefType.hasStaticShape()) return {};

    SmallVector<int64_t> offsets(memrefType.getRank(), 0);
    SmallVector<int64_t> strides(memrefType.getRank(), 1);
    return rewriter.create<memref::SubViewOp>(
        subview.getLoc(), subspan, offsets, memrefType.getShape(), strides);
  }
};

LogicalResult copyToWorkgroupMemory(OpBuilder &builder, Value src, Value dst) {
  Operation *copyOp = builder.create<memref::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

template <typename T>
using LinalgPromotionPattern =
    mlir::iree_compiler::IREE::LinalgExt::LinalgPromotionPattern<T>;
static void populatePromotionPatterns(RewritePatternSet &patterns,
                                      StringAttr replaceMarker) {
  MLIRContext *context = patterns.getContext();
  auto baseOptions =
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory)
          .setUseFullTileBuffers({false, false});
  auto promoteLHSOptions = baseOptions.setOperandsToPromote({0});
  auto promoteRHSOptions = baseOptions.setOperandsToPromote({1});
  auto promoteBothOptions = baseOptions.setOperandsToPromote({0, 1});

  IREE::LinalgExt::LinalgTransformationFilter promoteLHSFilter(
      {StringAttr::get(context, promoteLHSMarker)}, replaceMarker);
  IREE::LinalgExt::LinalgTransformationFilter promoteRHSFilter(
      {StringAttr::get(context, promoteRHSMarker)}, replaceMarker);
  IREE::LinalgExt::LinalgTransformationFilter promoteBothFilter(
      {StringAttr::get(context, promoteBothMarker)}, replaceMarker);

  patterns.insert<LinalgPromotionPattern<linalg::MatmulOp>,
                  LinalgPromotionPattern<linalg::BatchMatmulOp>>(
      patterns.getContext(), promoteLHSOptions, promoteLHSFilter);
  patterns.insert<LinalgPromotionPattern<linalg::MatmulOp>,
                  LinalgPromotionPattern<linalg::BatchMatmulOp>>(
      patterns.getContext(), promoteRHSOptions, promoteRHSFilter);
  patterns.insert<LinalgPromotionPattern<linalg::MatmulOp>,
                  LinalgPromotionPattern<linalg::BatchMatmulOp>>(
      patterns.getContext(), promoteBothOptions, promoteBothFilter);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {

class SPIRVTileAndPromotePass final
    : public SPIRVTileAndPromoteBase<SPIRVTileAndPromotePass> {
 public:
  SPIRVTileAndPromotePass(bool skipThreadLevel)
      : skipThreadLevel(skipThreadLevel) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override;

 private:
  // Whether to skip thread level tiling and distribution.
  bool skipThreadLevel = false;
};

}  // namespace

void SPIRVTileAndPromotePass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (failed(exportOp)) return;

  {  // Tile reduction dimensions.
    RewritePatternSet tilingPatterns(context);
    IREE::LinalgExt::LinalgTransformationFilter filter(
        ArrayRef<StringAttr>(),
        StringAttr::get(context, getWorkgroupKTiledMarker()));
    populateTilingReductionPatterns(tilingPatterns, filter);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(tilingPatterns)))) {
      funcOp.emitOpError() << "failed tiling reduction";
      return signalPassFailure();
    }

    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError() << "failed canonicalization after tiling reduction";
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After tiling reduction dimensions ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
      exportOp->getWorkgroupSize().value(),
      [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
  int64_t totalThreads = workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  int subgroupSize =
      getSPIRVTargetEnvAttr(funcOp).getResourceLimits().getSubgroupSize();

  funcOp.walk([&](Operation *op) {
    if (isa<linalg::FillOp, linalg::GenericOp>(op)) {
      op->setAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker,
                  StringAttr::get(context, getWorkgroupMemoryMarker()));
    } else if (isa<linalg::BatchMatmulOp, linalg::MatmulOp>(op)) {
      auto lhsType = op->getOperand(0).getType().cast<ShapedType>();
      auto rhsType = op->getOperand(1).getType().cast<ShapedType>();

      auto elementNumBits = lhsType.getElementTypeBitWidth();
      assert(kMaxVectorNumBits % elementNumBits == 0);
      const int vectorSize = kMaxVectorNumBits / elementNumBits;

      const bool canPromoteLHS = canPerformVectorAccessUsingAllThreads(
          lhsType.getShape(), totalThreads, vectorSize);
      const bool canPromoteRHS = canPerformVectorAccessUsingAllThreads(
          rhsType.getShape(), totalThreads, vectorSize);

      StringAttr promoteMarker =
          StringAttr::get(context, getWorkgroupMemoryMarker());
      if (canPromoteLHS && canPromoteRHS) {
        promoteMarker = StringAttr::get(context, promoteBothMarker);
      } else if (canPromoteLHS) {
        promoteMarker = StringAttr::get(context, promoteLHSMarker);
      } else if (canPromoteRHS) {
        promoteMarker = StringAttr::get(context, promoteRHSMarker);
      }
      op->setAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker,
                  promoteMarker);
    }
    return WalkResult::advance();
  });

  // Only promote to workgroup size if there are multiple warps.
  if (totalThreads > subgroupSize) {
    RewritePatternSet preparePatterns(&getContext());
    preparePatterns.add<CreateSubviewForPromotion<linalg::BatchMatmulOp>,
                        CreateSubviewForPromotion<linalg::MatmulOp>>(
        &getContext());
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(preparePatterns)))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After creating subview ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    RewritePatternSet promotionPatterns(&getContext());
    auto replaceMarker = StringAttr::get(context, getWorkgroupMemoryMarker());
    populatePromotionPatterns(promotionPatterns, replaceMarker);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(promotionPatterns)))) {
      return signalPassFailure();
    }

    // Insert barriers before and after copies to workgroup memory and skip
    // insert barriers between back to back copy to workgroup memory.
    OpBuilder builder(&getContext());
    funcOp.walk([&builder](memref::CopyOp copyOp) {
      if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
        Operation *prevOp = copyOp->getPrevNode();
        if (!prevOp || !hasMarker(prevOp, getCopyToWorkgroupMemoryMarker())) {
          builder.setInsertionPoint(copyOp);
          builder.create<gpu::BarrierOp>(copyOp.getLoc());
        }
        Operation *nextOp = copyOp->getNextNode();
        if (!nextOp || !hasMarker(nextOp, getCopyToWorkgroupMemoryMarker())) {
          builder.setInsertionPointAfter(copyOp);
          builder.create<gpu::BarrierOp>(copyOp.getLoc());
        }
      }
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After promotion ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  if (!skipThreadLevel) {  // Tile and distribute to invocations.
    RewritePatternSet tilingPatterns(&getContext());
    IREE::LinalgExt::LinalgTransformationFilter filter(
        {StringAttr::get(context, getWorkgroupMemoryMarker())}, llvm::None);
    populateTilingToInvocationPatterns(tilingPatterns, filter);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(tilingPatterns)))) {
      funcOp.emitOpError() << "failed tiling and distributing to invocations";
      return signalPassFailure();
    }

    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    populateFoldAffineMinInDistributedLoopsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      // TODO(#4759): This does not converge after the max number of iterations.
      // It indicates that some pattern upstream is generating ops even when the
      // pattern failed to match. Not related to correctness, but would be good
      // to figure out and fix.
      // return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTileAndPromotePass(
    bool skipThreadLevel) {
  return std::make_unique<SPIRVTileAndPromotePass>(skipThreadLevel);
}

}  // namespace iree_compiler
}  // namespace mlir
