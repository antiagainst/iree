// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVVectorizeGEMV.cpp ---------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-vectorize-gemv"

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

vector::ReductionOp findVectorReductionOp(func::FuncOp funcOp) {
  vector::ReductionOp reductionOp;
  funcOp.walk([&](vector::ReductionOp op) {
    reductionOp = op;
    return WalkResult::interrupt();
  });
  return reductionOp;
}

void subgroupReduce(vector::ReductionOp reductionOp,
                    unsigned subgroupSize) {
  IRRewriter rewriter(reductionOp->getContext());
  rewriter.setInsertionPoint(reductionOp);

  Location loc = reductionOp.getLoc();
  vector::CombiningKind kind = reductionOp.getKind();
  Value result = emitGPUGroupReduction(loc, rewriter, reductionOp.getVector(),
                                       kind, subgroupSize, subgroupSize);
  if (reductionOp.getAcc()) {
    result = vector::makeArithReduction(rewriter, loc, kind, result,
                                        reductionOp.getAcc());
  }
  rewriter.replaceAllUsesWith(reductionOp.getResult(), result);
}

class InsertElementToBroadcast final
    : public OpRewritePattern<vector::InsertElementOp> {
public:
  using OpRewritePattern<vector::InsertElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertElementOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp.getDestVectorType().getNumElements() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getSource());
    return success();
  }
};

LogicalResult preprocessReduction(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction);
  patterns.add<InsertElementToBroadcast>(context);
  // Trimming leading unit dims may generate broadcast/shape_cast ops.
  vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
  vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return funcOp.emitOpError("failed to appy canonicalization patterns");
  }
  debugPrint(funcOp, "preprocessing reduction");
  return success();
}

LogicalResult canonicalize(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  // Trimming leading unit dims may generate broadcast/shape_cast ops.
  vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
  vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return funcOp.emitOpError("failed to appy canonicalization patterns");
  }
  debugPrint(funcOp, "canonicalization");
  return success();
}

class SPIRVVectorizeGEMVPass final
    : public SPIRVVectorizeGEMVBase<SPIRVVectorizeGEMVPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (failed(preprocessReduction(funcOp)))
      return signalPassFailure();

    vector::ReductionOp reductionOp = findVectorReductionOp(funcOp);
    if (!reductionOp) {
      funcOp.emitError("missing vector.multi_reduction op");
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "reduction op: " << reductionOp << "\n");

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

    subgroupReduce(reductionOp, workgroupSize[0]);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVVectorizeGEMVPass() {
  return std::make_unique<SPIRVVectorizeGEMVPass>();
}

} // namespace iree_compiler
} // namespace mlir

