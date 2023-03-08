// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-flow-convert-standalone-extract-insert-slices"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Return `true` if the given op is contained in a DispatchRegionOp.
static bool isInDispatchRegion(Operation *op) {
  return op->getParentOfType<Flow::DispatchRegionOp>();
}

/// Wrap a single op in a DispatchRegionOp. When generateWorkloadRegion is
/// true, `workload_count` region is generated for dispatch.region.
static FailureOr<Flow::DispatchRegionOp> warpInRegionOp(
    mlir::TensorDimTrackingRewriter &rewriter, Operation *op,
    bool generateWorkloadRegion) {
  // Compute workload.
  Optional<Flow::WorkloadBuilder> workloadBuilder = std::nullopt;
  if (generateWorkloadRegion) {
    auto maybeBuilder =
        iree_compiler::IREE::Flow::getWorkloadBuilder(rewriter, op);
    if (failed(maybeBuilder)) return failure();
    workloadBuilder = *maybeBuilder;
  }

  // Simplify tensor::DimOps.
  SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
  if (failed(iree_compiler::IREE::Flow::simplifyDimOps(
          rewriter, rewriter.getTensorDimOps())))
    return failure();

  // Wrap operation.
  auto regionOp = Flow::wrapOpInDispatchRegion(rewriter, op, workloadBuilder);
  if (failed(regionOp)) return failure();
  if (failed(cloneProducersToRegion(rewriter, *regionOp))) return failure();
  return *regionOp;
}

/// Wrap all given ops in a DispatchRegionOp.
static FailureOr<SmallVector<Flow::DispatchRegionOp>> warpInRegionOp(
    mlir::TensorDimTrackingRewriter &rewriter, SmallVector<Operation *> rootOps,
    bool generateWorkloadRegion) {
  SmallVector<Flow::DispatchRegionOp> result;
  for (Operation *rootOp : rootOps) {
    auto regionOp = warpInRegionOp(rewriter, rootOp, generateWorkloadRegion);
    if (failed(regionOp)) return failure();
    result.push_back(*regionOp);
  }
  return result;
}

/// Wrap all ops of the given types that are direct children of the given op
/// in DispatchRegionOps.
template <typename... OpTys>
static FailureOr<SmallVector<Flow::DispatchRegionOp>> warpInRegionOp(
    mlir::TensorDimTrackingRewriter &rewriter, Operation *op,
    bool generateWorkloadRegion) {
  // Find ops of type OpTys.
  SmallVector<Operation *> rootOps;
  for (Region &r : op->getRegions())
    for (Block &b : r.getBlocks())
      for (Operation &op : b)
        if (isa<OpTys...>(&op)) rootOps.push_back(&op);

  // Wrap ops in DispatchRegionOps.
  return warpInRegionOp(rewriter, rootOps, generateWorkloadRegion);
}

/// Rewrite top-level InsertSliceOps to FlowUpdateOps or wrap them in a
/// dispatch region.
LogicalResult convertInsertSliceOps(mlir::TensorDimTrackingRewriter &rewriter,
                                    mlir::FunctionOpInterface funcOp,
                                    bool generateWorkloadRegion) {
  // Find eligible InsertSliceOps.
  SmallVector<tensor::InsertSliceOp> insertSliceOps;
  funcOp.walk([&](tensor::InsertSliceOp op) {
    if (!isInDispatchRegion(op)) insertSliceOps.push_back(op);
  });

  // Rewrite InsertSliceOps to FlowUpdateOps.
  SmallVector<Operation *> remainingInsertSliceOps;
  for (tensor::InsertSliceOp insertSliceOp : insertSliceOps) {
    if (failed(convertInsertSliceOpToFlowUpdateOp(rewriter, insertSliceOp))) {
      remainingInsertSliceOps.push_back(insertSliceOp);
    }
  }

  // Create a DispatchRegionOp for every remaining InsertSliceOp.
  FailureOr<SmallVector<Flow::DispatchRegionOp>> newRegionOps =
      warpInRegionOp(rewriter, remainingInsertSliceOps, generateWorkloadRegion);
  if (failed(newRegionOps)) return failure();

  return success();
}

/// Rewrite top-level ExtractSliceOps to FlowSliceOps or wrap them in a
/// dispatch region.
LogicalResult convertExtractSliceOps(mlir::TensorDimTrackingRewriter &rewriter,
                                     mlir::FunctionOpInterface funcOp,
                                     bool generateWorkloadRegion) {
  // Find eligible ExtractSliceOps.
  SmallVector<tensor::ExtractSliceOp> extractSliceOps;
  funcOp.walk([&](tensor::ExtractSliceOp op) {
    if (!isInDispatchRegion(op)) extractSliceOps.push_back(op);
  });

  // Rewrite ExtractSliceOps to FlowSliceOps.
  SmallVector<Operation *> remainingExtractSliceOps;
  for (tensor::ExtractSliceOp extractSliceOp : extractSliceOps) {
    if (failed(convertExtractSliceOpToFlowSliceOp(rewriter, extractSliceOp))) {
      remainingExtractSliceOps.push_back(extractSliceOp);
    }
  }

  // Create a DispatchRegionOp for every remaining ExtractSliceOp.
  FailureOr<SmallVector<Flow::DispatchRegionOp>> newRegionOps = warpInRegionOp(
      rewriter, remainingExtractSliceOps, generateWorkloadRegion);
  if (failed(newRegionOps)) return failure();

  return success();
}

namespace {
struct ConvertStandaloneExtractInsertSlicesPass
    : public ConvertStandaloneExtractInsertSlicesBase<
          ConvertStandaloneExtractInsertSlicesPass> {
  ConvertStandaloneExtractInsertSlicesPass(bool generateWorkloadRegion) {
    this->generateWorkloadRegion = generateWorkloadRegion;
  }
  ConvertStandaloneExtractInsertSlicesPass(
      const ConvertStandaloneExtractInsertSlicesPass &pass)
      : ConvertStandaloneExtractInsertSlicesPass(pass.generateWorkloadRegion) {}

  void runOnOperation() override;
};
}  // namespace

void ConvertStandaloneExtractInsertSlicesPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  mlir::TensorDimTrackingRewriter rewriter(funcOp);

  funcOp->walk([&](DispatchRegionOp regionOp) {
    if (failed(cloneProducersToRegion(rewriter, regionOp)))
      return signalPassFailure();
  });

  // Step 2: Rewrite InsertSliceOps to FlowUpdateOps.
  if (failed(convertInsertSliceOps(rewriter, funcOp, generateWorkloadRegion))) {
    funcOp->emitOpError(
        "failed to create dispatch region for `tensor.insert_slice`");
    return signalPassFailure();
  }

  // Step 3: Rewrite ExtractSliceOps to FlowUpdateOps.
  if (failed(
          convertExtractSliceOps(rewriter, funcOp, generateWorkloadRegion))) {
    funcOp->emitOpError(
        "failed to create dispatch region for `tensor.extract_slice`");
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertStandaloneExtractInsertSlicesPass(bool generateWorkloadRegion) {
  return std::make_unique<ConvertStandaloneExtractInsertSlicesPass>(
      generateWorkloadRegion);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
