// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- ConcretizeWorkloadOpsPass.cpp --------------------------------------===//
//
// This pass concretizes flow.dispatch.workgroup.* ops by replacing them with
// chosen constant values.
//
// During dispatch region formation in IREE Flow transformations, ops are tiled
// and distributed in an abstract way by using symbolic flow.dispatch.workgroup
// ops. That is because the same source region is compiled towards different
// target backends and each target backend could use different tiling and
// distribution schemes. However, after HAL interface materialization, the
// hal.executable.target is just meant for one target backend. We need to
// concretize the tiling and distribution in order to inject static information
// for further compilation.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/Common/LaunchConfig.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Builders.h"

#define DEBUG_TYPE "iree-codegen-concretize-workload-ops"

namespace mlir {
namespace iree_compiler {

namespace {

constexpr unsigned kWorkgroupDimCount = 3;

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// XXX: Hack. We need a better way to rediscover the original root op.
linalg::LinalgOp getRootLinalgOp(FuncOp funcOp) {
  linalg::LinalgOp rootOp;
  funcOp.walk([&rootOp](linalg::LinalgOp op) {
    if (op.getOperation()->hasAttr("iree.codegen.fushion.root_op")) {
      rootOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return rootOp;
}

// XXX: Hmm.. A better way to directly get the original workload would be nice.
llvm::Optional<SmallVector<int64_t, 4>> getWorkloadSize(FuncOp funcOp) {
  // There is no explicit workload size set in the IR that we can directly query
  // after HAL interface is materialized. So go through the IR use chain to
  // figure it out. Otherwise we need to make even more assumptions.

  linalg::LinalgOp rootOp = getRootLinalgOp(funcOp);
  if (!rootOp) {
    funcOp.emitError("unable to find root op");
    return llvm::None;
  }

  // Assume we only have one result.
  if (rootOp->getNumResults() != 1) return llvm::None;

  // Assume we only have one use.
  auto uses = rootOp->getResult(0).getUses();
  if (++uses.begin() != uses.end()) return llvm::None;

  // Then assume it's store op.
  auto storeOp =
      dyn_cast<IREE::Flow::DispatchOutputStoreOp>(uses.begin()->getOwner());
  if (!storeOp) return llvm::None;

  auto outputType =
      storeOp.target().getType().cast<IREE::Flow::DispatchOutputType>();

  // Flow/HAL processor id/size/count ops' are created using the reverse order.
  return llvm::to_vector<4>(llvm::reverse(outputType.getShape()));
}

// XXX: Hack. The following function makes this pass scoped to GPU..
// Remove it to make this pass common to CPU and GPU.
llvm::Optional<SmallVector<int64_t, 4>> getTileSize(FuncOp funcOp) {
  SPIRVCodegenOptions options;
  options.enableVectorization = true;
  options.usingLinalgOnTensors = true;

  linalg::LinalgOp rootOp = getRootLinalgOp(funcOp);
  if (!rootOp) {
    funcOp.emitError("unable to find root op");
    return llvm::None;
  }

  LLVM_DEBUG(llvm::dbgs() << "Root op: " << rootOp << "\n");

  SmallVector<linalg::LinalgOp, 4> linalgOps;
  auto ops = rootOp->getBlock()->getOps<linalg::LinalgOp>();
  linalgOps.assign(ops.begin(), ops.end());

  linalg::Aliases aliases;
  linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
  Optional<LaunchConfig> launchConfig = initGPULaunchConfig(
      funcOp.getContext(), dependenceGraph, options, linalgOps);
  if (!launchConfig) {
    funcOp.emitError("unable to find launch configuration");
    return llvm::None;
  }

  ArrayRef<int64_t> tileSize = launchConfig->getTileSizes(rootOp, 0);

  // Clean up internal markers that are set during launch configuration
  // preparation.
  launchConfig->finalize(funcOp);

  // The tile sizes are specified against the original dimension order of the
  // workload shape. But Flow/HAL processor id/size/count ops' are created using
  // the reverse order.
  return llvm::to_vector<4>(llvm::reverse(tileSize));
}

class ConcretizeWorkgroupIDOp final
    : public OpRewritePattern<IREE::Flow::DispatchWorkgroupIDOp> {
 public:
  ConcretizeWorkgroupIDOp(MLIRContext *context,
                          SmallVector<int64_t, 4> workloadSize,
                          SmallVector<int64_t, 4> tileSize,
                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(std::move(workloadSize)),
        tileSize(std::move(tileSize)) {}

  LogicalResult matchAndRewrite(IREE::Flow::DispatchWorkgroupIDOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex < kWorkgroupDimCount) {
      rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceWorkgroupIDOp>(
          op, op.getResult().getType(), op.dimensionAttr());
      return success();
    }

    // For all dimensions that are out of range, turn the workgroup ID to zero
    // to create a single workgroup, in order to undo the loop materialization.

    assert(tileSize[dimIndex] == 0 && "cannot tile out-of-range dim!");

    rewriter.replaceOpWithNewOp<ConstantOp>(op, rewriter.getIndexAttr(0));
    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadSize;
  SmallVector<int64_t, 4> tileSize;
};

class ConcretizeWorkgroupSizeOp final
    : public OpRewritePattern<IREE::Flow::DispatchWorkgroupSizeOp> {
 public:
  ConcretizeWorkgroupSizeOp(MLIRContext *context,
                            SmallVector<int64_t, 4> workloadSize,
                            SmallVector<int64_t, 4> tileSize,
                            PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(std::move(workloadSize)),
        tileSize(std::move(tileSize)) {}

  LogicalResult matchAndRewrite(IREE::Flow::DispatchWorkgroupSizeOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex < kWorkgroupDimCount) {
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, rewriter.getIndexAttr(tileSize[dimIndex]));
      return success();
    }

    // For all dimensions that are out of range, turn the workgroup size
    // to the workload size to create a single workgroup, in order to undo the
    // loop materialization.

    int64_t dimSize = workloadSize[dimIndex];
    assert(tileSize[dimIndex] == 0 && "cannot tile out-of-range dim!");

    if (dimSize == ShapedType::kDynamicSize) {
      // Need to reference to std.dim..
      return failure();
    } else {
      rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                              rewriter.getIndexAttr(dimSize));
    }
    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadSize;
  SmallVector<int64_t, 4> tileSize;
};

class ConcretizeWorkgroupCountOp final
    : public OpRewritePattern<IREE::Flow::DispatchWorkgroupCountOp> {
 public:
  ConcretizeWorkgroupCountOp(MLIRContext *context,
                             SmallVector<int64_t, 4> workloadSize,
                             SmallVector<int64_t, 4> tileSize,
                             PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(workloadSize),
        tileSize(std::move(tileSize)) {}

  LogicalResult matchAndRewrite(IREE::Flow::DispatchWorkgroupCountOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex >= kWorkgroupDimCount) {
      // For all dimensions that are out of range, turn the workgroup count
      // to one to create a single workgroup, in order to undo the loop
      // materialization.
      assert(tileSize[dimIndex] == 0 && "cannot tile out-of-range dim!");
      rewriter.replaceOpWithNewOp<ConstantOp>(op, rewriter.getIndexAttr(1));
      return success();
    }

    // Otherwise calculate the static count if possible.

    int64_t dimSize = workloadSize[dimIndex];
    int64_t dimTile = tileSize[dimIndex];

    if (dimSize == ShapedType::kDynamicSize) {
      rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceWorkgroupCountOp>(
          op, op.getResult().getType(), op.dimensionAttr());
    } else {
      int64_t count = ceilDiv(dimSize, dimTile);
      rewriter.replaceOpWithNewOp<ConstantOp>(op, rewriter.getIndexAttr(count));
    }
    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadSize;
  SmallVector<int64_t, 4> tileSize;
};

// Canonicalizes away a trip-one scf.for loop by inlining its body and removing
// the loop.
//
// This pattern is needed because in Flow abstract tiling and distribution we
// will create scf.for loops that distribute workload cyclically. After
// concretizing flow.dispatch.workgroup.* ops, these scf.for loops still remain,
// and they will be of the form:
//
//   %lb = mul %workgroup_id_{x|y|z}, %cst_tile_size_{x|y|z}
//   scf.for %iv = %lb to %cst_wokload_size_{x|y|z}
//                 step %cst_workload_size_{x|y|z} { ... }
//
// Such scf.for loops can be removed if %lb is smaller than upper bound.
class RemoveTripOneLoop final : public OpRewritePattern<scf::ForOp> {
 public:
  RemoveTripOneLoop(MLIRContext *context, SmallVector<int64_t, 4> workloadSize,
                    SmallVector<int64_t, 4> tileSize,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(workloadSize),
        tileSize(std::move(tileSize)) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // Get constant upper bound and step values.
    IntegerAttr ub, step;
    if (!matchPattern(op.upperBound(), m_Constant(&ub)) ||
        !matchPattern(op.step(), m_Constant(&step))) {
      return failure();
    }

    // Require that they are the same.
    if (ub != step) return failure();

    // Now make sure the lower bound is smaller than upper bound. The lower
    // bound should be multiplying the workgroup ID with some constant.
    auto mulOp = op.lowerBound().getDefiningOp<MulIOp>();
    if (!mulOp) return failure();

    auto idOp = mulOp.lhs().getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>();
    IntegerAttr multipler;
    if (!idOp || !matchPattern(mulOp.rhs(), m_Constant(&multipler)))
      return failure();

    // We just need to make sure the max value of the workgroup ID multipled by
    // the multipler is smaller than the upper bound to guarantee one trip.
    unsigned dimIndex = idOp.dimension().getZExtValue();
    int64_t dimSize = workloadSize[dimIndex];
    int64_t dimTile = tileSize[dimIndex];

    if (dimSize == ShapedType::kDynamicSize) return failure();

    int64_t count = ceilDiv(dimSize, dimTile);
    assert(count > 0 && "expected at least one tile!");

    // ID should be in range [0, count).
    if ((count - 1) * multipler.getInt() >= ub.getInt()) {
      // Dead loop. It can actually be removed entirely. But we aren't expecting
      // it to happen here. Do not canonicalize for such case.
      return failure();
    }

    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(op.getNumIterOperands() + 1);
    blockArgs.push_back(op.lowerBound());
    llvm::append_range(blockArgs, op.getIterOperands());

    Block *block = &op.getLoopBody().front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.mergeBlockBefore(block, op, blockArgs);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);

    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadSize;
  SmallVector<int64_t, 4> tileSize;
};

struct ConcretizeWorkloadOpsPass
    : public PassWrapper<ConcretizeWorkloadOpsPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
  void runOnOperation() override {
    IREE::HAL::ExecutableTargetOp targetOp = getOperation();
    ModuleOp module = targetOp.getInnerModule();

    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (!funcOp.isPublic()) return;

      MLIRContext &context = getContext();

      llvm::Optional<SmallVector<int64_t, 4>> tileSize = getTileSize(funcOp);
      if (!tileSize) {
        funcOp.emitError("failed to query tile size");
        return signalPassFailure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Queried tile size: ";
        llvm::interleaveComma(*tileSize, llvm::dbgs());
        llvm::dbgs() << "\n";
      });

      llvm::Optional<SmallVector<int64_t, 4>> workloadSize =
          getWorkloadSize(funcOp);
      if (!workloadSize) {
        funcOp.emitError("failed to query workload size");
        return signalPassFailure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Queried workload size: ";
        llvm::interleaveComma(*workloadSize, llvm::dbgs());
        llvm::dbgs() << "\n";
      });

      {
        OwningRewritePatternList patterns;
        patterns.insert<ConcretizeWorkgroupIDOp, ConcretizeWorkgroupSizeOp,
                        ConcretizeWorkgroupCountOp>(&context, *workloadSize,
                                                    *tileSize);

        (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
      }

      LLVM_DEBUG({
        llvm::dbgs()
            << "--- After concretizing flow.dispatch.workgroup.* ops ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });

      {
        OwningRewritePatternList patterns;
        patterns.insert<RemoveTripOneLoop>(&context, *workloadSize, *tileSize);

        (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
      }

      // XXX: Hack to add a num_workgroups so we can get this information later.
      {
        IREE::HAL::ExecutableEntryPointOp entryPointOp;
        for (auto op : targetOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
          if (op.sym_name() == funcOp.getName()) {
            entryPointOp = op;
            break;
          }
        }

        if (entryPointOp) {
          SmallVector<int64_t, 4> numWorkgroups;
          for (auto pair : llvm::zip(*workloadSize, *tileSize)) {
            auto total = std::get<0>(pair);
            auto tile = std::get<1>(pair);
            if (total != ShapedType::kDynamicSize && tile != 0) {
              numWorkgroups.push_back(ceilDiv(total, tile));
            } else {
              numWorkgroups.push_back(0);
            }
          }
          entryPointOp.getOperation()->setAttr(
              "num_workgroups", Builder(funcOp).getI64ArrayAttr(numWorkgroups));
        }
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createConcretizeWorkloadOpsPass() {
  return std::make_unique<ConcretizeWorkloadOpsPass>();
}

static PassRegistration<ConcretizeWorkloadOpsPass> pass(
    "iree-codegen-concretize-workload-ops",
    "Replace flow.dispatch.workgroup.* ops with constant values from chosen "
    "tiling and distribution scheme",
    [] { return std::make_unique<ConcretizeWorkloadOpsPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
