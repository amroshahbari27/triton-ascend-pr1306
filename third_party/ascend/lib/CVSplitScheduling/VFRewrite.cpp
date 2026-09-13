/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier:
 * MIT */
// Normalizes vector expressions before the shared SSA graph is constructed.
// Optional generated patterns expose SSA opportunities without scheduling.
#include "ascend/include/CVSplitScheduling/Pipeline.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#define DEBUG_TYPE "cv-split-vf-rewrite"
#define LDBG(MSG) LLVM_DEBUG(llvm::dbgs() << "[cv-split] " << MSG << '\n')
using namespace mlir;
namespace {
constexpr int64_t kVectorWidth = 64, kPackWidth = 16;
constexpr llvm::StringLiteral kRole = "cv_split.vf_role";
constexpr llvm::StringLiteral kConsumerGroupStart =
    "cv_split.vf_consumer_group_start";

bool shaped(Value value, int rank) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getRank() == rank;
}

bool isRowNormalizeExpCandidate(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op || op->hasAttr(kRole) || !shaped(value, 2))
    return false;
  auto type = cast<RankedTensorType>(value.getType());
  if (type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      type.getDimSize(1) % kVectorWidth)
    return false;
  auto exp = value.getDefiningOp<math::ExpOp>();
  if (!exp || !exp.getOperand().getDefiningOp<arith::SubFOp>())
    return false;
  bool castUser = false, reduceUser = false;
  for (Operation *user : value.getUsers()) {
    castUser |= isa<arith::TruncFOp>(user);
    reduceUser |= isa<linalg::ReduceOp>(user);
  }
  return castUser && reduceUser;
}

bool isOrderedAffineCandidate(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op || op->hasAttr(kRole) || !shaped(value, 1))
    return false;
  auto add = dyn_cast<arith::AddFOp>(op);
  return add && (add.getLhs().getDefiningOp<arith::MulFOp>() ||
                 add.getRhs().getDefiningOp<arith::MulFOp>());
}

Value cloneWithVFRole(PatternRewriter &rewriter, Value value, StringRef role) {
  Operation *clone = rewriter.clone(*value.getDefiningOp());
  clone->setAttr(kRole, rewriter.getStringAttr(role));
  return clone->getResult(0);
}

#include "CVSplitVFRewritePatterns.inc"

bool isProvenZero(Value value) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    value = fill.getInputs().front();
  return matchPattern(value, m_Zero()) || matchPattern(value, m_AnyZeroFloat());
}

bool isLocalTo(Value value, Block *body) {
  if (Operation *producer = value.getDefiningOp())
    return producer->getBlock() == body;
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner() == body;
}

// Restore an independent matrix result before graph construction.  This is an
// SSA expression rewrite; classification must never mutate the graph it reads.
LogicalResult normalizeDPSAccumulators(scf::ForOp loop) {
  SmallVector<linalg::MatmulOp> matches;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto matmul = dyn_cast<linalg::MatmulOp>(op);
    if (!matmul || matmul.getNumDpsInputs() != 2 ||
        matmul.getNumDpsInits() != 1 || matmul->getNumResults() != 1)
      continue;
    Value init = matmul.getDpsInitOperand(0)->get();
    if (isProvenZero(init))
      continue;
    if (!isLocalTo(init, loop.getBody())) {
      LDBG("VFRewrite rejected an external matrix accumulator");
      return failure();
    }
    auto type = dyn_cast<RankedTensorType>(init.getType());
    if (!type || !type.hasStaticShape() ||
        !isa<FloatType>(type.getElementType()) ||
        matmul.getResult(0).getType() != type)
      return failure();
    matches.push_back(matmul);
  }

  llvm::DenseMap<Type, Value> zeroByType;
  for (linalg::MatmulOp matmul : matches) {
    Value accumulator = matmul.getDpsInitOperand(0)->get();
    Type type = accumulator.getType();
    Value &zero = zeroByType[type];
    OpBuilder builder(matmul);
    if (!zero) {
      auto tensorType = cast<RankedTensorType>(type);
      zero = builder
                 .create<arith::ConstantOp>(
                     matmul.getLoc(), tensorType,
                     DenseElementsAttr::get(
                         tensorType,
                         builder.getZeroAttr(tensorType.getElementType())))
                 .getResult();
    }
    matmul.getDpsInitOperand(0)->set(zero);
    builder.setInsertionPointAfter(matmul);
    auto join = builder.create<arith::AddFOp>(matmul.getLoc(), accumulator,
                                              matmul.getResult(0));
    matmul.getResult(0).replaceAllUsesExcept(join.getResult(), join);
  }
  LDBG("VFRewrite normalized " << matches.size() << " DPS accumulator(s)");
  return success();
}

struct Lane {
  arith::MulFOp scale, scaleDenominator;
  linalg::ReduceOp maximumReduction, sumReduction;
  arith::MaximumFOp maximum;
  linalg::BroadcastOp maximumBroadcast;
  arith::SubFOp shift, alphaDifference;
  math::ExpOp probability, alpha;
  arith::TruncFOp castProbability;
  arith::AddFOp denominator;
  Value score, scaleTensor, oldMaximum, oldDenominator;
};

struct AccumulatorUpdate {
  linalg::BroadcastOp broadcast;
  arith::MulFOp multiply;
  arith::AddFOp add;
  Value previous;
  Value product;
};

Value other(Value lhs, Value rhs, Value known) {
  return lhs == known ? rhs : rhs == known ? lhs : Value();
}

FailureOr<AccumulatorUpdate> matchAccumulatorUpdate(Lane &lane) {
  std::optional<AccumulatorUpdate> match;
  for (Operation *user : lane.alpha->getUsers()) {
    auto broadcast = dyn_cast<linalg::BroadcastOp>(user);
    if (!broadcast || broadcast.getDpsInputs().size() != 1 ||
        broadcast.getDpsInputs().front() != lane.alpha.getResult() ||
        broadcast->getNumResults() != 1 ||
        !shaped(broadcast->getResult(0), 2))
      continue;
    if (!broadcast->getResult(0).hasOneUse())
      return failure();
    auto multiply = dyn_cast<arith::MulFOp>(
        (*broadcast->getResult(0).getUses().begin()).getOwner());
    Value previous = multiply ? other(multiply.getLhs(), multiply.getRhs(),
                                      broadcast->getResult(0))
                              : Value();
    if (!multiply || !previous || !shaped(previous, 2) ||
        !multiply.getResult().hasOneUse())
      return failure();
    auto add = dyn_cast<arith::AddFOp>(
        (*multiply.getResult().getUses().begin()).getOwner());
    Value product = add ? other(add.getLhs(), add.getRhs(), multiply.getResult())
                        : Value();
    if (!add || !product || !shaped(product, 2) ||
        previous.getType() != multiply.getResult().getType() ||
        product.getType() != multiply.getResult().getType() ||
        add.getResult().getType() != multiply.getResult().getType() || match)
      return failure();
    match = AccumulatorUpdate{broadcast, multiply, add, previous, product};
  }
  if (!match)
    return failure();
  for (Operation *user : lane.alpha->getUsers())
    if (user != lane.scaleDenominator.getOperation() &&
        user != match->broadcast.getOperation())
      return failure();
  return *match;
}

FailureOr<Lane> matchLane(triton::cv_split::VFTransferSite site) {
  Lane lane;
  lane.castProbability = site.source.getDefiningOp<arith::TruncFOp>();
  if (!site.source || !site.ready || !lane.castProbability)
    return failure();
  lane.probability = lane.castProbability.getIn().getDefiningOp<math::ExpOp>();
  auto role = lane.probability
                  ? lane.probability->getAttrOfType<StringAttr>(kRole)
                  : StringAttr();
  if (!role || role.getValue() != "row-normalize-exp")
    return failure();
  lane.shift = lane.probability.getOperand().getDefiningOp<arith::SubFOp>();
  lane.scale = lane.shift ? lane.shift.getLhs().getDefiningOp<arith::MulFOp>()
                          : arith::MulFOp();
  lane.maximumBroadcast =
      lane.shift ? lane.shift.getRhs().getDefiningOp<linalg::BroadcastOp>()
                 : linalg::BroadcastOp();
  if (!lane.scale || !lane.maximumBroadcast ||
      !shaped(lane.scale.getResult(), 2))
    return failure();
  lane.maximum = lane.maximumBroadcast.getDpsInputs()[0]
                     .getDefiningOp<arith::MaximumFOp>();
  if (!lane.maximum)
    return failure();
  lane.maximumReduction =
      lane.maximum.getLhs().getDefiningOp<linalg::ReduceOp>();
  if (!lane.maximumReduction)
    lane.maximumReduction =
        lane.maximum.getRhs().getDefiningOp<linalg::ReduceOp>();
  if (!lane.maximumReduction)
    return failure();
  lane.oldMaximum = other(lane.maximum.getLhs(), lane.maximum.getRhs(),
                          lane.maximumReduction.getResult(0));
  for (Operation *user : lane.probability->getUsers())
    if (auto reduction = dyn_cast<linalg::ReduceOp>(user))
      lane.sumReduction = reduction;
  if (!lane.sumReduction || !lane.oldMaximum)
    return failure();
  lane.score = lane.scale.getLhs();
  lane.scaleTensor = lane.scale.getRhs();
  if (lane.score.getDefiningOp<linalg::FillOp>())
    std::swap(lane.score, lane.scaleTensor);
  if (!shaped(lane.score, 2) ||
      lane.score.getType() != lane.scaleTensor.getType())
    return failure();
  Block *block = lane.maximum->getBlock();
  for (Operation &operation : *block) {
    auto difference = dyn_cast<arith::SubFOp>(&operation);
    if (!difference || difference.getLhs() != lane.oldMaximum ||
        difference.getRhs() != lane.maximum.getResult())
      continue;
    for (Operation *user : difference->getUsers())
      if (auto exp = dyn_cast<math::ExpOp>(user)) {
        lane.alphaDifference = difference;
        lane.alpha = exp;
        break;
      }
  }
  if (!lane.alpha)
    return failure();
  for (Operation *user : lane.alpha->getUsers()) {
    auto multiply = dyn_cast<arith::MulFOp>(user);
    if (!multiply)
      continue;
    lane.oldDenominator =
        other(multiply.getLhs(), multiply.getRhs(), lane.alpha.getResult());
    if (!lane.oldDenominator)
      continue;
    for (Operation *mulUser : multiply->getUsers()) {
      auto add = dyn_cast<arith::AddFOp>(mulUser);
      if (add && other(add.getLhs(), add.getRhs(), multiply.getResult()) ==
                     lane.sumReduction.getResult(0)) {
        lane.scaleDenominator = multiply;
        lane.denominator = add;
        break;
      }
    }
  }
  role = lane.denominator ? lane.denominator->getAttrOfType<StringAttr>(kRole)
                          : StringAttr();
  if (!role || role.getValue() != "ordered-affine")
    return failure();
  return lane;
}

Value empty(OpBuilder &builder, Location loc, ArrayRef<int64_t> shape,
            Type element) {
  return builder.create<tensor::EmptyOp>(loc, shape, element);
}

FailureOr<Value> reduce(OpBuilder &builder, Location loc,
                        linalg::ReduceOp model, Value input,
                        RankedTensorType resultType) {
  auto fill = model.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || fill.getInputs().size() != 1)
    return failure();
  Value init = builder
                   .create<linalg::FillOp>(
                       loc, fill.getInputs(),
                       ValueRange{empty(builder, loc, resultType.getShape(),
                                        resultType.getElementType())})
                   .getResult(0);
  IRMapping mapping;
  mapping.map(model.getDpsInputs()[0], input);
  mapping.map(model.getDpsInits()[0], init);
  Operation *copy = builder.clone(*model, mapping);
  copy->getResult(0).setType(resultType);
  return copy->getResult(0);
}

Value extract2(OpBuilder &builder, Location loc, Value input, Value row,
               int64_t column) {
  auto type = cast<RankedTensorType>(input.getType());
  auto slice = RankedTensorType::get({1, kVectorWidth}, type.getElementType());
  return builder
      .create<tensor::ExtractSliceOp>(
          loc, slice, input,
          SmallVector<OpFoldResult>{row, builder.getIndexAttr(column)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(1),
                                    builder.getIndexAttr(kVectorWidth)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(1),
                                    builder.getIndexAttr(1)})
      .getResult();
}

Value insert2(OpBuilder &builder, Location loc, Value slice, Value output,
              Value row, int64_t column) {
  return builder
      .create<tensor::InsertSliceOp>(
          loc, slice, output,
          SmallVector<OpFoldResult>{row, builder.getIndexAttr(column)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(1),
                                    builder.getIndexAttr(kVectorWidth)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(1),
                                    builder.getIndexAttr(1)})
      .getResult();
}

struct LaneResult {
  Value maximum, sum, packed, fullProbability;
};

void setSIMD(scope::ScopeOp scope) {
  OpBuilder builder(scope);
  scope->setAttr("noinline", builder.getUnitAttr());
  scope->setAttr("outline", builder.getBoolAttr(true));
  scope->setAttr("vector_mode", builder.getStringAttr("simd"));
  triton::cv_split::setOpEngineTypeAttr(scope,
                                        triton::cv_split::EngineType::VECTOR);
}

FailureOr<LaneResult> materializeLane(Lane &lane,
                                      triton::cv_split::VFTransferSite &site,
                                      Value sharedScaleRow, bool deferSum) {
  auto scoreType = cast<RankedTensorType>(lane.score.getType());
  int64_t rows = scoreType.getDimSize(0);
  int64_t width = scoreType.getDimSize(1);
  if (rows <= 0 || width <= 0 || width % kVectorWidth || width % kPackWidth)
    return failure();
  Location loc = lane.probability.getLoc();
  Type f32 = scoreType.getElementType();
  Type packedElement =
      cast<RankedTensorType>(site.source.getType()).getElementType();
  auto rowScalar = RankedTensorType::get({1}, f32);
  auto rowVector = RankedTensorType::get({1, kVectorWidth}, f32);
  if (!sharedScaleRow || sharedScaleRow.getType() != rowVector)
    return failure();
  auto maximumType = RankedTensorType::get({rows}, f32);
  auto scoreRowsType = RankedTensorType::get({rows, width}, f32);
  auto packedType = RankedTensorType::get(
      {width / kPackWidth, rows, kPackWidth}, packedElement);

  OpBuilder outer(lane.scale);
  Value maxInit = site.maximumDestination;
  Value scaledInit = site.scaledDestination;
  Value sumInit = site.sumDestination;
  Value packedInit = site.packedDestination;
  if (!maxInit || maxInit.getType() != maximumType || !scaledInit ||
      scaledInit.getType() != scoreRowsType || !sumInit ||
      sumInit.getType() != maximumType || !packedInit ||
      packedInit.getType() != packedType)
    return failure();
  auto vf = outer.create<scope::ScopeOp>(
      loc, TypeRange{maximumType, deferSum ? scoreRowsType : maximumType,
                     packedType});
  vf.getBodyRegion().emplaceBlock();
  setSIMD(vf);
  OpBuilder builder = OpBuilder::atBlockEnd(&vf.getBodyRegion().front());
  Value lower = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value upper = builder.create<arith::ConstantIntOp>(loc, rows, 32);
  Value step = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  auto maxLoop = builder.create<scf::ForOp>(loc, lower, upper, step,
                                            ValueRange{maxInit, scaledInit});
  if (!maxLoop.getBody()->empty())
    maxLoop.getBody()->back().erase();
  OpBuilder maxBuilder = OpBuilder::atBlockEnd(maxLoop.getBody());
  Value row = maxBuilder.create<arith::IndexCastOp>(
      loc, maxBuilder.getIndexType(), maxLoop.getInductionVar());
  Value maxVector;
  Value scaledRows = maxLoop.getRegionIterArgs()[1];
  for (int64_t column = 0; column < width; column += kVectorWidth) {
    Value scaled = maxBuilder.create<arith::MulFOp>(
        loc, extract2(maxBuilder, loc, lane.score, row, column),
        sharedScaleRow);
    scaledRows = insert2(maxBuilder, loc, scaled, scaledRows, row, column);
    maxVector =
        maxVector ? maxBuilder.create<arith::MaximumFOp>(loc, maxVector, scaled)
                        .getResult()
                  : scaled;
  }
  auto maxRow =
      reduce(maxBuilder, loc, lane.maximumReduction, maxVector, rowScalar);
  if (failed(maxRow))
    return failure();
  Value maxRows = maxBuilder.create<tensor::InsertSliceOp>(
      loc, *maxRow, maxLoop.getRegionIterArgs()[0],
      SmallVector<OpFoldResult>{row},
      SmallVector<OpFoldResult>{maxBuilder.getIndexAttr(1)},
      SmallVector<OpFoldResult>{maxBuilder.getIndexAttr(1)});
  maxBuilder.create<scf::YieldOp>(loc, ValueRange{maxRows, scaledRows});

  Value maximum = builder.create<arith::MaximumFOp>(loc, lane.oldMaximum,
                                                    maxLoop.getResult(0));
  auto token = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  auto mark = builder.create<annotation::MarkOp>(loc, token.getResult());
  mark->setAttr("SYNC_IN_VF", builder.getStringAttr("VST_VLD"));
  Value expInit = deferSum ? maxLoop.getResult(1) : sumInit;
  auto expLoop = builder.create<scf::ForOp>(loc, lower, upper, step,
                                            ValueRange{expInit, packedInit});
  if (!expLoop.getBody()->empty())
    expLoop.getBody()->back().erase();
  OpBuilder expBuilder = OpBuilder::atBlockEnd(expLoop.getBody());
  row = expBuilder.create<arith::IndexCastOp>(loc, expBuilder.getIndexType(),
                                              expLoop.getInductionVar());
  Value maximumRow = expBuilder.create<tensor::ExtractSliceOp>(
      loc, rowScalar, maximum, SmallVector<OpFoldResult>{row},
      SmallVector<OpFoldResult>{expBuilder.getIndexAttr(1)},
      SmallVector<OpFoldResult>{expBuilder.getIndexAttr(1)});
  Value maxBroadcast =
      expBuilder
          .create<linalg::BroadcastOp>(
              loc, maximumRow, empty(expBuilder, loc, {1, kVectorWidth}, f32),
              ArrayRef<int64_t>{1})
          ->getResult(0);
  Value aux = expLoop.getRegionIterArgs()[0];
  Value packed = expLoop.getRegionIterArgs()[1];
  Value sumVector;
  for (int64_t column = 0; column < width; column += kVectorWidth) {
    Value scaled = extract2(expBuilder, loc, maxLoop.getResult(1), row, column);
    Value shifted = expBuilder.create<arith::SubFOp>(loc, scaled, maxBroadcast);
    Value probability = expBuilder.create<math::ExpOp>(loc, shifted);
    if (deferSum)
      aux = insert2(expBuilder, loc, probability, aux, row, column);
    else
      sumVector =
          sumVector
              ? expBuilder.create<arith::AddFOp>(loc, sumVector, probability)
                    .getResult()
              : probability;

    auto shapeType = RankedTensorType::get({3}, expBuilder.getI64Type());
    Value shape = expBuilder.create<arith::ConstantOp>(
        loc, shapeType,
        DenseElementsAttr::get(
            shapeType,
            ArrayRef<int64_t>{kVectorWidth / kPackWidth, 1, kPackWidth}));
    auto packedFloatType =
        RankedTensorType::get({kVectorWidth / kPackWidth, 1, kPackWidth}, f32);
    Value packedFloat = expBuilder.create<tensor::ReshapeOp>(
        loc, packedFloatType, probability, shape);
    auto chunkType = RankedTensorType::get(
        {kVectorWidth / kPackWidth, 1, kPackWidth}, packedElement);
    Value chunk =
        expBuilder.create<arith::TruncFOp>(loc, chunkType, packedFloat);
    packed = expBuilder.create<tensor::InsertSliceOp>(
        loc, chunk, packed,
        SmallVector<OpFoldResult>{expBuilder.getIndexAttr(column / kPackWidth),
                                  row, expBuilder.getIndexAttr(0)},
        SmallVector<OpFoldResult>{
            expBuilder.getIndexAttr(kVectorWidth / kPackWidth),
            expBuilder.getIndexAttr(1), expBuilder.getIndexAttr(kPackWidth)},
        SmallVector<OpFoldResult>{expBuilder.getIndexAttr(1),
                                  expBuilder.getIndexAttr(1),
                                  expBuilder.getIndexAttr(1)});
  }
  if (!deferSum) {
    auto sum = reduce(expBuilder, loc, lane.sumReduction, sumVector, rowScalar);
    if (failed(sum))
      return failure();
    aux = expBuilder.create<tensor::InsertSliceOp>(
        loc, *sum, aux, SmallVector<OpFoldResult>{row},
        SmallVector<OpFoldResult>{expBuilder.getIndexAttr(1)},
        SmallVector<OpFoldResult>{expBuilder.getIndexAttr(1)});
  }
  expBuilder.create<scf::YieldOp>(loc, ValueRange{aux, packed});
  builder.create<scope::ReturnOp>(
      loc, ValueRange{maximum, expLoop.getResult(0), expLoop.getResult(1)});

  LaneResult result{vf.getResult(0), deferSum ? Value() : vf.getResult(1),
                    vf.getResult(2), deferSum ? vf.getResult(1) : Value()};
  if (deferSum) {
    OpBuilder deferredBuilder(site.ready);
    if (site.ready->hasTrait<OpTrait::IsTerminator>())
      deferredBuilder.setInsertionPoint(site.ready);
    else
      deferredBuilder.setInsertionPointAfter(site.ready);
    auto deferred =
        deferredBuilder.create<scope::ScopeOp>(loc, TypeRange{maximumType});
    deferred.getBodyRegion().emplaceBlock();
    setSIMD(deferred);
    OpBuilder deferredBody =
        OpBuilder::atBlockEnd(&deferred.getBodyRegion().front());
    Value deferredLower = deferredBody.create<arith::ConstantIntOp>(loc, 0, 32);
    Value deferredUpper =
        deferredBody.create<arith::ConstantIntOp>(loc, rows, 32);
    Value deferredStep = deferredBody.create<arith::ConstantIntOp>(loc, 1, 32);
    auto loop = deferredBody.create<scf::ForOp>(
        loc, deferredLower, deferredUpper, deferredStep, ValueRange{sumInit});
    if (!loop.getBody()->empty())
      loop.getBody()->back().erase();
    OpBuilder reduceBuilder = OpBuilder::atBlockEnd(loop.getBody());
    Value deferredRow = reduceBuilder.create<arith::IndexCastOp>(
        loc, reduceBuilder.getIndexType(), loop.getInductionVar());
    Value deferredSum;
    for (int64_t column = 0; column < width; column += kVectorWidth) {
      Value chunk = extract2(reduceBuilder, loc, result.fullProbability,
                             deferredRow, column);
      deferredSum =
          deferredSum
              ? reduceBuilder.create<arith::AddFOp>(loc, deferredSum, chunk)
                    .getResult()
              : chunk;
    }
    auto sum =
        reduce(reduceBuilder, loc, lane.sumReduction, deferredSum, rowScalar);
    if (failed(sum))
      return failure();
    Value rowsValue = reduceBuilder.create<tensor::InsertSliceOp>(
        loc, *sum, loop.getRegionIterArgs()[0],
        SmallVector<OpFoldResult>{deferredRow},
        SmallVector<OpFoldResult>{reduceBuilder.getIndexAttr(1)},
        SmallVector<OpFoldResult>{reduceBuilder.getIndexAttr(1)});
    reduceBuilder.create<scf::YieldOp>(loc, rowsValue);
    deferredBody.create<scope::ReturnOp>(loc, loop.getResult(0));
    result.sum = deferred.getResult(0);
  }

  lane.maximum.getResult().replaceAllUsesWith(result.maximum);
  lane.sumReduction.getResult(0).replaceAllUsesWith(result.sum);
  site.consumedInput = lane.score;
  site.consumptionComplete = vf;
  site.source = result.packed;
  for (Operation *operation :
       {lane.castProbability.getOperation(), lane.sumReduction.getOperation(),
        lane.probability.getOperation(), lane.shift.getOperation(),
        lane.maximumBroadcast.getOperation(), lane.maximum.getOperation(),
        lane.maximumReduction.getOperation(), lane.scale.getOperation()})
    if (operation->use_empty())
      operation->erase();
  return result;
}

LogicalResult
materializeGroup(MutableArrayRef<triton::cv_split::VFTransferSite> sites) {
  if (sites.size() < 2)
    return success();
  SmallVector<Lane> lanes;
  for (triton::cv_split::VFTransferSite site : sites) {
    auto lane = matchLane(site);
    if (failed(lane)) {
      LDBG("VFRewrite row materializer did not match the transfer group");
      return success();
    }
    lanes.push_back(*lane);
  }
  for (unsigned index = 1; index < lanes.size(); ++index)
    if (lanes[index].oldMaximum != lanes[index - 1].maximum.getResult() ||
        lanes[index].oldDenominator != lanes[index - 1].denominator.getResult())
      return success();

  Block *block = lanes.front().alpha->getBlock();
  SmallVector<AccumulatorUpdate> accumulatorUpdates;
  for (Lane &lane : lanes) {
    auto update = matchAccumulatorUpdate(lane);
    if (failed(update)) {
      LDBG("VFRewrite rejected a non-canonical accumulator update");
      return success();
    }
    Operation *product = update->product.getDefiningOp();
    if (!block || update->broadcast->getBlock() != block ||
        update->multiply->getBlock() != block ||
        update->add->getBlock() != block || !product ||
        product->getBlock() != block) {
      LDBG("VFRewrite rejected an accumulator update outside the loop body");
      return success();
    }
    for (const AccumulatorUpdate &previous : accumulatorUpdates)
      if (previous.product == update->product) {
        LDBG("VFRewrite rejected a reused direct product operand");
        return success();
      }
    accumulatorUpdates.push_back(*update);
  }
  for (unsigned index = 0; index < accumulatorUpdates.size(); ++index) {
    AccumulatorUpdate &update = accumulatorUpdates[index];
    if (index &&
        update.previous != accumulatorUpdates[index - 1].add.getResult()) {
      LDBG("VFRewrite rejected a broken accumulator recurrence");
      return success();
    }
    if (!update.add.getResult().hasOneUse()) {
      LDBG("VFRewrite rejected a multiply-used accumulator result");
      return success();
    }
    Operation *consumer = (*update.add.getResult().getUses().begin()).getOwner();
    if (index + 1 < accumulatorUpdates.size()
            ? consumer != accumulatorUpdates[index + 1].multiply.getOperation()
            : !consumer->hasTrait<OpTrait::IsTerminator>()) {
      LDBG("VFRewrite rejected a non-linear accumulator recurrence");
      return success();
    }
  }

  auto scaleFill = lanes.front().scaleTensor.getDefiningOp<linalg::FillOp>();
  if (!scaleFill || scaleFill.getInputs().size() != 1)
    return success();
  Value scaleScalar = scaleFill.getInputs().front();
  for (Lane &lane : lanes) {
    auto fill = lane.scaleTensor.getDefiningOp<linalg::FillOp>();
    if (!fill || fill.getInputs().size() != 1 ||
        fill.getInputs().front() != scaleScalar)
      return success();
  }
  OpBuilder scaleBuilder(lanes.front().scale);
  Type element =
      cast<RankedTensorType>(lanes.front().score.getType()).getElementType();
  Value sharedScaleRow =
      scaleBuilder
          .create<linalg::FillOp>(
              lanes.front().scale.getLoc(), ValueRange{scaleScalar},
              ValueRange{empty(scaleBuilder, lanes.front().scale.getLoc(),
                               {1, kVectorWidth}, element)})
          .getResult(0);

  SmallVector<LaneResult> values;
  for (auto [index, lane] : llvm::enumerate(lanes)) {
    auto result = materializeLane(lane, sites[index], sharedScaleRow,
                                  index + 1 == lanes.size());
    if (failed(result))
      return failure();
    values.push_back(*result);
    if (index + 1 < lanes.size())
      lanes[index + 1].oldMaximum = result->maximum;
  }

  Location loc = lanes.back().alpha.getLoc();
  auto type = cast<RankedTensorType>(values.front().maximum.getType());
  llvm::SmallPtrSet<Operation *, 16> recurrence;
  for (Lane &lane : lanes) {
    recurrence.insert(lane.alphaDifference);
    recurrence.insert(lane.alpha);
    recurrence.insert(lane.scaleDenominator);
    recurrence.insert(lane.denominator);
  }
  SmallVector<OpOperand *> outside;
  for (OpOperand &use : lanes.back().denominator.getResult().getUses())
    if (!recurrence.contains(use.getOwner()))
      outside.push_back(&use);
  if (outside.empty() || llvm::any_of(outside, [&](OpOperand *use) {
        return use->getOwner() != outside.front()->getOwner();
      }))
    return failure();

  // The grouped scope consumes every lane maximum, so it must be placed after
  // the final lane has produced its value.
  Operation *alphaAnchor = values.back().sum.getDefiningOp();
  if (!alphaAnchor || !block || alphaAnchor->getBlock() != block)
    return failure();
  OpBuilder alphaBuilder(alphaAnchor);
  alphaBuilder.setInsertionPointAfter(alphaAnchor);
  SmallVector<Type> alphaTypes(lanes.size(), type);
  auto alphaScope =
      alphaBuilder.create<scope::ScopeOp>(loc, TypeRange(alphaTypes));
  alphaScope.getBodyRegion().emplaceBlock();
  setSIMD(alphaScope);
  OpBuilder alphaBody =
      OpBuilder::atBlockEnd(&alphaScope.getBodyRegion().front());
  SmallVector<Value> alphas;
  Value previousMaximum = lanes.front().oldMaximum;
  for (LaneResult value : values) {
    Value difference =
        alphaBody.create<arith::SubFOp>(loc, previousMaximum, value.maximum);
    alphas.push_back(alphaBody.create<math::ExpOp>(loc, difference));
    previousMaximum = value.maximum;
  }
  alphaBody.create<scope::ReturnOp>(loc, alphas);

  // Keep each pure-SSA accumulator update contiguous.  The dependency
  // scheduler may legally hoist a later alpha broadcast, but doing so extends
  // a full matrix temporary across the preceding update after bufferization.
  Operation *cursor = alphaScope;
  for (AccumulatorUpdate &update : accumulatorUpdates) {
    Operation *product = update.product.getDefiningOp();
    if (cursor->isBeforeInBlock(product))
      cursor = product;
    update.broadcast->setAttr(kConsumerGroupStart,
                              UnitAttr::get(update.broadcast.getContext()));
    for (Operation *operation : {update.broadcast.getOperation(),
                                 update.multiply.getOperation(),
                                 update.add.getOperation()}) {
      operation->moveAfter(cursor);
      cursor = operation;
    }
  }
  LDBG("VFRewrite serialized " << accumulatorUpdates.size()
                                << " pure-SSA accumulator updates");

  OpBuilder affineBuilder(outside.front()->getOwner());
  auto affine = affineBuilder.create<scope::ScopeOp>(loc, TypeRange{type});
  affine.getBodyRegion().emplaceBlock();
  setSIMD(affine);
  OpBuilder affineBody = OpBuilder::atBlockEnd(&affine.getBodyRegion().front());
  SmallVector<std::pair<Value, Value>> segments;
  for (auto [index, value] : llvm::enumerate(values))
    segments.push_back({alphaScope.getResult(index), value.sum});
  while (segments.size() > 1) {
    SmallVector<std::pair<Value, Value>> next;
    for (unsigned index = 0; index < segments.size(); index += 2) {
      if (index + 1 == segments.size()) {
        next.push_back(segments[index]);
        continue;
      }
      auto left = segments[index];
      auto right = segments[index + 1];
      Value scale =
          affineBody.create<arith::MulFOp>(loc, left.first, right.first);
      Value offset = affineBody.create<arith::AddFOp>(
          loc, affineBody.create<arith::MulFOp>(loc, left.second, right.first),
          right.second);
      next.push_back({scale, offset});
    }
    segments = std::move(next);
  }
  Value result = affineBody.create<arith::AddFOp>(
      loc,
      affineBody.create<arith::MulFOp>(loc, lanes.front().oldDenominator,
                                       segments.front().first),
      segments.front().second);
  affineBody.create<scope::ReturnOp>(loc, result);
  for (OpOperand *use : outside)
    use->set(affine.getResult(0));
  for (auto [index, lane] : llvm::enumerate(lanes))
    lane.alpha.getResult().replaceAllUsesWith(alphaScope.getResult(index));
  for (Lane &lane : llvm::reverse(lanes))
    for (Operation *operation :
         {lane.denominator.getOperation(), lane.scaleDenominator.getOperation(),
          lane.alpha.getOperation(), lane.alphaDifference.getOperation()})
      if (operation->use_empty())
        operation->erase();
  LDBG("VFRewrite materialized " << lanes.size()
                                 << " row-SIMD lanes and affine tree");
  return success();
}
} // namespace
LogicalResult mlir::triton::cv_split::applyVFRewriteStage(scf::ForOp loop,
                                                          bool enabled) {
  if (!loop || failed(normalizeDPSAccumulators(loop)))
    return failure();
  if (!enabled) {
    LDBG("VFRewrite disabled");
    return success();
  }
  LDBG("VFRewrite applying generated patterns");
  RewritePatternSet patterns(loop.getContext());
  populateWithGenerated(patterns);
  FrozenRewritePatternSet frozen(std::move(patterns));
  walkAndApplyPatterns(loop.getOperation(), frozen);
  LDBG("VFRewrite completed");
  return success();
}

LogicalResult
mlir::triton::cv_split::materializeOptionalVFRewritesAfterRowSplit(
    MutableArrayRef<VFTransferSite> sites) {
  return materializeGroup(sites);
}
