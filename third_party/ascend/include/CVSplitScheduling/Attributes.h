/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier: MIT */
// Shared constants identify successful lowering and supported transfer geometry.
// Scheduling identity lives only in the canonical in-memory SSA graph.
#pragma once
#include "llvm/ADT/StringRef.h"
#include <cstdint>
namespace mlir::triton::cv_split {
inline constexpr char kAppliedAttr[] = "triton_ascend.cv_split_scheduling.applied";
// Successful CV-split lowering already owns sub-block mapping and the
// cross-engine event schedule.  The backend must preserve that contract.
inline constexpr char kPreserveExplicitScheduleAttr[] =
    "triton_ascend.cv_split_scheduling.preserve_explicit_schedule";
inline constexpr int64_t kNzTileSize = 16;
} // namespace mlir::triton::cv_split
