/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier: MIT */
// Provides overflow-safe byte accounting for the unified slot planner.
// Physical allocation and address-space policy remain in its single lowering.
#pragma once
#include <cstdint>
#include <limits>
namespace mlir::triton::cv_split {
inline bool checkedBufferAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
  result = lhs + rhs;
  return true;
}
inline bool checkedBufferMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs && rhs > std::numeric_limits<uint64_t>::max() / lhs) return false;
  result = lhs * rhs;
  return true;
}
} // namespace mlir::triton::cv_split
