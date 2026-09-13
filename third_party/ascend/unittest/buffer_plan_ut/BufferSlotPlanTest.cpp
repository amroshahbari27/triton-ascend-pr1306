/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "ascend/include/CVSplitScheduling/BufferSlotPlan.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

using namespace mlir::triton::cv_split;

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::cerr << "check failed at line " << __LINE__ << ": " #condition       \
                << '\n';                                                       \
      return 1;                                                                 \
    }                                                                           \
  } while (false)

int main() {
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  uint64_t bytes = 42;
  CHECK(checkedBufferAdd(maximum, 0, bytes) && bytes == maximum);
  bytes = 42;
  CHECK(!checkedBufferAdd(maximum, 1, bytes) && bytes == 42);
  CHECK(checkedBufferMultiply(0, maximum, bytes) && bytes == 0);
  CHECK(checkedBufferMultiply(maximum, 1, bytes) && bytes == maximum);
  bytes = 42;
  CHECK(!checkedBufferMultiply(maximum, 2, bytes) && bytes == 42);
  CHECK(checkedBufferMultiply(128 * 128, 4, bytes) && bytes == 65536);

  using Edge = std::tuple<unsigned, unsigned, bool>;
  for (unsigned lanes : {1u, 2u, 3u, 4u, 8u, 16u}) {
    const unsigned slots = std::min(2u, lanes);
    std::array<std::vector<unsigned>, 2> uses;
    for (unsigned lane = 0; lane < lanes; ++lane) {
      CHECK(rotatingBufferSlot(lane, slots) == lane % slots);
      uses[rotatingBufferSlot(lane, slots)].push_back(lane);
    }
    unsigned edges = 0;
    for (unsigned slot = 0; slot < slots; ++slot) {
      std::vector<Edge> actual;
      forEachCyclicBufferReuse(uses[slot], [&](unsigned from, unsigned to,
                                              bool loopCarried) {
        actual.emplace_back(from, to, loopCarried);
      });
      CHECK(actual.size() == uses[slot].size());
      for (unsigned index = 0; index + 1 < actual.size(); ++index) {
        CHECK(actual[index] == Edge(uses[slot][index], uses[slot][index + 1],
                                    false));
      }
      CHECK(actual.back() == Edge(uses[slot].back(), uses[slot].front(), true));
      edges += actual.size();
    }
    CHECK(edges == lanes);
  }
  std::vector<unsigned> empty;
  unsigned visits = 0;
  forEachCyclicBufferReuse(empty, [&](unsigned, unsigned, bool) { ++visits; });
  CHECK(visits == 0);
  std::cout << "BUFFER_SLOT_PLAN_TEST=PASS\n";
  return 0;
}
