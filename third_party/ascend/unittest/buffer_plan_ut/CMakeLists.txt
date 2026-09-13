# These memory-neutral primitives do not require a device, MLIR libraries or
# the optional cost model. Keep their unit test usable in a standalone build.
add_executable(CVSplitBufferSlotPlanTest BufferSlotPlanTest.cpp)
target_compile_features(CVSplitBufferSlotPlanTest PRIVATE cxx_std_17)
target_include_directories(CVSplitBufferSlotPlanTest PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../../..")
add_test(NAME CVSplitBufferSlotPlan COMMAND CVSplitBufferSlotPlanTest)
