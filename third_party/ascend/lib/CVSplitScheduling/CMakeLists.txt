set(LLVM_TARGET_DEFINITIONS VFRewritePatterns.td)
mlir_tablegen(CVSplitVFRewritePatterns.inc -gen-rewriters)
add_public_tablegen_target(CVSplitVFRewriteIncGen)

add_triton_library(CVSplitScheduling
  CVSplitScheduling.cpp
  classifyAllOps.cpp
  DependencyScheduler.cpp
  PreCheck.cpp
  SSAToBufferAllocation.cpp
  VFRewrite.cpp

  DEPENDS CVSplitSchedulingPassIncGen CVSplitVFRewriteIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  BiShengIRAnnotationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
)
