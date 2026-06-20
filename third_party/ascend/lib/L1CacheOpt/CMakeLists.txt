add_triton_library(L1CacheOpt
  L1CacheOptPass.cpp
  L1CacheAnalysis.cpp
  L1CacheTransform.cpp

  DEPENDS
  L1CacheOptConversionPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  BiShengIRAnnotationDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRBufferizationDialect
  MLIRLinalgDialect
  MLIRTensorDialect
)
