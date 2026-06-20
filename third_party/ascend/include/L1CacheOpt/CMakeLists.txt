set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name L1CacheOpt)
add_public_tablegen_target(L1CacheOptConversionPassIncGen)
