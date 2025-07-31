module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm_majo(%arg0: memref<500xi32>, %arg1: memref<750xi32>, %arg2: memref<600xi32>, %arg3: memref<1xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2123_i32 = arith.constant 2123 : i32
    %c32412_i32 = arith.constant 32412 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 20 {
      affine.for %arg5 = 0 to 30 {
        %0 = affine.for %arg6 = 0 to 25 iter_args(%arg7 = %c0_i32) -> (i32) {
          %5 = affine.load %arg0[%arg6 + %arg4 * 25] : memref<500xi32>
          %6 = affine.load %arg1[%arg5 + %arg6 * 30] : memref<750xi32>
          %7 = arith.muli %5, %6 : i32
          %8 = arith.addi %arg7, %7 : i32
          affine.yield %8 : i32
        }
        %1 = arith.muli %0, %c32412_i32 : i32
        %2 = affine.load %arg2[%arg5 + %arg4 * 30] : memref<600xi32>
        %3 = arith.muli %2, %c2123_i32 : i32
        %4 = arith.addi %1, %3 : i32
        affine.store %4, %arg2[%arg5 + %arg4 * 30] : memref<600xi32>
      }
    }
    return
  }
}
