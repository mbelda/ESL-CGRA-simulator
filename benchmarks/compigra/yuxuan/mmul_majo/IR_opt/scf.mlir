module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mmul_majo(%arg0: memref<500xi32>, %arg1: memref<750xi32>, %arg2: memref<600xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c20 = arith.constant 20 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c20 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c30 = arith.constant 30 : index
      %c1_1 = arith.constant 1 : index
      scf.for %arg4 = %c0_0 to %c30 step %c1_1 {
        %c0_2 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_3 = arith.constant 1 : index
        %0 = scf.for %arg5 = %c0_2 to %c25 step %c1_3 iter_args(%arg6 = %c0_i32) -> (i32) {
          %c25_5 = arith.constant 25 : index
          %3 = arith.muli %arg3, %c25_5 : index
          %4 = arith.addi %arg5, %3 : index
          %5 = memref.load %arg0[%4] : memref<500xi32>
          %c30_6 = arith.constant 30 : index
          %6 = arith.muli %arg5, %c30_6 : index
          %7 = arith.addi %arg4, %6 : index
          %8 = memref.load %arg1[%7] : memref<750xi32>
          %9 = arith.muli %5, %8 : i32
          %10 = arith.addi %arg6, %9 : i32
          scf.yield %10 : i32
        }
        %c30_4 = arith.constant 30 : index
        %1 = arith.muli %arg3, %c30_4 : index
        %2 = arith.addi %arg4, %1 : index
        memref.store %0, %arg2[%2] : memref<600xi32>
      }
    }
    return
  }
}

