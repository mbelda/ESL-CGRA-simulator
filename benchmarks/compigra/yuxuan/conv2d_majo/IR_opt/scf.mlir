module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d_majo(%arg0: memref<16384xi32>, %arg1: memref<9xi32>, %arg2: memref<16384xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c127 = arith.constant 127 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg3 = %c1 to %c127 step %c1_0 {
      %c1_1 = arith.constant 1 : index
      %c127_2 = arith.constant 127 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg4 = %c1_1 to %c127_2 step %c1_3 {
        %c-1 = arith.constant -1 : index
        %c2 = arith.constant 2 : index
        %c1_4 = arith.constant 1 : index
        %0 = scf.for %arg5 = %c-1 to %c2 step %c1_4 iter_args(%arg6 = %c0_i32) -> (i32) {
          %c-1_5 = arith.constant -1 : index
          %c2_6 = arith.constant 2 : index
          %c1_7 = arith.constant 1 : index
          %3 = scf.for %arg7 = %c-1_5 to %c2_6 step %c1_7 iter_args(%arg8 = %c0_i32) -> (i32) {
            %c128_8 = arith.constant 128 : index
            %5 = arith.muli %arg5, %c128_8 : index
            %6 = arith.addi %5, %arg4 : index
            %7 = arith.addi %6, %arg7 : index
            %c128_9 = arith.constant 128 : index
            %8 = arith.muli %arg3, %c128_9 : index
            %9 = arith.addi %7, %8 : index
            %10 = memref.load %arg0[%9] : memref<16384xi32>
            %c3 = arith.constant 3 : index
            %11 = arith.muli %arg5, %c3 : index
            %12 = arith.addi %arg7, %11 : index
            %c4 = arith.constant 4 : index
            %13 = arith.addi %12, %c4 : index
            %14 = memref.load %arg1[%13] : memref<9xi32>
            %15 = arith.muli %10, %14 : i32
            %16 = arith.addi %arg8, %15 : i32
            scf.yield %16 : i32
          }
          %4 = arith.addi %arg6, %3 : i32
          scf.yield %4 : i32
        }
        %c128 = arith.constant 128 : index
        %1 = arith.muli %arg3, %c128 : index
        %2 = arith.addi %arg4, %1 : index
        memref.store %0, %arg2[%2] : memref<16384xi32>
      }
    }
    return
  }
}

