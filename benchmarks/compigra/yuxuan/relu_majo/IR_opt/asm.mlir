Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @relu_majo(%arg0: memref<16384xi32>, %arg1: memref<1xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = cgra.lwd -> i32, {BaseAddr = "arg0"}
    %1 = cgra.lwd -> i32, {BaseAddr = "arg1"}
    %2 = cgra.lwi %1 : i32->i32
    %3 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    cf.br ^bb1(%3 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb4
    %5 = arith.addi %2, %c0_i32 : i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb5, ^bb2
  ^bb2:  // pred: ^bb1
    %6 = arith.muli %4, %c4_i32 : i32
    %7 = arith.addi %0, %6 : i32
    %8 = cgra.lwi %7 : i32->i32
    %9 = arith.addi %c0_i32_3, %c0_i32_4 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%8 : i32, %9 : i32], ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    %10 = arith.muli %4, %c4_i32 : i32
    %11 = arith.addi %0, %10 : i32
    %12 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    cgra.swi %12, %11 : i32, i32
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %13 = arith.addi %4, %c1_i32 : i32
    cf.br ^bb1(%13 : i32)
  ^bb5:  // pred: ^bb1
    return
  }
}

