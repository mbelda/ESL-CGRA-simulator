module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mmul_majo(%arg0: memref<500xi32>, %arg1: memref<750xi32>, %arg2: memref<600xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c20_i32 = arith.constant 20 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %c30_i32 = arith.constant 30 : i32
    %c30_i32_9 = arith.constant 30 : i32
    %c0_i32_10 = arith.constant 0 : i32
    %c25_i32 = arith.constant 25 : i32
    %c25_i32_11 = arith.constant 25 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = cgra.lwd -> i32, {BaseAddr = "arg0"}
    %1 = cgra.lwd -> i32, {BaseAddr = "arg1"}
    %2 = cgra.lwd -> i32, {BaseAddr = "arg2"}
    %3 = arith.addi %c0_i32_4, %c0_i32_5 {constant = 0 : i32} : i32
    cf.br ^bb1(%3 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb6
    %5 = arith.addi %c0_i32, %c20_i32 {constant = 20 : i32} : i32
    %6 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb7, ^bb2(%6 : i32)
  ^bb2(%7: i32):  // 2 preds: ^bb1, ^bb5
    %8 = arith.addi %c0_i32_8, %c30_i32 {constant = 30 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb6, ^bb3
  ^bb3:  // pred: ^bb2
    %9 = arith.addi %c0_i32_0, %c0_i32_1 {constant = 0 : i32} : i32
    %10 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    cf.br ^bb4(%9, %10 : i32, i32)
  ^bb4(%11: i32, %12: i32):  // 2 preds: ^bb3, ^bb4
    %13 = arith.muli %4, %c25_i32_11 : i32
    %14 = arith.addi %11, %13 : i32
    %15 = arith.muli %14, %c4_i32 : i32
    %16 = arith.addi %0, %15 : i32
    %17 = cgra.lwi %16 : i32->i32
    %18 = arith.muli %11, %c30_i32_9 : i32
    %19 = arith.addi %7, %18 : i32
    %20 = arith.muli %19, %c4_i32 : i32
    %21 = arith.addi %1, %20 : i32
    %22 = cgra.lwi %21 : i32->i32
    %23 = arith.muli %17, %22 : i32
    %24 = arith.addi %12, %23 : i32
    %25 = arith.addi %11, %c1_i32 : i32
    %26 = arith.addi %c0_i32_10, %c25_i32 {constant = 25 : i32} : i32
    cgra.cond_br<lt> [%25 : i32, %26 : i32], ^bb4(%25, %24 : i32, i32), ^bb5
  ^bb5:  // pred: ^bb4
    %27 = arith.muli %4, %c30_i32_9 : i32
    %28 = arith.addi %7, %27 : i32
    %29 = arith.muli %28, %c4_i32 : i32
    %30 = arith.addi %2, %29 : i32
    cgra.swi %24, %30 : i32, i32
    %31 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb2(%31 : i32)
  ^bb6:  // pred: ^bb2
    %32 = arith.addi %4, %c1_i32 : i32
    cf.br ^bb1(%32 : i32)
  ^bb7:  // pred: ^bb1
    return
  }
}

