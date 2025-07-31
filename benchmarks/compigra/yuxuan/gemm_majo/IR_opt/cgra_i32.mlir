module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm_majo(%arg0: memref<500xi32>, %arg1: memref<750xi32>, %arg2: memref<600xi32>, %arg3: memref<1xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %c3740_i32 = arith.constant 3740 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %0 = arith.addi %c7_i32, %c0_i32_8 : i32
    %c12_i32 = arith.constant 12 : i32
    %1 = arith.shli %0, %c12_i32 : i32
    %2 = arith.addi %c3740_i32, %1 {constant = 32412 : i32} : i32
    %c2123_i32 = arith.constant 2123 : i32
    %c0_i32_9 = arith.constant 0 : i32
    %c30_i32 = arith.constant 30 : i32
    %c30_i32_10 = arith.constant 30 : i32
    %c0_i32_11 = arith.constant 0 : i32
    %c25_i32 = arith.constant 25 : i32
    %c25_i32_12 = arith.constant 25 : i32
    %c4_i32 = arith.constant 4 : i32
    %3 = cgra.lwd -> i32, {BaseAddr = "arg0"}
    %4 = cgra.lwd -> i32, {BaseAddr = "arg1"}
    %5 = cgra.lwd -> i32, {BaseAddr = "arg2"}
    %6 = arith.addi %c0_i32_4, %c0_i32_5 {constant = 0 : i32} : i32
    cf.br ^bb1(%6 : i32)
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb6
    %8 = arith.addi %c0_i32, %c20_i32 {constant = 20 : i32} : i32
    %9 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb7, ^bb2(%9 : i32)
  ^bb2(%10: i32):  // 2 preds: ^bb1, ^bb5
    %11 = arith.addi %c0_i32_9, %c30_i32 {constant = 30 : i32} : i32
    cgra.cond_br<ge> [%10 : i32, %11 : i32], ^bb6, ^bb3
  ^bb3:  // pred: ^bb2
    %12 = arith.addi %c0_i32_0, %c0_i32_1 {constant = 0 : i32} : i32
    %13 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    cf.br ^bb4(%12, %13 : i32, i32)
  ^bb4(%14: i32, %15: i32):  // 2 preds: ^bb3, ^bb4
    %16 = arith.muli %7, %c25_i32_12 : i32
    %17 = arith.addi %14, %16 : i32
    %18 = arith.muli %17, %c4_i32 : i32
    %19 = arith.addi %3, %18 : i32
    %20 = cgra.lwi %19 : i32->i32
    %21 = arith.muli %14, %c30_i32_10 : i32
    %22 = arith.addi %10, %21 : i32
    %23 = arith.muli %22, %c4_i32 : i32
    %24 = arith.addi %4, %23 : i32
    %25 = cgra.lwi %24 : i32->i32
    %26 = arith.muli %20, %25 : i32
    %27 = arith.addi %15, %26 : i32
    %28 = arith.addi %14, %c1_i32 : i32
    %29 = arith.addi %c0_i32_11, %c25_i32 {constant = 25 : i32} : i32
    cgra.cond_br<lt> [%28 : i32, %29 : i32], ^bb4(%28, %27 : i32, i32), ^bb5
  ^bb5:  // pred: ^bb4
    %30 = arith.muli %27, %2 : i32
    %31 = arith.muli %7, %c30_i32_10 : i32
    %32 = arith.addi %10, %31 : i32
    %33 = arith.muli %32, %c4_i32 : i32
    %34 = arith.addi %5, %33 : i32
    %35 = cgra.lwi %34 : i32->i32
    %36 = arith.muli %35, %c2123_i32 : i32
    %37 = arith.addi %30, %36 : i32
    %38 = arith.muli %7, %c30_i32_10 : i32
    %39 = arith.addi %10, %38 : i32
    %40 = arith.muli %39, %c4_i32 : i32
    %41 = arith.addi %5, %40 : i32
    cgra.swi %37, %41 : i32, i32
    %42 = arith.addi %10, %c1_i32 : i32
    cf.br ^bb2(%42 : i32)
  ^bb6:  // pred: ^bb2
    %43 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb1(%43 : i32)
  ^bb7:  // pred: ^bb1
    return
  }
}

