module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d_majo(%arg0: memref<16384xi32>, %arg1: memref<9xi32>, %arg2: memref<16384xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c127_i32_1 = arith.constant 127 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c1_i32_4 = arith.constant 1 : i32
    %c1_i32_5 = arith.constant 1 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %c0_i32_9 = arith.constant 0 : i32
    %c0_i32_10 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32_11 = arith.constant 0 : i32
    %c-1_i32_12 = arith.constant -1 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32_14 = arith.constant 0 : i32
    %c2_i32_15 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c4_i32_16 = arith.constant 4 : i32
    %0 = cgra.lwd -> i32, {BaseAddr = "arg0"}
    %1 = cgra.lwd -> i32, {BaseAddr = "arg1"}
    %2 = cgra.lwd -> i32, {BaseAddr = "arg2"}
    %3 = arith.addi %c0_i32_3, %c1_i32_4 {constant = 1 : i32} : i32
    cf.br ^bb1(%3 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb8
    %5 = arith.addi %c0_i32_0, %c127_i32_1 {constant = 127 : i32} : i32
    %6 = arith.addi %c0_i32_2, %c1_i32 {constant = 1 : i32} : i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb9, ^bb2(%6 : i32)
  ^bb2(%7: i32):  // 2 preds: ^bb1, ^bb7
    %8 = arith.addi %c0_i32, %c127_i32 {constant = 127 : i32} : i32
    %9 = arith.addi %c0_i32_8, %c0_i32_9 {constant = 0 : i32} : i32
    %10 = arith.addi %c0_i32_11, %c-1_i32_12 {constant = -1 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb8, ^bb3(%10, %9 : i32, i32)
  ^bb3(%11: i32, %12: i32):  // 2 preds: ^bb2, ^bb6
    %13 = arith.addi %c0_i32_14, %c2_i32_15 {constant = 2 : i32} : i32
    cgra.cond_br<ge> [%11 : i32, %13 : i32], ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %14 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    %15 = arith.addi %c0_i32_10, %c-1_i32 {constant = -1 : i32} : i32
    cf.br ^bb5(%15, %14 : i32, i32)
  ^bb5(%16: i32, %17: i32):  // 2 preds: ^bb4, ^bb5
    %18 = arith.muli %11, %c128_i32 : i32
    %19 = arith.addi %18, %7 : i32
    %20 = arith.addi %19, %16 : i32
    %21 = arith.muli %4, %c128_i32 : i32
    %22 = arith.addi %20, %21 : i32
    %23 = arith.muli %22, %c4_i32_16 : i32
    %24 = arith.addi %0, %23 : i32
    %25 = cgra.lwi %24 : i32->i32
    %26 = arith.muli %11, %c3_i32 : i32
    %27 = arith.addi %16, %26 : i32
    %28 = arith.addi %27, %c4_i32 : i32
    %29 = arith.muli %28, %c4_i32_16 : i32
    %30 = arith.addi %1, %29 : i32
    %31 = cgra.lwi %30 : i32->i32
    %32 = arith.muli %25, %31 : i32
    %33 = arith.addi %17, %32 : i32
    %34 = arith.addi %16, %c1_i32_5 : i32
    %35 = arith.addi %c0_i32_13, %c2_i32 {constant = 2 : i32} : i32
    cgra.cond_br<lt> [%34 : i32, %35 : i32], ^bb5(%34, %33 : i32, i32), ^bb6
  ^bb6:  // pred: ^bb5
    %36 = arith.addi %12, %33 : i32
    %37 = arith.addi %11, %c1_i32_5 : i32
    cf.br ^bb3(%37, %36 : i32, i32)
  ^bb7:  // pred: ^bb3
    %38 = arith.muli %4, %c128_i32 : i32
    %39 = arith.addi %7, %38 : i32
    %40 = arith.muli %39, %c4_i32_16 : i32
    %41 = arith.addi %2, %40 : i32
    cgra.swi %12, %41 : i32, i32
    %42 = arith.addi %7, %c1_i32_5 : i32
    cf.br ^bb2(%42 : i32)
  ^bb8:  // pred: ^bb2
    %43 = arith.addi %4, %c1_i32_5 : i32
    cf.br ^bb1(%43 : i32)
  ^bb9:  // pred: ^bb1
    return
  }
}

