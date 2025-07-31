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
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mmul_majo(%arg0: memref<500xi32>, %arg1: memref<750xi32>, %arg2: memref<600xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
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
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb10
    %5 = arith.addi %c0_i32, %c20_i32 {constant = 20 : i32} : i32
    %6 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb11, ^bb2(%6 : i32)
  ^bb2(%7: i32):  // 2 preds: ^bb1, ^bb9
    %8 = arith.addi %c0_i32_8, %c30_i32 {constant = 30 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb10, ^bb3
  ^bb3:  // pred: ^bb2
    %9 = arith.addi %c0_i32_0, %c0_i32_1 {constant = 0 : i32} : i32
    %10 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    %c0_i32_12 = arith.constant 0 : i32
    %c0_i32_13 = arith.constant 0 : i32
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    %11 = arith.addi %9, %c0_i32_13 : i32
    %12 = arith.muli %4, %c25_i32_11 : i32
    %13 = arith.addi %c0_i32_10, %c25_i32 {constant = 25 : i32} : i32
    %14 = arith.addi %11, %12 : i32
    %15 = arith.muli %11, %c30_i32_9 : i32
    %16 = arith.addi %11, %c1_i32 : i32
    %17 = arith.muli %14, %c4_i32 : i32
    %18 = arith.addi %7, %15 : i32
    cgra.cond_br<ge> [%16 : i32, %13 : i32], ^bb8, ^bb5
  ^bb5:  // pred: ^bb4
    %19 = arith.muli %18, %c4_i32 : i32
    %20 = arith.addi %16, %c0_i32_13 : i32
    %21 = arith.muli %4, %c25_i32_11 : i32
    %22 = arith.addi %c0_i32_10, %c25_i32 {constant = 25 : i32} : i32
    %23 = arith.addi %0, %17 : i32
    %24 = arith.addi %1, %19 : i32
    %25 = arith.addi %20, %21 : i32
    %26 = arith.muli %20, %c30_i32_9 : i32
    %27 = arith.addi %20, %c1_i32 : i32
    %28 = arith.addi %10, %c0_i32_12 : i32
    %29 = cgra.lwi %23 : i32->i32
    %30 = cgra.lwi %24 : i32->i32
    %31 = arith.muli %25, %c4_i32 : i32
    %32 = arith.addi %7, %26 : i32
    cgra.cond_br<ge> [%27 : i32, %22 : i32], ^bb7(%29, %30, %28, %31, %32 : i32, i32, i32, i32, i32), ^bb6(%29, %30, %32, %27, %28, %31 : i32, i32, i32, i32, i32, i32)
  ^bb6(%33: i32, %34: i32, %35: i32, %36: i32, %37: i32, %38: i32):  // 2 preds: ^bb5, ^bb6
    %39 = arith.muli %33, %34 : i32
    %40 = arith.muli %35, %c4_i32 : i32
    %41 = arith.addi %36, %c0_i32_13 : i32
    %42 = arith.muli %4, %c25_i32_11 : i32
    %43 = arith.addi %c0_i32_10, %c25_i32 {constant = 25 : i32} : i32
    %44 = arith.addi %37, %39 : i32
    %45 = arith.addi %0, %38 : i32
    %46 = arith.addi %1, %40 : i32
    %47 = arith.addi %41, %42 : i32
    %48 = arith.muli %41, %c30_i32_9 : i32
    %49 = arith.addi %41, %c1_i32 : i32
    %50 = arith.addi %44, %c0_i32_12 : i32
    %51 = cgra.lwi %45 : i32->i32
    %52 = cgra.lwi %46 : i32->i32
    %53 = arith.muli %47, %c4_i32 : i32
    %54 = arith.addi %7, %48 : i32
    cgra.cond_br<lt> [%49 : i32, %43 : i32], ^bb6(%51, %52, %54, %49, %50, %53 : i32, i32, i32, i32, i32, i32), ^bb7(%51, %52, %50, %53, %54 : i32, i32, i32, i32, i32)
  ^bb7(%55: i32, %56: i32, %57: i32, %58: i32, %59: i32):  // 2 preds: ^bb5, ^bb6
    %60 = arith.muli %55, %56 : i32
    %61 = arith.addi %57, %60 : i32
    %62 = arith.addi %61, %c0_i32_12 : i32
    %63 = arith.addi %0, %58 : i32
    %64 = cgra.lwi %63 : i32->i32
    %65 = arith.muli %59, %c4_i32 : i32
    %66 = arith.addi %1, %65 : i32
    %67 = cgra.lwi %66 : i32->i32
    %68 = arith.muli %64, %67 : i32
    %69 = arith.addi %62, %68 : i32
    cf.br ^bb9(%69 : i32)
  ^bb8:  // pred: ^bb4
    %70 = arith.addi %10, %c0_i32_12 : i32
    %71 = arith.addi %0, %17 : i32
    %72 = cgra.lwi %71 : i32->i32
    %73 = arith.muli %18, %c4_i32 : i32
    %74 = arith.addi %1, %73 : i32
    %75 = cgra.lwi %74 : i32->i32
    %76 = arith.muli %72, %75 : i32
    %77 = arith.addi %70, %76 : i32
    cf.br ^bb9(%77 : i32)
  ^bb9(%78: i32):  // 2 preds: ^bb7, ^bb8
    %79 = arith.muli %4, %c30_i32_9 : i32
    %80 = arith.addi %79, %c0_i32 : i32
    %81 = arith.addi %80, %c0_i32 : i32
    %82 = arith.addi %7, %81 : i32
    %83 = arith.muli %82, %c4_i32 : i32
    %84 = arith.addi %2, %83 : i32
    %85 = arith.addi %84, %c0_i32 : i32
    %86 = arith.addi %85, %c0_i32 : i32
    cgra.swi %78, %86 : i32, i32
    %87 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb2(%87 : i32)
  ^bb10:  // pred: ^bb2
    %88 = arith.addi %4, %c1_i32 : i32
    cf.br ^bb1(%88 : i32)
  ^bb11:  // pred: ^bb1
    return
  }
}

