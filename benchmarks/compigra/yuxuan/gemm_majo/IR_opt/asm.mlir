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
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
Set parameter Username
Academic license - for non-commercial use only - expires 2026-07-25
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm_majo(%arg0: memref<500xi32>, %arg1: memref<750xi32>, %arg2: memref<600xi32>, %arg3: memref<1xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb10
    %8 = arith.addi %c0_i32, %c20_i32 {constant = 20 : i32} : i32
    %9 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb11, ^bb2(%9 : i32)
  ^bb2(%10: i32):  // 2 preds: ^bb1, ^bb9
    %11 = arith.addi %c0_i32_9, %c30_i32 {constant = 30 : i32} : i32
    cgra.cond_br<ge> [%10 : i32, %11 : i32], ^bb10, ^bb3
  ^bb3:  // pred: ^bb2
    %12 = arith.addi %c0_i32_0, %c0_i32_1 {constant = 0 : i32} : i32
    %13 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    %c0_i32_13 = arith.constant 0 : i32
    %c0_i32_14 = arith.constant 0 : i32
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    %14 = arith.addi %12, %c0_i32_14 : i32
    %15 = arith.muli %7, %c25_i32_12 : i32
    %16 = arith.addi %c0_i32_11, %c25_i32 {constant = 25 : i32} : i32
    %17 = arith.addi %14, %15 : i32
    %18 = arith.muli %14, %c30_i32_10 : i32
    %19 = arith.addi %14, %c1_i32 : i32
    %20 = arith.muli %17, %c4_i32 : i32
    %21 = arith.addi %10, %18 : i32
    cgra.cond_br<ge> [%19 : i32, %16 : i32], ^bb8, ^bb5
  ^bb5:  // pred: ^bb4
    %22 = arith.muli %21, %c4_i32 : i32
    %23 = arith.addi %19, %c0_i32_14 : i32
    %24 = arith.muli %7, %c25_i32_12 : i32
    %25 = arith.addi %c0_i32_11, %c25_i32 {constant = 25 : i32} : i32
    %26 = arith.addi %3, %20 : i32
    %27 = arith.addi %4, %22 : i32
    %28 = arith.addi %23, %24 : i32
    %29 = arith.muli %23, %c30_i32_10 : i32
    %30 = arith.addi %23, %c1_i32 : i32
    %31 = arith.addi %13, %c0_i32_13 : i32
    %32 = cgra.lwi %26 : i32->i32
    %33 = cgra.lwi %27 : i32->i32
    %34 = arith.muli %28, %c4_i32 : i32
    %35 = arith.addi %10, %29 : i32
    cgra.cond_br<ge> [%30 : i32, %25 : i32], ^bb7(%32, %33, %31, %34, %35 : i32, i32, i32, i32, i32), ^bb6(%32, %33, %35, %30, %31, %34 : i32, i32, i32, i32, i32, i32)
  ^bb6(%36: i32, %37: i32, %38: i32, %39: i32, %40: i32, %41: i32):  // 2 preds: ^bb5, ^bb6
    %42 = arith.muli %36, %37 : i32
    %43 = arith.muli %38, %c4_i32 : i32
    %44 = arith.addi %39, %c0_i32_14 : i32
    %45 = arith.muli %7, %c25_i32_12 : i32
    %46 = arith.addi %c0_i32_11, %c25_i32 {constant = 25 : i32} : i32
    %47 = arith.addi %40, %42 : i32
    %48 = arith.addi %3, %41 : i32
    %49 = arith.addi %4, %43 : i32
    %50 = arith.addi %44, %45 : i32
    %51 = arith.muli %44, %c30_i32_10 : i32
    %52 = arith.addi %44, %c1_i32 : i32
    %53 = arith.addi %47, %c0_i32_13 : i32
    %54 = cgra.lwi %48 : i32->i32
    %55 = cgra.lwi %49 : i32->i32
    %56 = arith.muli %50, %c4_i32 : i32
    %57 = arith.addi %10, %51 : i32
    cgra.cond_br<lt> [%52 : i32, %46 : i32], ^bb6(%54, %55, %57, %52, %53, %56 : i32, i32, i32, i32, i32, i32), ^bb7(%54, %55, %53, %56, %57 : i32, i32, i32, i32, i32)
  ^bb7(%58: i32, %59: i32, %60: i32, %61: i32, %62: i32):  // 2 preds: ^bb5, ^bb6
    %63 = arith.muli %58, %59 : i32
    %64 = arith.addi %60, %63 : i32
    %65 = arith.addi %64, %c0_i32_13 : i32
    %66 = arith.addi %3, %61 : i32
    %67 = cgra.lwi %66 : i32->i32
    %68 = arith.muli %62, %c4_i32 : i32
    %69 = arith.addi %4, %68 : i32
    %70 = cgra.lwi %69 : i32->i32
    %71 = arith.muli %67, %70 : i32
    %72 = arith.addi %65, %71 : i32
    cf.br ^bb9(%72 : i32)
  ^bb8:  // pred: ^bb4
    %73 = arith.addi %13, %c0_i32_13 : i32
    %74 = arith.addi %3, %20 : i32
    %75 = cgra.lwi %74 : i32->i32
    %76 = arith.muli %21, %c4_i32 : i32
    %77 = arith.addi %4, %76 : i32
    %78 = cgra.lwi %77 : i32->i32
    %79 = arith.muli %75, %78 : i32
    %80 = arith.addi %73, %79 : i32
    cf.br ^bb9(%80 : i32)
  ^bb9(%81: i32):  // 2 preds: ^bb7, ^bb8
    %82 = arith.addi %2, %c0_i32 : i32
    %83 = arith.addi %82, %c0_i32 : i32
    %84 = arith.addi %83, %c0_i32 : i32
    %85 = arith.muli %81, %84 : i32
    %86 = arith.muli %7, %c30_i32_10 : i32
    %87 = arith.addi %86, %c0_i32 : i32
    %88 = arith.addi %87, %c0_i32 : i32
    %89 = arith.addi %10, %88 : i32
    %90 = arith.muli %89, %c4_i32 : i32
    %91 = arith.addi %5, %90 : i32
    %92 = cgra.lwi %91 : i32->i32
    %93 = arith.muli %92, %c2123_i32 : i32
    %94 = arith.addi %85, %93 : i32
    %95 = arith.muli %7, %c30_i32_10 : i32
    %96 = arith.addi %95, %c0_i32 : i32
    %97 = arith.addi %96, %c0_i32 : i32
    %98 = arith.addi %10, %97 : i32
    %99 = arith.muli %98, %c4_i32 : i32
    %100 = arith.addi %5, %99 : i32
    cgra.swi %94, %100 : i32, i32
    %101 = arith.addi %10, %c1_i32 : i32
    cf.br ^bb2(%101 : i32)
  ^bb10:  // pred: ^bb2
    %102 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb1(%102 : i32)
  ^bb11:  // pred: ^bb1
    return
  }
}

