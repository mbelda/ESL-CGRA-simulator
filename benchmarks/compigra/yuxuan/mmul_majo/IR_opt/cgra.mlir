module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mmul_majo(%arg0: memref<500xi32>, %arg1: memref<750xi32>, %arg2: memref<600xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c30 = arith.constant 30 : index
    %c25 = arith.constant 25 : index
    %c4_i32 = arith.constant 4 : i32
    %0 = cgra.lwd -> i32, {BaseAddr = "arg0"}
    %1 = cgra.lwd -> i32, {BaseAddr = "arg1"}
    %2 = cgra.lwd -> i32, {BaseAddr = "arg2"}
    cf.br ^bb1(%c0 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb6
    %4 = arith.index_cast %3 : index to i32
    %5 = arith.index_cast %c20 : index to i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb7, ^bb2(%c0 : index)
  ^bb2(%6: index):  // 2 preds: ^bb1, ^bb5
    %7 = arith.index_cast %6 : index to i32
    %8 = arith.index_cast %c30 : index to i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb6, ^bb3
  ^bb3:  // pred: ^bb2
    cf.br ^bb4(%c0, %c0_i32 : index, i32)
  ^bb4(%9: index, %10: i32):  // 2 preds: ^bb3, ^bb4
    %11 = arith.muli %3, %c25 : index
    %12 = arith.addi %9, %11 : index
    %13 = arith.index_cast %12 : index to i32
    %14 = arith.muli %13, %c4_i32 : i32
    %15 = arith.addi %0, %14 : i32
    %16 = cgra.lwi %15 : i32->i32
    %17 = arith.muli %9, %c30 : index
    %18 = arith.addi %6, %17 : index
    %19 = arith.index_cast %18 : index to i32
    %20 = arith.muli %19, %c4_i32 : i32
    %21 = arith.addi %1, %20 : i32
    %22 = cgra.lwi %21 : i32->i32
    %23 = arith.muli %16, %22 : i32
    %24 = arith.addi %10, %23 : i32
    %25 = arith.addi %9, %c1 : index
    %26 = arith.index_cast %25 : index to i32
    %27 = arith.index_cast %c25 : index to i32
    cgra.cond_br<lt> [%26 : i32, %27 : i32], ^bb4(%25, %24 : index, i32), ^bb5
  ^bb5:  // pred: ^bb4
    %28 = arith.muli %3, %c30 : index
    %29 = arith.addi %6, %28 : index
    %30 = arith.index_cast %29 : index to i32
    %31 = arith.muli %30, %c4_i32 : i32
    %32 = arith.addi %2, %31 : i32
    cgra.swi %24, %32 : i32, i32
    %33 = arith.addi %6, %c1 : index
    cf.br ^bb2(%33 : index)
  ^bb6:  // pred: ^bb2
    %34 = arith.addi %3, %c1 : index
    cf.br ^bb1(%34 : index)
  ^bb7:  // pred: ^bb1
    return
  }
}

