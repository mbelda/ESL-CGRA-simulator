module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d_majo(%arg0: memref<16384xi32>, %arg1: memref<9xi32>, %arg2: memref<16384xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c127 = arith.constant 127 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c4_i32 = arith.constant 4 : i32
    %0 = cgra.lwd -> i32, {BaseAddr = "arg0"}
    %1 = cgra.lwd -> i32, {BaseAddr = "arg1"}
    %2 = cgra.lwd -> i32, {BaseAddr = "arg2"}
    cf.br ^bb1(%c1 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb8
    %4 = arith.index_cast %3 : index to i32
    %5 = arith.index_cast %c127 : index to i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb9, ^bb2(%c1 : index)
  ^bb2(%6: index):  // 2 preds: ^bb1, ^bb7
    %7 = arith.index_cast %6 : index to i32
    %8 = arith.index_cast %c127 : index to i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb8, ^bb3(%c-1, %c0_i32 : index, i32)
  ^bb3(%9: index, %10: i32):  // 2 preds: ^bb2, ^bb6
    %11 = arith.index_cast %9 : index to i32
    %12 = arith.index_cast %c2 : index to i32
    cgra.cond_br<ge> [%11 : i32, %12 : i32], ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%c-1, %c0_i32 : index, i32)
  ^bb5(%13: index, %14: i32):  // 2 preds: ^bb4, ^bb5
    %15 = arith.muli %9, %c128 : index
    %16 = arith.addi %15, %6 : index
    %17 = arith.addi %16, %13 : index
    %18 = arith.muli %3, %c128 : index
    %19 = arith.addi %17, %18 : index
    %20 = arith.index_cast %19 : index to i32
    %21 = arith.muli %20, %c4_i32 : i32
    %22 = arith.addi %0, %21 : i32
    %23 = cgra.lwi %22 : i32->i32
    %24 = arith.muli %9, %c3 : index
    %25 = arith.addi %13, %24 : index
    %26 = arith.addi %25, %c4 : index
    %27 = arith.index_cast %26 : index to i32
    %28 = arith.muli %27, %c4_i32 : i32
    %29 = arith.addi %1, %28 : i32
    %30 = cgra.lwi %29 : i32->i32
    %31 = arith.muli %23, %30 : i32
    %32 = arith.addi %14, %31 : i32
    %33 = arith.addi %13, %c1 : index
    %34 = arith.index_cast %33 : index to i32
    %35 = arith.index_cast %c2 : index to i32
    cgra.cond_br<lt> [%34 : i32, %35 : i32], ^bb5(%33, %32 : index, i32), ^bb6
  ^bb6:  // pred: ^bb5
    %36 = arith.addi %10, %32 : i32
    %37 = arith.addi %9, %c1 : index
    cf.br ^bb3(%37, %36 : index, i32)
  ^bb7:  // pred: ^bb3
    %38 = arith.muli %3, %c128 : index
    %39 = arith.addi %6, %38 : index
    %40 = arith.index_cast %39 : index to i32
    %41 = arith.muli %40, %c4_i32 : i32
    %42 = arith.addi %2, %41 : i32
    cgra.swi %10, %42 : i32, i32
    %43 = arith.addi %6, %c1 : index
    cf.br ^bb2(%43 : index)
  ^bb8:  // pred: ^bb2
    %44 = arith.addi %3, %c1 : index
    cf.br ^bb1(%44 : index)
  ^bb9:  // pred: ^bb1
    return
  }
}

