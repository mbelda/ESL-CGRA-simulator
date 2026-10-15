module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d_majo(%arg0: memref<16384xi32>, %arg1: memref<9xi32>, %arg2: memref<16384xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c127 = arith.constant 127 : index
    cf.br ^bb1(%c1 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
    %1 = arith.cmpi slt, %0, %c127 : index
    cf.cond_br %1, ^bb2(%c1 : index), ^bb9
  ^bb2(%2: index):  // 2 preds: ^bb1, ^bb7
    %3 = arith.cmpi slt, %2, %c127 : index
    cf.cond_br %3, ^bb3(%c-1, %c0_i32 : index, i32), ^bb8
  ^bb3(%4: index, %5: i32):  // 2 preds: ^bb2, ^bb6
    %6 = arith.cmpi slt, %4, %c2 : index
    cf.cond_br %6, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%c-1, %c0_i32 : index, i32)
  ^bb5(%7: index, %8: i32):  // 2 preds: ^bb4, ^bb5
    %9 = arith.muli %4, %c128 : index
    %10 = arith.addi %9, %2 : index
    %11 = arith.addi %10, %7 : index
    %12 = arith.muli %0, %c128 : index
    %13 = arith.addi %11, %12 : index
    %14 = memref.load %arg0[%13] : memref<16384xi32>
    %15 = arith.muli %4, %c3 : index
    %16 = arith.addi %7, %15 : index
    %17 = arith.addi %16, %c4 : index
    %18 = memref.load %arg1[%17] : memref<9xi32>
    %19 = arith.muli %14, %18 : i32
    %20 = arith.addi %8, %19 : i32
    %21 = arith.addi %7, %c1 : index
    %22 = arith.cmpi slt, %21, %c2 : index
    cf.cond_br %22, ^bb5(%21, %20 : index, i32), ^bb6
  ^bb6:  // pred: ^bb5
    %23 = arith.addi %5, %20 : i32
    %24 = arith.addi %4, %c1 : index
    cf.br ^bb3(%24, %23 : index, i32)
  ^bb7:  // pred: ^bb3
    %25 = arith.muli %0, %c128 : index
    %26 = arith.addi %2, %25 : index
    memref.store %5, %arg2[%26] : memref<16384xi32>
    %27 = arith.addi %2, %c1 : index
    cf.br ^bb2(%27 : index)
  ^bb8:  // pred: ^bb2
    %28 = arith.addi %0, %c1 : index
    cf.br ^bb1(%28 : index)
  ^bb9:  // pred: ^bb1
    return
  }
}

