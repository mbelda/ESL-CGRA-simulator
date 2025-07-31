module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm_majo(%arg0: memref<500xi32>, %arg1: memref<750xi32>, %arg2: memref<600xi32>, %arg3: memref<1xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c25 = arith.constant 25 : index
    %c30 = arith.constant 30 : index
    %c2123_i32 = arith.constant 2123 : i32
    %c32412_i32 = arith.constant 32412 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c20 = arith.constant 20 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb6
    %1 = arith.cmpi slt, %0, %c20 : index
    cf.cond_br %1, ^bb2(%c0 : index), ^bb7
  ^bb2(%2: index):  // 2 preds: ^bb1, ^bb5
    %3 = arith.cmpi slt, %2, %c30 : index
    cf.cond_br %3, ^bb3, ^bb6
  ^bb3:  // pred: ^bb2
    cf.br ^bb4(%c0, %c0_i32 : index, i32)
  ^bb4(%4: index, %5: i32):  // 2 preds: ^bb3, ^bb4
    %6 = arith.muli %0, %c25 : index
    %7 = arith.addi %4, %6 : index
    %8 = memref.load %arg0[%7] : memref<500xi32>
    %9 = arith.muli %4, %c30 : index
    %10 = arith.addi %2, %9 : index
    %11 = memref.load %arg1[%10] : memref<750xi32>
    %12 = arith.muli %8, %11 : i32
    %13 = arith.addi %5, %12 : i32
    %14 = arith.addi %4, %c1 : index
    %15 = arith.cmpi slt, %14, %c25 : index
    cf.cond_br %15, ^bb4(%14, %13 : index, i32), ^bb5
  ^bb5:  // pred: ^bb4
    %16 = arith.muli %13, %c32412_i32 : i32
    %17 = arith.muli %0, %c30 : index
    %18 = arith.addi %2, %17 : index
    %19 = memref.load %arg2[%18] : memref<600xi32>
    %20 = arith.muli %19, %c2123_i32 : i32
    %21 = arith.addi %16, %20 : i32
    %22 = arith.muli %0, %c30 : index
    %23 = arith.addi %2, %22 : index
    memref.store %21, %arg2[%23] : memref<600xi32>
    %24 = arith.addi %2, %c1 : index
    cf.br ^bb2(%24 : index)
  ^bb6:  // pred: ^bb2
    %25 = arith.addi %0, %c1 : index
    cf.br ^bb1(%25 : index)
  ^bb7:  // pred: ^bb1
    return
  }
}

