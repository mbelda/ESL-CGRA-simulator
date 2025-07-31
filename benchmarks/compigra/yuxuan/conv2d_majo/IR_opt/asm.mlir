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
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb14
    %5 = arith.addi %c0_i32_0, %c127_i32_1 {constant = 127 : i32} : i32
    %6 = arith.addi %c0_i32_2, %c1_i32 {constant = 1 : i32} : i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb15, ^bb2(%6 : i32)
  ^bb2(%7: i32):  // 2 preds: ^bb1, ^bb13
    %8 = arith.addi %c0_i32, %c127_i32 {constant = 127 : i32} : i32
    %9 = arith.addi %c0_i32_8, %c0_i32_9 {constant = 0 : i32} : i32
    %c0_i32_17 = arith.constant 0 : i32
    cgra.swi %9, %c0_i32_17 : i32, i32 {memLoc = 0 : i32}
    %10 = arith.addi %c0_i32_11, %c-1_i32_12 {constant = -1 : i32} : i32
    %11 = arith.addi %c0_i32_11, %c-1_i32_12 {constant = -1 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb14, ^bb3(%10, %11 : i32, i32)
  ^bb3(%12: i32, %13: i32):  // 2 preds: ^bb2, ^bb12
    %14 = arith.addi %c0_i32_14, %c2_i32_15 {constant = 2 : i32} : i32
    cgra.cond_br<ge> [%12 : i32, %14 : i32], ^bb13, ^bb4
  ^bb4:  // pred: ^bb3
    %15 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    %16 = arith.addi %c0_i32_10, %c-1_i32 {constant = -1 : i32} : i32
    %c0_i32_18 = arith.constant 0 : i32
    %c0_i32_19 = arith.constant 0 : i32
    cf.br ^bb5
  ^bb5:  // pred: ^bb4
    %17 = arith.muli %12, %c128_i32 : i32
    %18 = arith.muli %13, %c3_i32 : i32
    %19 = arith.addi %16, %c0_i32_19 : i32
    %20 = arith.addi %17, %7 : i32
    %21 = arith.addi %20, %19 : i32
    %22 = arith.muli %4, %c128_i32 : i32
    %23 = arith.addi %19, %18 : i32
    %24 = arith.addi %19, %c1_i32_5 : i32
    %25 = arith.addi %c0_i32_13, %c2_i32 {constant = 2 : i32} : i32
    %26 = arith.muli %12, %c128_i32 : i32
    %27 = arith.muli %13, %c3_i32 : i32
    %28 = arith.addi %21, %22 : i32
    %29 = arith.addi %23, %c4_i32 : i32
    %30 = arith.addi %24, %c0_i32_19 : i32
    %31 = arith.addi %26, %7 : i32
    cgra.cond_br<ge> [%24 : i32, %25 : i32], ^bb11, ^bb6
  ^bb6:  // pred: ^bb5
    %32 = arith.muli %28, %c4_i32_16 : i32
    %33 = arith.muli %29, %c4_i32_16 : i32
    %34 = arith.addi %31, %30 : i32
    %35 = arith.muli %4, %c128_i32 : i32
    %36 = arith.addi %30, %27 : i32
    %37 = arith.addi %30, %c1_i32_5 : i32
    %38 = arith.addi %c0_i32_13, %c2_i32 {constant = 2 : i32} : i32
    %39 = arith.muli %12, %c128_i32 : i32
    %40 = arith.muli %13, %c3_i32 : i32
    %41 = arith.addi %0, %32 : i32
    %42 = arith.addi %1, %33 : i32
    %43 = arith.addi %34, %35 : i32
    %44 = arith.addi %36, %c4_i32 : i32
    %45 = arith.addi %37, %c0_i32_19 : i32
    %46 = arith.addi %39, %7 : i32
    cgra.cond_br<ge> [%37 : i32, %38 : i32], ^bb10, ^bb7
  ^bb7:  // pred: ^bb6
    %47 = cgra.lwi %41 : i32->i32
    %48 = cgra.lwi %42 : i32->i32
    %49 = arith.muli %43, %c4_i32_16 : i32
    %50 = arith.muli %44, %c4_i32_16 : i32
    %51 = arith.addi %46, %45 : i32
    %52 = arith.muli %4, %c128_i32 : i32
    %53 = arith.addi %45, %40 : i32
    %54 = arith.addi %45, %c1_i32_5 : i32
    %55 = arith.addi %c0_i32_13, %c2_i32 {constant = 2 : i32} : i32
    %56 = arith.muli %12, %c128_i32 : i32
    %57 = arith.muli %13, %c3_i32 : i32
    %58 = arith.addi %15, %c0_i32_18 : i32
    %59 = arith.muli %47, %48 : i32
    %60 = arith.addi %0, %49 : i32
    %61 = arith.addi %1, %50 : i32
    %62 = arith.addi %51, %52 : i32
    %63 = arith.addi %53, %c4_i32 : i32
    %64 = arith.addi %54, %c0_i32_19 : i32
    %65 = arith.addi %56, %7 : i32
    cgra.cond_br<ge> [%54 : i32, %55 : i32], ^bb9(%58, %59, %60, %61, %62, %63 : i32, i32, i32, i32, i32, i32), ^bb8(%58, %59, %60, %61, %62, %63, %65, %64, %64, %57, %64 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
  ^bb8(%66: i32, %67: i32, %68: i32, %69: i32, %70: i32, %71: i32, %72: i32, %73: i32, %74: i32, %75: i32, %76: i32):  // 2 preds: ^bb7, ^bb8
    %77 = arith.addi %66, %67 : i32
    %78 = cgra.lwi %68 : i32->i32
    %79 = cgra.lwi %69 : i32->i32
    %80 = arith.muli %70, %c4_i32_16 : i32
    %81 = arith.muli %71, %c4_i32_16 : i32
    %82 = arith.addi %72, %73 : i32
    %83 = arith.muli %4, %c128_i32 : i32
    %84 = arith.addi %74, %75 : i32
    %85 = arith.addi %76, %c1_i32_5 : i32
    %86 = arith.addi %c0_i32_13, %c2_i32 {constant = 2 : i32} : i32
    %87 = arith.muli %12, %c128_i32 : i32
    %88 = arith.muli %13, %c3_i32 : i32
    %89 = arith.addi %77, %c0_i32_18 : i32
    %90 = arith.muli %78, %79 : i32
    %91 = arith.addi %0, %80 : i32
    %92 = arith.addi %1, %81 : i32
    %93 = arith.addi %82, %83 : i32
    %94 = arith.addi %84, %c4_i32 : i32
    %95 = arith.addi %85, %c0_i32_19 : i32
    %96 = arith.addi %87, %7 : i32
    cgra.cond_br<lt> [%85 : i32, %86 : i32], ^bb8(%89, %90, %91, %92, %93, %94, %96, %95, %95, %88, %95 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32), ^bb9(%89, %90, %91, %92, %93, %94 : i32, i32, i32, i32, i32, i32)
  ^bb9(%97: i32, %98: i32, %99: i32, %100: i32, %101: i32, %102: i32):  // 2 preds: ^bb7, ^bb8
    %103 = arith.addi %97, %98 : i32
    %104 = arith.addi %103, %c0_i32_18 : i32
    %105 = cgra.lwi %99 : i32->i32
    %106 = cgra.lwi %100 : i32->i32
    %107 = arith.muli %105, %106 : i32
    %108 = arith.addi %104, %107 : i32
    %109 = arith.addi %108, %c0_i32_18 : i32
    %110 = arith.muli %101, %c4_i32_16 : i32
    %111 = arith.addi %0, %110 : i32
    %112 = cgra.lwi %111 : i32->i32
    %113 = arith.muli %102, %c4_i32_16 : i32
    %114 = arith.addi %1, %113 : i32
    %115 = cgra.lwi %114 : i32->i32
    %116 = arith.muli %112, %115 : i32
    %117 = arith.addi %109, %116 : i32
    cf.br ^bb12(%117 : i32)
  ^bb10:  // pred: ^bb6
    %118 = arith.addi %15, %c0_i32_18 : i32
    %119 = cgra.lwi %41 : i32->i32
    %120 = cgra.lwi %42 : i32->i32
    %121 = arith.muli %119, %120 : i32
    %122 = arith.addi %118, %121 : i32
    %123 = arith.addi %122, %c0_i32_18 : i32
    %124 = arith.muli %43, %c4_i32_16 : i32
    %125 = arith.addi %0, %32 : i32
    %126 = cgra.lwi %41 : i32->i32
    %127 = arith.muli %44, %c4_i32_16 : i32
    %128 = arith.addi %1, %33 : i32
    %129 = cgra.lwi %42 : i32->i32
    %130 = arith.muli %126, %129 : i32
    %131 = arith.addi %123, %130 : i32
    cf.br ^bb12(%131 : i32)
  ^bb11:  // pred: ^bb5
    %132 = arith.addi %15, %c0_i32_18 : i32
    %133 = arith.muli %28, %c4_i32_16 : i32
    %134 = arith.addi %0, %133 : i32
    %135 = cgra.lwi %134 : i32->i32
    %136 = arith.muli %29, %c4_i32_16 : i32
    %137 = arith.addi %1, %136 : i32
    %138 = cgra.lwi %137 : i32->i32
    %139 = arith.muli %135, %138 : i32
    %140 = arith.addi %132, %139 : i32
    cf.br ^bb12(%140 : i32)
  ^bb12(%141: i32):  // 3 preds: ^bb9, ^bb10, ^bb11
    %c0_i32_20 = arith.constant 0 : i32
    %142 = cgra.lwi %c0_i32_20 : i32->i32
    %143 = arith.addi %142, %c0_i32 : i32
    %144 = arith.addi %143, %141 : i32
    %c0_i32_21 = arith.constant 0 : i32
    cgra.swi %144, %c0_i32_21 : i32, i32 {memLoc = 0 : i32}
    %145 = arith.addi %12, %c1_i32_5 : i32
    %146 = arith.addi %13, %c1_i32_5 : i32
    cf.br ^bb3(%145, %146 : i32, i32)
  ^bb13:  // pred: ^bb3
    %147 = arith.muli %4, %c128_i32 : i32
    %148 = arith.addi %7, %147 : i32
    %149 = arith.muli %148, %c4_i32_16 : i32
    %150 = arith.addi %2, %149 : i32
    %c0_i32_22 = arith.constant 0 : i32
    %151 = cgra.lwi %c0_i32_22 : i32->i32
    cgra.swi %151, %150 : i32, i32
    %152 = arith.addi %7, %c1_i32_5 : i32
    cf.br ^bb2(%152 : i32)
  ^bb14:  // pred: ^bb2
    %153 = arith.addi %4, %c1_i32_5 : i32
    cf.br ^bb1(%153 : i32)
  ^bb15:  // pred: ^bb1
    return
  }
}

