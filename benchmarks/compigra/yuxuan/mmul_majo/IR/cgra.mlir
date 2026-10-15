Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @MatMul(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, ...) attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2", "in3"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = [], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(12 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(12 : i32) : i32
    %3 = llvm.mlir.constant(4 : i32) : i32
    %4 = llvm.mlir.constant(2052 : i32) : i32
    %5 = llvm.mlir.constant(12 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(12 : i32) : i32
    %8 = llvm.mlir.constant(4 : i32) : i32
    %9 = llvm.mlir.constant(2068 : i32) : i32
    %10 = llvm.mlir.constant(12 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(12 : i32) : i32
    %13 = llvm.mlir.constant(4 : i32) : i32
    %14 = llvm.mlir.constant(2084 : i32) : i32
    %15 = llvm.mlir.constant(12 : i32) : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(12 : i32) : i32
    %18 = llvm.mlir.constant(4 : i32) : i32
    %19 = llvm.mlir.constant(2084 : i32) : i32
    %20 = llvm.mlir.constant(12 : i32) : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(12 : i32) : i32
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(1 : i32) : i32
    %30 = llvm.mlir.constant(1 : i32) : i32
    %31 = llvm.mlir.constant(1 : i32) : i32
    %32 = llvm.mlir.constant(0 : i32) : i32
    %33 = llvm.mlir.constant(0 : i32) : i32
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = llvm.mlir.constant(0 : i32) : i32
    %36 = llvm.mlir.constant(0 : i32) : i32
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = llvm.mlir.constant(2048 : i32) : i32
    %39 = llvm.add %0, %1 : i32
    %40 = llvm.shl %39, %2 : i32
    %41 = llvm.add %38, %40 {constant = 51200 : i32} : i32
    %42 = lwi %41 : i32->i32
    %43 = llvm.add %23, %32 {constant = 0 : i32} : i32
    %44 = llvm.add %24, %33 {constant = 0 : i32} : i32
    %45 = llvm.add %25, %34 {constant = 0 : i32} : i32
    %46 = llvm.add %26, %35 {constant = 0 : i32} : i32
    %47 = llvm.add %27, %36 {constant = 0 : i32} : i32
    %48 = llvm.add %28, %37 {constant = 0 : i32} : i32
    cond_br<ge> [%44 : i32, %42 : i32], ^bb15, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2(%43 : i32)
  ^bb2(%49: i32):  // 2 preds: ^bb1, ^bb13
    %50 = llvm.mul %49, %42 : i32
    %51 = llvm.add %15, %16 : i32
    %52 = llvm.shl %51, %17 : i32
    %53 = llvm.add %14, %52 {constant = 51236 : i32} : i32
    llvm.br ^bb3(%48 : i32)
  ^bb3(%54: i32):  // 2 preds: ^bb2, ^bb11
    %55 = llvm.add %54, %50 : i32
    %56 = llvm.mul %55, %13 : i32
    %57 = llvm.add %53, %56 : i32
    swi %45, %57 : i32, i32
    %58 = llvm.add %5, %6 : i32
    %59 = llvm.shl %58, %7 : i32
    %60 = llvm.add %4, %59 {constant = 51204 : i32} : i32
    %61 = llvm.add %10, %11 : i32
    %62 = llvm.shl %61, %12 : i32
    %63 = llvm.add %9, %62 {constant = 51220 : i32} : i32
    %64 = llvm.add %20, %21 : i32
    %65 = llvm.shl %64, %22 : i32
    %66 = llvm.add %19, %65 {constant = 51236 : i32} : i32
    %67 = llvm.mlir.constant(0 : i32) : i32
    %68 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %69 = llvm.add %46, %67 : i32
    %70 = llvm.add %69, %50 : i32
    %71 = llvm.mul %70, %3 : i32
    %72 = llvm.add %60, %71 : i32
    %73 = llvm.mul %69, %42 : i32
    %74 = llvm.add %73, %47 : i32
    %75 = llvm.mul %74, %8 : i32
    %76 = llvm.add %69, %31 : i32
    %77 = llvm.add %76, %67 : i32
    %78 = llvm.add %77, %50 : i32
    %79 = llvm.mul %77, %42 : i32
    %80 = llvm.add %77, %31 : i32
    cond_br<eq> [%76 : i32, %42 : i32], ^bb8, ^bb5
  ^bb5:  // pred: ^bb4
    %81 = lwi %72 : i32->i32
    %82 = llvm.add %63, %75 : i32
    %83 = lwi %82 : i32->i32
    %84 = llvm.mul %55, %18 : i32
    %85 = llvm.mul %78, %3 : i32
    %86 = llvm.add %60, %85 : i32
    %87 = llvm.add %79, %47 : i32
    %88 = llvm.mul %87, %8 : i32
    %89 = llvm.add %80, %67 : i32
    %90 = llvm.add %89, %50 : i32
    %91 = llvm.mul %89, %42 : i32
    %92 = llvm.add %89, %31 : i32
    cond_br<eq> [%80 : i32, %42 : i32], ^bb9, ^bb6
  ^bb6:  // pred: ^bb5
    %93 = lwi %86 : i32->i32
    %94 = llvm.add %63, %88 : i32
    %95 = lwi %94 : i32->i32
    %96 = llvm.mul %55, %18 : i32
    %97 = llvm.mul %90, %3 : i32
    %98 = llvm.add %60, %97 : i32
    %99 = llvm.add %91, %47 : i32
    %100 = llvm.mul %99, %8 : i32
    %101 = llvm.add %92, %67 : i32
    %102 = llvm.add %101, %50 : i32
    %103 = llvm.mul %101, %42 : i32
    %104 = llvm.add %101, %31 : i32
    %105 = llvm.add %47, %68 : i32
    %106 = llvm.mul %95, %93 : i32
    %107 = llvm.add %105, %106 : i32
    %108 = llvm.add %66, %96 : i32
    cond_br<eq> [%92 : i32, %42 : i32], ^bb10(%107, %108, %98, %100, %107, %108 : i32, i32, i32, i32, i32, i32), ^bb7(%107, %108, %98, %100, %102, %103, %107, %104, %104, %107 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
  ^bb7(%109: i32, %110: i32, %111: i32, %112: i32, %113: i32, %114: i32, %115: i32, %116: i32, %117: i32, %118: i32):  // 2 preds: ^bb6, ^bb7
    swi %109, %110 : i32, i32
    %119 = lwi %111 : i32->i32
    %120 = llvm.add %63, %112 : i32
    %121 = lwi %120 : i32->i32
    %122 = llvm.mul %55, %18 : i32
    %123 = llvm.mul %113, %3 : i32
    %124 = llvm.add %60, %123 : i32
    %125 = llvm.add %114, %115 : i32
    %126 = llvm.mul %125, %8 : i32
    %127 = llvm.add %117, %67 : i32
    %128 = llvm.add %127, %50 : i32
    %129 = llvm.mul %127, %42 : i32
    %130 = llvm.add %127, %31 : i32
    %131 = llvm.add %118, %68 : i32
    %132 = llvm.mul %121, %119 : i32
    %133 = llvm.add %131, %132 : i32
    %134 = llvm.add %66, %122 : i32
    cond_br<ne> [%116 : i32, %42 : i32], ^bb7(%133, %134, %124, %126, %128, %129, %133, %130, %130, %133 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32), ^bb10(%107, %108, %98, %100, %107, %108 : i32, i32, i32, i32, i32, i32)
  ^bb8:  // pred: ^bb4
    %135 = llvm.add %47, %68 : i32
    %136 = lwi %72 : i32->i32
    %137 = llvm.add %63, %75 : i32
    %138 = lwi %137 : i32->i32
    %139 = llvm.mul %138, %136 : i32
    %140 = llvm.add %135, %139 : i32
    %141 = llvm.mul %55, %18 : i32
    %142 = llvm.add %66, %141 : i32
    swi %140, %142 : i32, i32
    llvm.br ^bb11
  ^bb9:  // pred: ^bb5
    %143 = llvm.add %47, %68 : i32
    %144 = llvm.mul %83, %81 : i32
    %145 = llvm.add %143, %144 : i32
    %146 = llvm.add %66, %84 : i32
    swi %145, %146 : i32, i32
    %147 = llvm.add %145, %68 : i32
    %148 = lwi %86 : i32->i32
    %149 = llvm.add %63, %88 : i32
    %150 = lwi %149 : i32->i32
    %151 = llvm.mul %150, %148 : i32
    %152 = llvm.add %147, %151 : i32
    %153 = llvm.mul %55, %18 : i32
    %154 = llvm.add %66, %153 : i32
    swi %152, %154 : i32, i32
    llvm.br ^bb11
  ^bb10(%155: i32, %156: i32, %157: i32, %158: i32, %159: i32, %160: i32):  // 2 preds: ^bb6, ^bb7
    swi %155, %156 : i32, i32
    %161 = lwi %157 : i32->i32
    %162 = llvm.add %63, %158 : i32
    %163 = lwi %162 : i32->i32
    %164 = llvm.mul %55, %18 : i32
    swi %159, %160 : i32, i32
    llvm.br ^bb11
  ^bb11:  // 3 preds: ^bb8, ^bb9, ^bb10
    %165 = llvm.add %54, %30 : i32
    cond_br<ne> [%165 : i32, %42 : i32], ^bb3(%165 : i32), ^bb12
  ^bb12:  // pred: ^bb11
    llvm.br ^bb13
  ^bb13:  // pred: ^bb12
    %166 = llvm.add %49, %29 : i32
    cond_br<ne> [%166 : i32, %42 : i32], ^bb2(%166 : i32), ^bb14
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15
  ^bb15:  // 2 preds: ^bb0, ^bb14
    llvm.return
  }
}

