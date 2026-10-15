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
    %54 = lwi %41 : i32->i32
    %55 = llvm.add %23, %32 {constant = 0 : i32} : i32
    %56 = llvm.add %24, %33 {constant = 0 : i32} : i32
    %58 = llvm.add %26, %35 {constant = 0 : i32} : i32
    %59 = llvm.add %27, %36 {constant = 0 : i32} : i32
    %60 = llvm.add %28, %37 {constant = 0 : i32} : i32
    cond_br<ge> [%56 : i32, %54 : i32], ^bb10, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2(%55 : i32)
  ^bb2(%61: i32):  // 2 preds: ^bb1, ^bb8
    %100 = llvm.add %61, %32 : i32
    %103 = llvm.add %54, %36 : i32
    %62 = llvm.mul %100, %103 : i32
    llvm.br ^bb3(%60 : i32)
  ^bb3(%63: i32):  // 2 preds: ^bb2, ^bb6
    %101 = llvm.add %63, %32 : i32
    %102 = llvm.add %62, %32 : i32
    %64 = llvm.add %101, %102 : i32
    %65 = llvm.mul %64, %13 : i32
    %42 = llvm.add %5, %6 : i32
    %43 = llvm.shl %42, %7 : i32
    %44 = llvm.add %4, %43 {constant = 51204 : i32} : i32
    %45 = llvm.add %10, %11 : i32
    %46 = llvm.shl %45, %12 : i32
    %47 = llvm.add %9, %46 {constant = 51220 : i32} : i32
    %48 = llvm.add %15, %16 : i32
    %49 = llvm.shl %48, %17 : i32
    %50 = llvm.add %14, %49 {constant = 51236 : i32} : i32
    %66 = llvm.add %50, %65 : i32
    %57 = llvm.add %25, %34 {constant = 0 : i32} : i32
    swi %57, %66 : i32, i32
    %51 = llvm.add %20, %21 : i32
    %52 = llvm.shl %51, %22 : i32
    %53 = llvm.add %19, %52 {constant = 51236 : i32} : i32
    llvm.br ^bb4(%59, %58 : i32, i32)
  ^bb4(%67: i32, %68: i32):  // 2 preds: ^bb3, ^bb4
    %69 = llvm.add %68, %62 : i32
    %70 = llvm.mul %69, %3 : i32
    %71 = llvm.add %44, %70 : i32
    %72 = lwi %71 : i32->i32
    %73 = llvm.mul %68, %54 : i32
    %74 = llvm.add %73, %63 : i32
    %75 = llvm.mul %74, %8 : i32
    %76 = llvm.add %47, %75 : i32
    %77 = lwi %76 : i32->i32
    %78 = llvm.mul %77, %72 : i32
    %79 = llvm.add %67, %78 : i32
    %80 = llvm.mul %64, %18 : i32
    %81 = llvm.add %53, %80 : i32
    swi %79, %81 : i32, i32
    %82 = llvm.add %68, %31 : i32
    cond_br<ne> [%82 : i32, %54 : i32], ^bb4(%79, %82 : i32, i32), ^bb5
  ^bb5:  // pred: ^bb4
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    %83 = llvm.add %63, %30 : i32
    cond_br<ne> [%83 : i32, %54 : i32], ^bb3(%83 : i32), ^bb7
  ^bb7:  // pred: ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %84 = llvm.add %61, %29 : i32
    cond_br<ne> [%84 : i32, %54 : i32], ^bb2(%84 : i32), ^bb9
  ^bb9:  // pred: ^bb8
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb0, ^bb9
    llvm.return
  }
}

