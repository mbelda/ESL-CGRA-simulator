#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func local_unnamed_addr @MatMul(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.load %arg0 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %3 = llvm.icmp "sgt" %2, %0 : i32
    llvm.cond_br %3, ^bb1(%0 : i32), ^bb6
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb5
    %5 = llvm.mul %4, %2  : i32
    llvm.br ^bb2(%0 : i32)
  ^bb2(%6: i32):  // 2 preds: ^bb1, ^bb4
    %7 = llvm.add %6, %5  : i32
    %8 = llvm.getelementptr inbounds %arg3[%7] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %0, %8 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    llvm.br ^bb3(%0, %0 : i32, i32)
  ^bb3(%9: i32, %10: i32):  // 2 preds: ^bb2, ^bb3
    %11 = llvm.add %10, %5  : i32
    %12 = llvm.getelementptr inbounds %arg1[%11] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %13 = llvm.load %12 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %14 = llvm.mul %10, %2  : i32
    %15 = llvm.add %14, %6  : i32
    %16 = llvm.getelementptr inbounds %arg2[%15] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %17 = llvm.load %16 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %18 = llvm.mul %17, %13  : i32
    %19 = llvm.add %9, %18  : i32
    llvm.store %19, %8 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %20 = llvm.add %10, %1  : i32
    %21 = llvm.icmp "eq" %20, %2 : i32
    llvm.cond_br %21, ^bb4, ^bb3(%19, %20 : i32, i32) {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb3
    %22 = llvm.add %6, %1  : i32
    %23 = llvm.icmp "eq" %22, %2 : i32
    llvm.cond_br %23, ^bb5, ^bb2(%22 : i32) {loop_annotation = #loop_annotation}
  ^bb5:  // pred: ^bb4
    %24 = llvm.add %4, %1  : i32
    %25 = llvm.icmp "eq" %24, %2 : i32
    llvm.cond_br %25, ^bb6, ^bb1(%24 : i32) {loop_annotation = #loop_annotation}
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.return
  }
}
