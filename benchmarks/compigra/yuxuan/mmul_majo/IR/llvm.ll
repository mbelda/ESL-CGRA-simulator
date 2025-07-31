; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks//MatMul/MatMul.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks//MatMul/MatMul.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @MatMul(i32* nocapture noundef readonly %0, i32* nocapture noundef readonly %1, i32* nocapture noundef readonly %2, i32* nocapture noundef %3) local_unnamed_addr #0 {
  %5 = load i32, i32* %0, align 4, !tbaa !4
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %34

7:                                                ; preds = %4, %31
  %8 = phi i32 [ %32, %31 ], [ 0, %4 ]
  %9 = mul nsw i32 %8, %5
  br label %10

10:                                               ; preds = %28, %7
  %11 = phi i32 [ 0, %7 ], [ %29, %28 ]
  %12 = add nsw i32 %11, %9
  %13 = getelementptr inbounds i32, i32* %3, i32 %12
  store i32 0, i32* %13, align 4, !tbaa !4
  br label %14

14:                                               ; preds = %14, %10
  %15 = phi i32 [ 0, %10 ], [ %25, %14 ]
  %16 = phi i32 [ 0, %10 ], [ %26, %14 ]
  %17 = add nsw i32 %16, %9
  %18 = getelementptr inbounds i32, i32* %1, i32 %17
  %19 = load i32, i32* %18, align 4, !tbaa !4
  %20 = mul nsw i32 %16, %5
  %21 = add nsw i32 %20, %11
  %22 = getelementptr inbounds i32, i32* %2, i32 %21
  %23 = load i32, i32* %22, align 4, !tbaa !4
  %24 = mul nsw i32 %23, %19
  %25 = add nsw i32 %15, %24
  store i32 %25, i32* %13, align 4, !tbaa !4
  %26 = add nuw nsw i32 %16, 1
  %27 = icmp eq i32 %26, %5
  br i1 %27, label %28, label %14, !llvm.loop !8

28:                                               ; preds = %14
  %29 = add nuw nsw i32 %11, 1
  %30 = icmp eq i32 %29, %5
  br i1 %30, label %31, label %10, !llvm.loop !11

31:                                               ; preds = %28
  %32 = add nuw nsw i32 %8, 1
  %33 = icmp eq i32 %32, %5
  br i1 %33, label %34, label %7, !llvm.loop !12

34:                                               ; preds = %31, %4
  ret void
}

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"uwtable", i32 1}
!3 = !{!"clang version 14.0.6 (https://github.com/llvm/llvm-project.git f28c006a5895fc0e329fe15fead81e37457cb1d1)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = distinct !{!11, !9, !10}
!12 = distinct !{!12, !9, !10}
