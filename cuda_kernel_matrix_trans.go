package mt

const KER_MATRIX_TRANS = `
//
// Generated by NVIDIA NVVM Compiler
// Compiler built on Thu Mar 13 19:31:35 2014 (1394735495)
// Cuda compilation tools, release 6.0, V6.0.1
//

.version 4.0
.target sm_30
.address_size 64


.visible .entry matrixTrans(
	.param .u64 matrixTrans_param_0,
	.param .u64 matrixTrans_param_1,
	.param .u32 matrixTrans_param_2,
	.param .u32 matrixTrans_param_3,
	.param .u32 matrixTrans_param_4,
	.param .u32 matrixTrans_param_5,
	.param .u32 matrixTrans_param_6
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<14>;
	.reg .s64 	%rd<9>;
	.reg .f64 	%fd<2>;


	ld.param.u64 	%rd1, [matrixTrans_param_0];
	ld.param.u64 	%rd2, [matrixTrans_param_1];
	ld.param.u32 	%r5, [matrixTrans_param_2];
	ld.param.u32 	%r6, [matrixTrans_param_3];
	ld.param.u32 	%r7, [matrixTrans_param_4];
	ld.param.u32 	%r4, [matrixTrans_param_5];
	ld.param.u32 	%r8, [matrixTrans_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r5, %r10;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r6, %r12;
	mad.lo.s32 	%r3, %r2, %r7, %r1;
	setp.lt.s32	%p1, %r3, %r8;
	setp.lt.s32	%p2, %r1, %r7;
	and.pred  	%p3, %p1, %p2;
	@!%p3 bra 	BB0_2;
	bra.uni 	BB0_1;

BB0_1:
	cvta.to.global.u64 	%rd3, %rd1;
	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r13, %r1, %r4, %r2;
	mul.wide.s32 	%rd5, %r13, 8;
	add.s64 	%rd6, %rd4, %rd5;
	mul.wide.s32 	%rd7, %r3, 8;
	add.s64 	%rd8, %rd3, %rd7;
	ld.global.f64 	%fd1, [%rd6];
	st.global.f64 	[%rd8], %fd1;

BB0_2:
	ret;
}


`