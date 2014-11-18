package mt

const KER_MATRIX_ADD = `
//
// Generated by NVIDIA NVVM Compiler
// Compiler built on Thu Mar 13 19:31:35 2014 (1394735495)
// Cuda compilation tools, release 6.0, V6.0.1
//

.version 4.0
.target sm_30
.address_size 64


.visible .entry matrixAdd(
	.param .u64 matrixAdd_param_0,
	.param .u64 matrixAdd_param_1,
	.param .u64 matrixAdd_param_2,
	.param .u32 matrixAdd_param_3,
	.param .u32 matrixAdd_param_4,
	.param .u32 matrixAdd_param_5,
	.param .u32 matrixAdd_param_6
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<12>;
	.reg .s64 	%rd<11>;
	.reg .f64 	%fd<4>;


	ld.param.u64 	%rd1, [matrixAdd_param_0];
	ld.param.u64 	%rd2, [matrixAdd_param_1];
	ld.param.u64 	%rd3, [matrixAdd_param_2];
	ld.param.u32 	%r2, [matrixAdd_param_3];
	ld.param.u32 	%r3, [matrixAdd_param_4];
	ld.param.u32 	%r4, [matrixAdd_param_5];
	ld.param.u32 	%r5, [matrixAdd_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r8, %r6, %r3, %r7;
	mov.u32 	%r9, %ctaid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r11, %r9, %r4, %r10;
	mad.lo.s32 	%r1, %r11, %r2, %r8;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r8, %r2;
	and.pred  	%p3, %p1, %p2;
	@!%p3 bra 	BB0_2;
	bra.uni 	BB0_1;

BB0_1:
	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd3;
	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 8;
	add.s64 	%rd8, %rd6, %rd7;
	add.s64 	%rd9, %rd5, %rd7;
	ld.global.f64 	%fd1, [%rd9];
	ld.global.f64 	%fd2, [%rd8];
	add.rn.f64 	%fd3, %fd2, %fd1;
	add.s64 	%rd10, %rd4, %rd7;
	st.global.f64 	[%rd10], %fd3;

BB0_2:
	ret;
}


`
