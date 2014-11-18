// Packae mt
// Linear algebra functions to work with matrix

package mt

import (
	"math"
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

type CudaMatrix struct {
	m cu.DevicePtr
	w int
	h int
}

const DEBUG = false
var cudaDevice = 0

var currentBuff = "default"
var usedMem = map[string]map[cu.DevicePtr]bool{
	currentBuff: make(map[cu.DevicePtr]bool),
}

var addBiasTopMod, multMod, subMod, addMod, multAllMod, negMatrixMod, setBiasToZeroMod,
	multTransMod, multByMod, removeBiasTopMod, transMod, sigmoidMatrixMod,
	logMatrixMod, oneMinusMod, addBiasMod, removeBias, powTwoMod, sigmoidGradMod,
	sumAll cu.Function
var maxNumThreads int
var cudaInitialized = false
var ctx cu.Context
var dev cu.Device

func InitCuda() {
	if !cudaInitialized {
		var mod cu.Module

		cu.Init(0)
		dev = cu.DeviceGet(cudaDevice)
		maxNumThreads = dev.Attribute(cu.MAX_THREADS_PER_BLOCK)

		ctx = cu.CtxCreate(cu.CTX_SCHED_AUTO, dev)
		ctx.SetCurrent()

		if DEBUG {
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult.ptx")
			multMod = mod.GetFunction("matrixMul")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sub.ptx")
			subMod = mod.GetFunction("matrixSub")
			mod = cu.ModuleLoad("/cuda_modules/matrix_add.ptx")
			addMod = mod.GetFunction("matrixAdd")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_all.ptx")
			multAllMod = mod.GetFunction("matrixMultAll")
			mod = cu.ModuleLoad("/cuda_modules/matrix_neg.ptx")
			negMatrixMod = mod.GetFunction("matrixNeg")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_trans.ptx")
			multTransMod = mod.GetFunction("matrixMulTrans")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sigmoid.ptx")
			sigmoidMatrixMod = mod.GetFunction("matrixSigmoid")
			mod = cu.ModuleLoad("/cuda_modules/matrix_log.ptx")
			logMatrixMod = mod.GetFunction("matrixLog")
			mod = cu.ModuleLoad("/cuda_modules/matrix_one_minus.ptx")
			oneMinusMod = mod.GetFunction("matrixOneMinus")
			mod = cu.ModuleLoad("/cuda_modules/matrix_add_bias.ptx")
			addBiasMod = mod.GetFunction("matrixAddBias")
			mod = cu.ModuleLoad("/cuda_modules/matrix_remove_bias.ptx")
			removeBias = mod.GetFunction("matrixRemoveBias")
			mod = cu.ModuleLoad("/cuda_modules/matrix_pow_two.ptx")
			powTwoMod = mod.GetFunction("matrixPowTwo")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sigmoid_gradient.ptx")
			sigmoidGradMod = mod.GetFunction("matrixSigmoidGrad")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_by.ptx")
			multByMod = mod.GetFunction("matrixMultBy")
			mod = cu.ModuleLoad("/cuda_modules/matrix_add_bias_top.ptx")
			addBiasTopMod = mod.GetFunction("matrixAddBiasTop")
			mod = cu.ModuleLoad("/cuda_modules/matrix_remove_bias_top.ptx")
			removeBiasTopMod = mod.GetFunction("matrixRemoveBiasTop")
			mod = cu.ModuleLoad("/cuda_modules/matrix_trans.ptx")
			transMod = mod.GetFunction("matrixTrans")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sum_all.ptx")
			sumAll = mod.GetFunction("matrixSumAll")
			mod = cu.ModuleLoad("/cuda_modules/matrix_set_bias_to_zero.ptx")
			setBiasToZeroMod = mod.GetFunction("matrixSetBiasToZero")
		} else {
			mod = cu.ModuleLoadData(KER_MATRIX_MULT)
			multMod = mod.GetFunction("matrixMul")
			mod = cu.ModuleLoadData(KER_MATRIX_SUB)
			subMod = mod.GetFunction("matrixSub")
			mod = cu.ModuleLoadData(KER_MATRIX_ADD)
			addMod = mod.GetFunction("matrixAdd")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_ALL)
			multAllMod = mod.GetFunction("matrixMultAll")
			mod = cu.ModuleLoadData(KER_MATRIX_NEG)
			negMatrixMod = mod.GetFunction("matrixNeg")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_TRANS)
			multTransMod = mod.GetFunction("matrixMulTrans")
			mod = cu.ModuleLoadData(KER_MATRIX_SIGMOID)
			sigmoidMatrixMod = mod.GetFunction("matrixSigmoid")
			mod = cu.ModuleLoadData(KER_MATRIX_LOG)
			logMatrixMod = mod.GetFunction("matrixLog")
			mod = cu.ModuleLoadData(KER_MATRIX_ONE_MINUS)
			oneMinusMod = mod.GetFunction("matrixOneMinus")
			mod = cu.ModuleLoadData(KER_MATRIX_ADD_BIAS)
			addBiasMod = mod.GetFunction("matrixAddBias")
			mod = cu.ModuleLoadData(KER_MATRIX_REMOVE_BIAS)
			removeBias = mod.GetFunction("matrixRemoveBias")
			mod = cu.ModuleLoadData(KER_MATRIX_POW_TWO)
			powTwoMod = mod.GetFunction("matrixPowTwo")
			mod = cu.ModuleLoadData(KER_MATRIX_SIGMOID_GRADIENT)
			sigmoidGradMod = mod.GetFunction("matrixSigmoidGrad")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_BY)
			multByMod = mod.GetFunction("matrixMultBy")
			mod = cu.ModuleLoadData(KER_MATRIX_ADD_BIAS_TOP)
			addBiasTopMod = mod.GetFunction("matrixAddBiasTop")
			mod = cu.ModuleLoadData(KER_MATRIX_REMOVE_BIAS_TOP)
			removeBiasTopMod = mod.GetFunction("matrixRemoveBiasTop")
			mod = cu.ModuleLoadData(KER_MATRIX_TRANS)
			transMod = mod.GetFunction("matrixTrans")
			mod = cu.ModuleLoadData(KER_MATRIX_SUM_ALL)
			sumAll = mod.GetFunction("matrixSumAll")
			mod = cu.ModuleLoadData(KER_MATRIX_SET_BIAS_TO_ZERO)
			setBiasToZeroMod = mod.GetFunction("matrixSetBiasToZero")
		}

		cudaInitialized = true
	}

	// Ugly hack to prevent problems with the libraries and the context
	// handling
	if cu.CtxGetCurrent() == 0 && ctx != 0 {
		ctx.SetCurrent()
	}
}

func SetDevice(dev int) {
	cudaDevice = dev
}

func StartBufferingMem(buff string) {
	if buff != "" {
		currentBuff = buff
		usedMem[buff] = make(map[cu.DevicePtr]bool)
	}
}

func SetDefaultBuff() {
	currentBuff = "default"
}

func AddToBuff(ptr cu.DevicePtr) {
	usedMem[currentBuff][ptr] = true
}

func FreeAllMem() {
	for _, buff := range(usedMem) {
		for m, _ := range(buff) {
			cu.MemFree(m)
		}
	}
	usedMem = map[string]map[cu.DevicePtr]bool{
		currentBuff: make(map[cu.DevicePtr]bool),
	}
}

func FreeMem() {
	for m, _ := range(usedMem[currentBuff]) {
		cu.MemFree(m)
		delete(usedMem[currentBuff], m)
	}
}

func (p *CudaMatrix) Free() {
	delete(usedMem[currentBuff], p.m)
	cu.MemFree(p.m)
}

func CudaMemAlloc(bytes int64) (cu.DevicePtr) {
	p := cu.MemAlloc(bytes)
	AddToBuff(p)

	return p
}

func InitCudaMatrix(w int, h int) (p *CudaMatrix) {
	size := int64(w * h) * cu.SIZEOF_FLOAT64
	InitCuda()
	p = &CudaMatrix {
		w: w,
		h: h,
		m: CudaMemAlloc(size),
	}
	// Initialize this var to zeros
	aux := make([]float64, w * h)
	cu.MemcpyHtoD(p.m, unsafe.Pointer(&aux[0]), size)

	return
}

func (m *CudaMatrix) H() (int) {
	return m.h
}

func (m *CudaMatrix) W() (int) {
	return m.w
}

func (m *CudaMatrix) CopyTo(t *CudaMatrix) (*CudaMatrix) {
	size := int64(m.w * m.h) * cu.SIZEOF_FLOAT64
	InitCuda()
	if t.w == 0 && t.h == 0 {
		t.m = CudaMemAlloc(size)
		t.w = m.w
		t.h = m.h
	}
	cu.MemcpyDtoD(t.m, m.m, size)

	return t
}

func (m *CudaMatrix) Copy() (r *CudaMatrix) {
	size := int64(m.w * m.h) * cu.SIZEOF_FLOAT64

	InitCuda()
	r = &CudaMatrix {
		m: CudaMemAlloc(size),
		w: m.w,
		h: m.h,
	}
	cu.MemcpyDtoD(r.m, m.m, size)

	return
}

func MoveToCuda(m [][]float64, p *CudaMatrix) {
	linealM := make([]float64, len(m) * len(m[0]))
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			linealM[(i * p.w) + j] = m[i][j]
		}
	}
	size := int64(len(linealM)) * cu.SIZEOF_FLOAT64

	InitCuda()
	if p.w == 0 && p.h == 0 {
		size := int64(len(m[0]) * len(m)) * cu.SIZEOF_FLOAT64
		p.m = CudaMemAlloc(size)
		p.w = len(m[0])
		p.h = len(m)
	}
	cu.MemcpyHtoD(p.m, unsafe.Pointer(&linealM[0]), size)

	return
}

func GetCudaMatrix(m [][]float64) (p *CudaMatrix) {
	p = &CudaMatrix {
		w: len(m[0]),
		h: len(m),
	}

	linealM := make([]float64, len(m) * len(m[0]))
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			linealM[(i * p.w) + j] = m[i][j]
		}
	}
	size := int64(len(linealM)) * cu.SIZEOF_FLOAT64

	InitCuda()
	p.m = CudaMemAlloc(size)
	cu.MemcpyHtoD(p.m, unsafe.Pointer(&linealM[0]), size)

	return
}

func (p *CudaMatrix) TransOneDimMatrix() (*CudaMatrix) {
	p.w ^= p.h
	p.h ^= p.w
	p.w ^= p.h

	return p
}

func (p *CudaMatrix) GetMatrixFromCuda() (m [][]float64) {
	buff := make([]float64, p.h * p.w)
	m = make([][]float64, p.h)

	InitCuda()
	cu.MemcpyDtoH(unsafe.Pointer(&buff[0]), p.m, int64(len(buff)) * cu.SIZEOF_FLOAT64)
	for i := 0; i < p.h; i++ {
		m[i] = buff[i * p.w : (i + 1) * p.w]
	}

	return
}

// Returns the rm of Multiply the given two matrix
func CudaMultAllElemsTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(m1.w, m1.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),

		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	cu.CtxSynchronize()
	launchKernelSync(multAllMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)
	cu.CtxSynchronize()

	return rm
}

func CudaSync() {
	cu.CtxSynchronize()
}

// Returns the rm of Multiply the given two matrix
func CudaMultAllElems(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	rm = &CudaMatrix{
		w: m2.w,
		h: m1.h,
		m: CudaMemAlloc(int64(m2.w * m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(m1.w, m1.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),

		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multAllMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// Returns the rm of Multiply the given two matrix
func MultElems(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaMultAllElems(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

func (m *CudaMatrix) SetPosTo(val float64, x int, y int) (*CudaMatrix) {
	buff := make([]float64, m.h * m.w)
	buffPoint := unsafe.Pointer(&buff[0])
	size := int64(len(buff)) * cu.SIZEOF_FLOAT64
	InitCuda()
	cu.MemcpyDtoH(buffPoint, m.m, size)
	buff[(y * m.w) + x] = val
	cu.MemcpyHtoD(m.m, buffPoint, size)

	return m
}

func (m *CudaMatrix) RemoveBiasTo(rm *CudaMatrix) (*CudaMatrix) {
	if rm.w == 0 && rm.h == 0 {
		rm.w = m.w - 1
		rm.h = m.h
		rm.m = CudaMemAlloc(int64(rm.w * rm.h) * cu.SIZEOF_FLOAT64)
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(removeBias, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

func (m *CudaMatrix) RemoveBias() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.w - 1,
		h: m.h,
		m: CudaMemAlloc(int64((m.w - 1) * m.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(removeBias, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

func (m *CudaMatrix) TransTo(rm *CudaMatrix) (*CudaMatrix) {
	InitCuda()

	if rm.w + rm.h == 0 {
		rm.w = m.h
		rm.h = m.w
		rm.m = CudaMemAlloc(int64(m.w * m.h) * cu.SIZEOF_FLOAT64)
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&rm.h),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(transMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

func (m *CudaMatrix) Trans() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.h,
		h: m.w,
		m: CudaMemAlloc(int64(m.w * m.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&rm.h),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(transMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

func (m *CudaMatrix) AddBiasTo(rm *CudaMatrix) (*CudaMatrix) {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	launchKernelSync(addBiasMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

func (m *CudaMatrix) RemoveBiasTopTo(rm *CudaMatrix) (*CudaMatrix) {
	InitCuda()

	if rm.w + rm.h == 0 {
		rm.w = m.w
		rm.h = m.h - 1
		rm.m = CudaMemAlloc(int64(m.w * (m.h + 1)) * cu.SIZEOF_FLOAT64)
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),

		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(removeBiasTopMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

func (m *CudaMatrix) RemoveBiasTop() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.w,
		h: m.h - 1,
		m: CudaMemAlloc(int64(m.w * (m.h + 1)) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),

		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(removeBiasTopMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

func (m *CudaMatrix) AddBiasTopTo(rm *CudaMatrix) (*CudaMatrix) {
	InitCuda()

	if rm.w == 0 && rm.h == 0 {
		rm.h = m.h + 1
		rm.w = m.w
		rm.m = CudaMemAlloc(int64(m.w * (m.h + 1)) * cu.SIZEOF_FLOAT64)
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(addBiasTopMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

func (m *CudaMatrix) AddBiasTop() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.w,
		h: m.h + 1,
		m: CudaMemAlloc(int64(m.w * (m.h + 1)) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(addBiasTopMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

func (m *CudaMatrix) AddBias() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.w + 1,
		h: m.h,
		m: CudaMemAlloc(int64((m.w + 1) * m.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(addBiasMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

func (m *CudaMatrix) MultBy(by float64) (*CudaMatrix) {
	InitCuda()
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(m.w, m.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&by),
		unsafe.Pointer(&m.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multByMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return m
}

func (m *CudaMatrix) SumAll() (float64) {
	// Note: Don't split in blocks
	InitCuda()

	cu.CtxSynchronize()
	size := m.w * m.h
	matrixSplits := int(math.Ceil(float64(size) / float64(maxNumThreads)))

	sumP := cu.MemAllocHost(cu.SIZEOF_FLOAT64)
	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&matrixSplits),
		unsafe.Pointer(&size),
		unsafe.Pointer(&sumP),
	}

	threads := maxNumThreads
	if size < maxNumThreads {
		threads = maxNumThreads
	}
	launchKernelSync(sumAll, 1, 1, 1, threads, 1, 1, 0, 0, args)
	cu.CtxSynchronize()

	return *(*float64)(sumP)
}

func (m *CudaMatrix) SetBiasToZero() (*CudaMatrix) {
	InitCuda()
	var gridsH, resH int

	if m.h > maxNumThreads {
		gridsH = int(math.Ceil(float64(m.h) / float64(maxNumThreads)))
		resH = maxNumThreads
	} else {
		gridsH = 1
		resH = m.h
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&m.h),
		unsafe.Pointer(&m.w),
		unsafe.Pointer(&resH),
	}

	launchKernelSync(setBiasToZeroMod, 1, gridsH, 1, 1, resH, 1, 0, 0, args)

	return m
}

func (m *CudaMatrix) applyFunc(function cu.Function) {
	InitCuda()

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(m.w, m.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&m.w),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(function, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) Log() (*CudaMatrix) {
	m.applyFunc(logMatrixMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) SigmoidGradient() (*CudaMatrix) {
	m.applyFunc(sigmoidGradMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) Sigmoid() (*CudaMatrix) {
	m.applyFunc(sigmoidMatrixMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) OneMinus() (*CudaMatrix) {
	m.applyFunc(oneMinusMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) PowTwo() (*CudaMatrix) {
	m.applyFunc(powTwoMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) Neg() (*CudaMatrix) {
	m.applyFunc(negMatrixMod)

	return m
}

func Neg(m [][]float64) (rm [][]float64) {
	cm := GetCudaMatrix(m)

	cm.Neg()

	rm = cm.GetMatrixFromCuda()

	return
}

// Returns as rm a matrix with all the elemtns of the first one multiplyed
// by the second one elems
func MultElemsNoCuda(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] * m2[x][y]
		}
	}

	return
}

func getGridThreadsFromSize(w, h int) (resW, resH, gridsW, gridsH, size int) {
	if w > h {
		if w > maxNumThreads{
			resW = maxNumThreads
			resH = 1
		} else {
			resW = w
			resH = int(float64(maxNumThreads) / float64(resW))
		}
	} else {
		if h > maxNumThreads{
			resH = maxNumThreads
			resW = 1
		} else {
			resH = h
			resW = int(float64(maxNumThreads) / float64(resH))
		}
	}
	if resW > w {
		resW = w
	}
	if resH > w {
		resH = h
	}

	gridsW = int(math.Ceil(float64(w) / float64(resW)))
	gridsH = int(math.Ceil(float64(h) / float64(resH)))

	size = w * h

	return
}

// Returns the rm of Multiply the given two matrix
func CudaMult(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	var resH, resW int
	InitCuda()
	rm = &CudaMatrix{
		w: m2.w,
		h: m1.h,
		m: CudaMemAlloc(int64(m2.w * m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&m2.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

func CudaMultTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&m2.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	launchKernelSync(multMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Returns the rm of Multiply the given two matrix
func Mult(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaMult(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

// Returns the rm of multiply the given two matrix without use cuda
func MultNoCuda(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m2[0]))

		for y := 0; y < len(m2[0]); y++ {
			for k := 0; k < len(m2); k++ {
				rm[x][y] += m1[x][k] * m2[k][y]
			}
		}
	}

	return
}

// Returns the rm of Multiply the given two matrix
func CudaSubTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	cu.CtxSynchronize()
	launchKernelSync(subMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)
	cu.CtxSynchronize()

	return rm
}

// Returns the rm of Multiply the given two matrix
func CudaSub(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	rm = &CudaMatrix{
		w: m1.w,
		h: m1.h,
		m: CudaMemAlloc(int64(m1.w * m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(subMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// Returns the subtraction of the given two matrix
func Sub(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaSub(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

// Returns the subtraction of the given two matrix
func SubNoCuda(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] - m2[x][y]
		}
	}

	return
}

// Returns the rm of Multiply the given two matrix
func CudaSumTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	launchKernelSync(addMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Returns the rm of Multiply the given two matrix
func CudaSum(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	rm = &CudaMatrix{
		w: m1.w,
		h: m1.h,
		m: CudaMemAlloc(int64(m1.w * m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(addMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// Returns the subtraction of the given two matrix
func Sum(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaSum(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

func CudaMultTransTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	InitCuda()
	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),

		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),

		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multTransMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Returns the rm of Multiply the given two matrix
func CudaMultTrans(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	rm = &CudaMatrix{
		w: m2.h,
		h: m1.h,
		m: CudaMemAlloc(int64(m2.h * m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),

		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),

		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multTransMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// Multiply on matrix by the transpose of the second matrix
func MultTrans(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaMultTrans(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

// Returns the sum of the given two matrix
func SumNoCuda(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] + m2[x][y]
		}
	}

	return
}

// Calculates the determinant of the matrix
func Det(m [][]float64) (rm float64) {
	// Sum diagonals
	ml := len(m)
	sums := make([]float64, ml*2)
	for i := 0; i < len(sums); i++ {
		sums[i] = 1
	}

	for r := 0; r < ml; r++ {
		for c := 0; c < ml; c++ {
			if c-r < 0 {
				sums[ml+c-r] *= m[c][r]
			} else {
				sums[c-r] *= m[c][r]
			}

			if c+r >= ml {
				sums[c+r] *= m[c][r]
			} else {
				sums[c+r+ml] *= m[c][r]
			}
		}
	}

	to := len(sums)
	if ml == 2 {
		to = 2
		ml = 1
	}
	for i := 0; i < to; i++ {
		if i >= ml {
			rm -= sums[i]
		} else {
			rm += sums[i]
		}
	}
	return
}

// Returns the minors matrix
func Minors(m [][]float64) (rm [][]float64) {
	ml := len(m)
	rm = make([][]float64, ml)
	for r := 0; r < ml; r++ {
		rm[r] = make([]float64, ml)
		for c := 0; c < ml; c++ {
			auxM := [][]float64{}
			for ra := 0; ra < ml; ra++ {
				if ra != r {
					auxR := []float64{}
					for ca := 0; ca < ml; ca++ {
						if ca != c {
							auxR = append(auxR, m[ra][ca])
						}
					}
					auxM = append(auxM, auxR)
				}
			}
			rm[r][c] = Det(auxM)
		}
	}

	return
}

// Returns the cofactors matrix
func Cofactors(m [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m))
	for r := 0; r < len(m); r++ {
		rm[r] = make([]float64, len(m[0]))
		for c := 0; c < len(m[0]); c++ {
			if (c+r)%2 == 0 {
				rm[r][c] = m[r][c]
			} else {
				rm[r][c] = -m[r][c]
			}
		}
	}

	return
}

// Calculates the inverse matrix
func Inv(m [][]float64) (rm [][]float64) {
	dm := Det(m)
	adj := Trans(Cofactors(Minors(m)))

	rm = MultBy(adj, 1.0/dm)

	return
}

// Divide the first matrix by the second one
func Div(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	return Mult(m1, Inv(m2))
}

// Returns the rm of multiply all the elements of a matrix by a float number
func MultBy(m1 [][]float64, n float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] * n
		}
	}

	return
}

// Matrix Transpose, returns the transpose of the given square matrix
func Trans(m1 [][]float64) (rm [][]float64) {
	if len(m1) == 0 {
		return [][]float64{}
	}
	rm = make([][]float64, len(m1[0]))

	// Initialize the matrix
	for x := 0; x < len(m1[0]); x++ {
		rm[x] = make([]float64, len(m1))

		for y := 0; y < len(m1); y++ {
			rm[x][y] = m1[y][x]
		}
	}

	return
}

// Sum all the elements in a matrix
func SumAll(m [][]float64) (rm float64) {
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			rm += m[i][j]
		}
	}

	return
}

// Apply a function to all the elements of a matrix, the function will receive a
// float64 as param and returns a float64 too
func Apply(m [][]float64, f func(x float64) float64) (rm [][]float64) {
	rm = make([][]float64, len(m))

	// Initialize the matrix
	for x := 0; x < len(m); x++ {
		rm[x] = make([]float64, len(m[0]))
		for y := 0; y < len(m[0]); y++ {
			rm[x][y] = f(m[x][y])
		}
	}

	return
}

// Returns a copy of the matrix
func Copy(m [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m))

	for i := 0; i < len(m); i++ {
		rm[i] = make([]float64, len(m[i]))
		for j := 0; j < len(m[i]); j++ {
			rm[i][j] = m[i][j]
		}
	}

	return
}

// Concatenates two matrix elements, ex:
// m1 = (M111, M112, M113)
//      (M121, M122, M123)
//      (M131, M132, M133)
// m2 = (M211, M212, M213)
//      (M221, M222, M223)
//      (M231, M232, M233)
// rm = (M111, M112, M113, M221, M222, M223)
//      (M121, M122, M123, M221, M222, M223)
//      (M131, M132, M133, M231, M232, M233)
func Concat(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		rm[i] = make([]float64, len(m1[i]) + len(m2[i]))
		for j := 0; j < len(m1[i]); j++ {
			rm[i][j] = m1[i][j]
		}
		for j := 0; j < len(m2[i]); j++ {
			rm[i][j + len(m1[i])] = m2[i][j]
		}
	}

	return
}

func launchKernelSync(f cu.Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream cu.Stream, kernelParams []unsafe.Pointer) {
	cu.LaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams)
	cu.CtxSynchronize()
}
