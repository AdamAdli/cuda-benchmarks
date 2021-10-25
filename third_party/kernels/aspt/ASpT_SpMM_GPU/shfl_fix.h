#ifndef _H_SHFL_FIX
#define _H_SHFL_FIX

#define __shfl __shfl2
#define __shfl_down __shfl_down2
#define __shfl_up __shfl_up2
#define __shfl_xor __shfl_xor2

__device__ int __shfl2(int var, int srcLane, int width=32) {
	return __shfl_sync(0xffffffff, var, srcLane, width);
}

__device__ int __shfl_down2(int var, int delta, int width=32) {
	return __shfl_down_sync(0xffffffff, var, delta, width);
}

__device__ int __shfl_up2(int var, int delta, int width=32) {
	return __shfl_up_sync(0xffffffff, var, delta, width);
}

__device__ int __shfl_xor2(int var, int laneMask, int width=32) {
	return __shfl_xor_sync(0xffffffff, var, laneMask, width);
}

#endif //_H_SHFL_FIX
