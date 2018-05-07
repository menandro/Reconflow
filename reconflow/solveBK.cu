#include "CudaFlow.h"

__global__
void SolveBKKernel(const float *du, const float *dumed, const float *bku, int count, float *bkuout)
{
	const int pos = threadIdx.x + blockIdx.x * blockDim.x;

	if (pos >= count) return;

	bkuout[pos] = bku[pos] + du[pos] - dumed[pos];
	//sum[pos] = op1[pos] + op2[pos];
}


void CudaFlow::SolveBK(const float *du, const float *dumed, const float *bku, int count, float *bkuout)
{
	dim3 threads(256);
	dim3 blocks(iDivUp(count, threads.x));

	SolveBKKernel <<< blocks, threads >> >(du, dumed, bku, count, bkuout);
}
