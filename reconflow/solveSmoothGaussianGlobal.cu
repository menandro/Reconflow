#include "CudaFlow.h"

__global__ void SolveSmoothGaussianGlobalKernel5(float* u, float* v, float* bku, float* bkv,
	int width, int height, int stride,
	float *outputu, float *outputv,
	float *outputbku, float* outputbkv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float w[25] = { 0.0037, 0.0147, 0.0256, 0.0147, 0.0037,
		0.0147, 0.0586, 0.0952, 0.0586, 0.0147,
		0.0256, 0.0952, 0.1502, 0.0952, 0.0256,
		0.0147, 0.0586, 0.0952, 0.0586, 0.0147,
		0.0037, 0.0147, 0.0256, 0.0147, 0.0037 };

	float sumu = 0;
	float sumv = 0;
	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			//get values
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sumu = sumu + w[j * 5 + i] * u[col + stride*row];
				sumv = sumv + w[j * 5 + i] * v[col + stride*row];
			}
			//solve gaussian
		}
	}
	outputu[pos] = sumu;
	outputv[pos] = sumv;
	outputbku[pos] = bku[pos] + u[pos] - sumu;
	outputbkv[pos] = bkv[pos] + v[pos] - sumv;
}

__global__ void SolveSmoothGaussianGlobalKernel3(float* u, float* v, float* bku, float* bkv,
	int width, int height, int stride,
	float *outputu, float *outputv,
	float *outputbku, float* outputbkv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float w[9] = {0.0f, 0.1667f, 0.0f, 0.1667f, 0.3333f, 0.1667f, 0.0f, 0.1667f, 0.0f};

	float sumu = 0;
	float sumv = 0;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			//get values
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sumu = sumu + w[j * 3 + i] * u[col + stride*row];
				sumv = sumv + w[j * 3 + i] * v[col + stride*row];
			}
			//solve gaussian
		}
	}
	outputu[pos] = sumu;
	outputv[pos] = sumv;
	outputbku[pos] = bku[pos] + u[pos] - sumu;
	outputbkv[pos] = bkv[pos] + v[pos] - sumv;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////

void CudaFlow::SolveSmoothGaussianGlobal(float *inputu, float *inputv, float *inputbku, float *inputbkv,
	int w, int h, int s,
	float *outputu, float*outputv,
	float *outputbku, float *outputbkv,
	int kernelsize)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	if (kernelsize == 3) {
		SolveSmoothGaussianGlobalKernel3 << < blocks, threads >> > (inputu, inputv, inputbku, inputbkv,
			w, h, s, outputu, outputv, outputbku, outputbkv);
	}
	else if (kernelsize == 5) {
		SolveSmoothGaussianGlobalKernel5 << < blocks, threads >> > (inputu, inputv, inputbku, inputbkv,
			w, h, s, outputu, outputv, outputbku, outputbkv);
	}
	else {
		SolveSmoothGaussianGlobalKernel3 << < blocks, threads >> > (inputu, inputv, inputbku, inputbkv,
			w, h, s, outputu, outputv, outputbku, outputbkv);
	}
}
