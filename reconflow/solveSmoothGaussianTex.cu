#include "CudaFlow.h"

/// source image
texture<float, 2, cudaReadModeElementType> texSourceU;
texture<float, 2, cudaReadModeElementType> texSourceV;

__global__ void SolveSmoothGaussianTexKernel(int width, int height, int stride,
	float *outputu, float *outputv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float w[25] = { 0.0037, 0.0147, 0.0256, 0.0147, 0.0037,
					0.0147, 0.0586, 0.0952, 0.0586, 0.0147,
					0.0256, 0.0952, 0.1502, 0.0952, 0.0256,
					0.0147, 0.0586, 0.0952, 0.0586, 0.0147,
					0.0037, 0.0147, 0.0256, 0.0147, 0.0037};
	float sumu = 0;
	float sumv = 0;
	for (int j=0; j < 5; j++){
		for (int i = 0; i < 5; i++) {
			sumu = sumu + w[j * 5 + i] * tex2D(texSourceU, x + ((float)i - 2.0f)*dx, y + ((float)j - 2.0f)*dy);
			sumv = sumv + w[j * 5 + i] * tex2D(texSourceV, x + ((float)i - 2.0f)*dx, y + ((float)j - 2.0f)*dy);
		}
	}
	outputu[pos] = sumu;
	outputv[pos] = sumv;
}

void CudaFlow::SolveSmoothGaussianTex(float *inputu, float *inputv,
	int w, int h, int s,
	float *outputu, float*outputv)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texSourceU.addressMode[0] = cudaAddressModeBorder;
	texSourceU.addressMode[1] = cudaAddressModeBorder;
	texSourceU.filterMode = cudaFilterModeLinear;
	texSourceU.normalized = true;

	texSourceV.addressMode[0] = cudaAddressModeBorder;
	texSourceV.addressMode[1] = cudaAddressModeBorder;
	texSourceV.filterMode = cudaFilterModeLinear;
	texSourceV.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texSourceU, inputu, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texSourceV, inputv, w, h, s * sizeof(float));

	SolveSmoothGaussianTexKernel <<< blocks, threads >>> (w, h, s, outputu, outputv);
}
