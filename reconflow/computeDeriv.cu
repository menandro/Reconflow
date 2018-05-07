#include "CudaFlow.h"

/// source image
texture<float, 2, cudaReadModeElementType> texInput;

__global__ void ComputeDerivKernel(int width, int height, int stride,
	float *Ix, float *Iy)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float t0;
	// x derivative
	t0 = tex2D(texInput, x - 2.0f * dx, y);
	t0 -= tex2D(texInput, x - 1.0f * dx, y) * 8.0f;
	t0 += tex2D(texInput, x + 1.0f * dx, y) * 8.0f;
	t0 -= tex2D(texInput, x + 2.0f * dx, y);
	t0 /= 12.0f;

	Ix[pos] = t0;

	// y derivative
	t0 = tex2D(texInput, x, y - 2.0f * dy);
	t0 -= tex2D(texInput, x, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texInput, x, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texInput, x, y + 2.0f * dy);
	t0 /= 12.0f;

	Iy[pos] = (t0);
}

__global__ void ComputeDerivMaskKernel(int width, int height, int stride,
	float *mask, float threshold)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float t0;
	// x derivative
	float ixderiv = (tex2D(texInput, x - 2.0f * dx, y) - tex2D(texInput, x - 1.0f * dx, y) * 8.0f 
		+ tex2D(texInput, x + 1.0f * dx, y) * 8.0f - tex2D(texInput, x + 2.0f * dx, y))/12.0f;
	float ixderiv2 = (tex2D(texInput, x - 4.0f * dx, y) - tex2D(texInput, x - 2.0f * dx, y) * 8.0f
		+ tex2D(texInput, x + 2.0f * dx, y) * 8.0f - tex2D(texInput, x + 4.0f * dx, y)) / 12.0f;
	float ixderiv4 = (tex2D(texInput, x - 8.0f * dx, y) - tex2D(texInput, x - 4.0f * dx, y) * 8.0f
		+ tex2D(texInput, x + 4.0f * dx, y) * 8.0f - tex2D(texInput, x + 8.0f * dx, y)) / 12.0f;
	//float ixderiv8 = (tex2D(texInput, x - 16.0f * dx, y) - tex2D(texInput, x - 8.0f * dx, y) * 8.0f
	//	+ tex2D(texInput, x + 8.0f * dx, y) * 8.0f - tex2D(texInput, x + 16.0f * dx, y)) / 12.0f;

	// y derivative
	float iyderiv = (tex2D(texInput, x, y - 2.0f * dy) - tex2D(texInput, x, y - 1.0f * dy) * 8.0f
		+ tex2D(texInput, x, y + 1.0f * dy) * 8.0f - tex2D(texInput, x, y + 2.0f * dy))/12.0f;
	float iyderiv2 = (tex2D(texInput, x, y - 4.0f * dy) - tex2D(texInput, x, y - 2.0f * dy) * 8.0f
		+ tex2D(texInput, x, y + 2.0f * dy) * 8.0f - tex2D(texInput, x, y + 4.0f * dy)) / 12.0f;
	float iyderiv4 = (tex2D(texInput, x, y - 8.0f * dy) - tex2D(texInput, x, y - 4.0f * dy) * 8.0f
		+ tex2D(texInput, x, y + 4.0f * dy) * 8.0f - tex2D(texInput, x, y + 8.0f * dy)) / 12.0f;
	//float iyderiv8 = (tex2D(texInput, x, y - 16.0f * dy) - tex2D(texInput, x, y - 8.0f * dy) * 8.0f
	//	+ tex2D(texInput, x, y + 8.0f * dy) * 8.0f - tex2D(texInput, x, y + 16.0f * dy)) / 12.0f;

	if ((ixderiv > threshold) || (iyderiv > threshold) || (ixderiv2 > threshold) || (iyderiv2 > threshold)
		|| (ixderiv4 > threshold) || (iyderiv4 > threshold)){// || (ixderiv8 > threshold) || (iyderiv8 > threshold)) {
		mask[pos] = 1.0f;
	}
	else mask[pos] = 0.0f;
}


void CudaFlow::ComputeDeriv(float *I0,
	int w, int h, int s,
	float *Ix, float *Iy)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texInput.addressMode[0] = cudaAddressModeMirror;
	texInput.addressMode[1] = cudaAddressModeMirror;
	texInput.filterMode = cudaFilterModeLinear;
	texInput.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texInput, I0, w, h, s * sizeof(float));

	ComputeDerivKernel << < blocks, threads >> >(w, h, s, Ix, Iy);
}

void CudaFlow::ComputeDerivMask(float *I0, int w, int h, int s, float *mask, float threshold) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texInput.addressMode[0] = cudaAddressModeMirror;
	texInput.addressMode[1] = cudaAddressModeMirror;
	texInput.filterMode = cudaFilterModeLinear;
	texInput.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texInput, I0, w, h, s * sizeof(float));
	ComputeDerivMaskKernel << < blocks, threads >> >(w, h, s, mask, threshold);
}
