#include "CudaFlow.h"

///search image
texture<float, 2, cudaReadModeElementType> texKernelImage;
texture<float, 2, cudaReadModeElementType> texSearchImage;


__global__ void Correlation1x1Kernel(float kernel, float*searchSpace, float*output,
	int width, int height, int stride) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float diff = searchSpace[iy*stride + ix] - kernel;
	output[pos] = diff*diff;
}

__global__ void CorrelationKernel(float* kernel, float* searchSpace, float* output,
	int width, int height, int stride, int kernelSize) {

	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float total = 0.0f;
	int offset = (kernelSize - 1) / 2;
	for (int j = 0; j < kernelSize; j++) {
		for (int i = 0; i < kernelSize; i++) {
			int col = ix + i - offset;
			int row = iy + j - offset;
			//correlate
			float diff = searchSpace[row*stride + col] - kernel[j*kernelSize + i];
			total += diff*diff;
			//total += abs(searchSpace[row*stride + col] * kernel[j*kernelSize + i]);
		}
	}
	//normalize
	output[pos] = total;// / (kernelSize*kernelSize);
}

__global__ void CorrelationKernelSamplingKernel(int x, int y, float* kernel, int width, int height, int kernelSize) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * kernelSize;
	if (ix >= kernelSize || iy >= kernelSize) return;

	float texx = ((float)ix + (float)x + 0.5f - (kernelSize - 1) / 2) / (float)width;
	float texy = ((float)iy + (float)y + 0.5f - (kernelSize - 1) / 2) / (float)height;

	kernel[pos] = tex2D(texKernelImage, texx, texy);
}

__global__ void CorrelationSearchSamplingKernel(int x, int y, float* searchSpace, int maxSearchWidth, int maxSearchHeight, int corrStride, int width, int height) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * corrStride;
	if (ix >= maxSearchWidth || iy >= maxSearchHeight) return;

	float texx = ((float)ix + (float)x + 0.5f - (maxSearchWidth) / 2) / (float)width;
	float texy = ((float)iy + (float)y + 0.5f - (maxSearchHeight) / 2) / (float)height;

	searchSpace[pos] = tex2D(texSearchImage, texx, texy);
}



void CudaFlow::Correlation(float* kernel, float* searchSpace, float* output)
{
	dim3 threads(32, 12);
	dim3 blocks(iDivUp(corrMaxSearchWidth, threads.x), iDivUp(corrMaxSearchHeight, threads.y));

	CorrelationKernel << < blocks, threads >> > (kernel, searchSpace, output, corrMaxSearchWidth, corrMaxSearchHeight, corrStride, corrKernelSize);
}

void CudaFlow::Correlation1x1(float kernel, float* searchSpace, float* output)
{
	dim3 threads(32, 12);
	dim3 blocks(iDivUp(corrMaxSearchWidth, threads.x), iDivUp(corrMaxSearchHeight, threads.y));

	Correlation1x1Kernel << < blocks, threads >> > (kernel, searchSpace, output, corrMaxSearchWidth, corrMaxSearchHeight, corrStride);
}

void CudaFlow::CorrelationBindTextures(float* im0, float*im1, int w, int h, int s) {
	texSearchImage.addressMode[0] = cudaAddressModeClamp;
	texSearchImage.addressMode[1] = cudaAddressModeClamp;
	texSearchImage.filterMode = cudaFilterModeLinear;
	texSearchImage.normalized = true;

	texKernelImage.addressMode[0] = cudaAddressModeClamp;
	texKernelImage.addressMode[1] = cudaAddressModeClamp;
	texKernelImage.filterMode = cudaFilterModeLinear;
	texKernelImage.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texKernelImage, im0, w, h, s * sizeof(float)));
	checkCudaErrors(cudaBindTexture2D(0, texSearchImage, im1, w, h, s * sizeof(float)));
}

void CudaFlow::CorrelationKernelSampling(int x, int y, float* kernel, int w, int h) {
	dim3 threads(corrKernelSize, corrKernelSize);
	dim3 blocks(iDivUp(corrKernelSize, threads.x), iDivUp(corrKernelSize, threads.y));
	CorrelationKernelSamplingKernel << < blocks, threads >> > (x, y, kernel, w, h, corrKernelSize);
}

void CudaFlow::CorrelationSearchSampling(int x, int y, float* searchSpace) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(corrMaxSearchWidth, threads.x), iDivUp(corrMaxSearchHeight, threads.y));
	CorrelationSearchSamplingKernel << < blocks, threads >> > (x, y, searchSpace, corrMaxSearchWidth, corrMaxSearchHeight, corrStride, width, height);
}


__global__ void GetValueKernel(float *input, int idx, float &value) {
	value = input[idx];
}

void CudaFlow::GetValue(float *input, int idx, float &value) {
	GetValueKernel << <1, 1 >> > (input, idx, value);
}