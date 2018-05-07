#include "CudaFlow.h"

__global__
void rgbaToGrayKernel(uchar3 *d_iRgb, float *d_iGray, int width, int height, int stride)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((r < height) && (c < width))
	{
		int idx = c + stride * r;        // current pixel index 

		uchar3 pixel = d_iRgb[idx];

		//d_iGray[idx] = 0.2126f * (float)pixel.x + 0.7152f * (float)pixel.y + 0.0722f * (float)pixel.z;
		d_iGray[idx] = ((float)pixel.x + (float)pixel.y + (float)pixel.z)/3;
		d_iGray[idx] = d_iGray[idx] / 256.0f;
	}
}

__global__
void Cv8uToGrayKernel(uchar *d_iCv8u, float *d_iGray, int width, int height, int stride)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((r < height) && (c < width))
	{
		int idx = c + stride * r;        // current pixel index 

		//d_iGray[idx] = 0.2126f * (float)pixel.x + 0.7152f * (float)pixel.y + 0.0722f * (float)pixel.z;
		d_iGray[idx] = (float)d_iCv8u[idx] / 256.0f;
	}
}

__global__
void Cv16uToGrayKernel(ushort *d_iCv16u, float *d_iGray, int width, int height, int stride)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((r < height) && (c < width))
	{
		int idx = c + stride * r;        // current pixel index 

		//d_iGray[idx] = 0.2126f * (float)pixel.x + 0.7152f * (float)pixel.y + 0.0722f * (float)pixel.z;
		d_iGray[idx] = (float) d_iCv16u[idx] / 65536.0f;
	}
}

__global__
void Cv16uTo32fKernel(ushort *d_iCv16u, float *d_iGray, int width, int height, int stride)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((r < height) && (c < width))
	{
		int idx = c + stride * r;        // current pixel index 

										 //d_iGray[idx] = 0.2126f * (float)pixel.x + 0.7152f * (float)pixel.y + 0.0722f * (float)pixel.z;
		d_iGray[idx] = (float)d_iCv16u[idx];
	}
}

__global__
void Cv32fToGrayKernel(float *d_iCv32f, float *d_iGray, int width, int height, int stride)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((r < height) && (c < width))
	{
		int idx = c + stride * r;        // current pixel index 
		d_iGray[idx] = d_iCv32f[idx];
	}
}

//convert RGB image (0,255) to Floating point Grayscale (0,1)s with padding to fit BLOCK MODEL
void CudaFlow::rgbToGray(uchar3 * d_iRgb, float *d_iGray, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	rgbaToGrayKernel <<< blocks, threads >>>(d_iRgb, d_iGray, w, h, s);
}

void CudaFlow::Cv8uToGray(uchar * d_iCv8u, float *d_iGray, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	Cv8uToGrayKernel << < blocks, threads >> >(d_iCv8u, d_iGray, w, h, s);
}

void CudaFlow::Cv16uToGray(ushort * d_iCv16u, float *d_iGray, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	Cv16uToGrayKernel << < blocks, threads >> >(d_iCv16u, d_iGray, w, h, s);
}

void CudaFlow::Cv16uTo32f(ushort * d_iCv16u, float *d_iGray, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	Cv16uTo32fKernel << < blocks, threads >> >(d_iCv16u, d_iGray, w, h, s);
}

void CudaFlow::Cv32fToGray(float * d_iCv32f, float *d_iGray, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	Cv32fToGrayKernel << < blocks, threads >> >(d_iCv32f, d_iGray, w, h, s);
}