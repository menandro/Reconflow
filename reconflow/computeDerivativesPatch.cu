#include "CudaFlow.h"

/// source image
texture<float, 2, cudaReadModeElementType> texSource;
/// tracked image
texture<float, 2, cudaReadModeElementType> texTarget;

__global__ void ComputeDerivativesPatchKernel(int width, int height, int stride,
	float *Ix, float *Iy, float *Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;
	float x, y;
	float t0, t1;
	float Ixsub = 0.0f;
	float Iysub = 0.0f;
	float Izsub = 0.0f;

	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			//get values
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			x = ((float)col + 0.5f) * dx;
			y = ((float)row + 0.5f) * dy;

			t0 = tex2D(texSource, x - 2.0f * dx, y);
			t0 -= tex2D(texSource, x - 1.0f * dx, y) * 8.0f;
			t0 += tex2D(texSource, x + 1.0f * dx, y) * 8.0f;
			t0 -= tex2D(texSource, x + 2.0f * dx, y);
			t0 /= 12.0f;

			t1 = tex2D(texTarget, x - 2.0f * dx, y);
			t1 -= tex2D(texTarget, x - 1.0f * dx, y) * 8.0f;
			t1 += tex2D(texTarget, x + 1.0f * dx, y) * 8.0f;
			t1 -= tex2D(texTarget, x + 2.0f * dx, y);
			t1 /= 12.0f;

			Ixsub += (t0 + t1) * 0.5f;
			Izsub += tex2D(texTarget, x, y) - tex2D(texSource, x, y);

			t0 = tex2D(texSource, x, y - 2.0f * dy);
			t0 -= tex2D(texSource, x, y - 1.0f * dy) * 8.0f;
			t0 += tex2D(texSource, x, y + 1.0f * dy) * 8.0f;
			t0 -= tex2D(texSource, x, y + 2.0f * dy);
			t0 /= 12.0f;

			t1 = tex2D(texTarget, x, y - 2.0f * dy);
			t1 -= tex2D(texTarget, x, y - 1.0f * dy) * 8.0f;
			t1 += tex2D(texTarget, x, y + 1.0f * dy) * 8.0f;
			t1 -= tex2D(texTarget, x, y + 2.0f * dy);
			t1 /= 12.0f;

			Iysub += (t0 + t1) * 0.5f;
		}
	}
	Ix[pos] = Ixsub;
	Iy[pos] = Iysub;
	Iz[pos] = Izsub;
}

///CUDA CALL FUNCTIONS ***********************************************************
void CudaFlow::ComputeDerivativesPatch(float *I0, float *I1,
	int w, int h, int s,
	float *Ix, float *Iy, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texSource.addressMode[0] = cudaAddressModeMirror;
	texSource.addressMode[1] = cudaAddressModeMirror;
	texSource.filterMode = cudaFilterModeLinear;
	texSource.normalized = true;

	texTarget.addressMode[0] = cudaAddressModeMirror;
	texTarget.addressMode[1] = cudaAddressModeMirror;
	texTarget.filterMode = cudaFilterModeLinear;
	texTarget.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texSource, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texTarget, I1, w, h, s * sizeof(float));

	ComputeDerivativesPatchKernel << < blocks, threads >> >(w, h, s, Ix, Iy, Iz);
}

