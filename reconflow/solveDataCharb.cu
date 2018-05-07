#include "CudaFlow.h"

__global__
void SolveDataCharbKernel(const float *du0, const float *dv0,
	const float *dumed0, const float *dvmed0,
	const float *bku0, const float *bkv0,
	const float *Ix,
	const float *Iy,
	const float *It,
	int width, int height, int stride,
	float lambdam,
	float *du1,
	float *dv1)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	float beta, a11, a12, a21, a22, detA, b1, b2, innerSum;
	float ix, iy, it, du, dv, dumed, dvmed, bku, bkv;

	if ((r < height) && (c < width))
	{
		int i = c + stride * r;        // current pixel index 
		du = du0[i];
		dv = dv0[i];
		ix = Ix[i];
		iy = Iy[i];
		it = It[i];
		dumed = dumed0[i];
		dvmed = dvmed0[i];
		bku = bku0[i];
		bkv = bkv0[i];

		for (int iter = 0; iter < 2; iter++) {
			innerSum = ix * du + iy * dv + it;
			beta = 1.0f/sqrtf(innerSum*innerSum + 0.0001);
			
			//construct matrix
			a11 = ix * ix * beta + lambdam;
			a12 = ix * iy * beta;
			a21 = a12;
			a22 = iy * iy * beta + lambdam;

			detA = a11*a22 - a12*a21;

			b1 = lambdam*(dumed - bku) - ix * it * beta;
			b2 = lambdam*(dvmed - bkv) - iy * it * beta;

			du = (a22*b1 - a12*b2) / detA;
			dv = (-a21*b1 + a11*b2) / detA;
		}
		du1[i] = du;
		dv1[i] = dv;
	}

}

///////////////////////////////////////////////////////////////////////////////

void CudaFlow::SolveDataCharb(const float *du0, const float *dv0,
	const float *dumed, const float *dvmed,
	const float *bku, const float *bkv,
	const float *Ix,
	const float *Iy,
	const float *Iz,
	int w, int h, int s,
	float alpha,
	float *du1,
	float *dv1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveDataCharbKernel <<< blocks, threads >>> (du0, dv0, dumed, dvmed, bku, bkv,
		Ix, Iy, Iz,
		w, h, s, alpha, du1, dv1);
}
