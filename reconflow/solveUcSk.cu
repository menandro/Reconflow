#include "ReconFlow.h"

__global__
void SolveUcKernel(const float *X0, const float *Y0, const float *Z0,
	float R11, float R12, float R13,
	float R21, float R22, float R23,
	float R31, float R32, float R33,
	float t1, float t2, float t3,
	float focals, float camMidXs, float camMidYs,
	int width, int height, int stride,
	float *uc, float *vc)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 


	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;      // current pixel index
		
		float X = X0[pos];
		float Y = Y0[pos];
		float Z = Z0[pos];

		float X2 = R11*X + R12*Y + R13*Z + t1; 
		float Y2 = R21*X + R22*Y + R23*Z + t2;
		float Z2 = R31*X + R32*Y + R33*Z + t3;

		float xproj = focals*X2 / Z2 + camMidXs;
		float yproj = focals*Y2 / Z2 + camMidYs;

		uc[pos] = xproj - ix;
		vc[pos] = yproj - iy;
	}
}

void ReconFlow::SolveUc(const float *X0, const float *Y0, const float *Z0,
	double *R, double *t, double *K1,
	int w, int h, int s,
	float *uc, float *vc)
{
	float R11 = (float)R[0];
	float R12 = (float)R[1];
	float R13 = (float)R[2];
	float R21 = (float)R[3];
	float R22 = (float)R[4];
	float R23 = (float)R[5];
	float R31 = (float)R[6];
	float R32 = (float)R[7];
	float R33 = (float)R[8];
	float t1 = (float)t[0];
	float t2 = (float)t[1];
	float t3 = (float)t[2];
	float focals = (float)K1[0];
	float camMidXs = (float)K1[2];
	float camMidYs = (float)K1[5];
	//std::cout << R11 << " " << R12 << " " << R13 << std::endl;
	//std::cout << R21 << " " << R22 << " " << R23 << std::endl;
	//std::cout << R31 << " " << R32 << " " << R33 << std::endl;
	//std::cout << focals << " " << camMidXs << " " << camMidYs << std::endl;
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveUcKernel << < blocks, threads >> > (X0, Y0, Z0, 
		R11, R12, R13,
		R21, R22, R23,
		R31, R32, R33,
		t1, t2, t3,
		focals, camMidXs, camMidYs,
		w, h, s,
		uc, vc);
}


//*******************************
// Update Sk
//*******************************
__global__ void UpdateSkKernel(const float *sku0, const float *skv0,
	const float *u0, const float *v0,
	const float *du0, const float *dv0,
	const float *uproj0, const float *vproj0,
	int width, int height, int stride,
	float *sku1, float *skv1) 
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 


	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		sku1[pos] = sku0[pos] + u0[pos] + du0[pos] - uproj0[pos];
		skv1[pos] = skv0[pos] + v0[pos] + dv0[pos] - vproj0[pos];
	}
}

void ReconFlow::UpdateSk(const float *sku0, const float *skv0,
	const float *u0, const float *v0,
	const float *du0, const float *dv0,
	const float *uproj0, const float *vproj0,
	int w, int h, int s,
	float *sku1, float *skv1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	UpdateSkKernel << < blocks, threads >> > (sku0, skv0,
		u0, v0, du0, dv0, uproj0, vproj0,
		w, h, s,
		sku1, skv1);
}