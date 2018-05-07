#include "ReconFlow.h"


__global__
void SolveUVProjKernel(const float *u0, const float *v0,
	const float *du0, const float *dv0,
	const float *uc0, const float *vc0,
	const float *sku0, const float *skv0,
	int width, int height, int stride,
	float lambdaf, float alphaProj,
	float *uproj1, float *vproj1)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;      // current pixel index
		uproj1[pos] = (lambdaf*uc0[pos] + alphaProj*(u0[pos] + du0[pos] + sku0[pos])) / (lambdaf + alphaProj);
		vproj1[pos] = (lambdaf*vc0[pos] + alphaProj*(v0[pos] + dv0[pos] + skv0[pos])) / (lambdaf + alphaProj);
	}
}


void ReconFlow::SolveUVProj(const float *u0, const float *v0,
	const float *du0, const float *dv0,
	const float *uc0, const float *vc0,
	const float *sku0, const float *skv0,
	int w, int h, int s,
	float lambdaf, float alphaProj,
	float *uproj1, float *vproj1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveUVProjKernel <<< blocks, threads >>> (u0, v0,
		du0, dv0,
		uc0, vc0,
		sku0, skv0,
		w, h, s,
		lambdaf, alphaProj,
		uproj1, vproj1);
}
