#include "ReconFlow.h"


__global__
void SolveReconDataL1Kernel(const float *u0, const float *v0,
	const float *duhat0, const float *dvhat0,
	const float *uproj0, const float *vproj0,
	const float *sku0, const float *skv0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *It,
	int width, int height, int stride,
	float lambda, float alphaTv, float alphaProj,
	float *duhat1, float *dvhat1)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	float dix, diy, dit, duhat, dvhat, du, dv;

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;      // current pixel index
		dix = Ix[pos];
		diy = Iy[pos];
		dit = It[pos];
		float duhat = duhat0[pos];
		float dvhat = dvhat0[pos];
		float uproj = uproj0[pos];
		float vproj = vproj0[pos];
		float u = u0[pos];
		float v = v0[pos];
		float sku = sku0[pos];
		float skv = skv0[pos];

		float dusub = (alphaTv*duhat + alphaProj*(uproj - u - sku))/(alphaTv + alphaProj);
		float dvsub = (alphaTv*dvhat + alphaProj*(vproj - v - skv))/(alphaTv + alphaProj);

		//problem 1a
		float rho = (dix*dusub + diy*dvsub + dit);
		float upper = lambda*(dix*dix + diy*diy)/(alphaTv + alphaProj);
		float lower = -lambda*(dix*dix + diy*diy) / (alphaTv + alphaProj);

		if ((rho <= upper) && (rho >= lower)) {
			float magi = dix*dix + diy*diy;
			if (magi != 0) {
				du = dusub - rho*dix / magi;
				dv = dvsub - rho*diy / magi;
			}
			else {
				du = dusub;
				dv = dvsub;
			}

		}
		else if (rho < lower) {
			du = dusub + lambda*dix / (alphaTv + alphaProj);
			dv = dvsub + lambda*diy / (alphaTv + alphaProj);
		}
		else if (rho > upper) {
			du = dusub - lambda*dix / (alphaTv + alphaProj);
			dv = dvsub - lambda*diy / (alphaTv + alphaProj);
		}

		//problem 1b
		float divpu, divpv;
		int left = (ix - 1) + iy * stride;
		int right = (ix + 1) + iy * stride;
		int down = ix + (iy - 1) * stride;
		int up = ix + (iy + 1) * stride;

		if ((ix - 1) < 0) {
			if ((iy - 1) < 0) {
				//divpu = pu1[right] - pu1[pos] + pu2[up] - pu2[pos];
				//divpv = pv1[right] - pv1[pos] + pv2[up] - pv2[pos];
				divpu = pu1[pos] + pu2[pos];
				divpv = pv1[pos] + pv2[pos];
			}
			else {
				//divpu = pu1[right] - pu1[pos] + pu2[pos] - pu2[down];
				//divpv = pv1[right] - pv1[pos] + pv2[pos] - pv2[down];
				divpu = pu1[pos] + pu2[pos] - pu2[down];
				divpv = pv1[pos] + pv2[pos] - pv2[down];
			}
		}
		else {
			if ((iy - 1) < 0) {
				//divpu = pu1[pos] - pu1[left] + pu2[up] - pu2[pos];
				//divpv = pv1[pos] - pv1[left] + pv2[up] - pv2[pos];
				divpu = pu1[pos] - pu1[left] + pu2[pos];
				divpv = pv1[pos] - pv1[left] + pv2[pos];
			}
			else {
				divpu = pu1[pos] - pu1[left] + pu2[pos] - pu2[down];
				divpv = pv1[pos] - pv1[left] + pv2[pos] - pv2[down];
			}
		}

		duhat1[pos] = du + divpu / (alphaTv + alphaProj);
		dvhat1[pos] = dv + divpv / (alphaTv + alphaProj);
	}
}


void ReconFlow::SolveReconDataL1(const float *u0, const float *v0,
	const float *duhat0, const float *dvhat0,
	const float *uproj0, const float *vproj0,
	const float *sku0, const float *skv0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *Iz,
	int w, int h, int s,
	float lambda, float alphaTv, float alphaProj,
	float *duhat1, float *dvhat1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveReconDataL1Kernel << < blocks, threads >> > (u0, v0,
		duhat0, dvhat0,
		uproj0, vproj0,
		sku0, skv0,
		pu1, pu2,
		pv1, pv2,
		Ix, Iy, Iz,
		w, h, s,
		lambda, alphaTv, alphaProj,
		duhat1, dvhat1);
}


//************************************************
// DATA L1 for SOF3D with FlowNet as initial value
//************************************************

//dense
__global__
void SolveReconDataL1FlownetKernel(const float *u0, const float *v0,
	const float *duhat0, const float *dvhat0,
	const float *uproj0, const float *vproj0,
	const float *ufn0, const float *vfn0,
	const float *sku0, const float *skv0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *It,
	int width, int height, int stride,
	float lambda, float alphaTv, float alphaProj, float alphaFn,
	float *duhat1, float *dvhat1)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	float dix, diy, dit, duhat, dvhat, du, dv;

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;      // current pixel index
		dix = Ix[pos];
		diy = Iy[pos];
		dit = It[pos];
		float duhat = duhat0[pos];
		float dvhat = dvhat0[pos];
		float uproj = uproj0[pos];
		float vproj = vproj0[pos];
		float ufn = ufn0[pos];
		float vfn = vfn0[pos];
		float u = u0[pos];
		float v = v0[pos];
		float sku = sku0[pos];
		float skv = skv0[pos];

		float dusub = (alphaFn*(ufn - u) + alphaTv*duhat + alphaProj*(uproj - u - sku)) / (alphaTv + alphaProj + alphaFn);
		float dvsub = (alphaFn*(vfn - v) + alphaTv*dvhat + alphaProj*(vproj - v - skv)) / (alphaTv + alphaProj + alphaFn);

		//problem 1a
		float rho = (dix*dusub + diy*dvsub + dit);
		float upper = lambda*(dix*dix + diy*diy) / (alphaTv + alphaProj + alphaFn);
		float lower = -lambda*(dix*dix + diy*diy) / (alphaTv + alphaProj + alphaFn);

		if ((rho <= upper) && (rho >= lower)) {
			float magi = dix*dix + diy*diy;
			if (magi != 0) {
				du = dusub - rho*dix / magi;
				dv = dvsub - rho*diy / magi;
			}
			else {
				du = dusub;
				dv = dvsub;
			}

		}
		else if (rho < lower) {
			du = dusub + lambda*dix / (alphaTv + alphaProj + alphaFn);
			dv = dvsub + lambda*diy / (alphaTv + alphaProj + alphaFn);
		}
		else if (rho > upper) {
			du = dusub - lambda*dix / (alphaTv + alphaProj + alphaFn);
			dv = dvsub - lambda*diy / (alphaTv + alphaProj + alphaFn);
		}

		//problem 1b
		float divpu, divpv;
		int left = (ix - 1) + iy * stride;
		int right = (ix + 1) + iy * stride;
		int down = ix + (iy - 1) * stride;
		int up = ix + (iy + 1) * stride;

		if ((ix - 1) < 0) {
			if ((iy - 1) < 0) {
				//divpu = pu1[right] - pu1[pos] + pu2[up] - pu2[pos];
				//divpv = pv1[right] - pv1[pos] + pv2[up] - pv2[pos];
				divpu = pu1[pos] + pu2[pos];
				divpv = pv1[pos] + pv2[pos];
			}
			else {
				//divpu = pu1[right] - pu1[pos] + pu2[pos] - pu2[down];
				//divpv = pv1[right] - pv1[pos] + pv2[pos] - pv2[down];
				divpu = pu1[pos] + pu2[pos] - pu2[down];
				divpv = pv1[pos] + pv2[pos] - pv2[down];
			}
		}
		else {
			if ((iy - 1) < 0) {
				//divpu = pu1[pos] - pu1[left] + pu2[up] - pu2[pos];
				//divpv = pv1[pos] - pv1[left] + pv2[up] - pv2[pos];
				divpu = pu1[pos] - pu1[left] + pu2[pos];
				divpv = pv1[pos] - pv1[left] + pv2[pos];
			}
			else {
				divpu = pu1[pos] - pu1[left] + pu2[pos] - pu2[down];
				divpv = pv1[pos] - pv1[left] + pv2[pos] - pv2[down];
			}
		}

		duhat1[pos] = du + divpu / (alphaTv + alphaProj + alphaFn);
		dvhat1[pos] = dv + divpv / (alphaTv + alphaProj + alphaFn);
	}
}

// dense flownet
void ReconFlow::SolveReconDataL1Fn(const float *u0, const float *v0,
	const float *duhat0, const float *dvhat0,
	const float *uproj0, const float *vproj0,
	const float *ufn0, const float *vfn0,
	const float *sku0, const float *skv0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *Iz,
	int w, int h, int s,
	float lambda, float alphaTv, float alphaProj, float alphaFn,
	float *duhat1, float *dvhat1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveReconDataL1FlownetKernel << < blocks, threads >> > (u0, v0,
		duhat0, dvhat0,
		uproj0, vproj0,
		ufn0, vfn0,
		sku0, skv0,
		pu1, pu2,
		pv1, pv2,
		Ix, Iy, Iz,
		w, h, s,
		lambda, alphaTv, alphaProj, alphaFn,
		duhat1, dvhat1);
}

//******************************
//          sparse flownet
//******************************
__global__
void SolveReconDataL1FlownetKernel(const float *u0, const float *v0,
	const float *duhat0, const float *dvhat0,
	const float *uproj0, const float *vproj0,
	const float *ufn0, const float *vfn0,
	const float *sku0, const float *skv0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *It,
	int width, int height, int stride,
	float lambda, float alphaTv, float alphaProj, float alphaFnVal, float* fnmask,
	float *duhat1, float *dvhat1)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	float dix, diy, dit, duhat, dvhat, du, dv;

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;      // current pixel index
		dix = Ix[pos];
		diy = Iy[pos];
		dit = It[pos];
		float duhat = duhat0[pos];
		float dvhat = dvhat0[pos];
		float uproj = uproj0[pos];
		float vproj = vproj0[pos];
		float ufn = ufn0[pos];
		float vfn = vfn0[pos];
		float u = u0[pos];
		float v = v0[pos];
		float sku = sku0[pos];
		float skv = skv0[pos];
		float alphaFn = alphaFnVal*fnmask[pos];

		float dusub = (alphaFn*(ufn - u) + alphaTv*duhat + alphaProj*(uproj - u - sku)) / (alphaTv + alphaProj + alphaFn);
		float dvsub = (alphaFn*(vfn - v) + alphaTv*dvhat + alphaProj*(vproj - v - skv)) / (alphaTv + alphaProj + alphaFn);

		//problem 1a
		float rho = (dix*dusub + diy*dvsub + dit);
		float upper = lambda*(dix*dix + diy*diy) / (alphaTv + alphaProj + alphaFn);
		float lower = -lambda*(dix*dix + diy*diy) / (alphaTv + alphaProj + alphaFn);

		if ((rho <= upper) && (rho >= lower)) {
			float magi = dix*dix + diy*diy;
			if (magi != 0) {
				du = dusub - rho*dix / magi;
				dv = dvsub - rho*diy / magi;
			}
			else {
				du = dusub;
				dv = dvsub;
			}

		}
		else if (rho < lower) {
			du = dusub + lambda*dix / (alphaTv + alphaProj + alphaFn);
			dv = dvsub + lambda*diy / (alphaTv + alphaProj + alphaFn);
		}
		else if (rho > upper) {
			du = dusub - lambda*dix / (alphaTv + alphaProj + alphaFn);
			dv = dvsub - lambda*diy / (alphaTv + alphaProj + alphaFn);
		}

		//problem 1b
		float divpu, divpv;
		int left = (ix - 1) + iy * stride;
		int right = (ix + 1) + iy * stride;
		int down = ix + (iy - 1) * stride;
		int up = ix + (iy + 1) * stride;

		if ((ix - 1) < 0) {
			if ((iy - 1) < 0) {
				//divpu = pu1[right] - pu1[pos] + pu2[up] - pu2[pos];
				//divpv = pv1[right] - pv1[pos] + pv2[up] - pv2[pos];
				divpu = pu1[pos] + pu2[pos];
				divpv = pv1[pos] + pv2[pos];
			}
			else {
				//divpu = pu1[right] - pu1[pos] + pu2[pos] - pu2[down];
				//divpv = pv1[right] - pv1[pos] + pv2[pos] - pv2[down];
				divpu = pu1[pos] + pu2[pos] - pu2[down];
				divpv = pv1[pos] + pv2[pos] - pv2[down];
			}
		}
		else {
			if ((iy - 1) < 0) {
				//divpu = pu1[pos] - pu1[left] + pu2[up] - pu2[pos];
				//divpv = pv1[pos] - pv1[left] + pv2[up] - pv2[pos];
				divpu = pu1[pos] - pu1[left] + pu2[pos];
				divpv = pv1[pos] - pv1[left] + pv2[pos];
			}
			else {
				divpu = pu1[pos] - pu1[left] + pu2[pos] - pu2[down];
				divpv = pv1[pos] - pv1[left] + pv2[pos] - pv2[down];
			}
		}

		duhat1[pos] = du + divpu / (alphaTv + alphaProj + alphaFn);
		dvhat1[pos] = dv + divpv / (alphaTv + alphaProj + alphaFn);
	}
}

//sparse flownet
void ReconFlow::SolveReconDataL1Fn(const float *u0, const float *v0,
	const float *duhat0, const float *dvhat0,
	const float *uproj0, const float *vproj0,
	const float *ufn0, const float *vfn0,
	const float *sku0, const float *skv0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *Iz,
	int w, int h, int s,
	float lambda, float alphaTv, float alphaProj, float alphaFn, float* fnmask,
	float *duhat1, float *dvhat1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveReconDataL1FlownetKernel << < blocks, threads >> > (u0, v0,
		duhat0, dvhat0,
		uproj0, vproj0,
		ufn0, vfn0,
		sku0, skv0,
		pu1, pu2,
		pv1, pv2,
		Ix, Iy, Iz,
		w, h, s,
		lambda, alphaTv, alphaProj, alphaFn, fnmask,
		duhat1, dvhat1);
}
