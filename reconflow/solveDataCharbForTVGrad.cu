#include "CudaFlow.h"

__global__
void SolveDataCharbForTVGradKernel(const float *du0, const float *dv0,
	const float *duhat0, const float *dvhat0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *It,
	const float *Ixx, const float *Ixy, const float *Ixt,
	const float *Iyx, const float *Iyy, const float *Iyt,
	int width, int height, int stride,
	float lambda, float lambdagrad, float theta,
	float *du1, float *dv1,
	float *duhat1, float *dvhat1)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	float betax, betay, beta;
	float a11, a12, a21, a22, detA, b1, b2, innerSum, innerSumx, innerSumy;
	float dix, diy, dit, du, dv, duhat, dvhat, bku, bkv;
	float dixx, dixy, dixt;
	float diyx, diyy, diyt;

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;       // current pixel index
		du = du0[pos];
		dv = dv0[pos];

		dix = Ix[pos];
		diy = Iy[pos];
		dit = It[pos];

		dixx = Ixx[pos];
		dixy = Ixy[pos];
		dixt = Ixt[pos];

		diyx = Iyx[pos];
		diyy = Iyy[pos];
		diyt = Iyt[pos];
		duhat = duhat0[pos];
		dvhat = dvhat0[pos];

		for (int iter = 0; iter < 2; iter++) {
			innerSum = dix * du + diy * dv + dit;
			innerSumx = dixx * du + dixy * dv + dixt;
			innerSumy = diyx * du + diyy * dv + diyt;
			beta = lambda / sqrtf(innerSum*innerSum + 0.001);
			betax = lambdagrad / sqrtf(innerSumx*innerSumx + 0.001);
			betay = lambdagrad / sqrtf(innerSumy*innerSumy + 0.001);

			//construct matrix
			a11 = dix * dix * beta + dixx * dixx * betax + diyx * diyx * betay + 1.0f / theta;
			a12 = dix * diy * beta + dixx * dixy * betax + diyx * diyy * betay;
			a21 = a12;
			a22 = diy * diy * beta + dixy * dixy * betax + diyy * diyy * betay + 1.0f / theta;

			detA = a11*a22 - a12*a21;

			b1 = (1.0f / theta)*(duhat) - dix * dit * beta - dixx * dixt * betax - diyx * diyt * betay;
			b2 = (1.0f / theta)*(dvhat) - diy * dit * beta - dixy * dixt * betax - diyy * diyt * betay;

			du = (a22*b1 - a12*b2) / detA;
			dv = (-a21*b1 + a11*b2) / detA;
		}

		du1[pos] = du;
		dv1[pos] = dv;

		//problem1b
		float divpu, divpv;
		int left = (ix - 1) + iy * stride;
		int right = (ix + 1) + iy * stride;
		int down = ix + (iy - 1) * stride;
		int up = ix + (iy + 1) * stride;

		if ((ix - 1) < 0) {
			if ((iy - 1) < 0) {
				divpu = pu1[pos] + pu2[pos];
				divpv = pv1[pos] + pv2[pos];
			}
			else {
				divpu = pu1[pos] + pu2[pos] - pu2[down];
				divpv = pv1[pos] + pv2[pos] - pv2[down];
			}
		}
		else {
			if ((iy - 1) < 0) {
				divpu = pu1[pos] - pu1[left] + pu2[pos];
				divpv = pv1[pos] - pv1[left] + pv2[pos];
			}
			else {
				divpu = pu1[pos] - pu1[left] + pu2[pos] - pu2[down];
				divpv = pv1[pos] - pv1[left] + pv2[pos] - pv2[down];
			}
		}

		duhat1[pos] = du + theta*divpu;
		dvhat1[pos] = dv + theta*divpv;
	}

}

///////////////////////////////////////////////////////////////////////////////


void CudaFlow::SolveDataCharbForTVGrad(const float *du0, const float *dv0,
	const float *duhat, const float *dvhat,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *Iz,
	const float *Ixx, const float *Ixy, const float *Ixz,
	const float *Iyx, const float *Iyy, const float *Iyz,
	int w, int h, int s,
	float lambda, float lambdagrad, float theta,
	float *du1, float *dv1,
	float *duhat1, float *dvhat1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveDataCharbForTVGradKernel << < blocks, threads >> > (du0, dv0,
		duhat, dvhat,
		pu1, pu2, pv1, pv2,
		Ix, Iy, Iz,
		Ixx, Ixy, Ixz,
		Iyx, Iyy, Iyz,
		w, h, s,
		lambda, lambdagrad, theta,
		du1, dv1,
		duhat1, dvhat1);
}
