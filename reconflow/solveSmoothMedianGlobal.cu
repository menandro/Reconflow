#include "CudaFlow.h"

__global__ void SolveSmoothMedianGlobalKernel5(float* u, float* v, float* bku, float* bkv,
	int width, int height, int stride,
	float *outputu, float *outputv,
	float *outputbku, float* outputbkv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float mu[25] = { 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0,
					0, 0, 0, 0, 0,
					0, 0, 0, 0, 0,
					0, 0, 0, 0, 0 };

	float mv[25] = { 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0,
					0, 0, 0, 0, 0,
					0, 0, 0, 0, 0,
					0, 0, 0, 0, 0 };

	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			//get values
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[col + stride*row];
				mv[j * 5 + i] = v[col + stride*row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[stride*row];
				mv[j * 5 + i] = v[stride*row];
			}
			else if ((col > width) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[width - 1 + stride*row];
				mv[j * 5 + i] = v[width - 1 + stride*row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mu[j * 5 + i] = u[col];
				mv[j * 5 + i] = v[col];
			}
			else if ((col >= 0) && (col < width) && (row > height)) {
				mu[j * 5 + i] = u[col + stride*(height - 1)];
				mv[j * 5 + i] = v[col + stride*(height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpu, tmpv;
	for (int j = 0; j < 25; j++) {
		for (int i = j+1; i < 25; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
			}
			if (mv[j] > mv[i]) {
				//Swap the variables.
				tmpv = mv[j];
				mv[j] = mv[i];
				mv[i] = tmpv;
			}
		}
	}

	outputu[pos] = mu[12];
	outputv[pos] = mv[12];
	outputbku[pos] = bku[pos] + u[pos] - mu[12];
	outputbkv[pos] = bkv[pos] + v[pos] - mv[12];
}

__global__ void SolveSmoothMedianGlobalKernel3(float* u, float* v, float* bku, float* bkv, 
	int width, int height, int stride,
	float *outputu, float *outputv, 
	float *outputbku, float* outputbkv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float mu[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	float mv[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			//get values
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			int index = j * 3 + i;
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mu[index] = u[col + stride*row];
				mv[index] = v[col + stride*row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mu[index] = u[stride*row];
				mv[index] = v[stride*row];
			}
			else if ((col > width) && (row >= 0) && (row < height)) {
				mu[index] = u[width - 1 + stride*row];
				mv[index] = v[width - 1 + stride*row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mu[index] = u[col];
				mv[index] = v[col];
			}
			else if ((col >= 0) && (col < width) && (row > height)) {
				mu[index] = u[col + stride*(height - 1)];
				mv[index] = v[col + stride*(height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpu, tmpv;
	for (int j = 0; j < 9; j++) {
		for (int i = j + 1; i < 9; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
			}
			if (mv[j] > mv[i]) {
				//Swap the variables.
				tmpv = mv[j];
				mv[j] = mv[i];
				mv[i] = tmpv;
			}
		}
	}

	outputu[pos] = mu[4];
	outputv[pos] = mv[4];
	outputbku[pos] = bku[pos] + u[pos] - mu[4];
	outputbkv[pos] = bkv[pos] + v[pos] - mv[4];
}


////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
__global__ void SolveSmoothMedianApproxGlobalKernel5(float* u, float* v, float* bku, float* bkv,
	int width, int height, int stride,
	float *outputu, float *outputv,
	float *outputbku, float* outputbkv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;
	
	float binu[5] = { 0, 0, 0, 0, 0 }; //bin of values
	float binv[5] = { 0, 0, 0, 0, 0 };
	float medu[5] = { 0, 0, 0, 0, 0 }; //handler for median of each bin5
	float medv[5] = { 0, 0, 0, 0, 0 };

	float tmpu, tmpv;
	int m, n;
	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			//get values
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				binu[i] = u[col + stride*row];
				binv[i] = v[col + stride*row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				binu[i] = u[stride*row];
				binv[i] = v[stride*row];
			}
			else if ((col > width) && (row >= 0) && (row < height)) {
				binu[i] = u[width - 1 + stride*row];
				binv[i] = v[width - 1 + stride*row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				binu[i] = u[col];
				binv[i] = v[col];
			}
			else if ((col >= 0) && (col < width) && (row > height)) {
				binu[i] = u[col + stride*(height - 1)];
				binv[i] = v[col + stride*(height - 1)];
			}
		}
		//sort 5x1
		for (m = 0; m < 5; m++) {
			for (n = m + 1; n < 5; n++) {
				if (binu[m] > binu[n]) {
					//swap the variables.
					tmpu = binu[m];
					binu[m] = binu[n];
					binu[n] = tmpu;
				}
				if (binv[m] > binv[n]) {
					//swap the variables.
					tmpv = binv[m];
					binv[m] = binv[n];
					binv[n] = tmpv;
				}
			}
		}
		medu[j] = binu[2];
		medv[j] = binv[2];
	}

	//sort 5x1
	float medianu, medianv;
	for (m = 0; m < 5; m++) {
		for (n = m + 1; n < 5; n++) {
			if (medu[m] > medu[n]) {
				//Swap the variables.
				tmpu = medu[m];
				medu[m] = medu[n];
				medu[n] = tmpu;
			}
			if (medv[m] > medv[n]) {
				//Swap the variables.
				tmpv = medv[m];
				medv[m] = medv[n];
				medv[n] = tmpv;
			}
		}
	}
	outputu[pos] = medu[2];
	outputv[pos] = medv[2];
	outputbku[pos] = bku[pos] + u[pos] - medu[2];
	outputbkv[pos] = bkv[pos] + v[pos] - medv[2];
}


///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////

void CudaFlow::SolveSmoothMedianGlobal(float *inputu, float *inputv, float *inputbku, float *inputbkv,
	int w, int h, int s,
	float *outputu, float*outputv, 
	float *outputbku, float *outputbkv, 
	int kernelsize)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	if (kernelsize == 3) {
		SolveSmoothMedianGlobalKernel3 << < blocks, threads >> > (inputu, inputv, inputbku, inputbkv,
			w, h, s, outputu, outputv, outputbku, outputbkv);
	}
	else if (kernelsize == 5) {
		SolveSmoothMedianGlobalKernel5 << < blocks, threads >> > (inputu, inputv, inputbku, inputbkv,
			w, h, s, outputu, outputv, outputbku, outputbkv);
	}
	else if (kernelsize == 1) {
		SolveSmoothMedianApproxGlobalKernel5 << < blocks, threads >> > (inputu, inputv, inputbku, inputbkv,
			w, h, s, outputu, outputv, outputbku, outputbkv);
	}
	else {
		SolveSmoothMedianGlobalKernel3 << < blocks, threads >> > (inputu, inputv, inputbku, inputbkv,
			w, h, s, outputu, outputv, outputbku, outputbkv);
	}
	
}
