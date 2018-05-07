#include "ReconFlow.h"

__global__
void Solve3dMinimalAreaKernel(const float *uproj0, const float *vproj0,
	const float *X0, const float *Y0, const float *Z0,
	float P11, float P12, float P13, float P14, 
	float P21, float P22, float P23, float P24, 
	float P31, float P32, float P33, float P34,
	float Q11, float Q12, float Q13, float Q14, 
	float Q21, float Q22, float Q23, float Q24, 
	float Q31, float Q32, float Q33, float Q34,
	float lambdaf, float lambdams,
	int width, int height, int stride,
	float *X, float *Y, float *Z)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		float x1 = ix;
		float y1 = iy;
		float x2 = ix + uproj0[pos];
		float y2 = iy + vproj0[pos];

		float p1 = x1*P31 - P11;
		float p2 = x1*P32 - P12;
		float p3 = x1*P33 - P13;
		float p4 = x1*P34 - P14;

		float q1 = y1*P31 - P21;
		float q2 = y1*P32 - P22;
		float q3 = y1*P33 - P23;
		float q4 = y1*P34 - P24;

		float r1 = x2*Q31 - Q11;
		float r2 = x2*Q32 - Q12;
		float r3 = x2*Q33 - Q13;
		float r4 = x2*Q34 - Q14;

		float s1 = y2*Q31 - Q21;
		float s2 = y2*Q32 - Q22;
		float s3 = y2*Q33 - Q23;
		float s4 = y2*Q34 - Q24;

		float ax = lambdaf*(p1*p1 + q1*q1 + r1*r1 + s1*s1);
		float bx = lambdaf*(p1*p2 + q1*q2 + r1*r2 + s1*s2);
		float cx = lambdaf*(p1*p3 + q1*q3 + r1*r3 + s1*s3);
		float dx = lambdaf*(p1*p4 + q1*q4 + r1*r4 + s1*s4);

		float ay = lambdaf*(p2*p1 + q2*q1 + r2*r1 + s2*s1);
		float by = lambdaf*(p2*p2 + q2*q2 + r2*r2 + s2*s2);
		float cy = lambdaf*(p2*p3 + q2*q3 + r2*r3 + s2*s3);
		float dy = lambdaf*(p2*p4 + q2*q4 + r2*r4 + s2*s4);

		float az = lambdaf*(p3*p1 + q3*q1 + r3*r1 + s3*s1);
		float bz = lambdaf*(p3*p2 + q3*q2 + r3*r2 + s3*s2);
		float cz = lambdaf*(p3*p3 + q3*q3 + r3*r3 + s3*s3);
		float dz = lambdaf*(p3*p4 + q3*q4 + r3*r4 + s3*s4);

		//Solve Partial derivatives of X, Y, Z
		float Xx, Yx, Zx, Xy, Yy, Zy;
		if (ix + 1 > width) {
			Xx = -X0[ix - 1 + iy * stride] + X0[pos];
			Yx = -Y0[ix - 1 + iy * stride] + Y0[pos];
			Zx = -Z0[ix - 1 + iy * stride] + Z0[pos];
		}
		else {
			Xx = X0[ix + 1 + iy * stride] - X0[pos];
			Yx = Y0[ix + 1 + iy * stride] - Y0[pos];
			Zx = Z0[ix + 1 + iy * stride] - Z0[pos];
		}

		if (iy + 1 > height) {
			Xy = -X0[ix + (iy - 1) * stride] + X0[pos];
			Yy = -Y0[ix + (iy - 1) * stride] + Y0[pos];
			Zy = -Z0[ix + (iy - 1) * stride] + Z0[pos];
		}
		else {
			Xy = X0[ix + (iy + 1) * stride] - X0[pos];
			Yy = Y0[ix + (iy + 1) * stride] - Y0[pos];
			Zy = Z0[ix + (iy + 1) * stride] - Z0[pos];
		}

		float lambetams = lambdams / sqrt((Yx*Zy - Zx*Yy)*(Yx*Zy - Zx*Yy) + (Zx*Xy - Xx*Zy)*(Zx*Xy - Xx*Zy) + (Xx*Yy - Yx*Xy)*(Xx*Yy - Yx*Xy) + 0.001f);

		//Solve Xbars, Ybars and Zbars
		float Xibar, Yibar, Zibar, Xjbar, Yjbar, Zjbar;
		if (ix - 1 < 0) {
			Xibar = X0[ix + 1 + iy * stride];
			Yibar = Y0[ix + 1 + iy * stride];
			Zibar = Z0[ix + 1 + iy * stride];
		}
		else if (ix + 1 > width) {
			Xibar = X0[ix - 1 + iy * stride];
			Yibar = Y0[ix - 1 + iy * stride];
			Zibar = Z0[ix - 1 + iy * stride];
		}
		else {
			Xibar = 0.5f * X0[ix - 1 + iy * stride] + 0.5f * X0[ix + 1 + iy * stride];
			Yibar = 0.5f * Y0[ix - 1 + iy * stride] + 0.5f * Y0[ix + 1 + iy * stride];
			Zibar = 0.5f * Z0[ix - 1 + iy * stride] + 0.5f * Z0[ix + 1 + iy * stride];
		}

		if (iy - 1 < 0) {
			Xjbar = X0[ix + (iy + 1) * stride];
			Yjbar = Y0[ix + (iy + 1) * stride];
			Zjbar = Z0[ix + (iy + 1) * stride];
		}
		else if (iy + 1 > height) {
			Xjbar = X0[ix + (iy - 1) * stride];
			Yjbar = Y0[ix + (iy - 1) * stride];
			Zjbar = Z0[ix + (iy - 1) * stride];
		}
		else {
			Xjbar = 0.5f * X0[ix + (iy - 1) * stride] + 0.5f * X0[ix + (iy + 1) * stride];
			Yjbar = 0.5f * Y0[ix + (iy - 1) * stride] + 0.5f * Y0[ix + (iy + 1) * stride];
			Zjbar = 0.5f * Z0[ix + (iy - 1) * stride] + 0.5f * Z0[ix + (iy + 1) * stride];
		}


		float a11 = ax + lambetams*(Yx*Yx + Zx*Zx + Yy*Yy + Zy*Zy);
		float a12 = bx - lambetams*(Xx*Yx + Xy*Yy);
		float a13 = cx - lambetams*(Xx*Zx + Xy*Zy);
		float a21 = ay - lambetams*(Xx*Yx + Xy*Yy);
		float a22 = by + lambetams*(Xx*Xx + Zx*Zx + Xy*Xy + Zy*Zy);
		float a23 = cy - lambetams*(Yx*Zx + Yy*Zy);
		float a31 = az - lambetams*(Xx*Zx + Xy*Zy);
		float a32 = bz - lambetams*(Yx*Zx + Yy*Zy);
		float a33 = cz + lambetams*(Xx*Xx + Yx*Yx + Xy*Xy + Yy*Yy);

		float detA = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31);

		float E11 = a22*a33 - a23*a32;
		float E12 = a13*a32 - a12*a33;
		float E13 = a12*a23 - a13*a22;
		float E21 = a23*a31 - a21*a33;
		float E22 = a11*a33 - a13*a31;
		float E23 = a13*a21 - a11*a23;
		float E31 = a21*a32 - a22*a31;
		float E32 = a12*a31 - a11*a32;
		float E33 = a11*a22 - a12*a21;

		float b1 = lambetams*((Zy*Zy + Yy*Yy)*Xibar + (Zx*Zx + Yx*Yx)*Xjbar
			- (Xy*Yy)*Yibar - (Xx*Yx)*Yjbar
			- (Xy*Zy)*Zibar - (Xx*Zx)*Zjbar)
			- dx;
		float b2 = lambetams*(-(Xy*Yy)*Xibar - (Xx*Yx)*Xjbar
			+ (Xy*Xy + Zy*Zy)*Yibar + (Xx*Xx + Zx*Zx)*Yjbar
			- (Yy*Zy)*Zibar - (Yx*Zx)*Zjbar)
			- dy;
		float b3 = lambetams*(-(Xy*Zy)*Xibar - (Xx*Zx)*Xjbar
			- (Yy*Zy)*Yibar - (Yx*Zx)*Yjbar
			+ (Xy*Xy + Yy*Yy)*Zibar + (Xx*Xx + Yx*Yx)*Zjbar)
			- dz;

		float Xsub = (E11*b1 + E12*b2 + E13*b3) / detA;
		float Ysub = (E21*b1 + E22*b2 + E23*b3) / detA;
		float Zsub = (E31*b1 + E32*b2 + E33*b3) / detA;

		//if ((abs(Xsub) >  200) || (abs(Ysub) > 200) || (abs(Zsub) > 200)) {
		/*if ((isinf(Zsub)) || (isnan(Zsub))) {
			X[pos] = 0.0f;
			Y[pos] = 0.0f;
			Z[pos] = 0.0f;
		}
		else {
			X[pos] = Xsub;
			Y[pos] = Ysub;
			Z[pos] = Zsub;
		}*/
		
		if (isfinite(Xsub) && isfinite(Ysub) && isfinite(Zsub)) {
			X[pos] = Xsub;
			Y[pos] = Ysub;
			Z[pos] = Zsub;
		}
		else {
			X[pos] = X0[pos];
			Y[pos] = Y0[pos];
			Z[pos] = Z0[pos];
		}
		

		/*if (sqrt(uproj0[pos] * uproj0[pos] + vproj0[pos] * vproj0[pos]) < 2.0) {
			X[pos] = 100000.0f;
			Y[pos] = 100000.0f;
			Z[pos] = 100000.0f;
		}*/
	}
}

///////////////////////////////////////////////////////////////////////////////

void ReconFlow::Solve3dMinimalArea(const float *uproj0, const float *vproj0,
	const float *X0, const float *Y0, const float *Z0,
	double *P, double *Q, //camera matrices
	float lambdaf, float lambdams,
	int w, int h, int s,
	float *X, float *Y, float *Z)
{
	float P11 = (float)P[0];
	float P12 = (float)P[1];
	float P13 = (float)P[2];
	float P14 = (float)P[3];
	float P21 = (float)P[4];
	float P22 = (float)P[5];
	float P23 = (float)P[6];
	float P24 = (float)P[7];
	float P31 = (float)P[8];
	float P32 = (float)P[9];
	float P33 = (float)P[10];
	float P34 = (float)P[11];

	float Q11 = (float)Q[0];
	float Q12 = (float)Q[1];
	float Q13 = (float)Q[2];
	float Q14 = (float)Q[3];
	float Q21 = (float)Q[4];
	float Q22 = (float)Q[5];
	float Q23 = (float)Q[6];
	float Q24 = (float)Q[7];
	float Q31 = (float)Q[8];
	float Q32 = (float)Q[9];
	float Q33 = (float)Q[10];
	float Q34 = (float)Q[11];
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	Solve3dMinimalAreaKernel <<< blocks, threads >>> (uproj0, vproj0,
		X0, Y0, Z0,
		P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34,
		Q11, Q12, Q13, Q14, Q21, Q22, Q23, Q24, Q31, Q32, Q33, Q34,
		lambdaf, lambdams,
		w, h, s,
		X, Y, Z);
}


//********************************************
// Filter XYZ
//********************************************

__global__
void CleanUp3DKernel(const float *X0, const float *Y0, const float *Z0,
	const float *Xmed0, const float *Ymed0, const float *Zmed0,
	int width, int height, int stride,
	float *X1, float *Y1, float *Z1)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		float Xmed = Xmed0[pos];
		float Ymed = Ymed0[pos];
		float Zmed = Zmed0[pos];
		float X = X0[pos];
		float Y = Y0[pos];
		float Z = Z0[pos];
		
		if (isfinite(X) && isfinite(Y) && isfinite(Z) && isfinite(Xmed) && isfinite(Ymed) && isfinite(Zmed)) {
			if (abs(Xmed - X) > 5) {
				X1[pos] = Xmed;
			}
			else {
				X1[pos] = X;
			}

			if (abs(Ymed - Y) > 5) {
				Y1[pos] = Ymed;
			}
			else {
				Y1[pos] = Y;
			}

			if (abs(Zmed - Z) > 5) {
				//Z1[pos] = abs(Zmed);
				Z1[pos] = Zmed;
			}
			else {
				Z1[pos] = Z;
				//Z1[pos] = abs(Z);
			}
		}
		else {
			X1[pos] = 0.0f;
			Y1[pos] = 0.0f;
			Z1[pos] = 0.0f;
		}
	}
}

void ReconFlow::CleanUp3D(const float *X0, const float *Y0, const float *Z0,
	const float *Xmed, const float *Ymed, const float *Zmed,
	int w, int h, int s,
	float *X1, float *Y1, float *Z1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	CleanUp3DKernel <<< blocks, threads >>> (X0, Y0, Z0,
		Xmed, Ymed, Zmed,
		w, h, s,
		X1, Y1, Z1);
}


//********************************************
// Remove Edge Artifacts in Output 3D
//********************************************

__global__
void RemoveEdgeArtifacts3DKernel(const float *X0, const float *Y0, const float *Z0, 
	int surfaceWidth, float maxSurfaceArea,
	int width, int height, int stride,
	float *X1, float *Y1, float *Z1)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		float X = X0[pos];
		float Y = Y0[pos];
		float Z = Z0[pos];

		//********************************************
		//Remove points outside maximum surface area
		//********************************************
		float Xx, Yx, Zx, Xy, Yy, Zy;
		int range = surfaceWidth;
		if (ix + range > width) {
			Xx = -X0[ix - range + iy * stride] + X0[pos];
			Yx = -Y0[ix - range + iy * stride] + Y0[pos];
			Zx = -Z0[ix - range + iy * stride] + Z0[pos];
		}
		else {
			Xx = X0[ix + range + iy * stride] - X0[pos];
			Yx = Y0[ix + range + iy * stride] - Y0[pos];
			Zx = Z0[ix + range + iy * stride] - Z0[pos];
		}

		if (iy + range > height) {
			Xy = -X0[ix + (iy - range) * stride] + X0[pos];
			Yy = -Y0[ix + (iy - range) * stride] + Y0[pos];
			Zy = -Z0[ix + (iy - range) * stride] + Z0[pos];
		}
		else {
			Xy = X0[ix + (iy + range) * stride] - X0[pos];
			Yy = Y0[ix + (iy + range) * stride] - Y0[pos];
			Zy = Z0[ix + (iy + range) * stride] - Z0[pos];
		}

		float surfaceArea = sqrt((Yx*Zy - Zx*Yy)*(Yx*Zy - Zx*Yy) + 
			(Zx*Xy - Xx*Zy)*(Zx*Xy - Xx*Zy) + 
			(Xx*Yy - Yx*Xy)*(Xx*Yy - Yx*Xy));

		if (surfaceArea > maxSurfaceArea) {
			X1[pos] = 0.0f;
			Y1[pos] = 0.0f;
			Z1[pos] = 0.0f;
		}
		else {
			X1[pos] = X;
			Y1[pos] = Y;
			Z1[pos] = Z;
		}
	}
}

void ReconFlow::RemoveEdgeArtifacts3D(const float *X0, const float *Y0, const float *Z0, 
	int surfaceWidth, float maxSurfaceArea,
	int w, int h, int s,
	float *X1, float *Y1, float *Z1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	RemoveEdgeArtifacts3DKernel << < blocks, threads >> > (X0, Y0, Z0, 
		surfaceWidth, maxSurfaceArea,
		w, h, s,
		X1, Y1, Z1);
}