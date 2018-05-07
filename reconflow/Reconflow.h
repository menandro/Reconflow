#ifndef RECONFLOW_H
#define RECONFLOW_H

//#include "main.h"
#include "CudaFlow.h"
#include "CameraPose.h"
#include <cuda_runtime.h>

class ReconFlow : public CudaFlow {
public:
	ReconFlow();
	ReconFlow(int BlockWidth, int BlockHeight, int StrideAlignment);
	~ReconFlow();

	float *d_uproj;
	float *d_vproj;
	float *d_uc;
	float *d_vc;
	float *d_sku;
	float *d_skv;
	float *d_skus;
	float *d_skvs;

	float *d_X;
	float *d_Y;
	float *d_Z;
	float *d_Xs;
	float *d_Ys;
	float *d_Zs;
	float *d_Xmed;
	float *d_Ymed;
	float *d_Zmed;

	cv::Mat intrinsic0;
	cv::Mat intrinsic1;

	std::vector<double> pFocals;
	std::vector<double> pCamMidX1s;
	std::vector<double> pCamMidY1s;
	std::vector<cv::Mat> pK0;
	std::vector<cv::Mat> pK1;

	std::vector<double*> pP;
	std::vector<double*> pQ;

	static const int METHODR_TVL1_MS = 14;
	static const int METHODR_TVL1_MSPLANAR = 15;
	static const int METHODR_TVL1_3D = 16; //not implemented
	static const int METHODR_TVL1_MS_FN = 17; //tvl1 with 3dms and flownet initial value
	static const int METHODR_TVCHARBGRAD_MS = 18;
	static const int METHODR_TVCHARBGRAD_MS_FN = 19;
	static const int METHODR_TVL1_MS_FNSPARSE = 20; //tvl1 with 3dms and sparsified flownet as sparse input

													//theta for 3D
	float thetaProj;
	float alphaProj;
	float lambdaf;
	float lambdams;

	//initialization without large displacement
	int _initializeR(int width, int height, int channels, int nLevels, float scale, int method,
		float lambda, float lambdagrad, float lamdbaf, float lambdams,
		float alphaTv, float alphaProj, float tau,
		int nWarpIters, int nSolverIters);

	//initialization with large displacement
	int initializeR(int width, int height, int channels, int nLevels, float scale, int method,
		float lambda, float lambdagrad, float lamdbaf, float lambdams,
		float alphaTv, float alphaProj, float alphaFn, float tau,
		int nWarpIters, int nSolverIters);

	int copy3dToHost(cv::Mat &X, cv::Mat &Y, cv::Mat &Z);
	int setCameraMatrices(cv::Mat intrinsic);
	int setCameraMatrices(cv::Mat intrinsic0, cv::Mat intrinsic1);

	int solveReconFlow(cv::Mat R1, cv::Mat t1, float flowScale);
	int solveReconFlow(cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1, float flowScale);
	//int solveR(cv::Mat pose0, cv::Mat pose1, cv::Mat R1, cv::Mat t1, float flowScale);
	int solveR(float flowScale);
	int solveR();
	int solveR(cv::Mat R1, cv::Mat t1, float flowScale);
	int solveR(cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1, float flowScale);
	int removeEdgeArtifacts3D(int surfaceWidth, float maxSurfaceArea);

	//KERNELS
	void SolveReconDataL1(const float *u0, const float *v0,
		const float *duhat0, const float *dvhat0,
		const float *uproj0, const float *vproj0,
		const float *sku0, const float *skv0,
		const float *pu1, const float *pu2,
		const float *pv1, const float *pv2,
		const float *Ix, const float *Iy, const float *Iz,
		int w, int h, int s,
		float lambda, float alphaTv, float alphaProj,
		float *duhat1, float *dvhat1);

	//dense flownet
	void SolveReconDataL1Fn(const float *u0, const float *v0,
		const float *duhat0, const float *dvhat0,
		const float *uproj0, const float *vproj0,
		const float *ufn0, const float *vfn0,
		const float *sku0, const float *skv0,
		const float *pu1, const float *pu2,
		const float *pv1, const float *pv2,
		const float *Ix, const float *Iy, const float *Iz,
		int w, int h, int s,
		float lambda, float alphaTv, float alphaProj, float alphaFn,
		float *duhat1, float *dvhat1);

	//sparse flownet
	void SolveReconDataL1Fn(const float *u0, const float *v0,
		const float *duhat0, const float *dvhat0,
		const float *uproj0, const float *vproj0,
		const float *ufn0, const float *vfn0,
		const float *sku0, const float *skv0,
		const float *pu1, const float *pu2,
		const float *pv1, const float *pv2,
		const float *Ix, const float *Iy, const float *Iz,
		int w, int h, int s,
		float lambda, float alphaTv, float alphaProj, float alphaFn, float* fnmask,
		float *duhat1, float *dvhat1);

	void SolveUVProj(const float *u0, const float *v0,
		const float *duhat0, const float *dvhat0,
		const float *uc0, const float *vc0,
		const float *sku0, const float *skv0,
		int w, int h, int s,
		float lambdaf, float alphaProj,
		float *uproj1, float *vproj1);

	void Solve3dMinimalArea(const float *uproj0, const float *vproj0,
		const float *X0, const float *Y0, const float *Z0,
		double *P, double *Q, //camera matrices
		float lambdaf, float lambdams,
		int w, int h, int s,
		float *X, float *Y, float *Z);

	void SolveUc(const float *X, const float *Y, const float *Z,
		double *R, double *t, double *K1,
		int w, int h, int s,
		float *uc, float *vc);

	void UpdateSk(const float *sku0, const float *skv0,
		const float *u0, const float *v0,
		const float *du0, const float *dv0,
		const float *uproj0, const float *vproj0,
		int width, int height, int stride,
		float *sku1, float *skv1);

	void CleanUp3D(const float *X0, const float *Y0, const float *Z0,
		const float *Xmed, const float *Ymed, const float *Zmed,
		int w, int h, int s,
		float *X1, float *Y1, float *Z1);

	void RemoveEdgeArtifacts3D(const float *X0, const float *Y0, const float *Z0,
		int surfaceWidth, float maxSurfaceArea,
		int w, int h, int s,
		float *X1, float *Y1, float *Z1);
};

#endif