#ifndef CUDAFLOW_H
#define CUDAFLOW_H

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "common.h"

class CudaFlow
{
public:
	static const int METHOD_TVL1 = 0;
	static const int METHOD_TVCHARB = 1;
	static const int METHOD_TVCHARBGRAD = 2;//|I1 - I0| + |Igradx - Igradx0| + |Igrady - Igrady0| 
	static const int METHOD_TVL1PATCH = 3; //sum(Ix)u + sum(Iy)v + sum(It)
	static const int SCENEFLOW_KINECT_TVL1 = 4;
	static const int METHOD_TVL1_FN = 5; //Large displacent using FlowNet as initial value for UV
	static const int METHOD_TVCHARB_FN = 6;
	static const int METHOD_TVCHARBGRAD_FN = 7;
	static const int METHOD_TVL1_INPAINTING = 8;

public:
	CudaFlow();
	CudaFlow(int BlockWidth, int BlockHeight, int StrideAlignment);
	~CudaFlow();
	//////////////////////////////////////////////////////////////////////
	/// OPTICAL FLOW
	//////////////////////////////////////////////////////////////////////
	//Functions for Optical Flow
	int initialize(int width, int height, int channels, int cvType, int nLevels, float scale, int method,
		float lambda, float lambdagrad, float theta, float tau,
		int nWarpIters, int nSolverIters, bool withVisualization);
	int initialize(int width, int height, int channels, int cvType, int nLevels, float scale, int method,
		float lambda, float lambdagrad, float theta, float tau,
		int nWarpIters, int nSolverIters);
	int initialize(int width, int height, int channels, int nLevels, float scale, int method,
		float lambda, float lambdagrad, float theta, float tau,
		int nWarpIters, int nSolverIters);
	int initializeColorWheel();
	int initializeLargeDisplacement();
	int initializeCorrelation(int kernelSize, int maxSearchWidth, int maxSearchHeight);
	int close();

	int copyToDevice(cv::Mat i0, cv::Mat i1); //old
	int copyToHost(cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb); //old

	int copyMaskToDevice(cv::Mat mask0, cv::Mat mask1);
	int copyImagesToDevice(cv::Mat i0, cv::Mat i1);
	int copyOpticalFlowToHost(cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb);
	int copyOpticalFlowToHost(cv::Mat &u, cv::Mat &v);

	//Large Displacement
	int copyInitialOpticalFlowToDevice(cv::Mat u, cv::Mat v);
	int copySparseOpticalFlowToDevice(cv::Mat u, cv::Mat v);
	int copySparseOpticalFlowToDevice(cv::Mat u, cv::Mat v, cv::Mat mask);

	int copyDepthToDevice(cv::Mat depth0, cv::Mat depth1);
	int copySceneFlowToHost(cv::Mat &sceneflow);
	int copySceneFlowToHost(cv::Mat &sceneflow, cv::Mat &sceneflowrgb);
	int copySceneFlowAndPointCloudToHost(cv::Mat &sceneflow, cv::Mat &pcloud0, cv::Mat &pcloud1, cv::Mat &sceneflowrgb);

	//solve Correlation Based Patch Matching
	int solveCorrPatchMatch();
	int solveCorrPatchMatch(const char *picName, const char *flowName);
	int _solveCorrPatchMatch();
	int _solveCorrPatchMatch_cpu(const char *picName, const char *flowName);
	int solveOpticalFlowLdof();
	int _solveOpticalFlowLdof();
	int copyCorrPatchMatchToHost(cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb);

	//solveOpticalFlow
	int solveOpticalFlow();
	int solveOpticalFlow(float flowScale);
	int _solveOpticalFlow(float flowScale);

	//inpainting
	int solveInpainting(float flowScale);
	int _solveInpainting(float flowScale);

	int solveOpticalFlow(cv::Mat i0, cv::Mat i1, cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb);
	int solveOpticalFlow(cv::Mat i0, cv::Mat i1, cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb, float flowScale);
	int _solveOpticalFlow(cv::Mat i0, cv::Mat i1, cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb, float flowScale);

	//solveSceneFlow
	int solveSceneFlow();
	int solveSceneFlow(float focalx, float focaly, float cx, float cy, float flowScale);
	int solveSceneFlow(float focalx, float focaly, float cx, float cy, float opticalFlowScale, float sceneFlowScale);
	int _solveSceneFlow(float opticalFlowScale, float sceneFlowScale);

	int solveSceneFlowAndPointCloud(float focalx, float focaly, float cx, float cy, float opticalFlowScale, float sceneFlowScale);
	int _solveSceneFlowAndPointCloud(float opticalFlowScale, float sceneFlowScale);

	//CUDA KERNELS
	void MedianFilter(float *inputu, float *inputv, int w, int h, int s, float *outputu, float*outputv, int kernelsize);
	void MedianFilter3D(float *X, float *Y, float *Z, int w, int h, int s, float *X1, float*Y1, float* Z1, int kernelsize);
	void Add(const float *op1, const float *op2, int count, float *sum);
	void ComputeDerivatives(float *I0, float *I1, int w, int h, int s, float *Ix, float *Iy, float *Iz);
	void ComputeDerivativesPatch(float *I0, float *I1, int w, int h, int s, float *Ix, float *Iy, float *Iz);
	void ComputeDeriv(float *I0, int w, int h, int s, float *Ix, float *Iy);
	void ComputeDerivMask(float *I0, int w, int h, int s, float *mask, float threshold);
	void Downscale(const float *src, int width, int height, int stride, int newWidth, int newHeight, int newStride, float *out);
	void Downscale(const float *src, int width, int height, int stride, int newWidth, int newHeight, int newStride, float scale, float *out);
	void FlowToHSV(float* u, float * v, int w, int h, int s, float3 * uRGB, float flowscale);
	void SceneFlowToHSV(float3* sceneflow, int w, int h, int s, float3 * uRGB, float flowscale);
	void rgbToGray(uchar3 * d_iRgb, float *d_iGray, int w, int h, int s);
	void Cv8uToGray(uchar * d_iCv8u, float *d_iGray, int w, int h, int s);
	void Cv16uToGray(ushort * d_iCv16u, float *d_iGray, int w, int h, int s);
	void Cv16uTo32f(ushort * d_iCv16u, float *d_iGray, int w, int h, int s);
	void Cv32fToGray(float * d_iCv32f, float *d_iGray, int w, int h, int s);
	void SolveBK(const float *du, const float *dumed, const float *bku, int count, float *bkuout);
	void SolveDataCharb(const float *du0, const float *dv0, const float *dumed, const float *dvmed,
		const float *bku, const float *bkv, const float *Ix, const float *Iy, const float *Iz, int w, int h, int s,
		float alpha, float *du1, float *dv1);
	void SolveDataCharbForTV(const float *du0, const float *dv0,
		const float *duhat, const float *dvhat,
		const float *pu1, const float *pu2,
		const float *pv1, const float *pv2,
		const float *Ix, const float *Iy, const float *Iz,
		int w, int h, int s,
		float lambda, float theta,
		float *du1, float *dv1,
		float *duhat1, float *dvhat1);
	void SolveDataCharbForTVGrad(const float *du0, const float *dv0,
		const float *duhat, const float *dvhat,
		const float *pu1, const float *pu2,
		const float *pv1, const float *pv2,
		const float *Ix, const float *Iy, const float *Iz,
		const float *Ixx, const float *Ixy, const float *Ixz,
		const float *Iyx, const float *Iyy, const float *Iyz,
		int w, int h, int s,
		float lambda, float lambdagrad, float theta,
		float *du1, float *dv1,
		float *duhat1, float *dvhat1);
	void SolveDataL1(const float *duhat0, const float *dvhat0,
		const float *pu1, const float *pu2,
		const float *pv1, const float *pv2,
		const float *Ix, const float *Iy, const float *Iz,
		int w, int h, int s,
		float lambda, float theta,
		float *duhat1, float *dvhat1);
	void SolveDataL1Inpaint(const float *duhat0, const float *dvhat0,
		const float *mask0, const float *mask1,
		const float *pu1, const float *pu2,
		const float *pv1, const float *pv2,
		const float *Ix, const float *Iy, const float *Iz,
		int w, int h, int s,
		float lambda, float theta,
		float *duhat1, float *dvhat1);
	void SolveSmoothDualTVGlobal(float *duhat, float *dvhat,
		float *pu1, float *pu2, float *pv1, float *pv2,
		int w, int h, int s,
		float tau, float theta,
		float *pu1s, float*pu2s,
		float *pv1s, float *pv2s);
	void SolveSmoothGaussianGlobal(float *inputu, float *inputv, float *inputbku, float *inputbkv,
		int w, int h, int s,
		float *outputu, float*outputv,
		float *outputbku, float *outputbkv,
		int kernelsize);
	void SolveSmoothGaussianTex(float *inputu, float *inputv,
		int w, int h, int s,
		float *outputu, float*outputv);
	void SolveSmoothMedianGlobal(float *inputu, float *inputv, float *inputbku, float *inputbkv,
		int w, int h, int s,
		float *outputu, float*outputv,
		float *outputbku, float *outputbkv,
		int kernelsize);
	void Upscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void WarpImage(const float *src, int w, int h, int s,
		const float *u, const float *v, float *out);
	void SolveSceneFlow_(float *u, float *v, float* depth0, float* depth1,
		float fx, float fy, float cx, float cy,
		int w, int h, int s,
		float3 *sceneflow);
	void SolveSceneFlow_(float *u, float *v, float* depth0, float* depth1,
		float fx, float fy, float cx, float cy,
		int w, int h, int s,
		float3 *pcloud0, float3 *pcloud1, float3 *sceneflow);
	void Correlation(float* kernel, float* searchSpace, float* output);
	void Correlation1x1(float kernel, float* searchSpace, float* output);
	void CorrelationBindTextures(float* im0, float*im1, int w, int h, int s);
	void CorrelationKernelSampling(int x, int y, float* kernel, int w, int h);
	void CorrelationSearchSampling(int x, int y, float* searchSpace);
	void GetValue(float *input, int idx, float &value);

	int width;
	int height;
	int stride;
	int inputChannels;
	int inputType; //cv type of type
	float flowScale;
	bool withVisualization;

	float lambda, lambdagrad, theta, tau;
	int BlockWidth;
	int BlockHeight;
	int StrideAlignment;

	//theta for TV
	float thetaTv;
	float alphaTv;

	//alpha for large displacement
	float alphaFn;

	/* Number of desired pyramid levels*/
	int nLevels;
	float fScale;
	int nWarpIters;
	int nSolverIters;

	uchar3 *h_i0rgb, *h_i1rgb;

	//depth camera intrinsics
	float depthCameraFocalX;
	float depthCameraFocalY;
	float depthCameraPrincipalX;
	float depthCameraPrincipalY;

	/* CUDA pointers */
	std::vector<float*> pI0;
	std::vector<float*> pI1;
	std::vector<float*> pIx0;
	std::vector<float*> pIx1;
	std::vector<float*> pIy0;
	std::vector<float*> pIy1;
	std::vector<int> pW;
	std::vector<int> pH;
	std::vector<int> pS;
	std::vector<int> pDataSize;

	//inpainting
	std::vector<float*> pMask0;
	std::vector<float*> pMask1;
	float *d_mask0;
	float *d_mask1;

	int dataSize;
	int dataSize8uc3;
	int dataSize8u;
	int dataSize16u;
	int dataSize32f;
	int dataSize32fc3;

	float *d_i1warp;
	float *d_ix1warp;
	float *d_iy1warp;

	float *d_du;
	float *d_dv;
	float *d_dus;
	float *d_dvs;

	float *d_dumed;
	float *d_dvmed;
	float *d_dumeds;
	float *d_dvmeds;

	//dual TV
	float *d_pu1;
	float *d_pu2;
	float *d_pv1;
	float *d_pv2;
	//dual TV temps
	float *d_pu1s;
	float *d_pu2s;
	float *d_pv1s;
	float *d_pv2s;

	float *d_Ix;
	float *d_Iy;
	float *d_Iz;
	float *d_Ixx;
	float *d_Ixy;
	float *d_Ixz;
	float *d_Iyx;
	float *d_Iyy;
	float *d_Iyz;

	float *d_us;
	float *d_vs;

	//large displacement initial values
	float *d_ufn; //fn means flownet
	float *d_vfn;
	float *d_ufn_l;
	float *d_vfn_l;
	float *d_ufns;
	float *d_vfns;
	float *d_fnmask; //mask for sparsity
	float *d_fnmask_l; //pyramid level sampled mask

					   //outputs
	float *d_u; //optical flow x
	float *d_v; //optical flow y
	float3 *d_sceneflow; //scene flow

						 //inputs
						 // CV_8UC3
	uchar3 *d_i08uc3;
	uchar3 *d_i18uc3;
	//CV_8U
	uchar *d_i08u;
	uchar *d_i18u;
	//CV_16U
	ushort *d_i016u;
	ushort *d_i116u;
	// CV_32F
	float *d_i032f;
	float *d_i132f;
	// CV_16U for Depth
	ushort *d_depth016u;
	ushort *d_depth116u;
	float *d_depth032f;
	float *d_depth132f;
	float3 *d_pcloud0;
	float3 *d_pcloud1;


	//FOR CORRELATION
	float *d_icorr032f; //for correlation
	float *d_icorr132f; //for correlation
	float* d_corrKernel;
	float* d_corrSearchSpace;
	float* d_corrOutput;
	int corrMaxSearchWidth;
	int corrMaxSearchHeight;
	int corrKernelSize;
	int corrStride;
	float *h_icorr032f;
	float *h_icorr132f;
	float *d_ucorr; //sparse optical flow from correlation matching x
	float *d_vcorr; //sparse optical flow from correlation matching y
	float3 *d_uvrgbcorr;
	float *d_corrSparseMask; //mask and weight of the sparse matching result
	float* d_derivMask;

	// colored uv, for display only
	float3 *d_uvrgb;
	float3 *d_sceneflowrgb;
	float3 *d_colorwheel;

	int method;

	//functions
	inline int iAlignUp(int n);
	template<typename T> inline void Swap(T &a, T &b);
	template<typename T> inline void Swap(T &a, T &b, T &c, T &d);
	template<typename T> inline void Swap(T &a, T &b, T &c, T &d, T &e, T &f, T &g, T &h);
	int computePyramidLevels(int width, int height, int minWidth, float scale);

};

#endif