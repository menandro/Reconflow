#include "CudaFlow.h"

//Constructors
CudaFlow::CudaFlow() {
	this->BlockWidth = 32;
	this->BlockHeight = 12;
	this->StrideAlignment = 32;
}

CudaFlow::CudaFlow(int BlockWidth, int BlockHeight, int StrideAlignment) {
	this->BlockWidth = BlockWidth;
	this->BlockHeight = BlockHeight;
	this->StrideAlignment = StrideAlignment;
}

int CudaFlow::initialize(int width, int height, int channels, int nLevels, float scale, int method,
	float lambda = 100.0f, float lambdagrad = 400.0f, float theta = 0.33f, float tau = 0.25f,
	int nWarpIters = 1, int nSolverIters = 100)
{
	int cvType = CV_8UC3; //Assume a 3-channel uchar input if not specified
	bool withVisualization = true;
	return initialize(width, height, channels, cvType, nLevels, scale, method,
		lambda, lambdagrad, theta, tau,
		nWarpIters, nSolverIters, withVisualization);
}

int CudaFlow::initialize(int width, int height, int channels, int cvType, int nLevels, float scale, int method,
	float lambda = 100.0f, float lambdagrad = 400.0f, float theta = 0.33f, float tau = 0.25f,
	int nWarpIters = 1, int nSolverIters = 100)
{
	bool withVisualization = true;
	return initialize(width, height, channels, cvType, nLevels, scale, method,
		lambda, lambdagrad, theta, tau,
		nWarpIters, nSolverIters, withVisualization);
}


int CudaFlow::initializeColorWheel() {
	checkCudaErrors(cudaMalloc(&d_colorwheel, 55 * 3 * sizeof(float)));
	float colorwheel[165] = { 255, 0, 0,
		255, 17, 0,
		255, 34, 0,
		255, 51, 0,
		255, 68, 0,
		255, 85, 0,
		255, 102, 0,
		255, 119, 0,
		255, 136, 0,
		255, 153, 0,
		255, 170, 0,
		255, 187, 0,
		255, 204, 0,
		255, 221, 0,
		255, 238, 0,
		255, 255, 0,
		213, 255, 0,
		170, 255, 0,
		128, 255, 0,
		85, 255, 0,
		43, 255, 0,
		0, 255, 0,
		0, 255, 63,
		0, 255, 127,
		0, 255, 191,
		0, 255, 255,
		0, 232, 255,
		0, 209, 255,
		0, 186, 255,
		0, 163, 255,
		0, 140, 255,
		0, 116, 255,
		0, 93, 255,
		0, 70, 255,
		0, 47, 255,
		0, 24, 255,
		0, 0, 255,
		19, 0, 255,
		39, 0, 255,
		58, 0, 255,
		78, 0, 255,
		98, 0, 255,
		117, 0, 255,
		137, 0, 255,
		156, 0, 255,
		176, 0, 255,
		196, 0, 255,
		215, 0, 255,
		235, 0, 255,
		255, 0, 255,
		255, 0, 213,
		255, 0, 170,
		255, 0, 128,
		255, 0, 85,
		255, 0, 43 };
	checkCudaErrors(cudaMemcpy(colorwheel, d_colorwheel, 55 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}

int CudaFlow::initialize(int width, int height, int channels, int cvType, int nLevels, float scale, int method,
	float lambda = 100.0f, float lambdagrad = 400.0f, float theta = 0.33f, float tau = 0.25f,
	int nWarpIters = 1, int nSolverIters = 100, bool withVisualization = true)
{
	//allocate all memories
	this->width = width;
	this->height = height;
	this->stride = iAlignUp(width);
	this->inputType = cvType;

	this->fScale = scale;
	this->nLevels = nLevels;
	this->method = method;
	this->inputChannels = channels;
	this->nSolverIters = nSolverIters; //number of inner iteration (ROF loop)
	this->nWarpIters = nWarpIters;

	this->lambda = lambda;
	this->lambdagrad = lambdagrad;
	this->theta = theta;
	this->tau = tau;

	this->withVisualization = withVisualization;


	//this->BlockWidth = 32;
	//this->BlockHeight = 6;
	//this->StrideAlignment = 32;
	//pyramids

	pI0 = std::vector<float*>(nLevels);
	pI1 = std::vector<float*>(nLevels);
	if (method == METHOD_TVCHARBGRAD) {
		pIx0 = std::vector<float*>(nLevels);
		pIx1 = std::vector<float*>(nLevels);
		pIy0 = std::vector<float*>(nLevels);
		pIy1 = std::vector<float*>(nLevels);
	}
	if (method == METHOD_TVL1_INPAINTING) {
		pMask0 = std::vector<float*>(nLevels);
		pMask1 = std::vector<float*>(nLevels);
	}
	pW = std::vector<int>(nLevels);
	pH = std::vector<int>(nLevels);
	pS = std::vector<int>(nLevels);
	pDataSize = std::vector<int>(nLevels);

	int newHeight = height;
	int newWidth = width;
	int newStride = iAlignUp(width);
	//std::cout << "Pyramid Sizes: " << newWidth << " " << newHeight << " " << newStride << std::endl;
	for (int level = 0; level < nLevels; level++) {
		pDataSize[level] = newStride * newHeight * sizeof(float);
		checkCudaErrors(cudaMalloc(&pI0[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pI1[level], pDataSize[level]));
		if ((method == METHOD_TVCHARBGRAD) || (method == METHOD_TVCHARBGRAD_FN)) {
			checkCudaErrors(cudaMalloc(&pIx0[level], pDataSize[level]));
			checkCudaErrors(cudaMalloc(&pIx1[level], pDataSize[level]));
			checkCudaErrors(cudaMalloc(&pIy0[level], pDataSize[level]));
			checkCudaErrors(cudaMalloc(&pIy1[level], pDataSize[level]));
		}
		if (method == METHOD_TVL1_INPAINTING) {
			checkCudaErrors(cudaMalloc(&pMask0[level], pDataSize[level]));
			checkCudaErrors(cudaMalloc(&pMask1[level], pDataSize[level]));
		}
		pW[level] = newWidth;
		pH[level] = newHeight;
		pS[level] = newStride;
		newHeight = newHeight / fScale;
		newWidth = newWidth / fScale;
		newStride = iAlignUp(newWidth);
		//std::cout << "Pyramid Sizes: " << newWidth << " " << newHeight << " " << newStride << std::endl;
	}
	//runtime

	dataSize = stride * height * sizeof(float);
	dataSize8uc3 = stride * height * sizeof(uchar3);
	dataSize8u = stride * height * sizeof(uchar);
	dataSize16u = stride * height * sizeof(ushort);
	dataSize32f = dataSize;
	dataSize32fc3 = dataSize * 3;
	checkCudaErrors(cudaMalloc(&d_i1warp, dataSize));

	checkCudaErrors(cudaMalloc(&d_du, dataSize));
	checkCudaErrors(cudaMalloc(&d_dv, dataSize));
	checkCudaErrors(cudaMalloc(&d_dus, dataSize));
	checkCudaErrors(cudaMalloc(&d_dvs, dataSize));

	checkCudaErrors(cudaMalloc(&d_dumed, dataSize));
	checkCudaErrors(cudaMalloc(&d_dvmed, dataSize));
	checkCudaErrors(cudaMalloc(&d_dumeds, dataSize));
	checkCudaErrors(cudaMalloc(&d_dvmeds, dataSize));

	//dual TV
	checkCudaErrors(cudaMalloc(&d_pu1, dataSize));
	checkCudaErrors(cudaMalloc(&d_pu2, dataSize));
	checkCudaErrors(cudaMalloc(&d_pv1, dataSize));
	checkCudaErrors(cudaMalloc(&d_pv2, dataSize));
	//dual TV temps
	checkCudaErrors(cudaMalloc(&d_pu1s, dataSize));
	checkCudaErrors(cudaMalloc(&d_pu2s, dataSize));
	checkCudaErrors(cudaMalloc(&d_pv1s, dataSize));
	checkCudaErrors(cudaMalloc(&d_pv2s, dataSize));

	checkCudaErrors(cudaMalloc(&d_Ix, dataSize));
	checkCudaErrors(cudaMalloc(&d_Iy, dataSize));
	checkCudaErrors(cudaMalloc(&d_Iz, dataSize));

	if ((method == METHOD_TVCHARBGRAD) || (method == METHOD_TVCHARBGRAD_FN)) {
		checkCudaErrors(cudaMalloc(&d_Ixx, dataSize));
		checkCudaErrors(cudaMalloc(&d_Ixy, dataSize));
		checkCudaErrors(cudaMalloc(&d_Ixz, dataSize));
		checkCudaErrors(cudaMalloc(&d_Iyx, dataSize));
		checkCudaErrors(cudaMalloc(&d_Iyy, dataSize));
		checkCudaErrors(cudaMalloc(&d_Iyz, dataSize));

		checkCudaErrors(cudaMalloc(&d_ix1warp, dataSize));
		checkCudaErrors(cudaMalloc(&d_iy1warp, dataSize));
	}

	checkCudaErrors(cudaMalloc(&d_u, dataSize));
	checkCudaErrors(cudaMalloc(&d_v, dataSize));
	checkCudaErrors(cudaMalloc(&d_us, dataSize));
	checkCudaErrors(cudaMalloc(&d_vs, dataSize));

	if (method == METHOD_TVL1_INPAINTING) {
		checkCudaErrors(cudaMalloc(&d_mask0, dataSize32f));
		checkCudaErrors(cudaMalloc(&d_mask1, dataSize32f));
	}

	if (inputType == CV_8UC3) {
		checkCudaErrors(cudaMalloc(&d_i08uc3, dataSize8uc3));
		checkCudaErrors(cudaMalloc(&d_i18uc3, dataSize8uc3));
	}
	else if (inputType == CV_8U) {
		checkCudaErrors(cudaMalloc(&d_i08u, dataSize8u));
		checkCudaErrors(cudaMalloc(&d_i18u, dataSize8u));
	}
	else if (inputType == CV_16U) {
		checkCudaErrors(cudaMalloc(&d_i016u, dataSize16u));
		checkCudaErrors(cudaMalloc(&d_i116u, dataSize16u));
	}
	else if (inputType == CV_32F) {
		checkCudaErrors(cudaMalloc(&d_i032f, dataSize32f));
		checkCudaErrors(cudaMalloc(&d_i132f, dataSize32f));
	}
	else {
		std::cout << "Warning: Input Type unknown. Assuming CV_8UC3." << std::endl;
		checkCudaErrors(cudaMalloc(&d_i08uc3, dataSize8uc3));
		checkCudaErrors(cudaMalloc(&d_i18uc3, dataSize8uc3));
	}


	// grayscale inputs
	//checkCudaErrors(cudaMalloc(&d_i0, dataSize));
	//checkCudaErrors(cudaMalloc(&d_i1, dataSize));
	// colored uv, for display only
	checkCudaErrors(cudaMalloc(&d_uvrgb, dataSize * 3));

	if (method == SCENEFLOW_KINECT_TVL1) {
		checkCudaErrors(cudaMalloc(&d_depth016u, dataSize16u));
		checkCudaErrors(cudaMalloc(&d_depth116u, dataSize16u));
		checkCudaErrors(cudaMalloc(&d_depth032f, dataSize32f));
		checkCudaErrors(cudaMalloc(&d_depth132f, dataSize32f));
		checkCudaErrors(cudaMalloc(&d_sceneflow, dataSize32fc3));
		checkCudaErrors(cudaMalloc(&d_sceneflowrgb, dataSize32fc3));
		checkCudaErrors(cudaMalloc(&d_pcloud0, dataSize32fc3));
		checkCudaErrors(cudaMalloc(&d_pcloud1, dataSize32fc3));
	}
	return 0;
}


int CudaFlow::initializeLargeDisplacement() {
	checkCudaErrors(cudaMalloc(&d_ufn, dataSize));
	checkCudaErrors(cudaMalloc(&d_vfn, dataSize));
	checkCudaErrors(cudaMalloc(&d_ufn_l, dataSize));
	checkCudaErrors(cudaMalloc(&d_vfn_l, dataSize));
	checkCudaErrors(cudaMalloc(&d_fnmask, dataSize));
	checkCudaErrors(cudaMalloc(&d_fnmask_l, dataSize));
	return 0;
}

int CudaFlow::copyDepthToDevice(cv::Mat depth0, cv::Mat depth1) {
	checkCudaErrors(cudaMemcpy(d_depth016u, (ushort *)depth0.ptr(), dataSize16u, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_depth116u, (ushort *)depth1.ptr(), dataSize16u, cudaMemcpyHostToDevice));
	return 0;
}

int CudaFlow::copyImagesToDevice(cv::Mat i0, cv::Mat i1) {
	if (inputType == CV_8UC3) {
		checkCudaErrors(cudaMemcpy(d_i08uc3, (uchar3 *)i0.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18uc3, (uchar3 *)i1.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	}
	else if (inputType == CV_8U) {
		checkCudaErrors(cudaMemcpy(d_i08u, (uchar *)i0.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18u, (uchar *)i1.ptr(), dataSize8u, cudaMemcpyHostToDevice));
	}
	else if (inputType == CV_16U) {
		checkCudaErrors(cudaMemcpy(d_i016u, (ushort *)i0.ptr(), dataSize16u, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i116u, (ushort *)i1.ptr(), dataSize16u, cudaMemcpyHostToDevice));
	}
	else if (inputType == CV_32F) {
		h_icorr032f = (float *)i0.ptr(); //for correlation only
		checkCudaErrors(cudaMemcpy(d_i032f, (float *)i0.ptr(), dataSize32f, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i132f, (float *)i1.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	}
	else {
		checkCudaErrors(cudaMemcpy(d_i08uc3, (uchar3 *)i0.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18uc3, (uchar3 *)i1.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	}
	return 0;
}

int CudaFlow::copyOpticalFlowToHost(cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb) {
	checkCudaErrors(cudaMemcpy((float3 *)uvrgb.ptr(), d_uvrgb, stride * height * sizeof(float) * 3, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)u.ptr(), d_u, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)v.ptr(), d_v, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}

int CudaFlow::copyOpticalFlowToHost(cv::Mat &u, cv::Mat &v) {
	checkCudaErrors(cudaMemcpy((float *)u.ptr(), d_u, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)v.ptr(), d_v, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}

int CudaFlow::copyInitialOpticalFlowToDevice(cv::Mat u, cv::Mat v) {
	checkCudaErrors(cudaMemcpy(d_ufn, (float*)u.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vfn, (float*)v.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int CudaFlow::copySparseOpticalFlowToDevice(cv::Mat u, cv::Mat v) {
	cv::Mat mask = cv::Mat::ones(u.size(), CV_32F);
	copySparseOpticalFlowToDevice(u, v, mask);
	return 0;
}

int CudaFlow::copySparseOpticalFlowToDevice(cv::Mat u, cv::Mat v, cv::Mat mask) {
	checkCudaErrors(cudaMemcpy(d_ufn, (float*)u.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vfn, (float*)v.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fnmask, (float*)mask.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int CudaFlow::copySceneFlowToHost(cv::Mat &sceneflow) {
	checkCudaErrors(cudaMemcpy((float3 *)sceneflow.ptr(), d_sceneflow, dataSize32fc3, cudaMemcpyDeviceToHost));
	return 0;
}

int CudaFlow::copySceneFlowToHost(cv::Mat &sceneflow, cv::Mat &sceneflowrgb) {
	checkCudaErrors(cudaMemcpy((float3 *)sceneflow.ptr(), d_sceneflow, dataSize32fc3, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float3 *)sceneflowrgb.ptr(), d_sceneflowrgb, dataSize32fc3, cudaMemcpyDeviceToHost));
	return 0;
}

int CudaFlow::copySceneFlowAndPointCloudToHost(cv::Mat &sceneflow, cv::Mat &pcloud0, cv::Mat &pcloud1, cv::Mat &sceneflowrgb) {
	checkCudaErrors(cudaMemcpy((float3 *)sceneflow.ptr(), d_sceneflow, dataSize32fc3, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float3 *)pcloud0.ptr(), d_pcloud0, dataSize32fc3, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float3 *)pcloud1.ptr(), d_pcloud1, dataSize32fc3, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float3 *)sceneflowrgb.ptr(), d_sceneflowrgb, dataSize32fc3, cudaMemcpyDeviceToHost));
	return 0;
}

//OLD copytodevice and copytohost
int CudaFlow::copyToDevice(cv::Mat i0, cv::Mat i1) {
	if (inputType == CV_8UC3) {
		checkCudaErrors(cudaMemcpy(d_i08uc3, (uchar3 *)i0.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18uc3, (uchar3 *)i1.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	}
	else if (inputType == CV_8U) {
		checkCudaErrors(cudaMemcpy(d_i08u, (uchar *)i0.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18u, (uchar *)i1.ptr(), dataSize8u, cudaMemcpyHostToDevice));
	}
	else if (inputType == CV_16U) {
		checkCudaErrors(cudaMemcpy(d_i016u, (ushort *)i0.ptr(), dataSize16u, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i116u, (ushort *)i1.ptr(), dataSize16u, cudaMemcpyHostToDevice));
	}
	else if (inputType == CV_32F) {
		checkCudaErrors(cudaMemcpy(d_i032f, (float *)i0.ptr(), dataSize32f, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i132f, (float *)i1.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	}
	else {
		checkCudaErrors(cudaMemcpy(d_i08uc3, (uchar3 *)i0.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18uc3, (uchar3 *)i1.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	}
}
int CudaFlow::copyToHost(cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb) {
	checkCudaErrors(cudaMemcpy((float3 *)uvrgb.ptr(), d_uvrgb, stride * height * sizeof(float) * 3, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)u.ptr(), d_u, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)v.ptr(), d_v, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
}


//Solve Optical Flow
int CudaFlow::solveOpticalFlow(float flowScale = 5.0f) {
	return _solveOpticalFlow(flowScale);
}

int CudaFlow::solveOpticalFlow() {
	return _solveOpticalFlow(5.0f);
}

int CudaFlow::_solveOpticalFlow(float flowScale) {
	// Convert RGB to Gray
	if (inputType == CV_8UC3) {
		rgbToGray(d_i08uc3, pI0[0], width, height, stride);
		rgbToGray(d_i18uc3, pI1[0], width, height, stride);
	}
	else if ((inputType == CV_8U) || (inputType == CV_8UC1)) {
		Cv8uToGray(d_i08u, pI0[0], width, height, stride);
		Cv8uToGray(d_i18u, pI1[0], width, height, stride);
	}
	else if (inputType == CV_16U) {
		Cv16uToGray(d_i016u, pI0[0], width, height, stride);
		Cv16uToGray(d_i116u, pI1[0], width, height, stride);
	}
	else if (inputType == CV_32F) {
		Cv32fToGray(d_i032f, pI0[0], width, height, stride);
		Cv32fToGray(d_i132f, pI1[0], width, height, stride);
	}
	else {
		rgbToGray(d_i08uc3, pI0[0], width, height, stride);
		rgbToGray(d_i18uc3, pI1[0], width, height, stride);
	}

	if ((method == METHOD_TVCHARBGRAD) || (method == METHOD_TVCHARBGRAD_FN)) {
		ComputeDeriv(pI0[0], width, height, stride, pIx0[0], pIy0[0]);
		ComputeDeriv(pI1[0], width, height, stride, pIx1[0], pIy1[0]);
	}

	// construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pI0[level]);

		Downscale(pI1[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pI1[level]);

		if ((method == METHOD_TVCHARBGRAD) || (method == METHOD_TVCHARBGRAD_FN)) {
			Downscale(pIx0[level - 1],
				pW[level - 1], pH[level - 1], pS[level - 1],
				pW[level], pH[level], pS[level],
				pIx0[level]);

			Downscale(pIx1[level - 1],
				pW[level - 1], pH[level - 1], pS[level - 1],
				pW[level], pH[level], pS[level],
				pIx1[level]);

			Downscale(pIy0[level - 1],
				pW[level - 1], pH[level - 1], pS[level - 1],
				pW[level], pH[level], pS[level],
				pIy0[level]);

			Downscale(pIy1[level - 1],
				pW[level - 1], pH[level - 1], pS[level - 1],
				pW[level], pH[level], pS[level],
				pIy1[level]);
		}
	}

	// solve flow
	checkCudaErrors(cudaMemset(d_u, 0, dataSize));
	checkCudaErrors(cudaMemset(d_v, 0, dataSize));

	for (int level = nLevels - 1; level >= 0; level--) {
		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			//std::cout << level << std::endl;
			//initialize zeros
			checkCudaErrors(cudaMemset(d_du, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dv, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dus, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvs, 0, dataSize));

			checkCudaErrors(cudaMemset(d_dumed, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvmed, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dumeds, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvmeds, 0, dataSize));

			checkCudaErrors(cudaMemset(d_pu1, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pu2, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pv1, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pv2, 0, dataSize));

			//warp frame 1
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_u, d_v, d_i1warp);
			if ((method == METHOD_TVCHARBGRAD) || (method == METHOD_TVCHARBGRAD_FN)) {
				WarpImage(pIx1[level], pW[level], pH[level], pS[level], d_u, d_v, d_ix1warp);
				WarpImage(pIy1[level], pW[level], pH[level], pS[level], d_u, d_v, d_iy1warp);
			}

			//compute derivatives
			if (method == METHOD_TVL1PATCH) {
				ComputeDerivativesPatch(pI0[level], d_i1warp, pW[level], pH[level], pS[level], d_Ix, d_Iy, d_Iz);
			}
			else {
				ComputeDerivatives(pI0[level], d_i1warp, pW[level], pH[level], pS[level], d_Ix, d_Iy, d_Iz);
			}

			if ((method == METHOD_TVCHARBGRAD) || (method == METHOD_TVCHARBGRAD_FN)) {
				ComputeDerivatives(pIx0[level], d_ix1warp, pW[level], pH[level], pS[level], d_Ixx, d_Ixy, d_Ixz);
				ComputeDerivatives(pIy0[level], d_iy1warp, pW[level], pH[level], pS[level], d_Iyx, d_Iyy, d_Iyz);
			}

			//inner iteration
			for (int iter = 0; iter < nSolverIters; ++iter)
			{
				if ((method == METHOD_TVCHARBGRAD) || (method == METHOD_TVCHARBGRAD_FN)) {
					SolveDataCharbForTVGrad(d_du, d_dv, d_dumed, d_dvmed,
						d_pu1, d_pu2, d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						d_Ixx, d_Ixy, d_Ixz,
						d_Iyx, d_Iyy, d_Iyz,
						pW[level], pH[level], pS[level],
						lambda, lambdagrad, theta,
						d_dus, d_dvs,
						d_dumeds, d_dvmeds);
					Swap(d_du, d_dus);
					Swap(d_dv, d_dvs);
					Swap(d_dumed, d_dumeds);
					Swap(d_dvmed, d_dvmeds);
				}
				else if (method == METHOD_TVCHARB) {
					SolveDataCharbForTV(d_du, d_dv, d_dumed, d_dvmed,
						d_pu1, d_pu2, d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, theta,
						d_dus, d_dvs,
						d_dumeds, d_dvmeds);
					Swap(d_du, d_dus);
					Swap(d_dv, d_dvs);
					Swap(d_dumed, d_dumeds);
					Swap(d_dvmed, d_dvmeds);
				}
				else if ((method == METHOD_TVL1) || (method == SCENEFLOW_KINECT_TVL1)) {
					SolveDataL1(d_dumed, d_dvmed,
						d_pu1, d_pu2,
						d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, theta,
						d_dumeds, d_dvmeds); //du1 = duhat output
					Swap(d_dumed, d_dumeds);
					Swap(d_dvmed, d_dvmeds);
				}
				else {
					SolveDataL1(d_dumed, d_dvmed,
						d_pu1, d_pu2,
						d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, theta,
						d_dumeds, d_dvmeds); //du1 = duhat output
					Swap(d_dumed, d_dumeds);
					Swap(d_dvmed, d_dvmeds);
				}

				SolveSmoothDualTVGlobal(d_dumed, d_dvmed,
					d_pu1, d_pu2, d_pv1, d_pv2,
					pW[level], pH[level], pS[level],
					tau, theta,
					d_pu1s, d_pu2s, d_pv1s, d_pv2s);
				Swap(d_pu1, d_pu1s);
				Swap(d_pu2, d_pu2s);
				Swap(d_pv1, d_pv1s);
				Swap(d_pv2, d_pv2s);
				//***********************************

				/*MedianFilter(d_dumed, d_dvmed, pW[level], pH[level], pS[level],
				d_dumeds, d_dvmeds, 5);
				Swap(d_dumed, d_dumeds);
				Swap(d_dvmed, d_dvmeds);*/
			}
			// one median filtering
			MedianFilter(d_dumed, d_dvmed, pW[level], pH[level], pS[level],
				d_dumeds, d_dvmeds, 5);
			Swap(d_dumed, d_dumeds);
			Swap(d_dvmed, d_dvmeds);

			// update u, v
			Add(d_u, d_dumed, pH[level] * pS[level], d_u);
			Add(d_v, d_dvmed, pH[level] * pS[level], d_v);
			/*
			MedianFilter(d_u, d_v, pW[level], pH[level], pS[level],
			d_dumeds, d_dvmeds, 5);
			Swap(d_u, d_dumeds);
			Swap(d_v, d_dvmeds);*/
		}

		//upscale
		if (level > 0)
		{
			// scale uv
			//float scale = (float)pW[level + 1] / (float)pW[level];
			float scale = fScale;

			Upscale(d_u, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);

			//float scaleY = (float)pH[level + 1] / (float)pH[level];

			Upscale(d_v, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);

			Swap(d_u, d_us);
			Swap(d_v, d_vs);
		}
	}
	if (withVisualization) {
		FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	}
	//FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	//SolveSceneFlow(d_u, d_v, d_depth016u, d_depth116u, width, height, stride, d_sceneflow);
	//std::cout << stride << " " << height << " " << height << " " << inputChannels << std::endl;
	return 0;
}

//OLD
int CudaFlow::_solveOpticalFlow(cv::Mat i0, cv::Mat i1, cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb, float flowScale) {
	// Convert RGB to Gray
	h_i0rgb = (uchar3 *)i0.ptr();
	h_i1rgb = (uchar3 *)i1.ptr();
	checkCudaErrors(cudaMemcpy(d_i08uc3, h_i0rgb, dataSize8uc3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_i18uc3, h_i1rgb, dataSize8uc3, cudaMemcpyHostToDevice));

	rgbToGray(d_i08uc3, pI0[0], width, height, stride);
	rgbToGray(d_i08uc3, pI1[0], width, height, stride);

	if (method == METHOD_TVCHARBGRAD) {
		ComputeDeriv(pI0[0], width, height, stride, pIx0[0], pIy0[0]);
		ComputeDeriv(pI1[0], width, height, stride, pIx1[0], pIy1[0]);
	}

	// construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pI0[level]);

		Downscale(pI1[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pI1[level]);

		if (method == METHOD_TVCHARBGRAD) {
			Downscale(pIx0[level - 1],
				pW[level - 1], pH[level - 1], pS[level - 1],
				pW[level], pH[level], pS[level],
				pIx0[level]);

			Downscale(pIx1[level - 1],
				pW[level - 1], pH[level - 1], pS[level - 1],
				pW[level], pH[level], pS[level],
				pIx1[level]);

			Downscale(pIy0[level - 1],
				pW[level - 1], pH[level - 1], pS[level - 1],
				pW[level], pH[level], pS[level],
				pIy0[level]);

			Downscale(pIy1[level - 1],
				pW[level - 1], pH[level - 1], pS[level - 1],
				pW[level], pH[level], pS[level],
				pIy1[level]);
		}
	}

	// solve flow
	checkCudaErrors(cudaMemset(d_u, 0, dataSize));
	checkCudaErrors(cudaMemset(d_v, 0, dataSize));

	for (int level = nLevels - 1; level >= 0; level--) {
		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			//std::cout << level << std::endl;
			//initialize zeros
			checkCudaErrors(cudaMemset(d_du, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dv, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dus, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvs, 0, dataSize));

			checkCudaErrors(cudaMemset(d_dumed, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvmed, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dumeds, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvmeds, 0, dataSize));

			checkCudaErrors(cudaMemset(d_pu1, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pu2, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pv1, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pv2, 0, dataSize));

			//warp frame 1
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_u, d_v, d_i1warp);
			if (method == METHOD_TVCHARBGRAD) {
				WarpImage(pIx1[level], pW[level], pH[level], pS[level], d_u, d_v, d_ix1warp);
				WarpImage(pIy1[level], pW[level], pH[level], pS[level], d_u, d_v, d_iy1warp);
			}

			//compute derivatives
			if (method == METHOD_TVL1PATCH) {
				ComputeDerivativesPatch(pI0[level], d_i1warp, pW[level], pH[level], pS[level], d_Ix, d_Iy, d_Iz);
			}
			else {
				ComputeDerivatives(pI0[level], d_i1warp, pW[level], pH[level], pS[level], d_Ix, d_Iy, d_Iz);
			}

			if (method == METHOD_TVCHARBGRAD) {
				ComputeDerivatives(pIx0[level], d_ix1warp, pW[level], pH[level], pS[level], d_Ixx, d_Ixy, d_Ixz);
				ComputeDerivatives(pIy0[level], d_iy1warp, pW[level], pH[level], pS[level], d_Iyx, d_Iyy, d_Iyz);
			}

			//inner iteration
			for (int iter = 0; iter < nSolverIters; ++iter)
			{
				if (method == METHOD_TVCHARBGRAD) {
					SolveDataCharbForTVGrad(d_du, d_dv, d_dumed, d_dvmed,
						d_pu1, d_pu2, d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						d_Ixx, d_Ixy, d_Ixz,
						d_Iyx, d_Iyy, d_Iyz,
						pW[level], pH[level], pS[level],
						lambda, lambdagrad, theta,
						d_dus, d_dvs,
						d_dumeds, d_dvmeds);
					Swap(d_du, d_dus, d_dv, d_dvs, d_dumed, d_dumeds, d_dvmed, d_dvmeds);
				}
				else if (method == METHOD_TVCHARB) {
					SolveDataCharbForTV(d_du, d_dv, d_dumed, d_dvmed,
						d_pu1, d_pu2, d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, theta,
						d_dus, d_dvs,
						d_dumeds, d_dvmeds);
					Swap(d_du, d_dus, d_dv, d_dvs, d_dumed, d_dumeds, d_dvmed, d_dvmeds);
				}
				else if (method == METHOD_TVL1) {
					SolveDataL1(d_dumed, d_dvmed,
						d_pu1, d_pu2,
						d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, theta,
						d_dumeds, d_dvmeds); //du1 = duhat output
					Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);
				}
				else {
					SolveDataL1(d_dumed, d_dvmed,
						d_pu1, d_pu2,
						d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, theta,
						d_dumeds, d_dvmeds); //du1 = duhat output
					Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);
				}

				SolveSmoothDualTVGlobal(d_dumed, d_dvmed,
					d_pu1, d_pu2, d_pv1, d_pv2,
					pW[level], pH[level], pS[level],
					tau, theta,
					d_pu1s, d_pu2s, d_pv1s, d_pv2s);
				Swap(d_pu1, d_pu1s, d_pu2, d_pu2s, d_pv1, d_pv1s, d_pv2, d_pv2s);
				//***********************************
			}
			// one median filtering
			MedianFilter(d_dumed, d_dvmed, pW[level], pH[level], pS[level],
				d_dumeds, d_dvmeds, 5);
			Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);

			// update u, v
			Add(d_u, d_dumed, pH[level] * pS[level], d_u);
			Add(d_v, d_dvmed, pH[level] * pS[level], d_v);
		}

		//upscale
		if (level > 0)
		{
			// scale uv
			//float scaleX = (float)pW[level + 1] / (float)pW[level];
			float scale = fScale;

			Upscale(d_u, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);

			//float scaleY = (float)pH[level + 1] / (float)pH[level];

			Upscale(d_v, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);

			Swap(d_u, d_us);
			Swap(d_v, d_vs);
		}
	}
	FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	//std::cout << stride << " " << height << " " << height << " " << inputChannels << std::endl;
	checkCudaErrors(cudaMemcpy((float3 *)uvrgb.ptr(), d_uvrgb, stride * height * sizeof(float) * inputChannels, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)u.ptr(), d_u, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)v.ptr(), d_v, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}
int CudaFlow::solveOpticalFlow(cv::Mat i0, cv::Mat i1, cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb, float flowScale = 5.0f) {
	return _solveOpticalFlow(i0, i1, u, v, uvrgb, flowScale);
}
int CudaFlow::solveOpticalFlow(cv::Mat i0, cv::Mat i1, cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb) {
	return _solveOpticalFlow(i0, i1, u, v, uvrgb, 5.0f);
}


//Solve Scene Flow
int CudaFlow::solveSceneFlow(float focalx, float focaly, float cx, float cy, float flowScale = 5.0f) {
	this->depthCameraFocalX = focalx;
	this->depthCameraFocalY = focaly;
	this->depthCameraPrincipalX = cx;
	this->depthCameraPrincipalY = cy;
	return _solveSceneFlow(flowScale, 1.0f);
}

int CudaFlow::solveSceneFlow(float focalx, float focaly, float cx, float cy, float opticalFlowScale = 5.0f, float sceneFlowScale = 1.0f) {
	this->depthCameraFocalX = focalx;
	this->depthCameraFocalY = focaly;
	this->depthCameraPrincipalX = cx;
	this->depthCameraPrincipalY = cy;
	return _solveSceneFlow(opticalFlowScale, sceneFlowScale);
}

int CudaFlow::solveSceneFlow() {
	return _solveSceneFlow(5.0f, 1.0f);
}

int CudaFlow::_solveSceneFlow(float opticalFlowScale, float sceneFlowScale) {
	solveOpticalFlow(opticalFlowScale);
	//convert depths to 3d coordinates
	Cv16uTo32f(d_depth016u, d_depth032f, width, height, stride);
	Cv16uTo32f(d_depth116u, d_depth132f, width, height, stride);

	//solve sceneflow using optical flow and depth
	SolveSceneFlow_(d_u, d_v, d_depth032f, d_depth032f,
		depthCameraFocalX, depthCameraFocalY, depthCameraPrincipalX, depthCameraPrincipalY,
		width, height, stride, d_sceneflow);
	SceneFlowToHSV(d_sceneflow, width, height, stride, d_sceneflowrgb, sceneFlowScale);
	return 0;
}

int CudaFlow::solveSceneFlowAndPointCloud(float focalx, float focaly, float cx, float cy, float opticalFlowScale = 5.0f, float sceneFlowScale = 1.0f) {
	this->depthCameraFocalX = focalx;
	this->depthCameraFocalY = focaly;
	this->depthCameraPrincipalX = cx;
	this->depthCameraPrincipalY = cy;
	return _solveSceneFlowAndPointCloud(opticalFlowScale, sceneFlowScale);
}

int CudaFlow::_solveSceneFlowAndPointCloud(float opticalFlowScale, float sceneFlowScale) {
	solveOpticalFlow(opticalFlowScale);
	//convert depths to 3d coordinates
	Cv16uTo32f(d_depth016u, d_depth032f, width, height, stride);
	Cv16uTo32f(d_depth116u, d_depth132f, width, height, stride);

	//solve sceneflow using optical flow and depth
	SolveSceneFlow_(d_u, d_v, d_depth032f, d_depth032f,
		depthCameraFocalX, depthCameraFocalY, depthCameraPrincipalX, depthCameraPrincipalY,
		width, height, stride, d_pcloud0, d_pcloud1, d_sceneflow);
	SceneFlowToHSV(d_sceneflow, width, height, stride, d_sceneflowrgb, sceneFlowScale);
	return 0;
}


CudaFlow::~CudaFlow() {
	for (int i = 0; i < nLevels; i++)
	{
		checkCudaErrors(cudaFree(pI0[i]));
		checkCudaErrors(cudaFree(pI1[i]));
		if (method == METHOD_TVCHARBGRAD) {
			checkCudaErrors(cudaFree(pIx0[i]));
			checkCudaErrors(cudaFree(pIx1[i]));
			checkCudaErrors(cudaFree(pIy0[i]));
			checkCudaErrors(cudaFree(pIy1[i]));
		}
	}

	checkCudaErrors(cudaFree(d_i1warp));


	checkCudaErrors(cudaFree(d_du));
	checkCudaErrors(cudaFree(d_dv));
	checkCudaErrors(cudaFree(d_dus));
	checkCudaErrors(cudaFree(d_dvs));

	checkCudaErrors(cudaFree(d_dumed));
	checkCudaErrors(cudaFree(d_dvmed));
	checkCudaErrors(cudaFree(d_dumeds));
	checkCudaErrors(cudaFree(d_dvmeds));

	checkCudaErrors(cudaFree(d_pu1));
	checkCudaErrors(cudaFree(d_pu2));
	checkCudaErrors(cudaFree(d_pv1));
	checkCudaErrors(cudaFree(d_pv2));
	checkCudaErrors(cudaFree(d_pu1s));
	checkCudaErrors(cudaFree(d_pu2s));
	checkCudaErrors(cudaFree(d_pv1s));
	checkCudaErrors(cudaFree(d_pv2s));

	checkCudaErrors(cudaFree(d_Ix));
	checkCudaErrors(cudaFree(d_Iy));
	checkCudaErrors(cudaFree(d_Iz));

	if (method == METHOD_TVCHARBGRAD) {
		checkCudaErrors(cudaFree(d_Ixx));
		checkCudaErrors(cudaFree(d_Ixy));
		checkCudaErrors(cudaFree(d_Ixz));
		checkCudaErrors(cudaFree(d_Iyx));
		checkCudaErrors(cudaFree(d_Iyy));
		checkCudaErrors(cudaFree(d_Iyz));

		checkCudaErrors(cudaFree(d_ix1warp));
		checkCudaErrors(cudaFree(d_iy1warp));
	}

	checkCudaErrors(cudaFree(d_u));
	checkCudaErrors(cudaFree(d_v));
	checkCudaErrors(cudaFree(d_us));
	checkCudaErrors(cudaFree(d_vs));

	if (inputType == CV_8UC3) {
		checkCudaErrors(cudaFree(d_i08uc3));
		checkCudaErrors(cudaFree(d_i18uc3));
	}
	else if (inputType == CV_8U) {
		checkCudaErrors(cudaFree(d_i08u));
		checkCudaErrors(cudaFree(d_i18u));
	}
	else if (inputType == CV_16U) {
		checkCudaErrors(cudaFree(d_i016u));
		checkCudaErrors(cudaFree(d_i116u));
	}
	else if (inputType == CV_32F) {
		checkCudaErrors(cudaFree(d_i032f));
		checkCudaErrors(cudaFree(d_i132f));
	}
	//for correlation
	if (d_icorr032f != NULL) {
		checkCudaErrors(cudaFree(d_icorr032f));
		checkCudaErrors(cudaFree(d_icorr132f));
		checkCudaErrors(cudaFree(d_corrKernel));
		checkCudaErrors(cudaFree(d_corrSearchSpace));
		checkCudaErrors(cudaFree(d_ucorr));
		checkCudaErrors(cudaFree(d_vcorr));
		checkCudaErrors(cudaFree(d_corrSparseMask));
	}

	checkCudaErrors(cudaFree(d_uvrgb));

	if (method == SCENEFLOW_KINECT_TVL1) {
		checkCudaErrors(cudaFree(d_depth016u));
		checkCudaErrors(cudaFree(d_depth116u));
		checkCudaErrors(cudaFree(d_depth032f));
		checkCudaErrors(cudaFree(d_depth132f));
		checkCudaErrors(cudaFree(d_sceneflow));
		checkCudaErrors(cudaFree(d_pcloud0));
		checkCudaErrors(cudaFree(d_pcloud1));
		checkCudaErrors(cudaFree(d_sceneflowrgb));

	}

	if (method == METHOD_TVL1_FN) {
		checkCudaErrors(cudaFree(d_ufn));
		checkCudaErrors(cudaFree(d_vfn));
		if (d_fnmask != NULL) {
			checkCudaErrors(cudaFree(d_fnmask));
			checkCudaErrors(cudaFree(d_fnmask_l));
		}
	}
}

int CudaFlow::close() {

	return 0;
}
// Align up n to the nearest multiple of m
inline int CudaFlow::iAlignUp(int n)
{
	int m = this->StrideAlignment;
	int mod = n % m;

	if (mod)
		return n + m - mod;
	else
		return n;
}

// swap two values
template<typename T>
inline void CudaFlow::Swap(T &a, T &ax)
{
	T t = a;
	a = ax;
	ax = t;
}

//swap four values
template<typename T>
inline void CudaFlow::Swap(T &a, T &ax, T &b, T &bx)
{
	Swap(a, ax);
	Swap(b, bx);
}

//swap eight values
template<typename T>
inline void CudaFlow::Swap(T &a, T &ax, T &b, T &bx, T &c, T &cx, T &d, T &dx)
{
	Swap(a, ax);
	Swap(b, bx);
	Swap(c, cx);
	Swap(d, dx);
}

int CudaFlow::computePyramidLevels(int width, int height, int minWidth, float scale) {
	int nLevels = 1;
	int pHeight = (int)((float)height / scale);
	while (pHeight > minWidth) {
		nLevels++;
		pHeight = (int)((float)pHeight / scale);
	}
	std::cout << "Pyramid Levels: " << nLevels << std::endl;
	return nLevels;
}