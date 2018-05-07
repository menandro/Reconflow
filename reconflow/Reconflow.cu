#include "ReconFlow.h"

ReconFlow::ReconFlow() {
	this->BlockWidth = 32;
	this->BlockHeight = 12;
	this->StrideAlignment = 32;
}

ReconFlow::ReconFlow(int BlockWidth, int BlockHeight, int StrideAlignment) {
	this->BlockWidth = BlockWidth;
	this->BlockHeight = BlockHeight;
	this->StrideAlignment = StrideAlignment;
}

int ReconFlow::initializeR(int width, int height, int channels, int nLevels, float scale, int method,
	float lambda = 100.0f, float lambdagrad = 400.0f, float lambdaf = 100.0f, float lambdams = 100.0f,
	float alphaTv = 3.0f, float alphaProj = 1.0f, float alphaFn = 0.5f, float tau = 0.25f,
	int nWarpIters = 1, int nSolverIters = 100) {
	this->alphaFn = alphaFn;
	this->_initializeR(width, height, channels, nLevels, scale, method,
		lambda, lambdagrad, lambdaf, lambdams,
		alphaTv, alphaProj, tau,
		nWarpIters, nSolverIters);
	return this->initializeLargeDisplacement();
}

int ReconFlow::_initializeR(int width, int height, int channels, int nLevels, float scale, int method,
	float lambda = 100.0f, float lambdagrad = 400.0f, float lambdaf = 100.0f, float lambdams = 100.0f,
	float alphaTv = 3.0f, float alphaProj = 1.0f, float tau = 0.25f,
	int nWarpIters = 1, int nSolverIters = 100)
{
	float thetaTv = 1.0f / alphaTv;
	this->initialize(width, height, channels, nLevels, scale, method,
		lambda, lambdagrad, thetaTv, tau,
		nWarpIters, nSolverIters);
	this->thetaTv = thetaTv;
	this->alphaProj = alphaProj;
	this->alphaTv = alphaTv;
	this->lambdaf = lambdaf;
	this->lambdams = lambdams;

	checkCudaErrors(cudaMalloc(&d_X, dataSize));
	checkCudaErrors(cudaMalloc(&d_Y, dataSize));
	checkCudaErrors(cudaMalloc(&d_Z, dataSize));

	checkCudaErrors(cudaMalloc(&d_Xs, dataSize));
	checkCudaErrors(cudaMalloc(&d_Ys, dataSize));
	checkCudaErrors(cudaMalloc(&d_Zs, dataSize));

	checkCudaErrors(cudaMalloc(&d_Xmed, dataSize));
	checkCudaErrors(cudaMalloc(&d_Ymed, dataSize));
	checkCudaErrors(cudaMalloc(&d_Zmed, dataSize));

	checkCudaErrors(cudaMalloc(&d_uproj, dataSize));
	checkCudaErrors(cudaMalloc(&d_vproj, dataSize));
	checkCudaErrors(cudaMalloc(&d_uc, dataSize));
	checkCudaErrors(cudaMalloc(&d_vc, dataSize));
	checkCudaErrors(cudaMalloc(&d_sku, dataSize));
	checkCudaErrors(cudaMalloc(&d_skv, dataSize));
	checkCudaErrors(cudaMalloc(&d_skus, dataSize));
	checkCudaErrors(cudaMalloc(&d_skvs, dataSize));
	return 0;
}

int ReconFlow::copy3dToHost(cv::Mat &X, cv::Mat &Y, cv::Mat &Z) {
	checkCudaErrors(cudaMemcpy((float *)X.ptr(), d_X, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)Y.ptr(), d_Y, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)Z.ptr(), d_Z, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}

int ReconFlow::setCameraMatrices(cv::Mat intrinsic) {
	//K0 = (double*)intrinsic.ptr();
	//K1 = (double*)intrinsic.ptr();
	return 0;
}

int ReconFlow::setCameraMatrices(cv::Mat intrinsic0, cv::Mat intrinsic1) {
	this->intrinsic0 = intrinsic0;
	this->intrinsic1 = intrinsic1;
	double *K0 = (double*)intrinsic0.ptr();
	double *K1 = (double*)intrinsic1.ptr();
	//Construct Intrinsic Camera Pyramid
	//pFocals = std::vector<double>(this->nLevels);
	//pCamMidX1s = std::vector<double>(this->nLevels);
	//pCamMidY1s = std::vector<double>(this->nLevels);
	pK0 = std::vector<cv::Mat>(this->nLevels);
	pK1 = std::vector<cv::Mat>(this->nLevels);

	//pFocals[0] = K0[0];
	//pCamMidX1s[0] = K1[2];
	//pCamMidY1s[0] = K1[5];

	pK0[0] = cv::Mat(cv::Size(3, 3), CV_64F, K0).clone();
	pK1[0] = cv::Mat(cv::Size(3, 3), CV_64F, K1).clone();
	//std::cout << "here" << std::endl;
	//std::cout << pK0[0].at<double>(0, 2);
	for (int level = 1; level < nLevels; level++) {
		pK0[level] = cv::Mat::zeros(cv::Size(3, 3), CV_64F);
		pK1[level] = cv::Mat::zeros(cv::Size(3, 3), CV_64F);

		pK0[level].at<double>(0, 0) = pK0[level - 1].at<double>(0, 0) / this->fScale; //focal
		pK0[level].at<double>(0, 2) = pK0[level - 1].at<double>(0, 2) / this->fScale; //camMidX
		pK0[level].at<double>(1, 1) = pK0[level - 1].at<double>(1, 1) / this->fScale; //focal
		pK0[level].at<double>(1, 2) = pK0[level - 1].at<double>(1, 2) / this->fScale; //camMidY

		pK1[level].at<double>(0, 0) = pK1[level - 1].at<double>(0, 0) / this->fScale; //focal
		pK1[level].at<double>(0, 2) = pK1[level - 1].at<double>(0, 2) / this->fScale; //camMidX
		pK1[level].at<double>(1, 1) = pK1[level - 1].at<double>(1, 1) / this->fScale; //focal
		pK1[level].at<double>(1, 2) = pK1[level - 1].at<double>(1, 2) / this->fScale; //camMidY
																					  //std::cout << pK0[level].at<double>(0, 0) << " " << pK0[level].at<double>(0, 2) << " " << pK0[level].at<double>(1, 2) << std::endl;
																					  //std::cout << pK1[level].at<double>(0, 0) << " " << pK1[level].at<double>(0, 2) << " " << pK1[level].at<double>(1, 2) << std::endl;
	}

	return 0;
}

int ReconFlow::solveR(float flowScale) {
	//Obsolete
	cv::Mat R1, t1;
	return solveReconFlow(R1, t1, flowScale);
}

int ReconFlow::solveR() {
	//Obsolete
	cv::Mat R1, t1;
	return solveReconFlow(R1, t1, 5.0f);
}

int ReconFlow::solveR(cv::Mat R1, cv::Mat t1, float flowScale) {
	return solveReconFlow(R1, t1, flowScale);
}

int ReconFlow::solveR(cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1, float flowScale) {
	return solveReconFlow(R0, t0, R1, t1, flowScale);
}

int ReconFlow::solveReconFlow(cv::Mat rot1, cv::Mat tr1,//input pose
	float flowScale) //display normalization
{
	cv::Mat rot0 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat tr0 = cv::Mat::zeros(3, 1, CV_64F);
	return solveReconFlow(rot0, tr0, rot1, tr1, flowScale);
}

//Simultaneous optical flow estimation and 3d reconstruction
int ReconFlow::solveReconFlow(cv::Mat rot0, cv::Mat tr0, cv::Mat rot1, cv::Mat tr1,//input pose
	float flowScale) //display normalization
{
	cv::Mat Rt0, Rt1;
	cv::hconcat(rot0, tr0, Rt0);
	cv::hconcat(rot1, tr1, Rt1);

	double *R1 = (double*)rot1.ptr();
	double *t1 = (double*)tr1.ptr();
	double *R0 = (double*)rot0.ptr();
	double *t0 = (double*)tr0.ptr();
	//std::cout << (float)pose0.at<double>(0, 0) << std::endl;
	//std::cout << (float)R1[0] << " " << (float)R1[1] << std::endl;
	//std::cout << (float)R0[0] << " " << (float)R0[1] << std::endl;

	if (inputType == CV_8UC3) {
		rgbToGray(d_i08uc3, pI0[0], width, height, stride);
		rgbToGray(d_i18uc3, pI1[0], width, height, stride);
	}

	if ((method == METHODR_TVCHARBGRAD_MS) || (method == METHODR_TVCHARBGRAD_MS_FN)) {
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
	checkCudaErrors(cudaMemset(d_uproj, 0, dataSize));
	checkCudaErrors(cudaMemset(d_vproj, 0, dataSize));
	checkCudaErrors(cudaMemset(d_uc, 0, dataSize));
	checkCudaErrors(cudaMemset(d_vc, 0, dataSize));

	checkCudaErrors(cudaMemset(d_X, 0, dataSize));
	checkCudaErrors(cudaMemset(d_Y, 0, dataSize));
	checkCudaErrors(cudaMemset(d_Z, 0, dataSize));
	checkCudaErrors(cudaMemset(d_sku, 0, dataSize));
	checkCudaErrors(cudaMemset(d_skv, 0, dataSize));

	for (int level = nLevels - 1; level >= 0; level--) {
		//checkCudaErrors(cudaMemset(d_sku, 0, dataSize));
		//checkCudaErrors(cudaMemset(d_skv, 0, dataSize));
		cv::Mat pose0 = pK0[level] * Rt0;
		cv::Mat pose1 = pK1[level] * Rt1;
		double *P = (double*)pose0.ptr();
		double *Q = (double*)pose1.ptr();
		double *K0 = (double*)pK0[level].ptr();
		double *K1 = (double*)pK1[level].ptr();

		if ((method == METHODR_TVL1_MS_FN) || (method == METHODR_TVL1_MS_FNSPARSE)) {
			//downscale ufn and vfn
			//std::cout << level << std::endl;
			float scale = (float)pW[level] / (float)pW[0];
			Downscale(d_ufn, pW[0], pH[0], pS[0], pW[level], pH[level], pS[level], scale, d_ufn_l);
			Downscale(d_vfn, pW[0], pH[0], pS[0], pW[level], pH[level], pS[level], scale, d_vfn_l);
			Downscale(d_fnmask, pW[0], pH[0], pS[0], pW[level], pH[level], pS[level], d_fnmask_l);
			//std::cout << "herex" << std::endl;
			//Swap(d_ufn_l, d_ufns, d_vfn_l, d_vfns);
		}
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

			//checkCudaErrors(cudaMemset(d_sku, 0, dataSize));
			//checkCudaErrors(cudaMemset(d_skv, 0, dataSize));

			//warp frame 1
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_u, d_v, d_i1warp);
			if ((method == METHODR_TVCHARBGRAD_MS) || (method == METHODR_TVCHARBGRAD_MS_FN)) {
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

			if ((method == METHODR_TVCHARBGRAD_MS) || (method == METHODR_TVCHARBGRAD_MS_FN)) {
				ComputeDerivatives(pIx0[level], d_ix1warp, pW[level], pH[level], pS[level], d_Ixx, d_Ixy, d_Ixz);
				ComputeDerivatives(pIy0[level], d_iy1warp, pW[level], pH[level], pS[level], d_Iyx, d_Iyy, d_Iyz);
			}

			//inner iteration
			for (int iter = 0; iter < nSolverIters; ++iter)
			{
				//solve du
				if (method == METHODR_TVL1_MS) {
					SolveReconDataL1(d_u, d_v,
						d_dumed, d_dvmed,
						d_uproj, d_vproj,
						d_sku, d_skv,
						d_pu1, d_pu2,
						d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, alphaTv, alphaProj,
						d_dumeds, d_dvmeds); //du1 = duhat output
					Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);
				}

				else if (method == METHODR_TVL1_MS_FN) {
					SolveReconDataL1Fn(d_u, d_v,
						d_dumed, d_dvmed,
						d_uproj, d_vproj,
						d_ufn_l, d_vfn_l,
						d_sku, d_skv,
						d_pu1, d_pu2,
						d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, alphaTv, alphaProj, alphaFn,
						d_dumeds, d_dvmeds); //du1 = duhat output
					Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);
				}

				else if (method == METHODR_TVL1_MS_FNSPARSE) { //sparse flownet
					SolveReconDataL1Fn(d_u, d_v,
						d_dumed, d_dvmed,
						d_uproj, d_vproj,
						d_ufn_l, d_vfn_l,
						d_sku, d_skv,
						d_pu1, d_pu2,
						d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, alphaTv, alphaProj, alphaFn, d_fnmask_l,
						d_dumeds, d_dvmeds); //du1 = duhat output
					Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);
				}

				else {
					SolveReconDataL1(d_u, d_v,
						d_dumed, d_dvmed,
						d_uproj, d_vproj,
						d_sku, d_skv,
						d_pu1, d_pu2,
						d_pv1, d_pv2,
						d_Ix, d_Iy, d_Iz,
						pW[level], pH[level], pS[level],
						lambda, alphaTv, alphaProj,
						d_dumeds, d_dvmeds); //du1 = duhat output
					Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);
				}

				//solve TV
				SolveSmoothDualTVGlobal(d_dumed, d_dvmed,
					d_pu1, d_pu2, d_pv1, d_pv2,
					pW[level], pH[level], pS[level],
					tau, thetaTv,
					d_pu1s, d_pu2s, d_pv1s, d_pv2s);
				Swap(d_pu1, d_pu1s, d_pu2, d_pu2s, d_pv1, d_pv1s, d_pv2, d_pv2s);

				//solve uproj
				//float lambdaf_val = lambdaf* ((float)iter / (float)nSolverIters);
				SolveUVProj(d_u, d_v, d_dumed, d_dvmed,
					d_uc, d_vc, d_sku, d_skv,
					pW[level], pH[level], pS[level],
					lambdaf, alphaProj,
					d_uproj, d_vproj);


				//solve for X
				if (false) { //no ms
					Solve3dMinimalArea(d_uproj, d_vproj,
						d_X, d_Y, d_Z,
						P, Q, lambdaf, 0.0f,
						pW[level], pH[level], pS[level],
						d_Xs, d_Ys, d_Zs);
					Swap(d_X, d_Xs);
					Swap(d_Y, d_Ys);
					Swap(d_Z, d_Zs);
				}
				else {
					Solve3dMinimalArea(d_uproj, d_vproj,
						d_X, d_Y, d_Z,
						P, Q, lambdaf, lambdams,
						pW[level], pH[level], pS[level],
						d_Xs, d_Ys, d_Zs);
					Swap(d_X, d_Xs);
					Swap(d_Y, d_Ys);
					Swap(d_Z, d_Zs);
				}
				MedianFilter3D(d_X, d_Y, d_Z,
					pW[level], pH[level], pS[level],
					d_Xmed, d_Ymed, d_Zmed, 3);
				CleanUp3D(d_X, d_Y, d_Z, d_Xmed, d_Ymed, d_Zmed,
					pW[level], pH[level], pS[level],
					d_Xs, d_Ys, d_Zs);
				Swap(d_X, d_Xs);
				Swap(d_Y, d_Ys);
				Swap(d_Z, d_Zs);
				//Median filter the 3D

				//Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);

				//compute uc
				SolveUc(d_X, d_Y, d_Z, R1, t1, K1,
					pW[level], pH[level], pS[level],
					d_uc, d_vc);

				//update sk
				UpdateSk(d_sku, d_skv, d_u, d_v, d_dumed, d_dvmed, d_uproj, d_vproj,
					pW[level], pH[level], pS[level],
					d_skus, d_skvs);
				Swap(d_sku, d_skus, d_skv, d_skvs);

				//***********************************
			}
			// one median filtering
			MedianFilter(d_dumed, d_dvmed, pW[level], pH[level], pS[level],
				d_dumeds, d_dvmeds, 5);
			Swap(d_dumed, d_dumeds, d_dvmed, d_dvmeds);

			// update u, v
			Add(d_u, d_dumed, pH[level] * pS[level], d_u);
			Add(d_v, d_dvmed, pH[level] * pS[level], d_v);

			MedianFilter(d_u, d_v, pW[level], pH[level], pS[level],
				d_dumeds, d_dvmeds, 5);
			Swap(d_u, d_dumeds, d_v, d_dvmeds);
		}

		//upscale
		if (level > 0)
		{
			// scale uv
			//float scaleX = (float)pW[level + 1] / (float)pW[level];
			float scale = fScale;

			Upscale(d_u, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_v, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);

			Swap(d_u, d_us);
			Swap(d_v, d_vs);

			Upscale(d_uc, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_vc, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);
			Swap(d_uc, d_us);
			Swap(d_vc, d_vs);

			Upscale(d_uproj, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_vproj, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);
			Swap(d_uproj, d_us);
			Swap(d_vproj, d_vs);

			Upscale(d_sku, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_skv, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);
			Swap(d_sku, d_us);
			Swap(d_skv, d_vs);
		}
	}
	FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	//std::cout << stride << " " << height << " " << height << " " << inputChannels << std::endl;
	//checkCudaErrors(cudaMemcpy((float3 *)uvrgb.ptr(), d_uvrgb, stride * height * sizeof(float) * inputChannels, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy((float *)u.ptr(), d_u, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy((float *)v.ptr(), d_v, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}

int ReconFlow::removeEdgeArtifacts3D(int surfaceWidth, float maxSurfaceArea) {
	RemoveEdgeArtifacts3D(d_X, d_Y, d_Z,
		surfaceWidth, maxSurfaceArea,
		width, height, stride,
		d_Xs, d_Ys, d_Zs);
	Swap(d_X, d_Xs);
	Swap(d_Y, d_Ys);
	Swap(d_Z, d_Zs);
	return 0;
}

ReconFlow::~ReconFlow() {
	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_Y));
	checkCudaErrors(cudaFree(d_Z));

	checkCudaErrors(cudaFree(d_Xs));
	checkCudaErrors(cudaFree(d_Ys));
	checkCudaErrors(cudaFree(d_Zs));

	checkCudaErrors(cudaFree(d_Xmed));
	checkCudaErrors(cudaFree(d_Ymed));
	checkCudaErrors(cudaFree(d_Zmed));

	checkCudaErrors(cudaFree(d_uproj));
	checkCudaErrors(cudaFree(d_vproj));
	checkCudaErrors(cudaFree(d_uc));
	checkCudaErrors(cudaFree(d_vc));
	checkCudaErrors(cudaFree(d_sku));
	checkCudaErrors(cudaFree(d_skv));
	checkCudaErrors(cudaFree(d_skus));
	checkCudaErrors(cudaFree(d_skvs));

	if ((method == METHODR_TVL1_MS_FN) || (method == METHODR_TVL1_MS_FNSPARSE)) {
		checkCudaErrors(cudaFree(d_ufn));
		checkCudaErrors(cudaFree(d_vfn));
		if (d_fnmask != NULL) {
			checkCudaErrors(cudaFree(d_fnmask));
			checkCudaErrors(cudaFree(d_fnmask_l));
		}
	}
}