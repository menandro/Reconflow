#include "CudaFlow.h"

int CudaFlow::copyMaskToDevice(cv::Mat mask0, cv::Mat mask1) {
	checkCudaErrors(cudaMemcpy(d_mask0, (float *)mask0.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mask1, (float *)mask1.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int CudaFlow::solveInpainting(float flowScale) {
	return _solveInpainting(flowScale);
}

int CudaFlow::_solveInpainting(float flowScale) {
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

	Cv32fToGray(d_mask0, pMask0[0], width, height, stride);
	Cv32fToGray(d_mask1, pMask1[0], width, height, stride);

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

		//masks
		Downscale(pMask0[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pMask0[level]);
		Downscale(pMask1[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pMask1[level]);
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

			//compute derivatives
			ComputeDerivatives(pI0[level], d_i1warp, pW[level], pH[level], pS[level], d_Ix, d_Iy, d_Iz);

			//inner iteration
			for (int iter = 0; iter < nSolverIters; ++iter)
			{
				SolveDataL1Inpaint(d_dumed, d_dvmed,
					pMask0[level], pMask1[level],
					d_pu1, d_pu2,
					d_pv1, d_pv2,
					d_Ix, d_Iy, d_Iz,
					pW[level], pH[level], pS[level],
					lambda, theta,
					d_dumeds, d_dvmeds); //du1 = duhat output
				Swap(d_dumed, d_dumeds);
				Swap(d_dvmed, d_dvmeds);

				SolveSmoothDualTVGlobal(d_dumed, d_dvmed,
					d_pu1, d_pu2, d_pv1, d_pv2,
					pW[level], pH[level], pS[level],
					tau, theta,
					d_pu1s, d_pu2s, d_pv1s, d_pv2s);
				Swap(d_pu1, d_pu1s);
				Swap(d_pu2, d_pu2s);
				Swap(d_pv1, d_pv1s);
				Swap(d_pv2, d_pv2s);
			}
			// one median filtering
			MedianFilter(d_dumed, d_dvmed, pW[level], pH[level], pS[level],
				d_dumeds, d_dvmeds, 5);
			Swap(d_dumed, d_dumeds);
			Swap(d_dvmed, d_dvmeds);

			// update u, v
			Add(d_u, d_dumed, pH[level] * pS[level], d_u);
			Add(d_v, d_dvmed, pH[level] * pS[level], d_v);
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