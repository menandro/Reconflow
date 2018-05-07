#include "CudaFlow.h"
#include <opencv2/optflow.hpp>

int CudaFlow::initializeCorrelation(int kernelSize, int maxSearchWidth, int maxSearchHeight) {
	this->corrKernelSize = kernelSize;
	this->corrMaxSearchHeight = maxSearchHeight;
	this->corrMaxSearchWidth = maxSearchWidth;
	this->corrStride = iAlignUp(maxSearchWidth);
	checkCudaErrors(cudaMalloc(&d_icorr032f, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_icorr132f, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_corrKernel, corrKernelSize * corrKernelSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_corrSearchSpace, corrMaxSearchWidth * corrMaxSearchHeight * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_corrOutput, corrMaxSearchWidth * corrMaxSearchHeight * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_ucorr, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_vcorr, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_uvrgbcorr, dataSize32fc3));
	checkCudaErrors(cudaMalloc(&d_corrSparseMask, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_derivMask, dataSize32f));
	return 0;
}

int CudaFlow::solveCorrPatchMatch(const char *picName, const char *flowName) {
	//this->_solveCorrPatchMatch();
	this->_solveCorrPatchMatch_cpu(picName, flowName);
	return 0;
}

int CudaFlow::solveCorrPatchMatch() {
	//this->_solveCorrPatchMatch();
	this->_solveCorrPatchMatch_cpu("new", "new");
	return 0;
}

int CudaFlow::_solveCorrPatchMatch_cpu(const char *picName, const char *flowName) {
	//bind textures
	rgbToGray(d_i08uc3, d_icorr032f, width, height, stride);
	rgbToGray(d_i18uc3, d_icorr132f, width, height, stride);
	CorrelationBindTextures(d_icorr032f, d_icorr132f, width, height, stride);
	cv::Mat u = cv::Mat::zeros(cv::Size(stride, height), CV_32F);
	cv::Mat v = cv::Mat::zeros(cv::Size(stride, height), CV_32F);
	cv::Mat sMask = cv::Mat::zeros(cv::Size(stride, height), CV_32F);

	//solve derivative to accept only high derivative kernels
	cv::Mat derivMask = cv::Mat(cv::Size(stride, height), CV_32F);
	ComputeDerivMask(d_icorr032f, width, height, stride, d_derivMask, 0.001f);
	checkCudaErrors(cudaMemcpy((float*)derivMask.ptr(), d_derivMask, dataSize32f, cudaMemcpyDeviceToHost));

	//int total = 0;
	for (int j = 0; j < height; j += 1) {
		for (int i = 0; i < width; i += 1) {
			//5x5 correlation
			CorrelationKernelSampling(i, j, d_corrKernel, width, height);
			CorrelationSearchSampling(i, j, d_corrSearchSpace);
			Correlation(d_corrKernel, d_corrSearchSpace, d_corrOutput);
			///TODO: get maximum value and save the u, v
			//CPU Version
			cv::Mat corrResult = cv::Mat(cv::Size(corrMaxSearchWidth, corrMaxSearchHeight), CV_32F);
			checkCudaErrors(cudaMemcpy(corrResult.ptr(), d_corrOutput, corrMaxSearchHeight*corrMaxSearchWidth * sizeof(float), cudaMemcpyDeviceToHost));
			double minVal, maxVal;
			cv::Point minLoc, maxLoc;
			cv::minMaxLoc(corrResult, &minVal, &maxVal, &minLoc, &maxLoc);
			//cv::Scalar mean, stdev;
			//cv::meanStdDev(corrResult, mean, stdev);
			//std::cout << minLoc << " " << minVal << std::endl;
			//std::cout << stdev << std::endl;
			//if ((mean[0] - minVal) > stdev[0]/2) {
			if (derivMask.at<float>(j, i) == 1.0f) {
				u.at<float>(j, i) = minLoc.x - (corrMaxSearchWidth / 2);
				v.at<float>(j, i) = minLoc.y - (corrMaxSearchHeight / 2);
				sMask.at<float>(j, i) = minVal;
			}
			//}
			//std::cout << u.at<float>(j, i) << "," << v.at<float>(j, i) << " " << minVal << std::endl;
			//para kita lahat
			/*u.at<float>(j-1, i-1) = minLoc.x - (corrMaxSearchWidth / 2);
			v.at<float>(j-1, i-1) = minLoc.y - (corrMaxSearchHeight / 2);
			u.at<float>(j, i-1) = minLoc.x - (corrMaxSearchWidth / 2);
			v.at<float>(j, i-1) = minLoc.y - (corrMaxSearchHeight / 2);
			u.at<float>(j-1, i) = minLoc.x - (corrMaxSearchWidth / 2);
			v.at<float>(j-1, i) = minLoc.y - (corrMaxSearchHeight / 2);*/

		}
	}
	//std::cout << total << std::endl;
	//cv::Mat output = cv::Mat(cv::Size(corrMaxSearchWidth, corrMaxSearchHeight), CV_32F);
	//checkCudaErrors(cudaMemcpy(output.ptr(), d_corrOutput, corrMaxSearchWidth * corrMaxSearchHeight * sizeof(float), cudaMemcpyDeviceToHost));
	//cv::imshow("uv", sMask);
	//cv::waitKey();

	checkCudaErrors(cudaMemcpy(d_ucorr, (float*)u.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vcorr, (float*)v.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_corrSparseMask, (float*)sMask.ptr(), dataSize32f, cudaMemcpyHostToDevice));

	FlowToHSV(d_ucorr, d_vcorr, width, height, stride, d_uvrgbcorr, 50.0);
	cv::Mat uvrgb = cv::Mat(cv::Size(stride, height), CV_32FC3);
	checkCudaErrors(cudaMemcpy((float3*)uvrgb.ptr(), d_uvrgbcorr, dataSize32fc3, cudaMemcpyDeviceToHost));
	//cv::imshow(windowName, uvrgb);
	//cv::waitKey();
	std::cout << picName << std::endl;
	cv::Mat uvrgb8uc3;
	uvrgb = uvrgb * 256;
	uvrgb.convertTo(uvrgb8uc3, CV_8UC3);
	cv::imwrite(picName, uvrgb8uc3);

	std::vector<cv::Mat> channelForward;
	channelForward.push_back(u);
	channelForward.push_back(v);
	cv::Mat forward;
	cv::merge(channelForward, forward);
	cv::optflow::writeOpticalFlow(flowName, forward);
	return 0;
}

int CudaFlow::_solveCorrPatchMatch() {
	//bind textures
	rgbToGray(d_i08uc3, d_icorr032f, width, height, stride);
	rgbToGray(d_i18uc3, d_icorr132f, width, height, stride);
	CorrelationBindTextures(d_icorr032f, d_icorr132f, width, height, stride);
	//int total = 0;
	for (int j = 7; j < height; j += 16) {
		for (int i = 5; i < width; i += 12) {
			//5x5 correlation
			CorrelationKernelSampling(i, j, d_corrKernel, width, height);
			CorrelationSearchSampling(i, j, d_corrSearchSpace);
			Correlation(d_corrKernel, d_corrSearchSpace, d_corrOutput);
			///TODO: get maximum value and save the u, v
			//CPU Version
			cv::Mat corrResult = cv::Mat(cv::Size(corrMaxSearchWidth, corrMaxSearchHeight), CV_32F);
			checkCudaErrors(cudaMemcpy(corrResult.ptr(), d_corrOutput, corrMaxSearchHeight*corrMaxSearchWidth * sizeof(float), cudaMemcpyDeviceToHost));
			double minVal, maxVal;
			cv::Point minLoc, maxLoc;
			cv::minMaxLoc(corrResult, &minVal, &maxVal, &minLoc, &maxLoc);
			//std::cout << minLoc << " " << minVal << std::endl;


			//GPUMAT version test
			//cv::cuda::GpuMat corrResult_gpu(corrMaxSearchWidth, corrMaxSearchHeight, CV_32F, d_corrOutput);
			//cv::minMaxLoc(corrResult_gpu, &minVal, &maxVal, &minLoc, &maxLoc); --> not implemented in opencv


			///1x1 correlation -->>TODO: CREATE A SEPARATE FLOAT ARRAY IN CPU FOR THIS
			//float kernel = 1;
			//GetValue(d_icorr032f, j*stride + i, kernel);
			//CorrelationSearchSampling(i, j, d_corrSearchSpace);
			//Correlation1x1(kernel, d_corrSearchSpace, d_corrOutput);


			//total++;
			/*cv::Mat kernel = cv::Mat(cv::Size(corrKernelSize, corrKernelSize), CV_32F);
			checkCudaErrors(cudaMemcpy(kernel.ptr(), d_corrKernel, corrKernelSize * corrKernelSize * sizeof(float), cudaMemcpyDeviceToHost));
			cv::Mat kernelUp;
			cv::resize(kernel, kernelUp, cv::Size(corrKernelSize, corrKernelSize));
			cv::imshow("kernel", kernelUp);

			cv::Mat ss = cv::Mat(cv::Size(corrMaxSearchWidth, corrMaxSearchHeight), CV_32F);
			checkCudaErrors(cudaMemcpy(ss.ptr(), d_corrSearchSpace, corrMaxSearchWidth * corrMaxSearchHeight * sizeof(float), cudaMemcpyDeviceToHost));
			cv::imshow("ss", ss);

			cv::Mat output = cv::Mat(cv::Size(corrMaxSearchWidth, corrMaxSearchHeight), CV_32F);
			checkCudaErrors(cudaMemcpy(output.ptr(), d_corrOutput, corrMaxSearchWidth * corrMaxSearchHeight * sizeof(float), cudaMemcpyDeviceToHost));
			cv::imshow("output", output);

			cv::waitKey();*/
		}
	}
	//std::cout << total << std::endl;
	cv::Mat output = cv::Mat(cv::Size(corrMaxSearchWidth, corrMaxSearchHeight), CV_32F);
	checkCudaErrors(cudaMemcpy(output.ptr(), d_corrOutput, corrMaxSearchWidth * corrMaxSearchHeight * sizeof(float), cudaMemcpyDeviceToHost));
	//cv::imshow("output", output);
	//for each 5x5 kernel in im0
	//create kernel
	//create search space in im1 (texture sampling)
	//perform correlation
	//get max value and save the coordinate to ld(u,v)


	return 0;
}

int CudaFlow::solveOpticalFlowLdof() {
	return this->_solveOpticalFlowLdof();
}

int CudaFlow::_solveOpticalFlowLdof() {
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
	return 0;
}

int CudaFlow::copyCorrPatchMatchToHost(cv::Mat &u, cv::Mat &v, cv::Mat &uvrgb) {
	return 0;
}