// To work, the project needs:
// Right click Project -> Build Dependencies -> Build Customizations  -> Cuda8.0 (or higher versions) (so that cudart is found)
// Link to cudart.lib (Linker -> Input -> Additional Dependencies -> cudart.lib

#pragma once

#include "lib_link.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <reconflow/Reconflow.h>
#include <opencv2/optflow.hpp>

int getRtStereo(std::string filename, cv::Mat &R, cv::Mat &t, cv::Mat &K);

int main(int argc, char **argv)
{
	std::string mainfolder = "";
	std::string im1filename = "data/im319.png";
	std::string im2filename = "data/im318.png";
	std::string flownetfilename = "data/flo319.flo";
	std::string cameramatrix = mainfolder + "data/calib.txt";
	std::string outputfilename = mainfolder + "data/output/output3d";
	ReconFlow *flow = new ReconFlow(32, 12, 32);

	// Load Params
	float lambda, tau, alphaTv, alphaFn, alphaProj, lambdaf, lambdams, scale;
	int nWarpIters, iters, minWidth;
	std::string suffixFor3D = "oursfn";
	lambda = 50.0f;
	tau = 0.125f;
	alphaTv = 33.3f;
	alphaFn = 100.0f; //1
	alphaProj = 60.0f;
	lambdaf = 0.1f;//0.1
	lambdams = 100.0f;//100
	nWarpIters = 1;
	iters = 10000;
	scale = 2.0f;
	minWidth = 400;

	cv::Mat R, t, K;
	int isSetPose = getRtStereo(cameramatrix, R, t, K);

	// Check image size and compute pyramid nlevels
	std::string initialImage = mainfolder + im1filename;
	cv::Mat iset = cv::imread(initialImage.c_str());
	int width = iset.cols;
	int height = iset.rows;
	int nLevels = 1;
	int pHeight = (int)((float)height / scale);
	while (pHeight > minWidth) {
		nLevels++;
		pHeight = (int)((float)pHeight / scale);
	}
	std::cout << "Pyramid Levels: " << nLevels << std::endl;
	int stride = flow->iAlignUp(width);
	cv::Mat isetpad;
	cv::copyMakeBorder(iset, isetpad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	// Initialize handler matrices for display and output
	cv::Mat uvrgb = cv::Mat(isetpad.size(), CV_32FC3);
	cv::Mat u = cv::Mat(isetpad.size(), CV_32F);
	cv::Mat v = cv::Mat(isetpad.size(), CV_32F);
	cv::Mat X = cv::Mat(isetpad.size(), CV_32F);
	cv::Mat Y = cv::Mat(isetpad.size(), CV_32F);
	cv::Mat Z = cv::Mat(isetpad.size(), CV_32F);

	// Initialize ReconFlow
	flow->initializeR(width, height, 3, nLevels, scale, ReconFlow::METHODR_TVL1_MS_FNSPARSE,
		lambda, 0.0f, lambdaf, lambdams,
		alphaTv, alphaProj, alphaFn,
		tau, nWarpIters, iters);
	flow->setCameraMatrices(K, K);

	
	// Open input images
	cv::Mat i0rgb, i1rgb, flownet, i1rgbpad, i0rgbpad, flownetpad, RR, tt;
	std::string flownet2filename = mainfolder + flownetfilename;
	std::string image1filename = mainfolder + im1filename;
	std::string image2filename = mainfolder + im2filename;
	i0rgb = cv::imread(image1filename);
	i1rgb = cv::imread(image2filename);

	// Open initial matching (flownet)
	flownet = cv::optflow::readOpticalFlow(flownet2filename);
	if (flownet.empty()) {
		std::cerr << "Flownet file not found." << std::endl;
		return 0;
	}

	// Resize images by padding
	cv::copyMakeBorder(i0rgb, i0rgbpad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(i1rgb, i1rgbpad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(flownet, flownetpad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::Mat flownet2[2];   //split flownet channels
	cv::split(flownetpad, flownet2);

	// Copy data to GPU
	flow->copyImagesToDevice(i0rgbpad, i1rgbpad);
	flow->copySparseOpticalFlowToDevice(flownet2[0], flownet2[1]); //can set a mask as third argument

	// Calculate ReconFlow
	flow->solveR(R, t, 50.0f); //main computation iteration
	flow->copyOpticalFlowToHost(u, v, uvrgb); //uvrgb is an optical flow image
	flow->copy3dToHost(X, Y, Z); //3D points

	// Save output 3D as ply file
	std::vector<cv::Vec3b> colorbuffer(stride*height);
	cv::Mat colorMat = cv::Mat(static_cast<int>(colorbuffer.size()), 1, CV_8UC3, &colorbuffer[0]);
	std::vector<cv::Vec3f> buffer(stride*height);
	cv::Mat cloudMat = cv::Mat(static_cast<int>(buffer.size()), 1, CV_32FC3, &buffer[0]);
	colorbuffer.clear();
	buffer.clear();

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < stride; i++) {
			cv::Vec3b rgb = i0rgbpad.at<cv::Vec3b>(j, i);
			colorbuffer.push_back(rgb);

			float x = X.at<float>(j, i);
			float y = Y.at<float>(j, i);
			float z = Z.at<float>(j, i);
			if (((z <= 75) && (z > 10)) && ((x <= 75) && (y > -75)) && ((y <= 75) && (y > -75))) {
				buffer.push_back(cv::Vec3f(x, -y, -z));
			}
			else {
				x = std::numeric_limits<float>::quiet_NaN();
				y = std::numeric_limits<float>::quiet_NaN();
				z = std::numeric_limits<float>::quiet_NaN();
				buffer.push_back(cv::Vec3f(x, y, z));
			}
		}
	}
	cv::viz::WCloud cloud(cloudMat, colorMat);

	std::ostringstream output3d;
	output3d << outputfilename << suffixFor3D << ".ply";
	cv::viz::writeCloud(output3d.str(), cloudMat, colorMat);
	return 0;
}

int getRtStereo(std::string filename, cv::Mat &R, cv::Mat &t, cv::Mat &K) {
	double Re[9] = { 1,0,0,0,1,0,0,0,1 };
	double te[3] = { -1, 0.00001, 0.00001 };
	R = cv::Mat(3, 3, CV_64F, Re).clone();
	t = cv::Mat(3, 1, CV_64F, te).clone();
	//read from file
	std::fstream myfile(filename, std::ios_base::in);
	float a, b, c, d, e, f, g, h, i, j, k, l;
	char ch[3];
	myfile >> ch >> a >> b >> c >> d >> e >> f >> g >> h >> i >> j >> k >> l;
	double cammat[9] = { a, b, c, e, f, g, i, j, k };
	K = cv::Mat(3, 3, CV_64F, cammat).clone();
	std::cout << "Camera pose found for: " << filename << std::endl;
	std::cout << "K0: " << a << " " << b << " " << c << " " << e << " " << f << " " << g << " " << i << " " << j << " " << k << std::endl;
	return 0;
}