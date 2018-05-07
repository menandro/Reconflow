#include "CameraPose.h"

CameraPose::CameraPose() {

}

CameraPose::~CameraPose() {

}

int CameraPose::initialize(cv::Mat intrinsic, int minHessian) {
	this->intrinsic = intrinsic.clone();
	this->K = this->intrinsic;
	this->minHessian = minHessian;
	surf = cv::cuda::SURF_CUDA(this->minHessian);
	surf.hessianThreshold = minHessian;
	matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
	return 0;
}

int CameraPose::initialize(cv::Mat intrinsic) {
	minHessian = 2000;
	this->initialize(intrinsic, minHessian);
	return 0;
}

int CameraPose::initialize(double focal, cv::Point2d pp) {
	minHessian = 2000;
	this->initialize(focal, pp, minHessian);
	return 0;
}

int CameraPose::initialize(double focal, cv::Point2d pp, int minHessian) {
	double k[9] = { focal, 0.0, (double)pp.x, 0.0, focal, (double)pp.y, 0.0, 0.0, 1.0 };
	intrinsic = cv::Mat(3, 3, CV_64F, k);
	this->initialize(intrinsic, minHessian);
	return 0;
}

int CameraPose::initialize(double focalx, double focaly, cv::Point2d pp, int minHessian) {
	double k[9] = { focalx, 0.0, (double)pp.x, 0.0, focaly, (double)pp.y, 0.0, 0.0, 1.0 };
	intrinsic = cv::Mat(3, 3, CV_64F, k);
	this->initialize(intrinsic, minHessian);
	return 0;
}

int CameraPose::solvePose_8uc1(cv::Mat im1, cv::Mat im2, cv::Mat &R, cv::Mat &t) {
	this->im1 = im1;
	this->im2 = im2;
	d_im1.upload(im1); //first cuda call is always slow
	d_im2.upload(im2);

	surf(d_im1, cv::cuda::GpuMat(), d_keypoints_im1, d_descriptors_im1);
	surf(d_im2, cv::cuda::GpuMat(), d_keypoints_im2, d_descriptors_im2);

	if ((d_keypoints_im1.empty() || d_keypoints_im2.empty()) || (d_descriptors_im1.cols <= 1) || (d_descriptors_im2.cols <= 1)) {
		std::cout << "No keypoints found." << std::endl;
		return 1;
	}
	else {
		matcher->knnMatch(d_descriptors_im1, d_descriptors_im2, matches, 2);
		surf.downloadKeypoints(d_keypoints_im1, keypoints_im1);
		surf.downloadKeypoints(d_keypoints_im2, keypoints_im2);
		
		if (!matches.empty()) {
			matched_keypoints_im1 = std::vector< cv::KeyPoint >();
			matched_keypoints_im2 = std::vector< cv::KeyPoint >();
			matchedpoints_im1 = std::vector<cv::Point2f>();
			matchedpoints_im2 = std::vector<cv::Point2f>();
			matched_distance = std::vector< float >();
			for (int k = 0; k < (int)matches.size(); k++)
			{
				if ((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int)matches[k].size() <= 2 && (int)matches[k].size() > 0))
				{
					matched_keypoints_im1.push_back(keypoints_im1[matches[k][0].queryIdx]);
					matched_keypoints_im2.push_back(keypoints_im2[matches[k][0].trainIdx]);
					matchedpoints_im1.push_back(keypoints_im1[matches[k][0].queryIdx].pt);
					matchedpoints_im2.push_back(keypoints_im2[matches[k][0].trainIdx].pt);
					matched_distance.push_back(matches[k][0].distance);
				}
			}

			if (!matchedpoints_im1.empty() && (matchedpoints_im1.size() >= 5)) {
				//E = cv::findEssentialMat(cv::Mat(matchedpoints_im1), cv::Mat(matchedpoints_im2), K, CV_RANSAC, 0.999, 0.5, keypoint_mask);
				E = cv::findEssentialMat(cv::Mat(matchedpoints_im1), cv::Mat(matchedpoints_im2), K, CV_LMEDS, 0.999, 0.5, keypoint_mask);
				filtered_keypoints_im1 = std::vector< cv::KeyPoint >();
				filtered_keypoints_im2 = std::vector< cv::KeyPoint >();
				filtered_matchedpoints_im1 = std::vector < cv::Point2f >();
				filtered_matchedpoints_im2 = std::vector < cv::Point2f >();
				for (int k = 0; k < keypoint_mask.rows; k++) {
					if (keypoint_mask.at<bool>(0, k) == 1) {
						filtered_keypoints_im1.push_back(matched_keypoints_im1[k]);
						filtered_keypoints_im2.push_back(matched_keypoints_im2[k]);
						filtered_matchedpoints_im1.push_back(matchedpoints_im1[k]);
						filtered_matchedpoints_im2.push_back(matchedpoints_im2[k]);
					}
				}
				if (E.rows != 3) {
					std::cout << "Invalid essential matrix." << std::endl;
					return 1;
				}
				else {
					//cv::recoverPose(E, cv::Mat(matchedpoints_im1), cv::Mat(matchedpoints_im2), R, t, focal, pp);
					cv::recoverPose(E, cv::Mat(filtered_matchedpoints_im1), cv::Mat(filtered_matchedpoints_im2), K, R, t);
				}
				return 0;
			}
			else {
				std::cout << "Too few matches found." << std::endl;
				return 1;
			}
		}
		else {
			std::cout << "No matches found." << std::endl;
			return 1;
		}
	}
}

int CameraPose::solvePose_8uc3(cv::Mat im1, cv::Mat im2, cv::Mat &R, cv::Mat &t) {
	cv::Mat im1gray;
	cv::Mat im2gray;
	cv::cvtColor(im1, im1gray, CV_BGR2GRAY);
	cv::cvtColor(im2, im2gray, CV_BGR2GRAY);
	if (!this->solvePose_8uc1(im1gray, im2gray, R, t))
		return 0;
	else return 1;
}
