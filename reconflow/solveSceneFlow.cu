#include "CudaFlow.h"

/// image to warp
texture<float, 2, cudaReadModeElementType> depthMap0;
texture<float, 2, cudaReadModeElementType> depthMap1;

__global__ void SolveSceneFlowKernel(float *u, float *v,
	float fx, float fy, float cx, float cy,
	int width, int height, int stride,
	float3 *sceneflow)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float x0 = ((float)ix + 0.5f) / (float)width;
	float y0 = ((float)iy +  0.5f) / (float)height;
	float x1 = ((float)ix + u[pos] + 0.5f) / (float)width;
	float y1 = ((float)iy + v[pos] + 0.5f) / (float)height;

	float d0 = (float)tex2D(depthMap0, x0, y0);
	float d1 = (float)tex2D(depthMap1, x1, y1);
	//z3D = image_depth.at<ushort>(Point2d(u, v)) / 1000.0f;
	//x3D = (u - cx_d) * z3D / fx_d;
	//y3D = (v - cy_d) * z3D / fy_d;
	float pt0z = d0;
	float pt0x = ((float)ix - cx) * pt0z / fx;
	float pt0y = ((float)iy - cy) * pt0z / fy;
	float pt1z = d1;
	float pt1x = ((float)ix - cx) * pt1z / fx;
	float pt1y = ((float)iy - cy) * pt1z / fy;

	float sfx = (pt0x - pt1x);
	float sfy = (pt0y - pt1y);
	float sfz = (pt0z - pt1z);
	if ((d0 < 500) || (d0 > 2500)) {
		sfx = 0;
		sfy = 0;
		sfz = 0;
	}
	
	sceneflow[pos] = make_float3(sfx, sfy, sfz); //in millimeters
}


__global__ void SolveSceneFlowAndPointCloudKernel(float *u, float *v,
	float fx, float fy, float cx, float cy,
	int width, int height, int stride,
	float3 *pcloud0, float3 *pcloud1,
	float3 *sceneflow)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float x0 = ((float)ix + 0.5f) / (float)width;
	float y0 = ((float)iy + 0.5f) / (float)height;
	float x1 = ((float)ix + u[pos] + 0.5f) / (float)width;
	float y1 = ((float)iy + v[pos] + 0.5f) / (float)height;

	float d0 = (float)tex2D(depthMap0, x0, y0);
	float d1 = (float)tex2D(depthMap1, x1, y1);
	//z3D = image_depth.at<ushort>(Point2d(u, v)) / 1000.0f;
	//x3D = (u - cx_d) * z3D / fx_d;
	//y3D = (v - cy_d) * z3D / fy_d;
	float pt0z = d0;
	float pt0x = ((float)ix - cx) * pt0z / fx;
	float pt0y = ((float)iy - cy) * pt0z / fy;
	float pt1z = d1;
	float pt1x = ((float)ix - cx) * pt1z / fx;
	float pt1y = ((float)iy - cy) * pt1z / fy;

	float sfx = (pt0x - pt1x);
	float sfy = (pt0y - pt1y);
	float sfz = (pt0z - pt1z);
	if ((d0 < 500) || (d0 > 2500)) {
		sfx = 0;
		sfy = 0;
		sfz = 0;
	}

	pcloud0[pos] = make_float3(pt0x, pt0y, pt0z);
	pcloud1[pos] = make_float3(pt1x, pt1y, pt1z);
	sceneflow[pos] = make_float3(sfx, sfy, sfz); //in millimeters
}

///////////////////////////////////////////////////////////////////////////////

void CudaFlow::SolveSceneFlow_(float *u, float *v, float* depth0, float* depth1,
	float fx, float fy, float cx, float cy,
	int w, int h, int s,
	float3 *sceneflow)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	depthMap0.addressMode[0] = cudaAddressModeMirror;
	depthMap0.addressMode[1] = cudaAddressModeMirror;
	depthMap0.filterMode = cudaFilterModeLinear;
	depthMap0.normalized = true;

	depthMap1.addressMode[0] = cudaAddressModeMirror;
	depthMap1.addressMode[1] = cudaAddressModeMirror;
	depthMap1.filterMode = cudaFilterModeLinear;
	depthMap1.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, depthMap0, depth0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, depthMap1, depth1, w, h, s * sizeof(float));

	SolveSceneFlowKernel << <blocks, threads >> >(u, v, fx, fy, cx, cy, w, h, s, sceneflow);
}

void CudaFlow::SolveSceneFlow_(float *u, float *v, float* depth0, float* depth1,
	float fx, float fy, float cx, float cy,
	int w, int h, int s,
	float3 *pcloud0, float3 *pcloud1,
	float3 *sceneflow) {

	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	depthMap0.addressMode[0] = cudaAddressModeMirror;
	depthMap0.addressMode[1] = cudaAddressModeMirror;
	depthMap0.filterMode = cudaFilterModeLinear;
	depthMap0.normalized = true;

	depthMap1.addressMode[0] = cudaAddressModeMirror;
	depthMap1.addressMode[1] = cudaAddressModeMirror;
	depthMap1.filterMode = cudaFilterModeLinear;
	depthMap1.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, depthMap0, depth0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, depthMap1, depth1, w, h, s * sizeof(float));

	SolveSceneFlowAndPointCloudKernel << <blocks, threads >> >(u, v, fx, fy, cx, cy, w, h, s, pcloud0, pcloud1, sceneflow);
}
