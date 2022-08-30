#ifndef __PREPROCESS_CUDA_H__
#define __PREPROCESS_CUDA_H__

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <opencv2/core/core.hpp>

#ifdef _WIN32
#pragma warning(pop)
#endif

__constant__ float g_const_mean[3];
__constant__ float g_const_std[3];

void SetConstantValue(const float* const_mean, const float* const_std);
void PreProcess_cuda(const cv::Mat& input_images, void* Preprocess_cuda_in_buffer, float* GPU_In_buffer, int out_width, int out_height, cudaStream_t stream);

#endif
