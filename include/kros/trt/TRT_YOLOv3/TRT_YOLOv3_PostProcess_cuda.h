#ifndef __YOLOV3_CUDA_H__
#define __YOLOV3_CUDA_H__

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <vector>
#include <opencv2/opencv.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	class TRT_YOLOv3_PostProcess_cuda
	{
	public:
		struct st_Boxes
		{
			float box_confidence;
			std::vector<float> box_class_probs;
			std::vector<float> box_scores;
			float boxes[4];
		};

		struct st_argumets
		{
			int class_num;
			int num_batch;
			int num_super_batch;
			int width;
			int height;
			float threshold;
			int MidUnitLen;
			int OutUnitLen;
		};

		TRT_YOLOv3_PostProcess_cuda();
		~TRT_YOLOv3_PostProcess_cuda();
		void Init(const st_argumets& argumets, const int* GridWidth, const int* GridHeight);

		void Exec(const std::vector<void*>& gpu_buffers,
			const std::vector<cv::Size>& original_image_sizes,
			const int super_batch_id,
			std::vector<std::vector<std::vector<float>>>& output_result);

	private:
		int m_class_num = -1;
		int m_num_batch = -1;
		int m_num_super_batch = -1;
		int m_GridWidth[3];
		int m_GridHeight[3];
		int m_GridSize[3];
		//		int m_org_width;
		//		int m_org_height;
		int m_width = -1;
		int m_height = -1;
		float m_threshold = 0.0f;
		int m_MidUnitLen = -1;
		int m_OutUnitLen = -1;
		int m_MidAnchorSize[3];
		int m_OutAnchorSize[3];

		//		float* m_gpu_Mid1BufferPtr;
		//		float* m_gpu_Mid2BufferPtr;
		int m_gpuMid1Size = 0;
		int m_gpuMid2Size = 0;

		std::vector<int> m_MidSize_list;
		std::vector<int> m_OutSize_list;

		std::vector<thrust::device_vector<float>> m_Mid1Buffer;
		std::vector<thrust::device_vector<float>> m_Mid2Buffer;
	};

} // namespace kros
#endif
