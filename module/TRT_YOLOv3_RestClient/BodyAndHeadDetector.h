#pragma once

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{

	class BodyAndHeadDetector
	{
	private:
		std::unique_ptr<kros::TRT_YOLOv3> body_detector = nullptr;
		std::unique_ptr<kros::TRT_YOLOv3> head_detector = nullptr;

	public:
		BodyAndHeadDetector() {};
		~BodyAndHeadDetector() {};

		bool Init(const std::string& model_dir)
		{
			std::string input_layers = "000_net";
			std::vector<std::string> output_layers{ "082_convolutional", "094_convolutional", "106_convolutional" };
			body_detector = std::unique_ptr <kros::TRT_YOLOv3>(new kros::TRT_YOLOv3());
			if (!body_detector->Init("yolov3_body", model_dir, 80, 512, 512, 1, 1, 1, 0.5, 0.5, input_layers, output_layers)) {
				std::cout << "yolov3_body model error" << std::endl;
				return false;
			}

			head_detector = std::unique_ptr <kros::TRT_YOLOv3>(new kros::TRT_YOLOv3());
			if (!head_detector->Init("yolov3_head", model_dir, 1, 512, 512, 1, 1, 1, 0.5, 0.5)) {
				std::cout << "yolov3_head model error" << std::endl;
				return false;
			}
			return false;
		}

		void Infer(
			const cv::Mat& input_image,
			std::vector<std::vector<float> >& detection_result)
		{
			std::vector<cv::Mat> input_images;
			input_images.push_back(input_image);

			std::vector<std::vector<std::vector<float> > > body_detection_results;
			body_detector->InferParallel(input_images, body_detection_results);
			std::vector<std::vector<float> > body_detection_result_all = body_detection_results[0];

			std::vector<std::vector<float> > body_detection_result;
			for (int idx = 0; idx < (int)body_detection_result_all.size(); idx++)
			{
				// check class id to be person
				if (body_detection_result_all[idx][4] == 0) {
					body_detection_result.push_back(body_detection_result_all[idx]);
				}
			}

			std::vector<std::vector<std::vector<float> > > head_detection_results;
			head_detector->InferParallel(input_images, head_detection_results);
			std::vector<std::vector<float> > head_detection_result = head_detection_results[0];

			for (int idx = 0; idx < (int)head_detection_result.size(); idx++)
			{
				// replace class id to 1 from 0
				head_detection_result[idx][4] = 1;
			}

			detection_result.clear();
			detection_result.insert(detection_result.end(), body_detection_result.begin(), body_detection_result.end());
			detection_result.insert(detection_result.end(), head_detection_result.begin(), head_detection_result.end());

			return;
		}
	};
}