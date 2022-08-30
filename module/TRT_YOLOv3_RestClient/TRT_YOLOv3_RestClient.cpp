#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <kros/trt.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

#include "JsonPoster.h"
#include "BodyAndHeadDetector.h"

#ifdef _WIN32
#include "../LinkOpenCV.h"
#include "../LinkTensorRT.h"
#endif

int main(int argc, char** argv)
{
	const cv::String keys =
		"{help ? usage  |   | print this messag  }"
		"{c CameraID    | 0 | camera id          }"
		"{s SaveFlag    | 0 | save image flag (0 or otherwise) }"
		;

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("aaa");
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	bool camera_id = (bool)parser.get<int>("CameraID");
	bool save_flag = (bool)parser.get<int>("SaveFlag");

	if (!parser.check()) {
		parser.printErrors();
		return 1;
	}

	std::string root_dir = "../../data";
	std::string model_dir = root_dir + "/yolov3_rest_client_models";
	kros::BodyAndHeadDetector body_head_detector;
	body_head_detector.Init(model_dir);

	std::vector<std::string> label_list;
	label_list.push_back(std::string("body"));
	label_list.push_back(std::string("head"));

	JsonPoster json_poster("http://127.0.0.1:5000/");

	cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	if (!cap.isOpened()) return -1;

	while (1) {
		cv::Mat frame;
		cap >> frame;

		std::vector<cv::Mat> input_images;
		std::vector<std::vector<float> > detection_result;

		body_head_detector.Infer(frame, detection_result);
		kros::TRT_YOLOv3::DrawResult(frame, detection_result, label_list);

		json_poster.PostDetectedResults(detection_result);

		cv::imshow("Capture", frame);
		int key = cv::waitKey(1);
		if (key == 27) {// Esc
			break;
		}
	}

	return 0;
}
