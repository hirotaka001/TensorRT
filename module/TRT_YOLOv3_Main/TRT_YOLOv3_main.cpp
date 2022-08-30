#include <kros/trt.h>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef _WIN32
#include "../LinkOpenCV.h"
#include "../LinkTensorRT.h"
#if __has_include( "../LinkTRT.h")
#include "../LinkTRT.h"
#endif
#endif

#define TAG "TRT_YOLOv3"

int main(int argc, char** argv)
{
	std::string root_dir = "../../data";
	std::string model_name = "yolov3";
	std::string model_dir = root_dir + "/" + model_name + "_models";
	std::string dataset_dir = root_dir + "/" + model_name + "_inputs";
	std::string output_path = root_dir + "/" + model_name + "_outputs";
	std::string label_list_path = model_dir + "/coco.names";// only for drawing

	const cv::String keys =
		"{help ? usage           |     | print this message             }"
		"{gpu GPUID              | 0   | GPU ID                         }"
		"{b Batches              | 1   | number of batches              }"
		"{sb SuperBatches        | 1   | number of super batches        }"
		"{t Threads              | 1   | number of threads              }"
		"{m Model                | yolov3 | model filename              }"
		"{w Width                | 512 | image width                    }"
		"{h Height               | 512 | image height                   }"
		"{c Classes              | 80  | number of classes              }"
		"{ppc PreProcessCuda     | 1   | pre_process cuda flag (0 or otherwise)}"
		"{sc Score               | 0.5 | score threshold                }"
		"{n NMS                  | 0.5 | nms threshold                  }"
		"{d DrawFlag             | 1   | draw flag (0 or otherwise)     }"
		;

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about(TAG);
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	int gpu_id = parser.get<int>("GPUID");
	int batches = parser.get<int>("Batches");
	int super_batches = parser.get<int>("SuperBatches");
	int threads = parser.get<int>("Threads");
	std::string model_filename = parser.get<std::string>("Model");
	int width = parser.get<int>("Width");
	int height = parser.get<int>("Height");
	int nClasses = parser.get<int>("Classes");
	bool is_pre_process_cuda = (bool)parser.get<int>("PreProcessCuda");
	float score_threshold = parser.get<float>("Score");
	float nms_threshold = parser.get<float>("NMS");
	bool draw_flag = (bool)parser.get<int>("DrawFlag");

	if (!parser.check()) {
		parser.printErrors();
		return 1;
	}

	std::cout << "[" << TAG << "] Parameters" << std::endl;
	std::cout << "GPU ID              = " << gpu_id << std::endl;
	std::cout << "Batches             = " << batches << std::endl;
	std::cout << "SuperBatches        = " << super_batches << std::endl;
	std::cout << "Threads             = " << threads << std::endl;
	std::cout << "ModelName           = " << model_filename << std::endl;
	std::cout << "Width               = " << width << std::endl;
	std::cout << "Height              = " << height << std::endl;
	std::cout << "Classes             = " << nClasses << std::endl;
	std::cout << "PreProcessCuda      = " << is_pre_process_cuda << std::endl;
	std::cout << "Score Threshold     = " << score_threshold << std::endl;
	std::cout << "NMS Threshold       = " << nms_threshold << std::endl;

	std::cout << "[" << TAG << "] Loading input images ... ";
	std::vector<std::string> input_file_names;
	std::filesystem::directory_iterator end;
	for (std::filesystem::directory_iterator it(dataset_dir + "/"); it != end; ++it)
	{
		if (!std::filesystem::is_directory(*it)) {
			std::filesystem::path ext = it->path().extension();
			if (ext == ".bmp" || ext == ".png" || ext == ".jpg") {
				input_file_names.push_back(it->path().filename().string());
			}
		}
	}

	std::vector<cv::Mat> loaded_input_images;
	for (int i = 0; i < (int)input_file_names.size(); i++)
	{
		cv::Mat img = cv::imread(dataset_dir + "/" + input_file_names[i], cv::IMREAD_COLOR);
		CV_Assert(!img.empty());
		loaded_input_images.push_back(img);
	}
	std::cout << (int)loaded_input_images.size() << " images found" << std::endl;

	std::cout << "[" << TAG << "] Initializing ... " << std::endl;
	std::string input_layers = "000_net";
	std::vector<std::string> output_layers{ "082_convolutional", "094_convolutional", "106_convolutional" };
	kros::TRT_YOLOv3 cTRT_YOLOv3;
	if (!cTRT_YOLOv3.Init(gpu_id, model_filename, model_dir, nClasses, width, height, batches, super_batches, threads, is_pre_process_cuda,
		score_threshold, nms_threshold, input_layers, output_layers)) {
		exit(EXIT_FAILURE);
	}

	std::cout << "[" << TAG << "] Inferring ... ";
	std::vector<std::vector<std::vector<float> > > detection_result;
	cTRT_YOLOv3.InferParallel(loaded_input_images, detection_result);
	std::cout << "Completed" << std::endl;

	if (draw_flag)
	{
		std::cout << "[" << TAG << "] Drawing ... ";
		std::vector<std::string> label_list(0);
		std::ifstream ifs(label_list_path);
		std::string str;
		if (!ifs.fail())
		{
			while (getline(ifs, str)) {
				label_list.push_back(str);
			}
		}

		if (!std::filesystem::exists(output_path)) {
			std::filesystem::create_directory(output_path);
		}

		int num_images = (int)loaded_input_images.size();
		for (int Image_idx = 0; Image_idx < num_images; Image_idx++)
		{
			cv::Mat tmp;
			loaded_input_images[Image_idx].copyTo(tmp);
			kros::TRT_YOLOv3::DrawResult(tmp, detection_result[Image_idx], label_list);
			cv::imwrite(output_path + "/" + input_file_names[Image_idx], tmp);
		}
		std::cout << "Completed" << std::endl;
	}

	std::cout << "[" << TAG << "] All completed" << std::endl;
}
