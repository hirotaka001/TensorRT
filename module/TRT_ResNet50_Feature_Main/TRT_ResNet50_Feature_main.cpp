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

#define TAG "TRT_ResNet50_Feature"

int main(int argc, char** argv)
{
	std::string root_dir = "../../data";
	std::string model_name = "resnet50_feature";
	std::string model_dir = root_dir + "/" + model_name + "_models";
	std::string dataset_dir = root_dir + "/" + model_name + "_inputs";

	const cv::String keys =
		"{help ? usage           |     | print this message             }"
		"{gpu GPUID              | 0   | GPU ID                         }"
		"{b Batches              | 1   | number of batches              }"
		"{sb SuperBatches        | 1   | number of super batches        }"
		"{t Threads              | 1   | number of threads              }"
		"{m Model                | resnet50_feature | model filename    }"
		"{w Width                | 128 | image width                    }"
		"{h Height               | 256 | image height                   }"
		"{ppc PreProcessCuda     | 1   | pre_process cuda flag (0 or otherwise)}"
		"{p PrintFlag            | 1   | print flag (0 or otherwise)    }"
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
	bool is_pre_process_cuda = (bool)parser.get<int>("PreProcessCuda");
	bool print_flag = (bool)parser.get<int>("PrintFlag");

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
	std::cout << "PreProcessCuda      = " << is_pre_process_cuda << std::endl;

	std::cout << "[" << TAG << "] Loding input images ... ";
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
	kros::TRT_ResNet50_Feature cTRT_ResNet50_Feature;
	if (!cTRT_ResNet50_Feature.Init(gpu_id, model_filename, model_dir, width, height, batches, super_batches, threads, is_pre_process_cuda)) {
		exit(EXIT_FAILURE);
	}

	std::cout << "[" << TAG << "] Inferring ... ";
	std::vector<std::vector<float> > feature_vecs;
	cTRT_ResNet50_Feature.InferParallel(loaded_input_images, feature_vecs);
	std::cout << "Completed" << std::endl;

	if (print_flag) {
		std::cout << "[" << TAG << "] Printing ... " << std::endl;
		int num_images = (int)loaded_input_images.size();
		for (int Image_idx = 0; Image_idx < num_images; Image_idx++)
		{
			std::cout << input_file_names[Image_idx] << " : ";
			kros::TRT_ResNet50_Feature::PrintResult(feature_vecs[Image_idx]);
		}
	}

	std::cout << "[" << TAG << "] All completed" << std::endl;
}
