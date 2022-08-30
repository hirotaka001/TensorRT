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

#define TAG "TRT_CPN"

int main(int argc, char** argv)
{
	std::string model_name = "cpn";
	std::string root_dir = "../../data";
	std::string model_dir = root_dir + "/" + model_name + "_models";
	std::string dataset_dir = root_dir + "/" + model_name + "_inputs";
	std::string output_path = root_dir + "/" + model_name + "_outputs";

	int resize_margin = 10;
	bool keep_aspect = true;
	bool print_keypoint_score = true;

	const cv::String keys =
		"{help ? usage    |     | print this message             }"
		"{gpu GPUID       | 0   | GPU ID                         }"
		"{b Batches       | 1   | number of batches              }"
		"{sb SuperBatches | 1   | number of super batches        }"
		"{t Threads       | 1   | number of threads              }"
		"{m Model         | cpn | model filename                 }"
		"{w Width         | 192 | image width                    }"
		"{h Height        | 256 | image height                   }"
		"{d DrawFlag      | 1   | draw flag (0 or otherwise)     }"
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
	bool draw_flag = (bool)parser.get<int>("DrawFlag");

	if (!parser.check()) {
		parser.printErrors();
		return 1;
	}

	std::cout << "[" << TAG << "] Parameters" << std::endl;
	std::cout << "GPU ID          = " << gpu_id << std::endl;
	std::cout << "Batches         = " << batches << std::endl;
	std::cout << "SuperBatches    = " << super_batches << std::endl;
	std::cout << "Threads         = " << threads << std::endl;
	std::cout << "ModelName       = " << model_filename << std::endl;
	std::cout << "Width           = " << width << std::endl;
	std::cout << "Height          = " << height << std::endl;

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
	kros::TRT_CPN cTRT_CPN;
	if (!cTRT_CPN.Init(gpu_id, model_filename, model_dir, width, height, batches, super_batches, threads, resize_margin, keep_aspect)) {
		exit(EXIT_FAILURE);
	}

	std::cout << "[" << TAG << "] Inferring ... ";
	std::vector<std::vector<std::vector<float> > > detection_result;
	cTRT_CPN.InferParallel(loaded_input_images, detection_result);
	std::cout << "Completed" << std::endl;

	if (draw_flag)
	{
		std::cout << "[" << TAG << "] Drawing ... ";
		int num_images = (int)loaded_input_images.size();

		if (!std::filesystem::exists(output_path)) {
			std::filesystem::create_directory(output_path);
		}


		for (int Image_idx = 0; Image_idx < num_images; Image_idx++) {
			cv::Mat OutImage;
			cTRT_CPN.DrawResult(loaded_input_images[Image_idx], OutImage, detection_result[Image_idx], print_keypoint_score);
			cv::imwrite(output_path + "/" + input_file_names[Image_idx], OutImage);
		}
		std::cout << "Completed" << std::endl;
	}

	std::cout << "[" << TAG << "] All completed" << std::endl;
}
