#include <kros/trt.h>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <omp.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

#include "../utilities.h"

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
	std::string label_list_path = model_dir + "/coco.names";// only for drawing

	std::string output_path = root_dir + "/" + model_name + "_outputs";
	std::string benchmark_log = output_path + "/" + model_name + "_benchmark.csv";
	std::string gpu_memory_log = output_path + "/" + model_name + "_gpu_memory_usage_tmp.log";
	std::string gpu_checker_sh = "../../linux/utilities/gpu_memory_checker.sh";

	const cv::String keys =
		"{help ? usage           |     | print this message             }"
		"{gpu GPUID              | 0   | GPU ID                         }"
		"{b Batches              | 1   | number of batches              }"
		"{sb SuperBatches        | 1   | number of super batches        }"
		"{t Threads              | 1   | number of threads              }"
		"{l Loops                | 10  | number of loops                }"
		"{m Model                | yolov3 | model filename              }"
		"{w Width                | 512 | image width                    }"
		"{h Height               | 512 | image height                   }"
		"{c Classes              | 80  | number of classes              }"
		"{ppc PreProcessCuda     | 1   | pre_process cuda flag (0 or otherwise)}"
		"{sc Score               | 0.5 | score threshold                }"
		"{n NMS                  | 0.5 | nms threshold                  }"
		"{d DrawFlag             | 1   | draw flag (0 or otherwise)     }"
		"{o OutputLog            | 0   | Output Log (0 or otherwise)    }"
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
	int loops = parser.get<int>("Loops");
	std::string model_filename = parser.get<std::string>("Model");
	int width = parser.get<int>("Width");
	int height = parser.get<int>("Height");
	int classes = parser.get<int>("Classes");
	bool is_pre_process_cuda = (bool)parser.get<int>("PreProcessCuda");
	float score_threshold = parser.get<float>("Score");
	float nms_threshold = parser.get<float>("NMS");
	bool draw_flag = (bool)parser.get<int>("DrawFlag");
	bool output_log = (bool)parser.get<int>("OutputLog");

	if (!parser.check()) {
		parser.printErrors();
		return 1;
	}

	std::cout << "[" << TAG << "] Parameters" << std::endl;
	std::cout << "GPU ID              = " << gpu_id << std::endl;
	std::cout << "Batches             = " << batches << std::endl;
	std::cout << "SuperBatches        = " << super_batches << std::endl;
	std::cout << "Threads             = " << threads << std::endl;
	std::cout << "Loops               = " << loops << std::endl;
	std::cout << "ModelName           = " << model_filename << std::endl;
	std::cout << "Width               = " << width << std::endl;
	std::cout << "Height              = " << height << std::endl;
	std::cout << "Classes             = " << classes << std::endl;
	std::cout << "PreProcessCuda      = " << is_pre_process_cuda << std::endl;
	std::cout << "Score Threshold     = " << score_threshold << std::endl;
	std::cout << "NMS Threshold       = " << nms_threshold << std::endl;

#ifdef _OPENMP
	std::cout << "[" << TAG << "] Number of processors : " << omp_get_num_procs() << std::endl;
#endif

	std::cout << "[" << TAG << "] Loding input images ..." << std::endl;
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
	for (int i = 0; i < (int)input_file_names.size(); i++) {
		cv::Mat img = cv::imread(dataset_dir + "/" + input_file_names[i], cv::IMREAD_COLOR);
		CV_Assert(!img.empty());
		loaded_input_images.push_back(img);
	}

	std::vector < std::vector<std::vector<cv::Mat> > > input_images;
	int input_image_count = 0;
	for (int loop_idx = 0; loop_idx < loops; loop_idx++) {
		std::vector<std::vector<cv::Mat> > input_images_per_loop;

		// need to load (nBatchs * nSuperBatchs * nThreads) images
		for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
			std::vector<cv::Mat> input_images_per_thread;
			for (int i = 0; i < batches * super_batches; i++) {
				input_images_per_thread.push_back(loaded_input_images[input_image_count % (int)input_file_names.size()]);
				input_image_count++;
			}
			input_images_per_loop.push_back(input_images_per_thread);
		}
		input_images.push_back(input_images_per_loop);
	}

	const int frame_num_per_inferring = batches * super_batches * threads;
	const int frame_total_num = batches * super_batches * threads * loops;
	std::cout << "[" << TAG << "] Number of inputs per inferring (delay) : " << frame_num_per_inferring << std::endl;
	std::cout << "[" << TAG << "] Number of total inputs                 : " << frame_total_num << std::endl;

	std::cout << "[" << TAG << "] Initializing ..." << std::endl;
	const int pre_gpu_memory_usage = check_gpu_memory_usage(gpu_memory_log, gpu_checker_sh);

	// need to set the name of the input and output layers included in the onnx model
	// You can check them by using Netron (https://github.com/lutzroeder/netron)
	std::string input_layers = "000_net";
	std::vector<std::string> output_layers{ "082_convolutional", "094_convolutional", "106_convolutional" };
	kros::TRT_YOLOv3 cTRT_YOLOv3;

	if (!cTRT_YOLOv3.Init(gpu_id, model_filename, model_dir, classes, width, height, batches, super_batches, threads, is_pre_process_cuda,
		score_threshold, nms_threshold, input_layers, output_layers)) {
		exit(EXIT_FAILURE);
	}

	std::cout << "[" << TAG << "] Inferring ..." << std::endl;
	std::vector < std::vector< std::vector<std::vector<std::vector<float> > > > > detection_result_;
	for (int loop_idx = 0; loop_idx < loops; loop_idx++) {
		std::vector< std::vector<std::vector<std::vector<float> > > > detection_result(threads);
		detection_result_.push_back(detection_result);
	}

	// ignore the inference time at the first time
	for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
		int thread_id = (int)omp_get_thread_num();
		cTRT_YOLOv3.Infer(input_images[0][thread_id], detection_result_[0][thread_id], thread_id);
		detection_result_[0][thread_id].clear();
	}

	const auto st_time = cv::getTickCount();
	for (int loop_idx = 0; loop_idx < loops; loop_idx++) {

		// infer (nBatchs * nSuperBatchs * nThreads) images
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
			int thread_id = (int)omp_get_thread_num();
			cTRT_YOLOv3.Infer(input_images[loop_idx][thread_id], detection_result_[loop_idx][thread_id], thread_id);
		}

	}
	const auto ed_time = cv::getTickCount();
	std::cout << "[" << TAG << "] Completed" << std::endl;

	const auto elapsedTime = (ed_time - st_time) * 1000.0 / cv::getTickFrequency();
	const float average_time = (float)elapsedTime / frame_total_num;
	std::cout << "[" << TAG << "] Average Time     : " << average_time << " [ms/frame]" << std::endl;

	int gpu_memory_usage = -1;
	const int post_gpu_memory_usage = check_gpu_memory_usage(gpu_memory_log, gpu_checker_sh);
	if (post_gpu_memory_usage > 0) {
		gpu_memory_usage = post_gpu_memory_usage - pre_gpu_memory_usage;
		std::cout << "[" << TAG << "] Gpu Memory Usage : " << gpu_memory_usage << " [MB]" << std::endl;
	}

	if (output_log) {
		if (!std::filesystem::exists(output_path)) {
			std::filesystem::create_directory(output_path);
		}

		std::cout << "[" << TAG << "] Logging ..." << std::endl;
		std::ofstream ofs(benchmark_log, std::ios::app);
		ofs << frame_num_per_inferring << "," << batches << "," << super_batches << "," << threads << "," << width << "," << height << "," << average_time << "," << gpu_memory_usage << std::endl;
	}

	if (draw_flag) {
		std::cout << "[" << TAG << "] Drawing ..." << std::endl;
		std::vector<std::string> label_list(0);
		std::ifstream ifs(label_list_path);
		std::string str;
		if (!ifs.fail())
		{
			while (getline(ifs, str))
			{
				label_list.push_back(str);
			}
			std::cout << "[" << TAG << "] Found " << (int)label_list.size() << " labels" << std::endl;
		}

		if (!std::filesystem::exists(output_path)) {
			std::filesystem::create_directory(output_path);
		}

		int output_image_count = 0;
		for (int loop_idx = 0; loop_idx < loops; loop_idx++) {
			for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
				std::vector<cv::Mat> input_images_per_thread;
				for (int i = 0; i < batches * super_batches; i++) {
					cv::Mat tmp;
					input_images[loop_idx][thread_idx][i].copyTo(tmp);
					kros::TRT_YOLOv3::DrawResult(tmp, detection_result_[loop_idx][thread_idx][i], label_list);
					cv::imwrite(output_path + "/" + input_file_names[output_image_count], tmp);

					output_image_count++;
					if (output_image_count == (int)input_file_names.size()) {
						break;
					}
				}
				if (output_image_count == (int)input_file_names.size()) {
					break;
				}
			}
			if (output_image_count == (int)input_file_names.size()) {
				break;
			}
		}
	}

	std::cout << "[" << TAG << "] All completed" << std::endl;
}
