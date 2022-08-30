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

#define TAG "TRT_ResNet50"

int main(int argc, char** argv)
{
	std::string root_dir = "../../data";
	std::string model_name = "resnet50";
	std::string model_dir = root_dir + "/" + model_name + "_models";
	std::string dataset_dir = root_dir + "/" + model_name + "_inputs";

	std::string output_path = root_dir + "/" + model_name + "_outputs";
	std::string benchmark_log = output_path + "/" + model_name + "_benchmark.csv";
	std::string gpu_memory_log = output_path + "/" + model_name + "_gpu_memory_usage_tmp.log";
	std::string gpu_checker_sh = "../../linux/utilities/gpu_memory_checker.sh";

	std::string label_list_path = model_dir + "/labels.txt";// only for drawing

	const cv::String keys =
		"{help ? usage           |     | print this message             }"
		"{gpu GPUID              | 0   | GPU ID                         }"
		"{b Batches              | 10  | number of batches              }"
		"{sb SuperBatches        | 1   | number of super batches        }"
		"{t Threads              | 1   | number of threads              }"
		"{l Loops                | 10  | number of loops                }"
		"{m Model                | resnet50 | model filename            }"
		"{w Width                | 128 | image width                    }"
		"{h Height               | 256 | image height                   }"
		"{c Classes              | 3   | number of classes              }"
		"{ppc PreProcessCuda     | 1   | pre_process cuda flag (0 or otherwise)}"
		"{st ScoreTh             | 0.0 | Score Threshold                }"
		"{p PrintFlag            | 0   | print flag (0 or otherwise)    }"
		"{o OutputLog            | 0   | output Log (0 or otherwise)    }"
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
	float score_thresh = parser.get<float>("ScoreTh");
	bool print_flag = (bool)parser.get<int>("PrintFlag");
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
	std::cout << "nClasses            = " << classes << std::endl;
	std::cout << "PreProcessCuda      = " << is_pre_process_cuda << std::endl;
	std::cout << "ScoreThreshold      = " << score_thresh << std::endl;

#ifdef _OPENMP
	std::cout << "[" << TAG << "] Number of processors : " << omp_get_num_procs() << std::endl;
#endif

	std::cout << "[" << TAG << "] Loading input images ..." << std::endl;
	std::vector<std::string> input_file_names;
	std::filesystem::directory_iterator end;
	for (std::filesystem::directory_iterator it(dataset_dir + "/"); it != end; ++it)
	{
		if (!std::filesystem::is_directory(*it)) {
			input_file_names.push_back(it->path().filename().string());
			//std::cout << it->path().filename().string() << std::endl;
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
	kros::TRT_ResNet50 cTRT_ResNet50;
	if (!cTRT_ResNet50.Init(gpu_id, model_filename, model_dir, classes, width, height, batches, super_batches, threads, is_pre_process_cuda)) {
		exit(EXIT_FAILURE);
	}

	std::cout << "[" << TAG << "] Inferring ..." << std::endl;
	std::vector< std::vector<std::vector< std::pair<int, float> > > > output_classIDs_;
	std::vector< std::vector< std::vector<std::vector<float> > > > output_score_;

	for (int loop_idx = 0; loop_idx < loops; loop_idx++) {
		std::vector<std::vector< std::pair<int, float> > > output_classIDs(threads);
		output_classIDs_.push_back(output_classIDs);
		std::vector< std::vector<std::vector<float> > > output_score(threads);
		output_score_.push_back(output_score);
	}

	// ignore the inference time at the first time
	for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
		int thread_id = (int)omp_get_thread_num();
		cTRT_ResNet50.Infer(input_images[0][thread_id], output_classIDs_[0][thread_id], output_score_[0][thread_id], thread_id, score_thresh);
		output_classIDs_[0][thread_id].clear();
	}

	const auto st_time = cv::getTickCount();
	for (int loop_idx = 0; loop_idx < loops; loop_idx++) {

		// infer (nBatchs * nSuperBatchs * nThreads) images
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
			int thread_id = (int)omp_get_thread_num();
			cTRT_ResNet50.Infer(input_images[loop_idx][thread_id], output_classIDs_[loop_idx][thread_id], output_score_[loop_idx][thread_id], thread_id, score_thresh);
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

	if (print_flag) {
		std::cout << "[" << TAG << "] Printing ... " << std::endl;
		std::vector<std::string> label_list(0);
		std::ifstream ifs(label_list_path);
		std::string str;
		if (!ifs.fail())
		{
			while (getline(ifs, str)) {
				label_list.push_back(str);
			}
		}
		input_image_count = 0;
		for (int loop_idx = 0; loop_idx < loops; loop_idx++) {
			for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
				for (int i = 0; i < batches * super_batches; i++) {
					std::cout << input_file_names[input_image_count % (int)input_file_names.size()] << " : ";
					kros::TRT_ResNet50::PrintResult(output_classIDs_[loop_idx][thread_idx][i].first, output_score_[loop_idx][thread_idx][i], label_list);
					input_image_count++;
				}
			}
		}
	}

	std::cout << "[" << TAG << "] All completed" << std::endl;
}
