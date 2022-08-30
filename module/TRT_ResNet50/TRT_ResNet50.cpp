#define TAG "TRT_ResNet50"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/TRT_ResNet50/TRT_ResNet50.h>
#include <kros/trt/Common/PreProcess_cuda.h>
#include <omp.h>
#include <cmath>

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	TRT_ResNet50::TRT_ResNet50()
	{
		m_outputSize_list.push_back(3);
	}

	TRT_ResNet50::~TRT_ResNet50()
	{
		Release();
	}

	bool TRT_ResNet50::Init(
		const int gpu_id,
		const std::string& model_name,
		const std::string& model_dir,
		const int class_num,
		const int width,
		const int height,
		const int batch,
		const int super_batch,
		const int thread_num,
		const bool is_pre_process_cuda,
		const std::string& input_layers,
		const std::string& output_layers)

	{
		m_batch = batch;
		m_super_batch = super_batch;
		m_thread_num = thread_num;

		int ndevice = 0;
		cudaGetDeviceCount(&ndevice);
		m_gpu_id = gpu_id;
		if (0 <= m_gpu_id && m_gpu_id < ndevice)
		{
			CHECK(cudaSetDevice(m_gpu_id));
		}
		else
		{
			std::cout << "[" << TAG << "] Error : invalid gpu id" << std::endl;
			return false;
		}

		m_is_pre_process_cuda = is_pre_process_cuda;

		if (m_is_pre_process_cuda)
		{
			std::cout << "PreProcess  = CUDA" << std::endl;
		}
		else
		{
			std::cout << "PreProcess  = CPU" << std::endl;
		}

		m_is_post_process_cuda = false;  //YOLOv4はBBOXまで、Infer内で行う為、推論結果はホスト側に戻す。(ここをtrueにするのは現状YOLOv3でTHRUST使用時)

#ifdef _OPENMP
		if (m_thread_num > omp_get_num_procs()) {
			std::cout << "[" << TAG << "] Error : thread_num must be set within the number of processors" << std::endl;
			return false;
		}
		omp_set_num_threads(m_thread_num);
#endif


		//        std::cout << "m_const_mean[0]:" << m_const_mean[0] <<  "  m_const_std[0]:" << m_const_std[0] << std::endl; 
		//        std::cout << "m_const_mean[1]:" << m_const_mean[1] <<  "  m_const_std[1]:" << m_const_std[1] << std::endl; 
		//        std::cout << "m_const_mean[2]:" << m_const_mean[2] <<  "  m_const_std[2]:" << m_const_std[2] << std::endl; 
		//        cudaMemcpyToSymbol( g_const_mean, m_const_mean, sizeof( float) *3,0,cudaMemcpyHostToDevice);
		//       cudaMemcpyToSymbol( g_const_std, m_const_std, sizeof( float) *3,0,cudaMemcpyHostToDevice);
		//		float r_const_mean[3];
		//		float r_const_std[3];
		//        cudaMemcpyToSymbol( g_const_mean, r_const_mean, sizeof( float) *3,0,cudaMemcpyDeviceToHost);
		//        cudaMemcpyToSymbol( g_const_std, r_const_std, sizeof( float) *3,0,cudaMemcpyDeviceToHost);

		//        std::cout << "r_const_mean[0]:" << r_const_mean[0] <<  "  r_const_std[0]:" << r_const_std[0] << std::endl; 
		//        std::cout << "r_const_mean[1]:" << r_const_mean[1] <<  "  r_const_std[1]:" << r_const_std[1] << std::endl; 
		//        std::cout << "r_const_mean[2]:" << r_const_mean[2] <<  "  r_const_std[2]:" << r_const_std[2] << std::endl; 

		//PreprocessのCUDA版で使用する推論用入力パッファとバッファサイズの記録領域
		mPreprocess_cuda_in_buffer.clear();
		mPreprocess_cuda_in_buffer_size.clear();
		if (m_is_pre_process_cuda)
		{
			//CPU上のコンスタントメモリーに係数を積む。
			SetConstantValue(m_const_mean, m_const_std);
			//thread単位の入力GPUバッファ
			for (int i = 0; i < m_thread_num; i++)
			{
				mPreprocess_cuda_in_buffer.push_back(nullptr);
				mPreprocess_cuda_in_buffer_size.push_back(0);
			}
		}

		//クラス数を付与
		std::string model_name2 = model_name + "_c" + std::to_string(class_num);

		std::vector<std::string> output_layers2{ output_layers };
		return TRT::Init(gpu_id, model_name2, model_dir, input_layers, output_layers2, width, height, batch, super_batch, thread_num,
			m_is_post_process_cuda, m_is_pre_process_cuda);
	}

	void TRT_ResNet50::Release() {

		int num_buffer = (int)mPreprocess_cuda_in_buffer.size();
		for (int i = 0; i < num_buffer; i++)
		{
			cudaFree(mPreprocess_cuda_in_buffer[i]);
		}
		mPreprocess_cuda_in_buffer.clear();
		TRT::Release();
	}


	bool TRT_ResNet50::InferParallel(
		const std::vector<cv::Mat>& input_images,
		std::vector<std::pair<int, float>>& output_classIDs,
		std::vector<std::vector<float>>& output_scores,
		const float score_threshold)
	{
		int num_images = (int)input_images.size();   //ここは入力そのままのcv::Matの1次元vector

		int num_frm_per_infer = m_batch * m_super_batch; //1回のInferで処理する

		int num_infer = num_images / num_frm_per_infer;
		int remain_Frm = num_images % num_frm_per_infer;

		std::vector<std::vector<cv::Mat> > infer_input_images;
		int input_image_count = 0;

		//1スレッド分で纏める
		for (int i = 0; i < num_infer; i++) {
			std::vector<cv::Mat> infer_input_images_;
			for (int i = 0; i < num_frm_per_infer; i++) {
				infer_input_images_.push_back(input_images[input_image_count]);
				input_image_count++;
			}
			infer_input_images.push_back(infer_input_images_);
		}

		//残りの中途半端となった分を積む
		if (remain_Frm > 0) {
			std::vector<cv::Mat> infer_input_images_;
			for (int i = 0; i < remain_Frm; i++) {
				infer_input_images_.push_back(input_images[input_image_count]);
				input_image_count++;
			}
			infer_input_images.push_back(infer_input_images_);
			num_infer++;
		}

		std::vector < std::vector<std::pair<int, float> > > output_classIDs_(num_infer);
		std::vector<std::vector<std::vector<float> > > output_scores_(num_infer);

#pragma omp parallel for num_threads(m_thread_num) schedule(static)
		for (int loop_idx = 0; loop_idx < num_infer; loop_idx++) {
			int thread_id = (int)omp_get_thread_num();
			Infer(infer_input_images[loop_idx], output_classIDs_[loop_idx], output_scores_[loop_idx], thread_id, score_threshold);
		}

		for (int loop_idx = 0; loop_idx < num_infer; loop_idx++) {
			int nFrm = (int)output_scores_[loop_idx].size();
			for (int FrmNo = 0; FrmNo < nFrm; FrmNo++) {
				output_classIDs.push_back(output_classIDs_[loop_idx][FrmNo]);
				output_scores.push_back(output_scores_[loop_idx][FrmNo]);
			}
		}

		return true;
	}

	bool TRT_ResNet50::Infer(
		const std::vector<cv::Mat>& input_images,
		std::vector<std::pair<int, float>>& output_classIDs,
		std::vector<std::vector<float>>& output_scores,
		const int thread_id,
		const float score_threshold)
	{
		std::vector<std::vector<std::vector<float> > > output_results;
		bool ret = TRT::Infer(input_images, output_results, thread_id);

		if (ret) {
			std::vector<float> intermid_scores;
			output_scores.clear();
			output_classIDs.clear();
			for (int super_batch_idx = 0; super_batch_idx < (int)output_results.size(); super_batch_idx++) {
				for (int frame_idx = 0; frame_idx < (int)output_results[super_batch_idx].size(); frame_idx++) {
					intermid_scores.clear();
					float denomi_softmax = 0.0;
					float sum = 0.0;
					std::vector<float> numerator_softmax((int)output_results[super_batch_idx][frame_idx].size());
					for (int class_idx = 0; class_idx < (int)output_results[super_batch_idx][frame_idx].size(); class_idx++) {
						denomi_softmax += numerator_softmax[class_idx] = std::exp(output_results[super_batch_idx][frame_idx][class_idx]);
						sum += output_results[super_batch_idx][frame_idx][class_idx];
					}
					float MaxScore = 0.0f;
					int   MaxScoreIdx = INT_MAX;
					for (int class_idx = 0; class_idx < (int)output_results[super_batch_idx][frame_idx].size(); class_idx++) {
						float score = numerator_softmax[class_idx] / denomi_softmax;
						intermid_scores.push_back(score);
						if (MaxScore < score && score > score_threshold) {
							MaxScore = score;
							MaxScoreIdx = class_idx;
						}
					}
					output_scores.push_back(intermid_scores);
					if (MaxScoreIdx == INT_MAX) {
						output_classIDs.push_back(std::make_pair(-1, MaxScore));  //unknown
					}
					else {
						output_classIDs.push_back(std::make_pair(MaxScoreIdx, MaxScore));
					}
				}
			}
			return true;
		}
		else {
			return false;
		}
		return false;
	}

	bool TRT_ResNet50::PrintResult(
		const int output_classID,
		const std::vector<float>& output_score,
		const std::vector<std::string>& label)
	{
		int numLabels = (int)label.size();

		std::cout << "scores = [";
		int NumClass = (int)output_score.size();
		for (int i = 0; i < NumClass; i++) {
			std::cout << std::setw(6) << std::fixed << std::setprecision(3) << output_score[i];
			if (i < NumClass - 1) {
				std::cout << ",";
			}
		}
		std::cout << " ], max_score_class = ";
		if (numLabels > 0) {
			//ラベルリストが存在すること。
			if (output_classID == -1) {
				//スコアの閾値に引っかかった物は、unknownと判断する。(-1)がセットされて来る。
				std::cout << "unknown (" << output_classID << ")";
			}
			else if (output_classID < numLabels && output_classID >= 0) {
				//output_classID < numLabels   : 判定されたクラスIDが、ラベルリスト数以下であること。
				//output_classID >= 0          : 判定されたクラスIDが、0以上で有ること。負の数値が来た場合のガード。基本的にバグでしか存在しないはず。
				std::cout << label[output_classID] << " (" << output_classID << ")";
			}
			else {
				//ラベルリストに入らないクラスIDは未知のクラスとして"unlabeled"を表示
				std::cout << "unlabeled (" << output_classID << ")";
			}
		}
		else {
			std::cout << std::setw(2) << output_classID;
		}
		std::cout << std::endl;
		return true;
	}

	void TRT_ResNet50::PreProcess(
		const std::vector<cv::Mat>& input_images,
		TRT::CPU_buffer& buffer,
		float* GPU_In_buffers,
		const cudaStream_t stream,
		const int thread_id)
	{
		const int NumInputData = (int)input_images.size();
		buffer.numInput = NumInputData;

		if (m_is_pre_process_cuda)
		{
			float* inbuf_GPU_pos = GPU_In_buffers;
			for (int i = 0; i < NumInputData; i++)
			{
				int input_size = input_images[i].rows * input_images[i].cols;
				//最大の入力画像のサイズ合わせて再アロケーション
				if (input_size > mPreprocess_cuda_in_buffer_size[thread_id])
				{
					if (cudaFree(mPreprocess_cuda_in_buffer[thread_id]))
					{
						CHECK(cudaFree(mPreprocess_cuda_in_buffer[thread_id]));
					}
					CHECK(cudaMalloc(&mPreprocess_cuda_in_buffer[thread_id], input_size * 3 * sizeof(float))); //3channel固定
					mPreprocess_cuda_in_buffer_size[thread_id] = input_size;
				}

				PreProcess_cuda(input_images[i], mPreprocess_cuda_in_buffer[thread_id], inbuf_GPU_pos, m_width, m_height, stream);
				inbuf_GPU_pos += m_width * m_height * 3;
			}
		}
		else
		{
			cv::Mat resized;
			vector<cv::Mat> planes;
			cv::Mat planeR;
			cv::Mat planeG;
			cv::Mat planeB;

			float* inbufpos = buffer.in_buffer;
			for (int i = 0; i < NumInputData; i++) {
				//リサイズ
				cv::resize(input_images[i], resized, cv::Size(m_width, m_height), cv::INTER_LINEAR);
				// 3つのチャネルB, G, Rに分離。
				cv::split(resized, planes);
				//正規化 （8bit実数から32bit整数変換も同時に行う。
				planes[0].convertTo(planeB, CV_32F, 1.0 / 255); //B
				planeB = (planeB - m_const_mean[2]) / m_const_std[2];

				planes[1].convertTo(planeG, CV_32F, 1.0 / 255); //G
				planeG = (planeG - m_const_mean[1]) / m_const_std[1];

				planes[2].convertTo(planeR, CV_32F, 1.0 / 255); //R
				planeR = (planeR - m_const_mean[0]) / m_const_std[0];

				//1次元配列へ変換。Channel firstで、RGBの順番に変更
				memcpy(inbufpos, planeR.data, planeR.total() * planeR.elemSize());
				inbufpos += planeR.total();

				memcpy(inbufpos, planeG.data, planeG.total() * planeR.elemSize());
				inbufpos += planeG.total();

				memcpy(inbufpos, planeB.data, planeB.total() * planeR.elemSize());
				inbufpos += planeB.total();
			}
		}
	}

	void TRT_ResNet50::PostProcess(
		const TRT::CPU_buffer& buffer,
		const std::vector<void*>& gpu_buffers,
		const std::vector<cv::Size>& original_image_sizes,
		const int super_batch_id,
		const int thread_id,
		std::vector< std::vector<std::vector<float> > >& output_result) const
	{
		std::vector<std::vector<float> > output_features;
		for (int i = 0; i < buffer.numInput; i++) {
			std::vector<float> features;
			for (int j = 0; j < m_outputSize_list[0]; j++) {
				features.push_back(buffer.out_buffer[0][m_outputSize_list[0] * i + j]);
			}
			output_features.push_back(features);
		}
		output_result.push_back(output_features);
	}
}
