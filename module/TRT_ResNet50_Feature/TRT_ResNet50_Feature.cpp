#define TAG "TRT_ResNet50_Feature"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/TRT_ResNet50_Feature/TRT_ResNet50_Feature.h>
#include <kros/trt/Common/PreProcess_cuda.h>
#include <omp.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	TRT_ResNet50_Feature::TRT_ResNet50_Feature()
	{
		m_outputSize_list.push_back(2048);
	}

	TRT_ResNet50_Feature::~TRT_ResNet50_Feature()
	{
		Release();
	}

	bool TRT_ResNet50_Feature::Init(
		const int gpu_id,
		const std::string& model_name,
		const std::string& model_dir,
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

		std::vector<std::string> output_layers2{ output_layers };
		return TRT::Init(gpu_id, model_name, model_dir, input_layers, output_layers2, width, height, batch, super_batch, thread_num,
			m_is_post_process_cuda, m_is_pre_process_cuda);
	}

	void TRT_ResNet50_Feature::Release() {

		int num_buffer = (int)mPreprocess_cuda_in_buffer.size();
		for (int i = 0; i < num_buffer; i++)
		{
			cudaFree(mPreprocess_cuda_in_buffer[i]);
		}
		mPreprocess_cuda_in_buffer.clear();
		TRT::Release();
	}

	bool TRT_ResNet50_Feature::InferParallel(
		const std::vector<cv::Mat>& input_images,
		std::vector<std::vector<float> >& output_features)
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

		std::vector<std::vector<std::vector<float> > > output_features_(num_infer);

#pragma omp parallel for num_threads(m_thread_num) schedule(static)
		for (int loop_idx = 0; loop_idx < num_infer; loop_idx++) {
			int thread_id = (int)omp_get_thread_num();
			Infer(infer_input_images[loop_idx], output_features_[loop_idx], thread_id);
		}

		for (int loop_idx = 0; loop_idx < num_infer; loop_idx++) {
			int nFrm = (int)output_features_[loop_idx].size();
			for (int FrmNo = 0; FrmNo < nFrm; FrmNo++) {
				output_features.push_back(output_features_[loop_idx][FrmNo]);
			}
		}

		return true;
	}

	bool TRT_ResNet50_Feature::Infer(
		const std::vector<cv::Mat>& input_images,
		std::vector<std::vector<float> >& output_features,
		const int thread_id)
	{
		std::vector<std::vector<std::vector<float> > > output_results;
		bool ret = TRT::Infer(input_images, output_results, thread_id);

		if (ret) {
			output_features.clear();
			for (int super_batch_idx = 0; super_batch_idx < (int)output_results.size(); super_batch_idx++) {
				//std::cout << output_results[super_batch_idx].size() << std::endl;
				output_features.insert(output_features.end(), output_results[super_batch_idx].begin(), output_results[super_batch_idx].end());
			}
			return true;
		}
		else {
			return false;
		}
		return false;
	}

	void TRT_ResNet50_Feature::PrintResult(
		const std::vector<float>& output_features)
	{
		std::cout << "[";
		for (int i = 0; i < 3; i++) {
			std::cout << output_features[i] << ", ";

		}
		std::cout << "... , ";

		for (int i = 2044; i < (int)output_features.size(); i++) {
			std::cout << output_features[i];
			if (i < (int)output_features.size() - 1) {
				std::cout << ", ";
			}
		}
		std::cout << "]" << std::endl;
	}

	void TRT_ResNet50_Feature::PreProcess(
		const std::vector<cv::Mat>& input_images,
		TRT::CPU_buffer& buffer,
		float* GPU_In_buffers,
		const cudaStream_t stream,
		const int thread_id)
	{
		//最適化方法 参考
		//img = cv2.resize(img, size)
		// img2 = img.transpose(2, 0, 1)
		// img2 = np.array(img2, dtype=np.float32, order='C')
		// img2 = img2 / 255.0
		// for ch in range(3):
		//     img2[ch] = (img2[ch] - mean[ch]) / std[ch]
		// image = img2

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

	void TRT_ResNet50_Feature::PostProcess(
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
