#define TAG "TRT"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt.h>
#include <kros/trt/TRT/TRT.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include <NvOnnxParser.h>
#include <NvUffParser.h>
#include <cudnn.h>

#if __has_include(<NvInferVersion.h>)
#include <NvInferVersion.h>
#endif

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	static Logger gLogger;

	TRT::TRT()
	{
	}

	TRT::~TRT()
	{
		Release();
	}

	bool TRT::Init(
		const int gpu_id,
		const std::string& model_name,
		const std::string& model_dir,
		const std::string& inputIndex_name,
		const std::vector<std::string>& outputIndex_names,
		const int width,
		const int height,
		const int batch,
		const int super_batch,
		const int thread_num,
		const bool is_post_proces_cuda,
		const bool is_pre_process_cuda)
	{
		if (m_init_flg)
		{
			std::cout << "[" << TAG << "] Already initialized" << std::endl;
			return false;
		}

		//               std::string s_GPUID = std::to_string(GPUID);
		//               setenv("CUDA_VISIBLE_DEVICES", s_GPUID.c_str(), 1);
		std::cout << "--------------------------------------------------------" << std::endl;
		std::cout << "TRT           : " << TRT_VERSION_MAJOR << "." << TRT_VERSION_MINOR << std::endl;
		int runtimeVersion;
		cudaRuntimeGetVersion(&runtimeVersion);
		std::cout << "CUDA          : " << runtimeVersion << std::endl;
		std::cout << "cuDNN         : " << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << std::endl;
		std::cout << "TensorRT      : " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;
		int ndevice = 0;
		cudaGetDeviceCount(&ndevice);
		std::cout << "CudaDeviceNum : " << ndevice << std::endl;
		for (int igpu = 0; igpu < ndevice; igpu++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, igpu);
			printf("GPU %d : %s (CC : %d.%d / Memory : %.1f [GB])\n", igpu, prop.name, prop.major, prop.minor, (float)(prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0));
		}
		std::cout << "--------------------------------------------------------" << std::endl;

		//    1. cudaGetDevicePropertiesでcudaDeviceProp.majorとminorを取得し、ccを特定
		//    2. *_w128_h256_b40_cc72.trtが存在していれば、ロードし、
		//       一連の初期化処理（バッファの確保等）の後、return true;
		//    3. *_w128_h256.onnxが存在していればロードし、
		//       builder->platformHasFastFp16 ()がtrueならfp16でビルドする、falseならfp32でビルドした後
		//　　　 一連の初期化処理（バッファの確保等）の後、return true;
		//    4. onnxファイルが存在していなければreturn false

		if (0 <= gpu_id && gpu_id < ndevice)
		{
			m_gpu_id = gpu_id;
			CHECK(cudaSetDevice(m_gpu_id));
		}
		else
		{
			std::cout << "[" << TAG << "] Error : invalid gpu id" << std::endl;
			return false;
		}

		m_width = width;
		m_height = height;
		m_batch = batch;
		m_super_batch = super_batch;
		m_thread_num = thread_num;
		m_is_post_process_cuda = is_post_proces_cuda;
		m_is_pre_process_cuda = is_pre_process_cuda;

		//std::cout << "m_use_device_to_host : " << m_use_device_to_host << std::endl;
		m_inputIndex_name = inputIndex_name;
		std::copy(outputIndex_names.begin(), outputIndex_names.end(), back_inserter(m_outputIndex_names));

		std::cout << "gpu_id      = " << gpu_id << std::endl;
		std::cout << "model_name  = " << model_name << std::endl;
		std::cout << "model_dir   = " << model_dir << std::endl;
		std::cout << "width       = " << width << std::endl;
		std::cout << "height      = " << height << std::endl;
		std::cout << "batch       = " << batch << std::endl;
		std::cout << "super_batch = " << super_batch << std::endl;
		std::cout << "thread_num  = " << thread_num << std::endl;

		for (int i = 0; i < m_thread_num; i++)
		{
			m_current_super_batches.push_back(0);
		}

		// [thread_id][super_batch_id][buffer_id]
		m_original_image_sizes.clear();
		for (int i = 0; i < m_thread_num; i++)
		{
			std::vector<std::vector<cv::Size>> tmp2;
			for (int j = 0; j < m_super_batch; j++)
			{
				std::vector<cv::Size> tmp;
				for (int k = 0; k < m_batch; k++)
				{
					tmp.push_back(cv::Size(0, 0));
				}
				tmp2.push_back(tmp);
			}
			m_original_image_sizes.push_back(tmp2);
		}

		cudaDeviceProp dev;
		cudaGetDeviceProperties(&dev, 0);
		char temp_name[128];
		std::string Model_name;
		std::string trt_name;
		sprintf(temp_name, "%s/%s", model_dir.c_str(), model_name.c_str());
		Model_name = temp_name;
		sprintf(temp_name, "%s/%s_w%d_h%d_b%d_cc%1d%1d.trt", model_dir.c_str(), model_name.c_str(), width, height, batch, dev.major, dev.minor);
		trt_name = temp_name;
		std::cout << "Model_name  = " << Model_name << std::endl;
		std::cout << "trt_name    = " << trt_name << std::endl;

		// trt_engine作成
		m_trt_engine = GetEngine(trt_name, Model_name, width, height);
		if (m_trt_engine == nullptr)
		{
			return false;
		}

		// 入出力のインデックスをチェック
//		std::cout << "getNbBindings()    = " << m_trt_engine->getNbBindings() << std::endl;

		for (int b = 0; b < m_trt_engine->getNbBindings(); ++b)
		{
			char* name = (char*)m_trt_engine->getBindingName(b);
			int BufIdx = m_trt_engine->getBindingIndex(name);

			if (strcmp(name, inputIndex_name.c_str()) == 0)
			{
				m_inputIndex = BufIdx;
				//std::cout << m_inputIndex << std::endl;
			}
			else
			{
				bool found = false;
				for (int out_idx = 0; out_idx < (int)outputIndex_names.size(); out_idx++)
				{
					if (strcmp(name, outputIndex_names[out_idx].c_str()) == 0)
					{
						m_outputIndex_list.push_back(BufIdx);
						//std::cout << m_outputIndex_list.back() << std::endl;
						found = true;
						break;
					}
				}

				if (!found)
				{
					std::cout << "[" << TAG << "] Buffer name invalid : " << name << std::endl;
					return false;
				}
			}
		}
		if ((int)m_outputIndex_list.size() != (int)m_outputSize_list.size())
		{
			std::cout << "[" << TAG << "] Output size must be" << (int)m_outputIndex_list.size() << "(" << (int)m_outputSize_list.size() << ")" << std::endl;
			//			return false;
		}
		this->m_outputBuffer_num = (int)m_outputSize_list.size();

		// コンテキスト
		for (int j = 0; j < m_super_batch; j++)
		{
			IExecutionContext* context = m_trt_engine->createExecutionContext();
			m_contexts.push_back(context);
		}

		// ストリーム
		for (int j = 0; j < m_super_batch; j++)
		{
			cudaStream_t stream;
			CHECK(cudaStreamCreate(&stream));
			m_streams.push_back(stream);
		}

		// GPUメモリー
		for (int j = 0; j < m_super_batch; j++)
		{
			std::vector<void*> GPUbuffers;

			// 出力バッファ数＋入力バッファ数(1)
			for (int buffer_idx = 0; buffer_idx < (int)outputIndex_names.size() + 1; buffer_idx++)
			{
				void* tmp = NULL;
				GPUbuffers.push_back(tmp);
			}

			m_GPU_buffers.push_back(GPUbuffers);
			CHECK(cudaMalloc(&m_GPU_buffers.back()[m_inputIndex], batch * 3 * height * width * sizeof(float)));

			for (int out_idx = 0; out_idx < m_outputBuffer_num; out_idx++)
			{
				CHECK(cudaMalloc(&m_GPU_buffers.back()[m_outputIndex_list[out_idx]], batch * this->m_outputSize_list[out_idx] * sizeof(float)));
			}
		}
		//PreprocessのCUDA版で使用する推論用入力パッファ
		if (m_is_pre_process_cuda)
		{
			//thread単位の入力GPUバッファ
			for (int i = 0; i < m_thread_num; i++)
			{
				std::vector<void*> GPU_In_buffers_super_batch;
				for (int j = 0; j < m_super_batch; j++)
				{
					void* pGPU_In_buffer;
					CHECK(cudaMalloc(&pGPU_In_buffer, batch * 3 * height * width * sizeof(float)));
					GPU_In_buffers_super_batch.push_back(pGPU_In_buffer);
				}
				m_GPU_In_buffers.push_back(GPU_In_buffers_super_batch);
			}
		}

		if (m_is_post_process_cuda)
		{
			//thread単位のGPUバッファ
			for (int i = 0; i < m_thread_num; i++)
			{
				std::vector<std::vector<void*>> GPU_Out_buffers2;
				for (int j = 0; j < m_super_batch; j++)
				{
					std::vector<void*> GPU_Out_buffers1;
					for (int out_idx = 0; out_idx < m_outputBuffer_num; out_idx++)
					{
						void* Out_GPU_buffer;
						CHECK(cudaMalloc(&Out_GPU_buffer, batch * this->m_outputSize_list[out_idx] * sizeof(float)));
						GPU_Out_buffers1.push_back(Out_GPU_buffer);
					}
					GPU_Out_buffers2.push_back(GPU_Out_buffers1);
				}
				m_GPU_Out_buffers.push_back(GPU_Out_buffers2);
			}
		}

		// スレッド毎・スーパーバッチ毎にメモリ空間を確保
		for (int i = 0; i < m_thread_num; i++)
		{
			// 入出力用のメインメモリー(CPU)
			std::vector<CPU_buffer> buffer_vec;
			for (int j = 0; j < m_super_batch; j++)
			{
				CPU_buffer IO_buffer;
				IO_buffer.numInput = 0;
				IO_buffer.in_buffer = (float*)malloc(batch * width * height * 3 * sizeof(float));

				IO_buffer.out_buffer = (float**)malloc(m_outputBuffer_num * sizeof(float*));
				for (int out_idx = 0; out_idx < m_outputBuffer_num; out_idx++)
				{
					IO_buffer.out_buffer[out_idx] = (float*)malloc(batch * this->m_outputSize_list[out_idx] * sizeof(float));
				}
				buffer_vec.push_back(IO_buffer);
			}
			m_buffers.push_back(buffer_vec);
		}

		// スタッフィング用ダミーバッファ 3チャンネル分
		m_imgSize = width * height * 3;
		m_DummyBuff = (float*)malloc(m_imgSize * sizeof(float));
		m_init_flg = true;

		return true;
	}

	bool TRT::Infer(
		const std::vector<cv::Mat>& input_images,
		std::vector<std::vector<std::vector<float>>>& output_results,
		const int thread_id)
	{
		CHECK(cudaSetDevice(m_gpu_id));

		//      GPI ID test用	
		// 	    int use_GPU;
		//	    CHECK(cudaGetDevice(&use_GPU));
		//	    std::cout << "Using GPU     : " << use_GPU << std::endl;

		if (!this->PreProcess_SuperBatch(input_images, thread_id))
		{
			return false;
		}

		if (m_current_super_batches[thread_id] == 0)
		{
			std::cout << "[" << TAG << "] There is no super batch  " << m_current_super_batches[thread_id] << std::endl;
			return false;
		}

#pragma omp critical(trt_parallel)
		{

			for (int super_batch_id = 0; super_batch_id < m_current_super_batches[thread_id]; super_batch_id++)
			{
				if (m_is_pre_process_cuda)
				{
					CHECK(cudaMemcpyAsync(
						m_GPU_buffers[super_batch_id][m_inputIndex],
						m_GPU_In_buffers[thread_id][super_batch_id],
						m_batch * m_height * m_width * 3 * sizeof(float),
						cudaMemcpyDeviceToDevice,
						m_streams[super_batch_id]));
				}
				else
				{
					CHECK(cudaMemcpyAsync(
						m_GPU_buffers[super_batch_id][m_inputIndex],
						m_buffers[thread_id][super_batch_id].in_buffer,
						m_batch * m_height * m_width * 3 * sizeof(float),
						cudaMemcpyHostToDevice,
						m_streams[super_batch_id]));
				}

				m_contexts[super_batch_id]->enqueue(
					m_batch,
					m_GPU_buffers[super_batch_id].data(),
					m_streams[super_batch_id],
					nullptr);
			}

			for (int super_batch_id = 0; super_batch_id < m_current_super_batches[thread_id]; super_batch_id++)
			{
				if (m_is_post_process_cuda)
				{
					for (int out_idx = 0; out_idx < m_outputBuffer_num; out_idx++)
					{
						CHECK(cudaMemcpyAsync(
							m_GPU_Out_buffers[thread_id][super_batch_id][out_idx],
							m_GPU_buffers[super_batch_id][m_outputIndex_list[out_idx]],
							m_batch * this->m_outputSize_list[out_idx] * sizeof(float),
							cudaMemcpyDeviceToDevice,
							m_streams[super_batch_id]));
					}
				}
				else
				{
					for (int out_idx = 0; out_idx < m_outputBuffer_num; out_idx++)
					{
						CHECK(cudaMemcpyAsync(
							m_buffers[thread_id][super_batch_id].out_buffer[out_idx],
							m_GPU_buffers[super_batch_id][m_outputIndex_list[out_idx]],
							m_batch * this->m_outputSize_list[out_idx] * sizeof(float),
							cudaMemcpyDeviceToHost,
							m_streams[super_batch_id]));
					}
				}

				cudaStreamSynchronize(m_streams[super_batch_id]);
			}
		}
		return this->PostProcess_SuperBatch(output_results, thread_id);
	}

	void TRT::PreProcess(
		const std::vector<cv::Mat>& input_images,
		const int super_batch_id,
		const int thread_id)
	{
		//    1. Init()未実施ならreturn false
		//    2. input_images.size() > batch_sizeならreturn false
		//    3. input_images.size() < batch_sizeなら、入力枚数を
		//　　　 メンバ変数に確保すると共に、batch_sizeに到達するまでzeros画像を追加する
		//　　4. 一連の事前処理(BGR→RGB→正規化→平均分散画像)を適用した結果を
		//　　　 buffers[thread_id]のinput領域に格納

		const int NumInputData = (int)input_images.size();
		for (int i = 0; i < NumInputData; i++)
		{
			m_original_image_sizes[thread_id][super_batch_id][i] = cv::Size(input_images[i].cols, input_images[i].rows);
		}
		//m_is_pre_process_cuda=falseの時、m_GPU_In_buffersはnullptrのままのはずなので、ここでブランチさせる必要は本来不要だが、
		//今後の事を考えて敢えて明示的に不要のバッファにnullptrをセットしておく。
		//Preprocess_cudaを使用する場合でも、m_buffers内のnumInput（バッファ内の有効データ数）は使用している。
		//m_buffersは以下の構造体となっている。
		//struct CPU_buffer
		//{
		//	int numInput;
		//	float* in_buffer;
		//	float** out_buffer;
		//};
		//PreprocessをCPUで使用する場合は、in_bufferを使用するが、GPUを使用する場合は、in_bufferの替りにm_GPU_In_buffers側を使用する。

		if (m_is_pre_process_cuda)
		{
			PreProcess(input_images, m_buffers[thread_id][super_batch_id], (float*)m_GPU_In_buffers[thread_id][super_batch_id], m_streams[super_batch_id], thread_id);  // ←小クラス側のPreProcessが呼ばれる。
		}
		else
		{
			PreProcess(input_images, m_buffers[thread_id][super_batch_id], nullptr, nullptr, 0);  // ←小クラス側のPreProcessが呼ばれる。
		}
		//バッチの足りない部分をダミーデータで埋める。
		if (m_is_pre_process_cuda)
		{
			//本来m_GPU_In_buffersはvoidポインタだが、それだとアドレッシングが出来ないので、一旦floatポインタにキャストする。
			//void*のままだとerror: ‘__gnu_cxx::__alloc_traits<std::allocator<void*>, void*>::value_type’ {aka ‘void*’} is not a pointer-to-object type
			//でビルドエラーとなる。
			float* inbufpos_GPU = &(((float*)m_GPU_In_buffers[thread_id][super_batch_id])[NumInputData * m_imgSize]);
			for (int i = NumInputData; i < m_batch; i++)
			{
				m_original_image_sizes[thread_id][super_batch_id][i] = cv::Size(0, 0);
				CHECK(cudaMemcpyAsync(
					inbufpos_GPU,
					m_DummyBuff,
					m_imgSize * sizeof(float),
					cudaMemcpyHostToDevice,
					m_streams[super_batch_id]));
				inbufpos_GPU += m_imgSize;
			}
		}
		else
		{

			float* inbufpos = &(m_buffers[thread_id][super_batch_id].in_buffer[NumInputData * m_imgSize]);
			for (int i = NumInputData; i < m_batch; i++)
			{
				m_original_image_sizes[thread_id][super_batch_id][i] = cv::Size(0, 0);
				memcpy(inbufpos, m_DummyBuff, m_imgSize * sizeof(float));
				inbufpos += m_imgSize;
			}
		}
	}

	void TRT::PostProcess(
		std::vector<std::vector<std::vector<float>>>& output_result,
		const int super_batch_id,
		const int thread_id)
	{
		//    1. buffers[thread_id]のoutput領域が空ならreturn false
		//    2. buffers[thread_id]のoutput領域に事後処理を適用し、
		//       元々の入力枚数のみをoutput_featuresに格納する

		//	PostProcess(m_buffers[thread_id][super_batch_id], m_original_image_sizes[thread_id][super_batch_id], output_result);

		std::vector<void*> InferOutGPUbuffers;

		//InferOutGPUbuffersを使用しない場合の回避
		if (m_is_post_process_cuda)
		{
			for (int out_idx = 0; out_idx < m_outputBuffer_num; out_idx++)
			{
				InferOutGPUbuffers.push_back(m_GPU_Out_buffers[thread_id][super_batch_id][out_idx]);
			}
		}
		PostProcess(m_buffers[thread_id][super_batch_id], InferOutGPUbuffers, m_original_image_sizes[thread_id][super_batch_id],
			super_batch_id, thread_id, output_result);
		m_buffers[thread_id][super_batch_id].numInput = 0;
	}

	bool TRT::PreProcess_SuperBatch(
		const std::vector<cv::Mat>& input_images,
		const int thread_id)
	{
		if (m_init_flg == false)
		{
			std::cout << "[" << TAG << "] Error : not initialized" << std::endl;
			return false;
		}

		if (m_current_super_batches[thread_id] > 0)
		{
			std::cout << "[" << TAG << "] there are some super batches that has not been post-processed " << m_current_super_batches[thread_id] << std::endl;
			return false;
		}
		m_current_super_batches[thread_id] = (int)std::ceil((double)input_images.size() / m_batch);
		if (m_current_super_batches[thread_id] > m_super_batch)
		{
			std::cout << "[" << TAG << "] current_super_batch must be less than super_batch " << m_current_super_batches[thread_id] << "<=" << m_super_batch << std::endl;
			m_current_super_batches[thread_id] = 0;
			return false;
		}

		std::vector<std::vector<cv::Mat>> input_images_per_super_batches;
		for (int super_batch_idx = 0; super_batch_idx < m_current_super_batches[thread_id]; super_batch_idx++)
		{
			std::vector<cv::Mat> batch_images;
			for (int batch_idx = 0; batch_idx < m_batch; batch_idx++)
			{
				int idx = super_batch_idx * m_batch + batch_idx;
				if (idx >= (int)input_images.size())
				{
					break;
				}
				batch_images.push_back(input_images[idx]);
			}

			int NumInputData = (int)batch_images.size();
			if (NumInputData > m_batch)
			{
				std::cout << "[" << TAG << "] Error : NumInputData > m_batch" << std::endl;
				ClearSuperBatch(thread_id);
				return false;
			}

			//まだPostProcessしていないスレッドIDのデータがある。この判定で仕掛かり中データの上書きを防止。
			if (m_buffers[thread_id][super_batch_idx].numInput > 0)
			{
				std::cout << "[" << TAG << "] Error : Duplicate  thread_id" << std::endl;
				ClearSuperBatch(thread_id);
				return false;
			}
			PreProcess(batch_images, super_batch_idx, thread_id);
		}
		return true;
	}

	bool TRT::PostProcess_SuperBatch(
		std::vector<std::vector<std::vector<float>>>& output_results,
		const int thread_id)
	{

		if (m_init_flg == false)
		{
			std::cout << "[" << TAG << "] Error : not initialized" << std::endl;
			return false;
		}

		output_results.clear();

		for (int super_batch_id = 0; super_batch_id < m_current_super_batches[thread_id]; super_batch_id++)
		{
			std::vector<std::vector<std::vector<float>>> output_results_per_super_batch;
			this->PostProcess(output_results_per_super_batch, super_batch_id, thread_id);
			output_results.insert(output_results.end(), output_results_per_super_batch.begin(), output_results_per_super_batch.end());
		}
		//std::cout << output_features.size() << std::endl;
		m_current_super_batches[thread_id] = 0;
		return true;
	}

	ICudaEngine* TRT::GetEngine(
		const std::string& trt_filename,
		const std::string& model_filename,
		const int width,
		const int height)
	{
		// deserialize the engine
		IRuntime* runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);
		/*
			if (gUseDLACore >= 0)
			{
				runtime->setDLACore(gUseDLACore);
			}
			*/

		ICudaEngine* engine = NULL;

		std::ifstream ifs(trt_filename, std::ios::binary);
		if (ifs.is_open())
		{
			std::cout << "Loading engine ... " << std::endl;

			ifs.seekg(0, fstream::end);
			uint eofPos = (uint)ifs.tellg();
			ifs.clear();
			ifs.seekg(0, fstream::beg);
			uint begPos = (uint)ifs.tellg();
			uint size = eofPos - begPos;

			char* p = (char*)malloc(size);
			ifs.read(p, size);
			ifs.close();
			engine = runtime->deserializeCudaEngine(p, size, nullptr);
			assert(engine != nullptr);
			free(p);
		}
		else
		{
			std::cout << "Creating engine ... " << std::endl;

			IHostMemory* trtModelStream{ nullptr };

			std::string model_name;
			bool model_found = false;
			model_name = model_filename + "_w" + std::to_string(width) + "_h" + std::to_string(height) + ".onnx";
			if (!model_found && std::filesystem::exists(model_name))
			{
				std::cout << "model_name  = " << model_name << std::endl;
				OnnxToTRTModel(model_name, m_batch, trtModelStream);
				model_found = true;
			}

			model_name = model_filename + "_w" + std::to_string(width) + "_h" + std::to_string(height) + ".uff";
			if (!model_found && std::filesystem::exists(model_name))
			{
				std::cout << "model_name  = " << model_name << std::endl;
				UffToTRTModel(model_name, m_batch, trtModelStream);
				model_found = true;
			}

			model_name = model_filename + ".wts";
			if (!model_found && std::filesystem::exists(model_name))
			{
				std::cout << "model_name  = " << model_name << std::endl;
				CustomCreateEngine(model_name, m_batch, trtModelStream);
				model_found = true;
			}

			if (!model_found)
			{
				std::cout << "[" << TAG << "] Failed to open model_filename ： " << model_filename << std::endl;
				return nullptr;
			}
			assert(trtModelStream != nullptr);
			//std::cout << "trtModelStream->size : " << trtModelStream->size() << std::endl;
			std::ofstream ofs(trt_filename, std::ios::binary);
			if (ofs)
			{
				ofs.write((char*)trtModelStream->data(), trtModelStream->size());
				ofs.close();
			}
			else
			{
				std::cout << "[" << TAG << "] Failed to open trt_filename ： " << trt_filename << std::endl;
			}
			engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
			assert(engine != nullptr);
			trtModelStream->destroy();
		}

		std::cout << "Completed" << std::endl;
		runtime->destroy();
		return engine;
	}
	void TRT::OnnxToTRTModel(
		const std::string& modelFile, // name of the onnx model
		unsigned int maxBatchSize,	// batch size - NB must be at least as large as the batch we want to run with
		IHostMemory*& trtModelStream) // output buffer for the TensorRT model
	{
		int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

		// create the builder
		IBuilder* builder = createInferBuilder(gLogger);
#if NV_TENSORRT_MAJOR == 6		
		//tensorrt6 対応
		//		nvinfer1::INetworkDefinition* network = builder->createNetworkV2(nvinfer1::EngineCapability::kDEFAULT);
		//   	kDEFAULTがどれを使うのか不明。とりあえず０を設定して置く
		nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
		//		以下のコードを試してみたが、ビルドは通るもエンジン作成に失敗する。
		//		https://github.com/onnx/onnx-tensorrt/blob/master/README.md
		//		const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		//		nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
#else
		nvinfer1::INetworkDefinition* network = builder->createNetwork();
#endif

		auto parser = nvonnxparser::createParser(*network, gLogger);

		// 以下のコードがエラーとなる。どうも存在しないらしい。
		//		auto parser = nvonnxparser::createONNXParser(*network, gLogger);
		//		parser->reportParsingInfo();

				//Optional - uncomment below lines to view network layer information
				//config->setPrintLayerInfo(true);
				//parser->reportParsingInfo();

		if (!parser->parseFromFile(modelFile.c_str(), verbosity))
		{
			string msg("[TRT::OnnxToTRTModel()] Failed to parse onnx file");
			gLogger.log(ILogger::Severity::kERROR, msg.c_str());
			exit(EXIT_FAILURE);
		}
		builder->setMaxBatchSize(maxBatchSize);

#if (NV_TENSORRT_MAJOR == 6) || (NV_TENSORRT_MAJOR == 7)	
		nvinfer1::IBuilderConfig* BuilderConfig = builder->createBuilderConfig();
		BuilderConfig->setMaxWorkspaceSize(16 * (1 << 20));
		if (builder->platformHasFastFp16())
		{
			BuilderConfig->setFlag(BuilderFlag::kFP16);
			std::cout << "FP16 Mode True" << std::endl;
		}
		ICudaEngine* engine = builder->buildEngineWithConfig(*network, *BuilderConfig);
		assert(engine);
#else
		builder->setMaxWorkspaceSize(1 << 20);
		if (builder->platformHasFastFp16())
		{
			builder->setFp16Mode(true);
			std::cout << "FP16 Mode True" << std::endl;
		}
		//		samplesCommon::enableDLA(builder, /*gUseDLACore*/ -1);
		ICudaEngine* engine = builder->buildCudaEngine(*network);
		assert(engine);
#endif


		// we can destroy the parser
		parser->destroy();

		// serialize the engine, then close everything down
		trtModelStream = engine->serialize();
		engine->destroy();
		network->destroy();
		builder->destroy();
	}

	void TRT::UffToTRTModel(
		const std::string& modelFile, // name of the MNIST model
		unsigned int maxBatchSize,	// batch size - NB must be at least as large as the batch we want to run with
		IHostMemory*& trtModelStream) // output buffer for the TensorRT model
	{
		//		int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
		// create the builder
		IBuilder* builder = createInferBuilder(gLogger);

#if NV_TENSORRT_MAJOR == 6		
		//tensorrt6 対応
		//		nvinfer1::INetworkDefinition* network = builder->createNetworkV2(nvinfer1::EngineCapability::kDEFAULT);
		//   	kDEFAULTがどれを使うのか不明。とりあえず０を設定して置く
		nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
#else
		nvinfer1::INetworkDefinition* network = builder->createNetwork();
#endif
		auto parser = nvuffparser::createUffParser();

		parser->registerInput(m_inputIndex_name.c_str(), Dims3(3, m_height, m_width), nvuffparser::UffInputOrder::kNCHW);
		parser->registerOutput(m_outputIndex_names[0].c_str());

		if (!parser->parse(modelFile.c_str(), *network, nvinfer1::DataType::kFLOAT))
		{
			string msg("[TRT::UffToTRTModel()] Failed to parse uff file");
			gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
			exit(EXIT_FAILURE);
		}

		builder->setMaxBatchSize(maxBatchSize);
#if (NV_TENSORRT_MAJOR == 6) || (NV_TENSORRT_MAJOR == 7)	
		nvinfer1::IBuilderConfig* BuilderConfig = builder->createBuilderConfig();
		BuilderConfig->setMaxWorkspaceSize(1 << 20);
		if (builder->platformHasFastFp16())
		{
			BuilderConfig->setFlag(BuilderFlag::kFP16);
			std::cout << "FP16 Mode True" << std::endl;
		}
		ICudaEngine* engine = builder->buildEngineWithConfig(*network, *BuilderConfig);
		assert(engine);
#else		
		builder->setMaxWorkspaceSize(1 << 20);

		if (builder->platformHasFastFp16())
		{
			builder->setFp16Mode(true);
			std::cout << "FP16 Mode True" << std::endl;
		}

		//		samplesCommon::enableDLA(builder, /*gUseDLACore*/ -1);
		ICudaEngine* engine = builder->buildCudaEngine(*network);
		assert(engine);
#endif
		// we can destroy the parser
		parser->destroy();

		// serialize the engine, then close everything down
		trtModelStream = engine->serialize();
		engine->destroy();
		network->destroy();
		builder->destroy();
	}

	void TRT::ClearSuperBatch(const int thread_id)
	{
		for (int i = 0; i < m_super_batch; i++)
		{
			m_buffers[thread_id][i].numInput = 0;
		}
		m_current_super_batches[thread_id] = 0;
	}

	void TRT::Release()
	{
		if (m_init_flg)
			std::cout << "Releasing Cuda Memory" << std::endl;

		for (int i = 0; i < (int)m_buffers.size(); i++)
		{
			for (int j = 0; j < (int)m_buffers[i].size(); j++)
			{
				free(m_buffers[i][j].in_buffer);
				for (int out_idx = 0; out_idx < m_outputBuffer_num; out_idx++)
				{
					free(m_buffers[i][j].out_buffer[out_idx]);
				}
				free(m_buffers[i][j].out_buffer);
			}
		}
		m_buffers.clear();
		m_outputIndex_list.clear();
		m_outputSize_list.clear();

		for (int i = 0; i < (int)m_streams.size(); i++)
		{
			cudaStreamDestroy(m_streams[i]);
		}
		m_streams.clear();

		// MUST NOT release gpu buffers
		//	for (int i = 0; i < (int)m_GPU_buffers.size(); i++) {
		//		for (int j = 0; j < (int)m_GPU_buffers[i].size(); j++) {
		//			cudaFree(&m_GPU_buffers[i][j]);
		//		}
		//	}
		// Infer時のIOで使用するバッファ。
		m_GPU_buffers.clear();

		//m_GPU_In_buffersはPreprocessがCUDA使用時に使用するバッファ
		for (int i = 0; i < (int)m_GPU_In_buffers.size(); i++)
		{
			for (int j = 0; j < (int)m_GPU_In_buffers[i].size(); j++)
			{
				cudaFree(&m_GPU_In_buffers[i][j]);
			}
		}
		m_GPU_In_buffers.clear();

		//m_GPU_Out_buffersはPostprocessがCUDA使用時に使用するバッファ
		for (int i = 0; i < (int)m_GPU_Out_buffers.size(); i++)
		{
			for (int j = 0; j < (int)m_GPU_Out_buffers[i].size(); j++)
			{
				for (int k = 0; k < (int)m_GPU_Out_buffers[i][j].size(); k++)
				{
					cudaFree(&m_GPU_Out_buffers[i][j][k]);
				}
			}
		}
		m_GPU_Out_buffers.clear();

		if (m_DummyBuff)
		{
			free(m_DummyBuff);
			m_DummyBuff = nullptr;
		}

		for (int i = 0; i < (int)m_contexts.size(); i++)
		{
			m_contexts[i]->destroy();
		}
		m_contexts.clear();

		if (m_trt_engine)
		{
			m_trt_engine->destroy();
			m_trt_engine = nullptr;
		}

		m_current_super_batches.clear();
		//	cudaDeviceReset();
		m_init_flg = false;
	}

} // namespace kros
