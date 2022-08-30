#ifndef __TRT_H__
#define __TRT_H__

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <vector>
#include <opencv2/core/core.hpp>
#include <NvInfer.h>

#include "common.h"

#ifdef _WIN32
#pragma warning(pop)
#endif

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

namespace kros
{
	class TRT
	{
	public:
		TRT();
		~TRT();

		bool Init(
			const int GPUID,									// 使用するGPUのID（0以上の整数）
			const std::string& model_name,						// モデルファイルの識別子
			const std::string& model_dir,						// モデルファイルとTRTエンジンの保存先ディレクトリ
			const std::string& inputIndex_name,					// モデルファイル内で指定された入力層（単一個）の名前
			const std::vector<std::string>& m_outputIndex_names,// モデルファイル内で指定された出力層（単一個もしくは複数個）の名前
			const int width,									// 入力層の横幅（任意のサイズの入力画像はInfer()時にこのサイズにリサイズされる）
			const int height,									// 入力層の縦幅（任意のサイズの入力画像はInfer()時にこのサイズにリサイズされる）
			const int batch,									// バッチ数
			const int super_batch = 1,							// スーパーバッチ数（造語）
			const int thread_num = 1,							// OpenMPスレッド数
			const bool m_is_post_proces_cuda = false,			// PostProcess前に推論結果をCPU側へ転送するか否か（falseの場合転送し、PostProcessはCPUで実行）
			const bool m_is_pre_process_cuda = false);			// PreProcess前に入力画像をCPU側へ転送するか否か（falseの場合転送せず、PreProcessはCPUで実行）

		bool Infer(
			const std::vector<cv::Mat>& input_images,					  // BGR・3チャンネルの入力画像のバッチ分
			std::vector<std::vector<std::vector<float>>>& output_results, // 推論結果（std::vector<std::vector<float> >）の配列
			const int thread_id = 0);									  // OpenMPスレッド数

		void Release();

	protected:
		struct CPU_buffer
		{
			int numInput;
			float* in_buffer;
			float** out_buffer;
		};

		// 入力画像のリスト（バッチ枚数）を入力バッファに格納する
		virtual void PreProcess(
			const std::vector<cv::Mat>& input_images,	// BGR・3チャンネルの入力画像のバッチ分、input_images.size()はバッチサイズ以下となっている
			TRT::CPU_buffer& buffer,					// 入力データの格納先のバッファ
			float* GPU_In_buffers,
			const cudaStream_t stream,
			const int thread_id) = 0;					// Preprosess_cudaでCUDAメモリの再アロケーションを行う為にconstは外した。

		// 出力バッファからバッチ枚数分の推論結果を取り出す
		// 　出力バッファにはCPU版とGPU版のいずれかを用いることができる
		// 　Init()で一時バッファをメンバ変数に格納する場合は、スーパーバッチ×スレッド分作成すること
		virtual void PostProcess(
			const TRT::CPU_buffer& buffer,					   // 出力バッファ（use_device_to_hostがtrueの場合に格納）
			const std::vector<void*>& gpu_buffers,			   // 出力バッファ（CUDAの生ポインタ）
			const std::vector<cv::Size>& original_image_sizes, // オリジナルの画像サイズ
			const int super_batch_id,						   // スーパーバッチ数（造語）
			const int thread_id,							   // OpenMPスレッド数
			std::vector<std::vector<std::vector<float>>>& output_result) const = 0;

		virtual void CustomCreateEngine(
			const std::string& modelFile,	// モデルファイルの拡張子を除く部分
			unsigned int maxBatchSize,		// バッチ数
			IHostMemory*& trtModelStream)
		{
			trtModelStream = nullptr;

		}; // output buffer for the TensorRT model

		void PreProcess(
			const std::vector<cv::Mat>& input_images, // BGR・3チャンネルの入力画像の配列
			const int super_batch_id = 0,
			const int thread_id = 0);

		void PostProcess(
			std::vector<std::vector<std::vector<float>>>& output_result,
			const int super_batch_id = 0,
			const int thread_id = 0);

		bool PreProcess_SuperBatch(
			const std::vector<cv::Mat>& input_images, // BGR・3チャンネルの入力画像の配列
			const int thread_id);

		bool PostProcess_SuperBatch(
			std::vector<std::vector<std::vector<float>>>& output_results, // 推論結果(std::vector<std::vector<float> >)の配列
			const int thread_id);

		ICudaEngine* GetEngine(
			const std::string& trt_filename,
			const std::string& model_filename,
			const int width,
			const int height);

		void OnnxToTRTModel(
			const std::string& modelFile,	// name of the onnx model
			unsigned int maxBatchSize,		// batch size - NB must be at least as large as the batch we want to run with
			IHostMemory*& trtModelStream);	// output buffer for the TensorRT model

		void UffToTRTModel(
			const std::string& modelFile,	// name of the MNIST model
			unsigned int maxBatchSize,		// batch size - NB must be at least as large as the batch we want to run with
			IHostMemory*& trtModelStream);	// output buffer for the TensorRT model

		void ClearSuperBatch(const int thread_id);
		//	void Release();

		ICudaEngine* m_trt_engine = nullptr;
		std::vector<std::vector<CPU_buffer>> m_buffers;					// vectorの各要素はinput/ouput用のバッファ[thread_id][super_batch_id]
		std::vector<std::vector<void*>> m_GPU_buffers;					// [super_batch_id][buffer_id]
		std::vector<std::vector<std::vector<void*>>> m_GPU_Out_buffers;	// [thread_id][super_batch_id][buffer_id]
		std::vector<std::vector<void*>> m_GPU_In_buffers;				// [thread_id][super_batch_id]

		std::vector<IExecutionContext*> m_contexts;						// [super_batch_id]
		std::vector<cudaStream_t> m_streams;							// [super_batch_id]

		std::vector<std::vector<std::vector<cv::Size>>> m_original_image_sizes; // [thread_id][super_batch_id][buffer_id]

		int m_inputIndex = -1;
		std::vector<int> m_outputIndex_list;
		std::vector<int> m_outputSize_list;
		int m_outputBuffer_num = -1; // m_outputBuffer_numと(int)m_outputIndex_list.size()とm_outputSize_list.size()は同一

		int m_width = -1;
		int m_height = -1;
		int m_batch = -1;
		int m_super_batch = -1;
		std::vector<int> m_current_super_batches; // [thread_id]
		int m_thread_num = -1;
		float* m_DummyBuff = NULL;
		int m_imgSize = -1;

		bool m_init_flg = false;
		bool m_is_pre_process_cuda = false;// trueの場合はPreProcessのCUDA版を使用
		bool m_is_post_process_cuda = false;// trueの場合はPostProcessのCUDA版を使用
		int m_gpu_id = 0;

		std::string m_inputIndex_name; // ONNXファイル内で指定された入力層の名前
		std::vector<std::string> m_outputIndex_names;
	};

} // namespace kros

#endif
