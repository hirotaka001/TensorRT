#ifndef __TRT_RESNET50_FEATURE_H__
#define __TRT_RESNET50_FEATURE_H__

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/TRT/TRT.h>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	class TRT_ResNet50_Feature : public TRT
	{
	public:
		TRT_ResNet50_Feature();
		~TRT_ResNet50_Feature();

		bool Init(
			const int gpu_id,								// 使用するGPUのID（0以上の整数）
			const std::string& mode_name,					// モデルファイル（.onnx）の識別子
			const std::string& model_dir,					// モデルファイルとTRTエンジンの保存先ディレクトリ
			const int width,								// 入力層の横幅（任意のサイズの入力画像はInfer()時にこのサイズにリサイズされる）
			const int height,								// 入力層の横幅（任意のサイズの入力画像はInfer()時にこのサイズにリサイズされる）
			const int batch,								// バッチ数
			const int super_batch = 1,						// スーパーバッチ数（造語）
			const int thread_num = 1,						// OpenMPスレッド数
			const bool is_pre_process_cuda = true,			// PreprocessにCUDAを使用する場合はtrue（入力画像のサイズによっては遅くなる可能性あり）
			const std::string& input_layers = "0",			// ONNXファイル内で指定された入力層の名前
			const std::string& output_layers = "441");		// ONNXファイル内で指定された出力層の名前

		bool InferParallel(
			const std::vector<cv::Mat>& input_images,			// BGR・3チャンネルの入力画像×画像枚数
			std::vector<std::vector<float> >& output_features);	// 2048次元の特徴ベクトルの配列

		bool Infer(
			const std::vector<cv::Mat>& input_images,			// BGR・3チャンネルの入力画像の配列
			std::vector<std::vector<float> >& output_features,	// 2048次元の特徴ベクトルの配列
			const int thread_id = 0);

		static void PrintResult(
			const std::vector<float>& output_features);
		// インスタンスを残したまま可能な限りGPUメモリーを解放する
		// 再度利用する場合にはInit()が必要
		void Release();

	private:
		void PreProcess(
			const std::vector<cv::Mat>& input_images,
			TRT::CPU_buffer& buffer,
			float* GPU_In_buffers = nullptr,
			cudaStream_t stream = nullptr,
			const int thread_id = 0) override;         //Preprosess_cudaでCUDAメモリの再アロケーションを行う為にconstは外した。

		void PostProcess(
			const TRT::CPU_buffer& buffer,
			const std::vector<void*>& gpu_buffers,
			const std::vector<cv::Size>& original_image_sizes,
			const int super_batch_id,
			const int thread_id,
			std::vector< std::vector<std::vector<float> > >& output_result) const override;

		int m_batch = -1;
		int m_super_batch = -1;
		int m_thread_num = -1;
		bool m_is_pre_process_cuda = true;
		bool m_is_post_process_cuda = false;
		std::vector<void*> mPreprocess_cuda_in_buffer;
		std::vector<int> mPreprocess_cuda_in_buffer_size;
		const float m_const_mean[3] = { 0.485f, 0.456f, 0.406f };
		const float m_const_std[3] = { 0.229f, 0.224f, 0.225f };
		int m_gpu_id;
	};
}

#endif
