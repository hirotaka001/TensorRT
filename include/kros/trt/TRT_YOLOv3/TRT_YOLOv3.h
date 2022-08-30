#ifndef __TRT_YOLOV3_H__
#define __TRT_YOLOV3_H__

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/TRT/TRT.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "TRT_YOLOv3_PostProcess_cuda.h"

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	class TRT_YOLOv3 : public TRT
	{
	public:
		TRT_YOLOv3();
		~TRT_YOLOv3();

		bool Init(
			const int gpu_id,						// 使用するGPUのID（0以上の整数）
			const std::string& mode_name,			// モデルファイル（.onnx）の識別子
			const std::string& model_dir,			// モデルファイルとTRTエンジンの保存先ディレクトリ
			const int class_num,					// クラス数
			const int width,						// 入力層の横幅（任意のサイズの入力画像はInfer()時にこのサイズにリサイズされる）
			const int height,						// 入力層の横幅（任意のサイズの入力画像はInfer()時にこのサイズにリサイズされる）
			const int batch,						// バッチ数
			const int super_batch = 1,				// スーパーバッチ数（造語）
			const int thread_num = 1,				// OpenMPスレッド数
			const bool is_pre_process_cuda = true,	// PreprocessにCUDAを使用する場合はtrue（入力画像のサイズによっては遅くなる可能性あり）
			const float score_threshold = 0.5f,		// スコアがscore_threshold以上の検出結果のみを残す
			const float nms_threshold = 0.5f,		// 重複率（IoU：AND/OR）がnms_threshold以上の検出結果が一つに統合される、NMSを使用しない場合は0を指定
			const std::string& input_layers = "input1", 																	// ONNXファイル内で指定された入力層の名前
			const std::vector<std::string>& output_layers = std::vector<std::string>{ "output1", "output2", "output3" }); 	// ONNXファイル内で指定された出力層の名前

		bool InferParallel(
			const std::vector<cv::Mat>& input_images,							// BGR・3チャンネルの入力画像×画像枚数
			std::vector<std::vector<std::vector<float> > >& detection_result);	// (左上のx座標, 左上のy座標, 矩形の横幅, 矩形の縦幅, クラスID, スコア) の6列×画像中の検出数×画像枚数

		bool Infer(
			const std::vector<cv::Mat>& input_images,							// BGR・3チャンネルの入力画像×画像枚数
			std::vector<std::vector<std::vector<float> > >& detection_result,	// (左上のx座標, 左上のy座標, 矩形の横幅, 矩形の縦幅, クラスID, スコア) の6列×画像中の検出数×画像枚数
			const int thread_id = 0);											// OpenMPスレッドID（シングルスレッド時は0を指定）

		static bool DoNMS(
			const std::vector<std::vector<std::vector<float> > >& in_detection_result,	// NMS前の検出結果
			const float nms_threshold, 													// 重複率（IoU：AND/OR）がnms_threshold以上の検出結果が一つに統合される
			std::vector<std::vector<std::vector<float> > >& out_detection_result,		// NMS後の検出結果
			const bool sort_input = true,												//　NMS前に検出結果をスコアで昇順にソートするか否か
			const int overlap_method = 0);												// 0: A∩B/A∪B, 1: A∩B/B, 2: A∩B/A ※　スコアが高いB（配列の末尾）に対して、重複率の計算対象がA、最終的にAが消去される

		static void DrawResult(
			std::vector<cv::Mat>& images,											// 描画対象の画像
			const std::vector<std::vector<std::vector<float> > >& detection_result,	// 画像上に描画する検出結果
			const std::vector<std::string>& label_list);							// クラスラベルのリスト

		static void DrawResult(
			cv::Mat& image,												// 描画対象の画像
			const std::vector<std::vector<float> >& detection_result,	// 画像上に描画する検出結果
			const std::vector<std::string>& label_list);				// クラスラベルのリスト

		// インスタンスを残したまま可能な限りGPUメモリーを解放する
		// 再度利用する場合にはInit()が必要
		void Release();

	private:
		void PreProcess(
			const std::vector<cv::Mat>& input_images,
			TRT::CPU_buffer& buffer,
			float* GPU_In_buffers = nullptr,
			const cudaStream_t stream = nullptr,
			const int thread_id = 0) override;         //Preprosess_cudaでCUDAメモリの再アロケーションを行う為にconstは外した。

		void PostProcess(
			const TRT::CPU_buffer& buffer,
			const std::vector<void*>& gpu_buffers,
			const std::vector<cv::Size>& original_image_sizes,
			const int super_batch_id,
			const int thread_id,
			std::vector< std::vector<std::vector<float> > >& output_result) const override;

		void PostProcess_random(
			const TRT::CPU_buffer& buffer,
			const std::vector<void*>& gpu_buffers,
			const std::vector<cv::Size>& original_image_sizes,
			const int super_batch_id,
			std::vector< std::vector<std::vector<float> > >& output_result) const;

		void PostProcess_sequence(
			const TRT::CPU_buffer& buffer,
			const std::vector<void*>& gpu_buffers,
			const std::vector<cv::Size>& original_image_sizes,
			const int super_batch_id,
			std::vector< std::vector<std::vector<float> > >& output_result) const;

		std::vector<TRT_YOLOv3_PostProcess_cuda*> mp_TRT_YOLOv3_PostProcess_cuda;

		static cv::Scalar HSV2RGB(int classID);

		const std::vector<std::vector<int>> m_yolo_masks{ {6, 7, 8}, {3, 4, 5}, {0, 1, 2} };
		const std::vector<std::vector<int>> m_yolo_anchors{ {10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45}, {59, 119}, {116, 90}, {156, 198}, {373, 326} };

		float m_score_threshold = 0.0;
		float m_nms_threshold = 0.0;
		int m_class_num = -1;
		int m_batch = -1;
		int m_super_batch = -1;
		int m_base_grid_w = -1;
		int m_base_grid_h = -1;
		int m_thread_num = -1;
		std::vector<int> m_OutSize_list;

		struct st_Boxes {
			float box_confidence;
			std::vector<float> box_class_probs;
			std::vector<float> box_scores;
			float boxes[4];
		};

		//const のPostProcess内で使用される。
		int m_anchor_size[3];
		int m_grid_width[3];
		int m_grid_height[3];
		int m_GridSize[3];
		int m_OutAnchorSize[3];
		int m_UnitLen = -1;
		int m_OutUnitLen = -1;
		bool m_is_pre_process_cuda = false;
		bool m_is_post_process_cuda = true;
		std::vector<void*> mPreprocess_cuda_in_buffer;
		std::vector<int> mPreprocess_cuda_in_buffer_size;
		const float m_const_mean[3] = { 0.0f, 0.0f, 0.0f };
		const float m_const_std[3] = { 1.0f, 1.0f, 1.0f };
		int m_gpu_id;
	};

}

#endif
