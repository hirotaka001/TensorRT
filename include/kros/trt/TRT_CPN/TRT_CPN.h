#ifndef __TRT_CPN_H__
#define __TRT_CPN_H__

#include <kros/trt/TRT/TRT.h>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <vector>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	class TRT_CPN : public TRT
	{
	public:
		TRT_CPN();
		~TRT_CPN();

		bool Init(
			const int gpu_id,				// 使用するGPUのID（0以上の整数）
			const std::string& mode_name,	// *_w128_h256.onnxや*_w128_h256_b40_cc72.trtの*の部分
			const std::string& model_dir,	// *_w128_h256.onnxや*_w128_h256_b40_cc72.trtの保存先ディレクトリ
			const int width,				// 入力層の横幅（任意のサイズの入力画像はInfer()時にこのサイズにリサイズされる）
			const int height,				// 入力層の縦幅（任意のサイズの入力画像はInfer()時にこのサイズにリサイズされる）
			const int batch,				// バッチ数
			const int super_batch = 1,		// スーパーバッチ数（造語）
			const int thread_num = 1,		// OpenMPスレッド数
			const int resize_margin = 10,	// リサイズ後の上下左右の最小マージン [px]
			const bool keep_aspect = true,	// trueならアスペクトを維持したままリサイズ
			const std::string& input_layers = "Placeholder",
			const std::string& output_layers = "refine_out/Conv2D");

		bool InferParallel(
			const std::vector<cv::Mat>& input_images,							// BGR・3チャンネルの入力画像×画像枚数
			std::vector<std::vector<std::vector<float> > >& detection_result);	// 関節毎の(x座標, y座標, スコア)の3列×関節の17点×画像枚数

		bool Infer(
			const std::vector<cv::Mat>& input_images,							// BGR・3チャンネルの入力画像×画像枚数
			std::vector<std::vector<std::vector<float> > >& detection_result,	// 関節毎の(x座標, y座標, スコア)の3列×関節の17点×画像枚数
			const int thread_id = 0);											// OpenMPスレッドID（シングルスレッド時は0を指定）

		void DrawResult(
			const cv::Mat& input_images,								// 描画対象の入力画像
			cv::Mat& output_images,										// 描画後の出力画像
			const std::vector<std::vector<float> >& detection_result,	// 画像上に描画する検出結果
			const bool print_keypoint_score = true);					// truneなら画像の右側領域に各関節の推論スコアを描画 

		void DrawResult(
			const std::vector<cv::Mat>& input_images,								// 描画対象の入力画像
			std::vector<cv::Mat>& output_images,									// 描画後の出力画像
			const std::vector<std::vector<std::vector<float> > >& detection_result,	// 画像上に描画する検出結果
			bool print_keypoint_score = true);										// truneなら画像の右側領域に各関節の推論スコアを描画 

		static std::vector<float> GetKeyPoint(
			const int keypoint_idx,										// 関節のインデックス(kros::TRT_CPN::NOSEなど)
			const std::vector<std::vector<float> >& detection_results); // 17点分の関節情報

		// インスタンスを残したまま可能な限りGPUメモリーを解放する
		// 再度利用する場合にはInit()が必要
		void Release();

		enum KEYPOINTS {
			NOSE = 0,
			L_EYE = 1,
			R_EYE = 2,
			L_EAR = 3,
			R_EAR = 4,
			L_SHOULDER = 5,
			R_SHOULDER = 6,
			L_ELBOW = 7,
			R_ELBOW = 8,
			L_WRIST = 9,
			R_WRIST = 10,
			L_HIP = 11,
			R_HIP = 12,
			L_KNEE = 13,
			R_KNEE = 14,
			L_ANKLE = 15,
			R_ANKLE = 16
		};

	private:

		void PreProcess(
			const std::vector<cv::Mat>& input_images,
			TRT::CPU_buffer& buffer,
			float* GPU_In_buffers = nullptr,
			cudaStream_t stream = nullptr,
			const int thread_id = 0) override;  //Preprosess_cudaでCUDAメモリの再アロケーションを行う為にconstは外した。

		void PostProcess(
			const TRT::CPU_buffer& buffer,
			const std::vector<void*>& gpu_buffers,
			const std::vector<cv::Size>& original_image_sizes,
			const int super_batch_id,
			const int thread_id,								// OpenMPスレッド数
			std::vector< std::vector<std::vector<float> > >& detection_result) const override;

		void keep_aspect_resize(
			const cv::Mat& inImg,
			cv::Mat& outImg) const;

		//オリジナル画像のkeypoint位置へ変換
		void calc_org_img_pos(
			const std::vector<cv::Size>& original_image_sizes,
			std::vector< std::vector<std::vector<float> > >& detection_result) const;

		//内部処理用サイズへの変換
		void calc_resize_size(
			const cv::Size& OrgSize,
			cv::Rect& ResizeSize) const;

		void DrawKeyPoints(const cv::Mat& input_images, const std::vector<std::vector<float> >& detection_result, cv::Mat& output_images) const;

		void DrawPointList(const cv::Mat& input_images, const std::vector<std::vector<float> >& detection_result, cv::Mat& output_images) const;

		const std::vector<float> m_means{ 102.9801f, 115.9465f, 122.7717f };
		int m_output_c;
		int m_output_h;
		int m_output_w;

		enum CPNInferResultParseMethod {
			HIGHEST_POINT = 1,
			SECOND_POINT_MODIFICATION = 2,
			GAUSSIAN_FILTER_AND_SECOND_POINT_MODIFICATION = 3
		};

		CPNInferResultParseMethod m_method;
		const float m_delta = 0.25;
		const float m_min_value_of_cpn_output_modfiy_by_second_highest = 0.001f;
		int  m_resize_margin = 10;
		bool m_keep_aspect = true;
		int m_inside_w;
		int m_inside_h;
		float m_aspect;
		cv::Mat m_bachImg;
		int m_batch = -1;
		int m_super_batch = -1;
		int m_thread_num = -1;

		const std::vector<std::string> m_keyname = {
		   "NOSE",
		   "L_EYE",
		   "R_EYE",
		   "L_EAR",
		   "R_EAR",
		   "L_SHOULDER",
		   "R_SHOULDER",
		   "L_ELBOW",
		   "R_ELBOW",
		   "L_WRIST",
		   "R_WRIST",
		   "L_HIP",
		   "R_HIP",
		   "L_KNEE",
		   "R_KNEE",
		   "L_ANKLE",
		   "R_ANKLE" };

		// 17種類の色コード（BGR）HSVでHを360/17した物をシャッフルした物。
		// 並び順はBGR
		const int m_keycol[17][3] = {
				{0, 89, 255},    //NOSE
				{0, 179, 255},   //L_EYE
				{29, 255, 0},    //R_EYE
				{255, 0, 239},   //L_EAR
				{119, 60,128},   //R_EAR
				{255, 30, 0},    //L_SHOULDER
				{255, 0, 255},   //R_SHOULDER
				{255,78,129},    //L_ELBOW
				{100, 0, 200},   //R_ELBOW
				{128, 255, 150}, //L_WRIST
				{200, 128, 0},   //R_WRIST
				{32,103,154},    //L_HIP
				{60 , 210, 0},   //R_HIP
				{150, 100, 60},  //L_KNEE
				{255, 0, 159},   //R_KNEE
				{0, 0, 255},     //L_ANKLE
				{173,107,0} };    //R_ANKLE

		const int m_linecol[3][3] = {
				{0,  255, 255},   //Right
				{255,  0, 255},   //Left
				{255,255,   0} };  //Other

		const int m_link_line[18][3] = {
				{R_EAR,R_EYE,0},
				{R_EYE,NOSE,0},
				{NOSE,L_EYE,1},
				{L_EYE,L_EAR,1},
				{R_EAR,R_SHOULDER,0},
				{L_EAR,L_SHOULDER,1},
				{R_SHOULDER,L_SHOULDER,2},
				{R_SHOULDER,R_ELBOW,0},
				{R_ELBOW,R_WRIST,0},
				{L_SHOULDER,L_ELBOW,1},
				{L_ELBOW,L_WRIST,1},
				{R_SHOULDER,R_HIP,0},
				{R_HIP,R_KNEE,0},
				{R_KNEE,R_ANKLE,0},
				{L_SHOULDER,L_HIP,1},
				{L_HIP,L_KNEE,1},
				{L_KNEE,L_ANKLE,1},
				{R_HIP,L_HIP,2} };
	};
}
#endif
