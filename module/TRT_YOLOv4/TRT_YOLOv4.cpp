#define TAG "TRT_YOLOv4"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/TRT_YOLOv4/TRT_YOLOv4.h>
#include <kros/trt/Common/PreProcess_cuda.h>
#include <kros/trt/Common/utils.hpp>
#include <kros/trt/Common/nms.hpp>
#include <omp.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "yololayer.h"
#include "mish.h"

#ifdef _WIN32
#pragma warning(pop)
#endif

#define SIGMOID(val) \
		(1.0 / (1.0 + std::exp(-(val))))

#define EXPONENTIAL(val) \
		std::exp(val)

REGISTER_TENSORRT_PLUGIN(MishPluginCreator);
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

//実体は、yololayer.cu内にある。yololayerのプラグインは、tensorRTの中でインスタンスが動的に作られる為、
//画像サイズを設定するには、インスタンスが作られる前に決定している必要があり、苦肉の策
extern int g_width;
extern int g_height;
extern int g_class_num;

namespace kros
{
	static Logger gLogger;

	TRT_YOLOv4::TRT_YOLOv4()
	{
	}

	TRT_YOLOv4::~TRT_YOLOv4()
	{
		Release();
	}

	bool TRT_YOLOv4::Init(
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
		const float score_threshold,
		const float nms_threshold,
		const std::string& input_layers,
		const std::string& output_layers)
	{
		m_width = width;
		m_height = height;
		g_width = width;
		g_height = height;
		g_class_num = class_num;

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

		m_input_layers = input_layers;
		m_output_layers.push_back(output_layers);    //YOLOv3に合わせる為（複数出力レイヤー）にvectorに積む
		int nOutLayers = (int)m_output_layers.size();

		if (width % 16 != 0 || height % 16 != 0) {
			std::cout << "[" << TAG << "] Error : the both of width and height must be divided by 16" << std::endl;
			return false;
		}

		if (nOutLayers != 1) {
			std::cout << "[" << TAG << "] Error : the number of output layers must be 1 (" << nOutLayers << ")" << std::endl;
			return false;
		}
		m_class_num = class_num;
		m_score_threshold = score_threshold;
		m_nms_threshold = nms_threshold;
		m_batch = batch;
		m_super_batch = super_batch;
		m_thread_num = thread_num;

		//PreprocessのCUDA版で使用する推論用入力パッファとバッファサイズの記録領域
		mPreprocess_cuda_in_buffer.clear();
		mPreprocess_cuda_in_buffer_size.clear();

		if (m_is_pre_process_cuda)
		{
			//PReprocssのCUDA版で使用する正規化係数のセット
			SetConstantValue(m_const_mean, m_const_std);
			//thread単位の入力GPUバッファ
			for (int i = 0; i < m_thread_num; i++)
			{
				mPreprocess_cuda_in_buffer.push_back(nullptr);
				mPreprocess_cuda_in_buffer_size.push_back(0);
			}
		}
		//                std::cout << "TRT_YOLOv4::Init SetConstantValue" << std::endl;
		//               sleep(10);


#ifdef _OPENMP
		if (m_thread_num > omp_get_num_procs()) {
			std::cout << "[" << TAG << "] Error : thread_num must be set within the number of processors" << std::endl;
			return false;
		}
		omp_set_num_threads(m_thread_num);
#endif
		//クラス数を付与
		std::string model_name2 = model_name + "_c" + std::to_string(class_num);

		m_detection_size = sizeof(Yolo::Detection) / sizeof(float);
		m_output_size = Yolo::MAX_OUTPUT_BBOX_COUNT * m_detection_size + 1;  // we assume 
		m_outputSize_list.push_back(m_output_size);
		return TRT::Init(gpu_id, model_name2, model_dir, m_input_layers, m_output_layers, width, height, batch,
			super_batch, thread_num, m_is_post_process_cuda, m_is_pre_process_cuda);
	}

	void TRT_YOLOv4::Release() {

		int num_buffer = (int)mPreprocess_cuda_in_buffer.size();

		for (int i = 0; i < num_buffer; i++)
		{
			cudaFree(mPreprocess_cuda_in_buffer[i]);
		}
		mPreprocess_cuda_in_buffer.clear();
		TRT::Release();
	}

	bool TRT_YOLOv4::InferParallel(
		const std::vector<cv::Mat>& input_images,
		std::vector<std::vector<std::vector<float> > >& detection_result)  //nImage, nBOX, 6Items
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

		std::vector<std::vector<std::vector<std::vector<float> > > >detection_result_(num_infer);

#pragma omp parallel for num_threads(m_thread_num) schedule(static)
		for (int loop_idx = 0; loop_idx < num_infer; loop_idx++) {    //num_inferは何回Inferを行うか。Infer　1回はバッチ分一気に処理
			int thread_id = (int)omp_get_thread_num();
			Infer(infer_input_images[loop_idx], detection_result_[loop_idx], thread_id);
		}


		for (int loop_idx = 0; loop_idx < num_infer; loop_idx++) {
			int nFrm = (int)detection_result_[loop_idx].size();
			for (int FrmNo = 0; FrmNo < nFrm; FrmNo++) {
				detection_result.push_back(detection_result_[loop_idx][FrmNo]);
			}
		}

		return true;
	}

	bool TRT_YOLOv4::Infer(
		const std::vector<cv::Mat>& input_images,
		std::vector<std::vector<std::vector<float> > >& detection_result,
		const int thread_id)
	{
		if (m_nms_threshold > 0) {
			std::vector<std::vector<std::vector<float> > > mid_detection_result;
			bool ret = TRT::Infer(input_images, mid_detection_result, thread_id);

			if (ret) {
				DoNMS(mid_detection_result, m_nms_threshold, detection_result, true, 0);
				return true;
			}
			else {
				return false;
			}
		}
		else {
			bool ret = TRT::Infer(input_images, detection_result, thread_id);
			if (ret) {
				return true;
			}
			else {
				return false;
			}
		}
		return false;
	}

	bool TRT_YOLOv4::DoNMS(
		const std::vector<std::vector<std::vector<float> > >& in_detection_result,
		const float nms_threshold, //メンバ変数にあるので、この引数無くてもOKだが、将来Infer単位に可変させる事があるかも知れないので、この引数は残す。
		std::vector<std::vector<std::vector<float> > >& out_detection_result,
		const bool sort_input,
		const int overlap_method)
	{
		int num_frame = (int)in_detection_result.size();
		for (int i = 0; i < num_frame; i++) {
#if 0
			//クラス単位版
			std::vector<std::vector<float>> boxes = in_detection_result[i];
			std::vector<std::vector<float>> class_boxes;
			std::vector<std::vector<float>> acc_reducedRectangle;
			while (boxes.size()) {
				auto itr = boxes.begin();
				class_boxes.clear();
				float classno = (*itr)[4];
				while (itr != boxes.end()) {
					std::vector<float> box = *itr;
					if (classno == box[4]) {
						class_boxes.push_back(box);
						boxes.erase(itr);
					}
					else {
						itr++;
					}
				}
				std::vector<std::vector<float> > reducedRectangle = nms(class_boxes, nms_threshold);
				acc_reducedRectangle.insert(acc_reducedRectangle.end(), reducedRectangle.begin(), reducedRectangle.end());
			}
			out_detection_result.push_back(acc_reducedRectangle);
#else
			//オリジナル版若しくはスコアソート版 (nms.cpp内で切り替える)
			std::vector<std::vector<float> > reducedRectangle = nms(in_detection_result[i], nms_threshold, sort_input, overlap_method);
			out_detection_result.push_back(reducedRectangle);
#endif
		}

		return true;
	}

	void TRT_YOLOv4::DrawResult(
		std::vector<cv::Mat>& images,
		const std::vector<std::vector<std::vector<float>>>& detection_result,
		const std::vector<std::string>& label_list)
	{
		CV_Assert((int)images.size() == (int)detection_result.size());
		for (int i = 0; i < (int)images.size(); i++) {
			TRT_YOLOv4::DrawResult(images[i], detection_result[i], label_list);
		}
	}

	void TRT_YOLOv4::DrawResult(
		cv::Mat& image,
		const std::vector<std::vector<float>>& detection_result,
		const std::vector<std::string>& label_list)
	{
		const int box_num = (int)detection_result.size();
		const int label_num = (int)label_list.size();

		for (int i = 0; i < box_num; i++) {
			cv::Scalar Color = HSV2RGB((int)detection_result[i][4]);

			cv::Point st_Point{ (int)detection_result[i][0], (int)detection_result[i][1] };
			cv::Point ed_Point{
				(int)detection_result[i][0] + (int)detection_result[i][2],
				(int)detection_result[i][1] + (int)detection_result[i][3] };
			cv::rectangle(image, st_Point, ed_Point, Color, 1);

			std::ostringstream ss;
			if (label_num > (int)detection_result[i][4]) {
				ss << label_list[(int)detection_result[i][4]];
			}
			else {
				ss << std::setw(2) << std::setfill('0') << (int)detection_result[i][4];
			}
			ss << " " << std::setprecision(3) << detection_result[i][5];
			std::string class_label(ss.str());

			int baseline = 0;
			double fontSize = 0.5;
			cv::Size txtSize = cv::getTextSize(class_label, cv::FONT_HERSHEY_SIMPLEX, fontSize, 1, &baseline);
			cv::Point st_rectPoint{ (int)detection_result[i][0], (int)detection_result[i][1] - txtSize.height - 6 };
			cv::Point ed_rectPoint{ (int)detection_result[i][0] + txtSize.width, (int)detection_result[i][1] };
			cv::rectangle(image, st_rectPoint, ed_rectPoint, Color, -1);

			cv::Point st_txtPoint{ st_Point.x, st_Point.y - 3 };
			//		cv::Scalar txtColor {255-Color.val[0], 255-Color.val[1], 255-Color.val[2] };  //文字は反対色にしてみたが、思った程見やすくはならなかった。

			cv::putText(image, class_label, st_txtPoint, cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
		}
	}

	cv::Scalar TRT_YOLOv4::HSV2RGB(int classID) {
		int R;
		int G;
		int B;
		int v = 150;
		int s = 255;
		int max = v;
		int min = max - (int)((s / 255.0f) * max);

		int H = (classID * 28) % 360;
		if (H <= 60) {
			R = max;
			G = (int)(((float)H / 60.0f) * (max - min) + min);
			B = min;
		}
		else if (H <= 120) {
			R = (int)(((float)(120 - H) / 60.0f) * (max - min) + min);
			G = max;
			B = min;
		}
		else if (H <= 180) {
			R = min;
			G = max;
			B = (int)(((float)(H - 120) / 60.0f) * (max - min) + min);
		}
		else if (H <= 240) {
			R = min;
			G = (int)(((float)(240 - H) / 60.0f) * (max - min) + min);
			B = max;
		}
		else if (H <= 300) {
			R = (int)(((float)(H - 240) / 60.0f) * (max - min) + min);
			G = min;
			B = max;
		}
		else {
			R = max;
			G = min;
			B = (int)(((float)(360 - H) / 60.0f) * 255.0f);
		}
		return 	cv::Scalar(R, G, B);
	}

	void TRT_YOLOv4::PreProcess(
		const std::vector<cv::Mat>& input_images,
		TRT::CPU_buffer& buffer,
		float* GPU_In_buffers,
		cudaStream_t stream,
		const int thread_id)
	{
		const int NumInputData = (int)input_images.size();
		buffer.numInput = NumInputData;   //GPUを使うケースであっても、buffer.numInputだけは流用する点に注意！

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
				cv::resize(input_images[i], resized, cv::Size(m_width, m_height), cv::INTER_AREA);

				// 3つのチャネルB, G, Rに分離。
				cv::split(resized, planes);
				//正規化 （8bit実数から32bit整数変換も同時に行う。
				planes[0].convertTo(planeB, CV_32F, 1.0 / 255); //B
				planes[1].convertTo(planeG, CV_32F, 1.0 / 255); //G
				planes[2].convertTo(planeR, CV_32F, 1.0 / 255); //R

				//1次元配列へ変換。Channel firstで、RGBの順番に変更
				memcpy(inbufpos, planeR.data, planeR.total() * planeR.elemSize());
				inbufpos += planeR.total();

				memcpy(inbufpos, planeG.data, planeG.total() * planeR.elemSize());
				inbufpos += planeG.total();

				memcpy(inbufpos, planeB.data, planeB.total() * planeR.elemSize());
				inbufpos += planeB.total();

				//1次元配列へ変換。Channel firstで、RGBの順番に変更
			}
		}
	}

	void TRT_YOLOv4::PostProcess(
		const TRT::CPU_buffer& buffer,
		const std::vector<void*>& gpu_buffers,
		const std::vector<cv::Size>& original_image_sizes,
		const int super_batch_id,
		const int thread_id,
		std::vector< std::vector<std::vector<float> > >& output_result) const
	{

		//閾値以上の物に絞る
		for (int batch_idx = 0; batch_idx < buffer.numInput; batch_idx++) {
			int numBBOXs = (int)buffer.out_buffer[0][batch_idx * m_output_size];
			std::vector<std::vector<float>> ResBoxes;
			for (int i = 0; i < numBBOXs; i++) {
				Yolo::Detection det;
				memcpy(&det, &buffer.out_buffer[0][batch_idx * m_output_size + 1 + m_detection_size * i], m_detection_size * sizeof(float));
				if (det.class_confidence >= m_score_threshold) {
					//リサイズに寄るオリジナルサイズとの比計算
					float r_w = (float)original_image_sizes[batch_idx].width / (float)m_width;
					float r_h = (float)original_image_sizes[batch_idx].height / (float)m_height;

					//YOLOv4の推論結果は、BBOXのセンター座標が返るので、一旦左上と右下の座標を求める
					float left = (float)det.bbox[0] - (float)det.bbox[2] / 2.f;
					float right = (float)det.bbox[0] + (float)det.bbox[2] / 2.f;
					float top = (float)det.bbox[1] - (float)det.bbox[3] / 2.f;
					float bottom = (float)det.bbox[1] + (float)det.bbox[3] / 2.f;
					//リサイズ比を掛けてX,Y,W,Hを求める
					float x = left * r_w;
					float y = top * r_h;
					float w = (right - left) * r_w;
					float h = (bottom - top) * r_h;

					//TRTのフォーマットで出力バッファにセット
					std::vector<float> ResBox;
					ResBox.push_back(x);
					ResBox.push_back(y);
					ResBox.push_back(w);
					ResBox.push_back(h);
					ResBox.push_back((float)det.class_id);
					ResBox.push_back(det.class_confidence);
					ResBoxes.push_back(ResBox);
				}
			}
			output_result.push_back(ResBoxes);
		}
	}

	// TensorRT weight files have a simple space delimited format:
	// [type] [size] <data x size in hex>
	std::map<std::string, Weights> TRT_YOLOv4::loadWeights(const std::string file)
	{
		std::map<std::string, Weights> weightMap;

		// Open weights file
		std::ifstream input(file);
		assert(input.is_open() && "Unable to load weight file.");

		// Read number of weight blobs
		int32_t count;
		input >> count;
		assert(count > 0 && "Invalid weight map file.");

		while (count--)
		{
			Weights wt{ DataType::kFLOAT, nullptr, 0 };
			uint32_t size;

			// Read name and type of blob
			std::string name;
			input >> name >> std::dec >> size;
			wt.type = DataType::kFLOAT;

			// Load blob
			uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
			for (uint32_t x = 0, y = size; x < y; ++x)
			{
				input >> std::hex >> val[x];
			}
			wt.values = val;

			wt.count = size;
			weightMap[name] = wt;
		}
		return weightMap;
	}

	IScaleLayer* TRT_YOLOv4::addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
	{
		float* gamma = (float*)weightMap[lname + ".weight"].values;
		float* beta = (float*)weightMap[lname + ".bias"].values;
		float* mean = (float*)weightMap[lname + ".running_mean"].values;
		float* var = (float*)weightMap[lname + ".running_var"].values;
		int len = (int)weightMap[lname + ".running_var"].count;

		float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
		for (int i = 0; i < len; i++) {
			scval[i] = gamma[i] / sqrt(var[i] + eps);
		}
		Weights scale{ DataType::kFLOAT, scval, len };

		float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
		for (int i = 0; i < len; i++) {
			shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
		}
		Weights shift{ DataType::kFLOAT, shval, len };

		float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
		for (int i = 0; i < len; i++) {
			pval[i] = 1.0;
		}
		Weights power{ DataType::kFLOAT, pval, len };

		weightMap[lname + ".scale"] = scale;
		weightMap[lname + ".shift"] = shift;
		weightMap[lname + ".power"] = power;
		IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
		assert(scale_1);
		return scale_1;
	}


	ILayer* TRT_YOLOv4::convBnMish(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx)
	{
		Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
		IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
		assert(conv1);
		conv1->setStrideNd(DimsHW{ s, s });
		conv1->setPaddingNd(DimsHW{ p, p });

		IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", (float)1e-4);

		auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
		const PluginFieldCollection* pluginData = creator->getFieldNames();
		IPluginV2* pluginObj = creator->createPlugin(("mish" + std::to_string(linx)).c_str(), pluginData);
		ITensor* inputTensors[] = { bn1->getOutput(0) };
		auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
		return mish;
	}

	ILayer* TRT_YOLOv4::convBnLeaky(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx)
	{
		Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
		IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
		assert(conv1);
		conv1->setStrideNd(DimsHW{ s, s });
		conv1->setPaddingNd(DimsHW{ p, p });

		IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", (float)1e-4);

		auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
		lr->setAlpha(0.1f);

		return lr;
	}


	void TRT_YOLOv4::CustomCreateEngine(
		const std::string& modelFile,  // name of the wts model
		unsigned int maxBatchSize,	 // batch size - NB must be at least as large as the batch we want to run with
		IHostMemory*& trtModelStream)
	{
		IBuilder* builder = createInferBuilder(gLogger);

		INetworkDefinition* network = builder->createNetworkV2(0U);

		ITensor* data = network->addInput(m_input_layers.c_str(), DataType::kFLOAT, Dims3{ 3, m_height, m_width });

		assert(data);

		std::map<std::string, Weights> weightMap = loadWeights(modelFile);
		Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

		// define each layer.
		auto l0 = convBnMish(network, weightMap, *data, 32, 3, 1, 1, 0);
		auto l1 = convBnMish(network, weightMap, *l0->getOutput(0), 64, 3, 2, 1, 1);
		auto l2 = convBnMish(network, weightMap, *l1->getOutput(0), 64, 1, 1, 0, 2);
		auto l3 = l1;
		auto l4 = convBnMish(network, weightMap, *l3->getOutput(0), 64, 1, 1, 0, 4);
		auto l5 = convBnMish(network, weightMap, *l4->getOutput(0), 32, 1, 1, 0, 5);
		auto l6 = convBnMish(network, weightMap, *l5->getOutput(0), 64, 3, 1, 1, 6);
		auto ew7 = network->addElementWise(*l6->getOutput(0), *l4->getOutput(0), ElementWiseOperation::kSUM);
		auto l8 = convBnMish(network, weightMap, *ew7->getOutput(0), 64, 1, 1, 0, 8);

		ITensor* inputTensors9[] = { l8->getOutput(0), l2->getOutput(0) };
		auto cat9 = network->addConcatenation(inputTensors9, 2);

		auto l10 = convBnMish(network, weightMap, *cat9->getOutput(0), 64, 1, 1, 0, 10);
		auto l11 = convBnMish(network, weightMap, *l10->getOutput(0), 128, 3, 2, 1, 11);
		auto l12 = convBnMish(network, weightMap, *l11->getOutput(0), 64, 1, 1, 0, 12);
		auto l13 = l11;
		auto l14 = convBnMish(network, weightMap, *l13->getOutput(0), 64, 1, 1, 0, 14);
		auto l15 = convBnMish(network, weightMap, *l14->getOutput(0), 64, 1, 1, 0, 15);
		auto l16 = convBnMish(network, weightMap, *l15->getOutput(0), 64, 3, 1, 1, 16);
		auto ew17 = network->addElementWise(*l16->getOutput(0), *l14->getOutput(0), ElementWiseOperation::kSUM);
		auto l18 = convBnMish(network, weightMap, *ew17->getOutput(0), 64, 1, 1, 0, 18);
		auto l19 = convBnMish(network, weightMap, *l18->getOutput(0), 64, 3, 1, 1, 19);
		auto ew20 = network->addElementWise(*l19->getOutput(0), *ew17->getOutput(0), ElementWiseOperation::kSUM);
		auto l21 = convBnMish(network, weightMap, *ew20->getOutput(0), 64, 1, 1, 0, 21);

		ITensor* inputTensors22[] = { l21->getOutput(0), l12->getOutput(0) };
		auto cat22 = network->addConcatenation(inputTensors22, 2);

		auto l23 = convBnMish(network, weightMap, *cat22->getOutput(0), 128, 1, 1, 0, 23);
		auto l24 = convBnMish(network, weightMap, *l23->getOutput(0), 256, 3, 2, 1, 24);
		auto l25 = convBnMish(network, weightMap, *l24->getOutput(0), 128, 1, 1, 0, 25);
		auto l26 = l24;
		auto l27 = convBnMish(network, weightMap, *l26->getOutput(0), 128, 1, 1, 0, 27);
		auto l28 = convBnMish(network, weightMap, *l27->getOutput(0), 128, 1, 1, 0, 28);
		auto l29 = convBnMish(network, weightMap, *l28->getOutput(0), 128, 3, 1, 1, 29);
		auto ew30 = network->addElementWise(*l29->getOutput(0), *l27->getOutput(0), ElementWiseOperation::kSUM);
		auto l31 = convBnMish(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
		auto l32 = convBnMish(network, weightMap, *l31->getOutput(0), 128, 3, 1, 1, 32);
		auto ew33 = network->addElementWise(*l32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
		auto l34 = convBnMish(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
		auto l35 = convBnMish(network, weightMap, *l34->getOutput(0), 128, 3, 1, 1, 35);
		auto ew36 = network->addElementWise(*l35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
		auto l37 = convBnMish(network, weightMap, *ew36->getOutput(0), 128, 1, 1, 0, 37);
		auto l38 = convBnMish(network, weightMap, *l37->getOutput(0), 128, 3, 1, 1, 38);
		auto ew39 = network->addElementWise(*l38->getOutput(0), *ew36->getOutput(0), ElementWiseOperation::kSUM);
		auto l40 = convBnMish(network, weightMap, *ew39->getOutput(0), 128, 1, 1, 0, 40);
		auto l41 = convBnMish(network, weightMap, *l40->getOutput(0), 128, 3, 1, 1, 41);
		auto ew42 = network->addElementWise(*l41->getOutput(0), *ew39->getOutput(0), ElementWiseOperation::kSUM);
		auto l43 = convBnMish(network, weightMap, *ew42->getOutput(0), 128, 1, 1, 0, 43);
		auto l44 = convBnMish(network, weightMap, *l43->getOutput(0), 128, 3, 1, 1, 44);
		auto ew45 = network->addElementWise(*l44->getOutput(0), *ew42->getOutput(0), ElementWiseOperation::kSUM);
		auto l46 = convBnMish(network, weightMap, *ew45->getOutput(0), 128, 1, 1, 0, 46);
		auto l47 = convBnMish(network, weightMap, *l46->getOutput(0), 128, 3, 1, 1, 47);
		auto ew48 = network->addElementWise(*l47->getOutput(0), *ew45->getOutput(0), ElementWiseOperation::kSUM);
		auto l49 = convBnMish(network, weightMap, *ew48->getOutput(0), 128, 1, 1, 0, 49);
		auto l50 = convBnMish(network, weightMap, *l49->getOutput(0), 128, 3, 1, 1, 50);
		auto ew51 = network->addElementWise(*l50->getOutput(0), *ew48->getOutput(0), ElementWiseOperation::kSUM);
		auto l52 = convBnMish(network, weightMap, *ew51->getOutput(0), 128, 1, 1, 0, 52);

		ITensor* inputTensors53[] = { l52->getOutput(0), l25->getOutput(0) };
		auto cat53 = network->addConcatenation(inputTensors53, 2);

		auto l54 = convBnMish(network, weightMap, *cat53->getOutput(0), 256, 1, 1, 0, 54);
		auto l55 = convBnMish(network, weightMap, *l54->getOutput(0), 512, 3, 2, 1, 55);
		auto l56 = convBnMish(network, weightMap, *l55->getOutput(0), 256, 1, 1, 0, 56);
		auto l57 = l55;
		auto l58 = convBnMish(network, weightMap, *l57->getOutput(0), 256, 1, 1, 0, 58);
		auto l59 = convBnMish(network, weightMap, *l58->getOutput(0), 256, 1, 1, 0, 59);
		auto l60 = convBnMish(network, weightMap, *l59->getOutput(0), 256, 3, 1, 1, 60);
		auto ew61 = network->addElementWise(*l60->getOutput(0), *l58->getOutput(0), ElementWiseOperation::kSUM);
		auto l62 = convBnMish(network, weightMap, *ew61->getOutput(0), 256, 1, 1, 0, 62);
		auto l63 = convBnMish(network, weightMap, *l62->getOutput(0), 256, 3, 1, 1, 63);
		auto ew64 = network->addElementWise(*l63->getOutput(0), *ew61->getOutput(0), ElementWiseOperation::kSUM);
		auto l65 = convBnMish(network, weightMap, *ew64->getOutput(0), 256, 1, 1, 0, 65);
		auto l66 = convBnMish(network, weightMap, *l65->getOutput(0), 256, 3, 1, 1, 66);
		auto ew67 = network->addElementWise(*l66->getOutput(0), *ew64->getOutput(0), ElementWiseOperation::kSUM);
		auto l68 = convBnMish(network, weightMap, *ew67->getOutput(0), 256, 1, 1, 0, 68);
		auto l69 = convBnMish(network, weightMap, *l68->getOutput(0), 256, 3, 1, 1, 69);
		auto ew70 = network->addElementWise(*l69->getOutput(0), *ew67->getOutput(0), ElementWiseOperation::kSUM);
		auto l71 = convBnMish(network, weightMap, *ew70->getOutput(0), 256, 1, 1, 0, 71);
		auto l72 = convBnMish(network, weightMap, *l71->getOutput(0), 256, 3, 1, 1, 72);
		auto ew73 = network->addElementWise(*l72->getOutput(0), *ew70->getOutput(0), ElementWiseOperation::kSUM);
		auto l74 = convBnMish(network, weightMap, *ew73->getOutput(0), 256, 1, 1, 0, 74);
		auto l75 = convBnMish(network, weightMap, *l74->getOutput(0), 256, 3, 1, 1, 75);
		auto ew76 = network->addElementWise(*l75->getOutput(0), *ew73->getOutput(0), ElementWiseOperation::kSUM);
		auto l77 = convBnMish(network, weightMap, *ew76->getOutput(0), 256, 1, 1, 0, 77);
		auto l78 = convBnMish(network, weightMap, *l77->getOutput(0), 256, 3, 1, 1, 78);
		auto ew79 = network->addElementWise(*l78->getOutput(0), *ew76->getOutput(0), ElementWiseOperation::kSUM);
		auto l80 = convBnMish(network, weightMap, *ew79->getOutput(0), 256, 1, 1, 0, 80);
		auto l81 = convBnMish(network, weightMap, *l80->getOutput(0), 256, 3, 1, 1, 81);
		auto ew82 = network->addElementWise(*l81->getOutput(0), *ew79->getOutput(0), ElementWiseOperation::kSUM);
		auto l83 = convBnMish(network, weightMap, *ew82->getOutput(0), 256, 1, 1, 0, 83);

		ITensor* inputTensors84[] = { l83->getOutput(0), l56->getOutput(0) };
		auto cat84 = network->addConcatenation(inputTensors84, 2);

		auto l85 = convBnMish(network, weightMap, *cat84->getOutput(0), 512, 1, 1, 0, 85);
		auto l86 = convBnMish(network, weightMap, *l85->getOutput(0), 1024, 3, 2, 1, 86);
		auto l87 = convBnMish(network, weightMap, *l86->getOutput(0), 512, 1, 1, 0, 87);
		auto l88 = l86;
		auto l89 = convBnMish(network, weightMap, *l88->getOutput(0), 512, 1, 1, 0, 89);
		auto l90 = convBnMish(network, weightMap, *l89->getOutput(0), 512, 1, 1, 0, 90);
		auto l91 = convBnMish(network, weightMap, *l90->getOutput(0), 512, 3, 1, 1, 91);
		auto ew92 = network->addElementWise(*l91->getOutput(0), *l89->getOutput(0), ElementWiseOperation::kSUM);
		auto l93 = convBnMish(network, weightMap, *ew92->getOutput(0), 512, 1, 1, 0, 93);
		auto l94 = convBnMish(network, weightMap, *l93->getOutput(0), 512, 3, 1, 1, 94);
		auto ew95 = network->addElementWise(*l94->getOutput(0), *ew92->getOutput(0), ElementWiseOperation::kSUM);
		auto l96 = convBnMish(network, weightMap, *ew95->getOutput(0), 512, 1, 1, 0, 96);
		auto l97 = convBnMish(network, weightMap, *l96->getOutput(0), 512, 3, 1, 1, 97);
		auto ew98 = network->addElementWise(*l97->getOutput(0), *ew95->getOutput(0), ElementWiseOperation::kSUM);
		auto l99 = convBnMish(network, weightMap, *ew98->getOutput(0), 512, 1, 1, 0, 99);
		auto l100 = convBnMish(network, weightMap, *l99->getOutput(0), 512, 3, 1, 1, 100);
		auto ew101 = network->addElementWise(*l100->getOutput(0), *ew98->getOutput(0), ElementWiseOperation::kSUM);
		auto l102 = convBnMish(network, weightMap, *ew101->getOutput(0), 512, 1, 1, 0, 102);

		ITensor* inputTensors103[] = { l102->getOutput(0), l87->getOutput(0) };
		auto cat103 = network->addConcatenation(inputTensors103, 2);

		auto l104 = convBnMish(network, weightMap, *cat103->getOutput(0), 1024, 1, 1, 0, 104);

		// ---------
		auto l105 = convBnLeaky(network, weightMap, *l104->getOutput(0), 512, 1, 1, 0, 105);
		auto l106 = convBnLeaky(network, weightMap, *l105->getOutput(0), 1024, 3, 1, 1, 106);
		auto l107 = convBnLeaky(network, weightMap, *l106->getOutput(0), 512, 1, 1, 0, 107);

		auto pool108 = network->addPoolingNd(*l107->getOutput(0), PoolingType::kMAX, DimsHW{ 5, 5 });
		pool108->setPaddingNd(DimsHW{ 2, 2 });
		pool108->setStrideNd(DimsHW{ 1, 1 });

		auto l109 = l107;

		auto pool110 = network->addPoolingNd(*l109->getOutput(0), PoolingType::kMAX, DimsHW{ 9, 9 });
		pool110->setPaddingNd(DimsHW{ 4, 4 });
		pool110->setStrideNd(DimsHW{ 1, 1 });

		auto l111 = l107;

		auto pool112 = network->addPoolingNd(*l111->getOutput(0), PoolingType::kMAX, DimsHW{ 13, 13 });
		pool112->setPaddingNd(DimsHW{ 6, 6 });
		pool112->setStrideNd(DimsHW{ 1, 1 });

		ITensor* inputTensors113[] = { pool112->getOutput(0), pool110->getOutput(0), pool108->getOutput(0), l107->getOutput(0) };
		auto cat113 = network->addConcatenation(inputTensors113, 4);

		auto l114 = convBnLeaky(network, weightMap, *cat113->getOutput(0), 512, 1, 1, 0, 114);
		auto l115 = convBnLeaky(network, weightMap, *l114->getOutput(0), 1024, 3, 1, 1, 115);
		auto l116 = convBnLeaky(network, weightMap, *l115->getOutput(0), 512, 1, 1, 0, 116);
		auto l117 = convBnLeaky(network, weightMap, *l116->getOutput(0), 256, 1, 1, 0, 117);

		float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
		for (int i = 0; i < 256 * 2 * 2; i++) {
			deval[i] = 1.0;
		}
		Weights deconvwts118{ DataType::kFLOAT, deval, 256 * 2 * 2 };
		IDeconvolutionLayer* deconv118 = network->addDeconvolutionNd(*l117->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts118, emptywts);
		assert(deconv118);
		deconv118->setStrideNd(DimsHW{ 2, 2 });
		deconv118->setNbGroups(256);
		weightMap["deconv118"] = deconvwts118;

		auto l119 = l85;
		auto l120 = convBnLeaky(network, weightMap, *l119->getOutput(0), 256, 1, 1, 0, 120);

		ITensor* inputTensors121[] = { l120->getOutput(0), deconv118->getOutput(0) };
		auto cat121 = network->addConcatenation(inputTensors121, 2);

		auto l122 = convBnLeaky(network, weightMap, *cat121->getOutput(0), 256, 1, 1, 0, 122);
		auto l123 = convBnLeaky(network, weightMap, *l122->getOutput(0), 512, 3, 1, 1, 123);
		auto l124 = convBnLeaky(network, weightMap, *l123->getOutput(0), 256, 1, 1, 0, 124);
		auto l125 = convBnLeaky(network, weightMap, *l124->getOutput(0), 512, 3, 1, 1, 125);
		auto l126 = convBnLeaky(network, weightMap, *l125->getOutput(0), 256, 1, 1, 0, 126);
		auto l127 = convBnLeaky(network, weightMap, *l126->getOutput(0), 128, 1, 1, 0, 127);

		Weights deconvwts128{ DataType::kFLOAT, deval, 128 * 2 * 2 };
		IDeconvolutionLayer* deconv128 = network->addDeconvolutionNd(*l127->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts128, emptywts);
		assert(deconv128);
		deconv128->setStrideNd(DimsHW{ 2, 2 });
		deconv128->setNbGroups(128);

		auto l129 = l54;
		auto l130 = convBnLeaky(network, weightMap, *l129->getOutput(0), 128, 1, 1, 0, 130);

		ITensor* inputTensors131[] = { l130->getOutput(0), deconv128->getOutput(0) };
		auto cat131 = network->addConcatenation(inputTensors131, 2);

		auto l132 = convBnLeaky(network, weightMap, *cat131->getOutput(0), 128, 1, 1, 0, 132);
		auto l133 = convBnLeaky(network, weightMap, *l132->getOutput(0), 256, 3, 1, 1, 133);
		auto l134 = convBnLeaky(network, weightMap, *l133->getOutput(0), 128, 1, 1, 0, 134);
		auto l135 = convBnLeaky(network, weightMap, *l134->getOutput(0), 256, 3, 1, 1, 135);
		auto l136 = convBnLeaky(network, weightMap, *l135->getOutput(0), 128, 1, 1, 0, 136);
		auto l137 = convBnLeaky(network, weightMap, *l136->getOutput(0), 256, 3, 1, 1, 137);
		IConvolutionLayer* conv138 = network->addConvolutionNd(*l137->getOutput(0), 3 * (g_class_num + 5), DimsHW{ 1, 1 }, weightMap["module_list.138.Conv2d.weight"], weightMap["module_list.138.Conv2d.bias"]);
		assert(conv138);
		// 139 is yolo layer

		auto l140 = l136;
		auto l141 = convBnLeaky(network, weightMap, *l140->getOutput(0), 256, 3, 2, 1, 141);

		ITensor* inputTensors142[] = { l141->getOutput(0), l126->getOutput(0) };
		auto cat142 = network->addConcatenation(inputTensors142, 2);

		auto l143 = convBnLeaky(network, weightMap, *cat142->getOutput(0), 256, 1, 1, 0, 143);
		auto l144 = convBnLeaky(network, weightMap, *l143->getOutput(0), 512, 3, 1, 1, 144);
		auto l145 = convBnLeaky(network, weightMap, *l144->getOutput(0), 256, 1, 1, 0, 145);
		auto l146 = convBnLeaky(network, weightMap, *l145->getOutput(0), 512, 3, 1, 1, 146);
		auto l147 = convBnLeaky(network, weightMap, *l146->getOutput(0), 256, 1, 1, 0, 147);
		auto l148 = convBnLeaky(network, weightMap, *l147->getOutput(0), 512, 3, 1, 1, 148);
		IConvolutionLayer* conv149 = network->addConvolutionNd(*l148->getOutput(0), 3 * (g_class_num + 5), DimsHW{ 1, 1 }, weightMap["module_list.149.Conv2d.weight"], weightMap["module_list.149.Conv2d.bias"]);
		assert(conv149);
		// 150 is yolo layer

		auto l151 = l147;
		auto l152 = convBnLeaky(network, weightMap, *l151->getOutput(0), 512, 3, 2, 1, 152);

		ITensor* inputTensors153[] = { l152->getOutput(0), l116->getOutput(0) };
		auto cat153 = network->addConcatenation(inputTensors153, 2);

		auto l154 = convBnLeaky(network, weightMap, *cat153->getOutput(0), 512, 1, 1, 0, 154);
		auto l155 = convBnLeaky(network, weightMap, *l154->getOutput(0), 1024, 3, 1, 1, 155);
		auto l156 = convBnLeaky(network, weightMap, *l155->getOutput(0), 512, 1, 1, 0, 156);
		auto l157 = convBnLeaky(network, weightMap, *l156->getOutput(0), 1024, 3, 1, 1, 157);
		auto l158 = convBnLeaky(network, weightMap, *l157->getOutput(0), 512, 1, 1, 0, 158);
		auto l159 = convBnLeaky(network, weightMap, *l158->getOutput(0), 1024, 3, 1, 1, 159);
		IConvolutionLayer* conv160 = network->addConvolutionNd(*l159->getOutput(0), 3 * (g_class_num + 5), DimsHW{ 1, 1 }, weightMap["module_list.160.Conv2d.weight"], weightMap["module_list.160.Conv2d.bias"]);
		assert(conv160);
		// 161 is yolo layer

		auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
		const PluginFieldCollection* pluginData = creator->getFieldNames();
		IPluginV2* pluginObj = creator->createPlugin("yololayer", pluginData);
		ITensor* inputTensors_yolo[] = { conv138->getOutput(0), conv149->getOutput(0), conv160->getOutput(0) };
		auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

		yolo->getOutput(0)->setName(m_output_layers[0].c_str());
		network->markOutput(*yolo->getOutput(0));


		// Build engine
		builder->setMaxBatchSize(maxBatchSize);


		//        config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
		//#ifdef USE_FP16
		//        std::cout << "USE_FP16" << std::endl;
		//        config->setFlag(BuilderFlag::kFP16);
		//#endif
		//        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
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
		builder->setMaxWorkspaceSize(16 * (1 << 20));
		if (builder->platformHasFastFp16())
		{
			builder->setFp16Mode(true);
			std::cout << "FP16 Mode True" << std::endl;
		}
		//		samplesCommon::enableDLA(builder, /*gUseDLACore*/ -1);
		ICudaEngine* engine = builder->buildCudaEngine(*network);
		assert(engine);
#endif


		trtModelStream = engine->serialize();
		// Don't need the network any more
		network->destroy();
		engine->destroy();
		builder->destroy();

		// Release host memory
		for (auto& mem : weightMap)
		{
			free((void*)(mem.second.values));
		}

	}
}
