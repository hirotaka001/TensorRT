#define TAG "TRT_YOLOv3"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/TRT_YOLOv3/TRT_YOLOv3.h>
#include <kros/trt/Common/PreProcess_cuda.h>
#include <kros/trt/Common/utils.hpp>
#include <kros/trt/Common/nms.hpp>
#include <omp.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

#define SIGMOID(val) \
		(1.0 / (1.0 + std::exp(-(val))))

#define EXPONENTIAL(val) \
		std::exp(val)

namespace kros
{
	struct operator1 {
		float** m_in_buffer;
		std::vector<float>& m_midvec;
		//	int m_layerIdx;
		int m_UnitLen;  //detection結果と求める為の一番小さい単位（x,y,w,h,conf,class1...classn）クラスを１だとすると、x,y,w,h,conf,class1で6となる。
		int m_GridSize[3];
		int m_InBufSize[3];
		int m_numbatch;
		int m_AnchorSize[3];
		int m_1BatchLen;
		int m_layerBatchSize[3];  //leyer内の１バッチのサイズ。１バッチのサイズはlayerに寄って異なる。
		operator1(float** in_Buffer, std::vector<float>& mid_vec, int UnitLen, int nBatch, const int* GridSize, const int* AnchorSize) :
			m_in_buffer(in_Buffer), m_midvec(mid_vec), m_UnitLen(UnitLen), m_numbatch(nBatch) {
			m_GridSize[0] = GridSize[0];
			m_GridSize[1] = GridSize[1];
			m_GridSize[2] = GridSize[2];
			m_AnchorSize[0] = AnchorSize[0];
			m_AnchorSize[1] = AnchorSize[1];
			m_AnchorSize[2] = AnchorSize[2];
			m_layerBatchSize[0] = m_AnchorSize[0] * m_GridSize[0];
			m_layerBatchSize[1] = m_AnchorSize[1] * m_GridSize[1];
			m_layerBatchSize[2] = m_AnchorSize[2] * m_GridSize[2];
			m_InBufSize[0] = m_layerBatchSize[0] * m_numbatch;
			m_InBufSize[1] = m_layerBatchSize[1] * m_numbatch;
			m_InBufSize[2] = m_layerBatchSize[2] * m_numbatch;
			m_1BatchLen = m_layerBatchSize[0] + m_layerBatchSize[1] + m_layerBatchSize[2];
			return;
		}
		//ValueはInBuffer側で回す。Leyer単位でtranform
		float operator()(float& Value) {

			int Cur_pos = (int)(&Value - &m_midvec[0]);

			int BatchNo = Cur_pos / m_1BatchLen;
			int BatchInterIdx = Cur_pos % m_1BatchLen;
			int BatchOffset;
			//    	int LayerNo;
			float* in_Buffer;
			int layerBatchSize;
			int AnchorSize;
			int GridSize;
			if (BatchInterIdx < m_layerBatchSize[0]) {
				//    		LayerNo   = 0;
				in_Buffer = m_in_buffer[0];
				BatchOffset = BatchInterIdx;
				layerBatchSize = m_layerBatchSize[0];
				AnchorSize = m_AnchorSize[0];
				GridSize = m_GridSize[0];
			}
			else if (BatchInterIdx < (m_layerBatchSize[0] + m_layerBatchSize[1])) {
				//    		LayerNo   = 1;
				in_Buffer = m_in_buffer[1];
				BatchOffset = (BatchInterIdx - m_layerBatchSize[0]);
				layerBatchSize = m_layerBatchSize[1];
				AnchorSize = m_AnchorSize[1];
				GridSize = m_GridSize[1];
			}
			else {
				//    		LayerNo = 2;
				in_Buffer = m_in_buffer[2];
				BatchOffset = (BatchInterIdx - m_layerBatchSize[0] - m_layerBatchSize[1]);
				layerBatchSize = m_layerBatchSize[2];
				AnchorSize = m_AnchorSize[2];
				GridSize = m_GridSize[2];

			}
			int InAnchorSize = m_UnitLen * GridSize;
			int GridNo = BatchOffset / AnchorSize;
			int AnchorIdx = (BatchOffset % AnchorSize) / m_UnitLen;
			int ItemPos = (BatchOffset % AnchorSize) % m_UnitLen;

			float val = in_Buffer[(BatchNo * layerBatchSize) + (AnchorIdx * InAnchorSize) + (ItemPos * GridSize) + GridNo];
			return val;  //セット先はmidbuffer
		}
	};

	struct operator2 {
		float* m_midTopPtr;
		float* m_outTopPtr;
		int m_MidUnitLen;  //Ope1のユニット長
		int m_OutUnitLen;  //Ope2のユニット長

		int m_GridWidth[3];
		int m_GridHeight[3];
		int m_GridSize[3];

		int m_MidAnchorSize[3];
		int m_OutAnchorSize[3];

		int m_MidLayerSize[3];  //leyer内の１バッチのサイズ。１バッチのサイズはlayerに寄って異なる。
		int m_OutLayerSize[3];  //leyer内の１バッチのサイズ。１バッチのサイズはlayerに寄って異なる。

		int m_Mid1BatchLen;
		int m_Out1BatchLen;

		int m_AnchorX[9];
		int m_AnchorY[9];
		int m_OrgWidth;
		int m_OrgHeight;
		int m_Width;
		int m_Height;
		int m_ClassNum;
		int m_cnt;

		operator2(float* midTopPtr, float* outTopPtr, int MidUnitLen, int OutUnitLen, const int* GridWidth, const int* GridHeight, const int* GridSize,
			const int* MidAnchorSize, const int* OutAnchorSize, int OrgWidth, int OrgHeight, int Width, int Height, int ClassNum) :
			m_midTopPtr(midTopPtr), m_outTopPtr(outTopPtr), m_MidUnitLen(MidUnitLen), m_OutUnitLen(OutUnitLen), m_OrgWidth(OrgWidth), m_OrgHeight(OrgHeight),
			m_Width(Width), m_Height(Height), m_ClassNum(ClassNum) {
			//Gridサイズは入出力で共通
			m_GridWidth[0] = GridWidth[0];	//16
			m_GridWidth[1] = GridWidth[1];	//32
			m_GridWidth[2] = GridWidth[2];	//64

			m_GridHeight[0] = GridHeight[0];	//16
			m_GridHeight[1] = GridHeight[1];	//32
			m_GridHeight[2] = GridHeight[2];	//64

			m_GridSize[0] = GridSize[0];	//256
			m_GridSize[1] = GridSize[1];	//1024
			m_GridSize[2] = GridSize[2];	//4096

			m_MidAnchorSize[0] = MidAnchorSize[0];	//18   Mid側 （X,Y,W,H,conf,class） x anchor数3
			m_MidAnchorSize[1] = MidAnchorSize[1];	//18
			m_MidAnchorSize[2] = MidAnchorSize[2];	//18

			m_MidLayerSize[0] = m_MidAnchorSize[0] * m_GridSize[0]; //4608   Mid側 1layer内1Batchサイズ
			m_MidLayerSize[1] = m_MidAnchorSize[1] * m_GridSize[1];	//18432  Mid側 1layer内1Batchサイズ
			m_MidLayerSize[2] = m_MidAnchorSize[2] * m_GridSize[2];	//73278  Mid側 1layer内1Batchサイズ

			m_Mid1BatchLen = m_MidLayerSize[0] + m_MidLayerSize[1] + m_MidLayerSize[2];	//96768 Mid側 1Batch分サイズ layerを3つ纏めた物

			m_OutAnchorSize[0] = OutAnchorSize[0];	//21   Out側 （X,Y,W,H,ClassID,prob,score） x anchor数3
			m_OutAnchorSize[1] = OutAnchorSize[1];	//21
			m_OutAnchorSize[2] = OutAnchorSize[2];	//21

			m_OutLayerSize[0] = m_OutAnchorSize[0] * m_GridSize[0];  	//5376   Out側 1layer内1Batchサイズ
			m_OutLayerSize[1] = m_OutAnchorSize[1] * m_GridSize[1];  	//21504  Out側 1layer内1Batchサイズ
			m_OutLayerSize[2] = m_OutAnchorSize[2] * m_GridSize[2];  	//96016  Out側 1layer内1Batchサイズ

			m_Out1BatchLen = m_OutLayerSize[0] + m_OutLayerSize[1] + m_OutLayerSize[2]; //112896 Out側 1Batch分サイズ layerを3つ纏めた物

			m_AnchorX[0] = 116;
			m_AnchorX[1] = 156;
			m_AnchorX[2] = 373;
			m_AnchorX[3] = 30;
			m_AnchorX[4] = 62;
			m_AnchorX[5] = 59;
			m_AnchorX[6] = 10;
			m_AnchorX[7] = 16;
			m_AnchorX[8] = 33;
			m_AnchorY[0] = 90;
			m_AnchorY[1] = 198;
			m_AnchorY[2] = 326;
			m_AnchorY[3] = 61;
			m_AnchorY[4] = 45;
			m_AnchorY[5] = 119;
			m_AnchorY[6] = 13;
			m_AnchorY[7] = 30;
			m_AnchorY[8] = 23;
			m_cnt = 0;

		}
		//ValueはInBuffer側で回す。Leyer単位でtranform
		float operator () (float& Value) {
			int OutLayerOffset;
			int MidLayerOffset;
			int MidAnchorSize;
			int OutAnchorSize;
			int GridWidth;
			int GridHeight;
			int LayerNo;
			//Out側の現在位置
			int Cur_pos = (int)(&Value - m_outTopPtr);

			//出力側のどのバッチかを示す
			int BatchNo = Cur_pos / m_Out1BatchLen;
			//バッチ内先頭からのポジション（先頭からの位置なので、どのlayerかはこの段階では不明。Layerのサイズが異なるため)
			int BatchInterIdx = Cur_pos % m_Out1BatchLen;

			//どのLayerかを特定
			if (BatchInterIdx < m_OutLayerSize[0]) {
				//Layer1
				LayerNo = 0;
				OutLayerOffset = BatchInterIdx;   											//Out側第1レーヤー先頭からのオフセット値
				OutAnchorSize = m_OutAnchorSize[0];											//Out側 （X,Y,W,H,conf,class） x anchor数3   18
				GridWidth = m_GridWidth[0];												//16
				GridHeight = m_GridHeight[0];											//16
				MidAnchorSize = m_MidAnchorSize[0];											//Mid側 （X,Y,W,H,ClassID,prob,score) x anchor数3 21
				MidLayerOffset = 0;
			}
			else if (BatchInterIdx < (m_OutLayerSize[0] + m_OutLayerSize[1])) {
				//Layer2
				LayerNo = 1;
				OutLayerOffset = (BatchInterIdx - m_OutLayerSize[0]);    							//第2レーヤー先頭からのオフセット値
				OutAnchorSize = m_OutAnchorSize[1];
				GridWidth = m_GridWidth[1];
				GridHeight = m_GridHeight[1];
				MidAnchorSize = m_MidAnchorSize[1];
				MidLayerOffset = m_MidLayerSize[0];
			}
			else {
				//Layer3
				LayerNo = 2;
				OutLayerOffset = (BatchInterIdx - m_OutLayerSize[0] - m_OutLayerSize[1]);	//第2レーヤー先頭からのオフセット値
				OutAnchorSize = m_OutAnchorSize[2];
				GridWidth = m_GridWidth[2];
				GridHeight = m_GridHeight[2];
				MidAnchorSize = m_MidAnchorSize[2];
				MidLayerOffset = m_MidLayerSize[0] + m_MidLayerSize[1];

			}

			int OutGridNo = OutLayerOffset / OutAnchorSize;										//Layer内Grid番号
			int OutAnchorNo = (OutLayerOffset % OutAnchorSize) / m_OutUnitLen;						//Grid内Anchor位置
			int ItemPos = (OutLayerOffset % OutAnchorSize) % m_OutUnitLen;						//Anchor内要素位置

			float res;
			switch (ItemPos) {
			case 0:
			{
				//X値計算
				float val = (float)SIGMOID(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 0]);
				int WidthPos = OutGridNo % GridWidth;
				val += (float)WidthPos;
				val /= (float)GridWidth;

				//Wの先行読み込み
				int yolo_anchorW = m_AnchorX[LayerNo * 3 + OutAnchorNo];
				float W = EXPONENTIAL(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 2]) * yolo_anchorW;
				W /= m_Width;
				val -= (float)(W / 2.0);
				res = val;
				//			res = val * m_OrgWidth;
				break;
			}
			case 1:  //Y
			{
				float val = (float)SIGMOID(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 1]);
				int HeightPos = OutGridNo / GridWidth;
				val += (float)HeightPos;
				val /= (float)GridHeight;
				//Hの先行読み込み
				int yolo_anchorH = m_AnchorY[LayerNo * 3 + OutAnchorNo];
				float H = EXPONENTIAL(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 3]) * yolo_anchorH;
				H /= m_Height;
				val -= (float)(H / 2.0);
				res = val;
				//			res = val * m_OrgHeight;
				break;
			}
			case 2:  //W
			{
				int yolo_anchorW = m_AnchorX[LayerNo * 3 + OutAnchorNo];
				float val = EXPONENTIAL(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 2]) * yolo_anchorW;
				val /= m_Width;
				res = val;
				//			res = val * m_OrgWidth;
				break;
			}
			case 3:  //H
			{
				int yolo_anchorH = m_AnchorY[LayerNo * 3 + OutAnchorNo];
				float val = EXPONENTIAL(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 3]) * yolo_anchorH;
				val /= m_Height;
				res = val;
				//			res = val * m_OrgHeight;
				break;
			}
			case 4:  //ClassID
			case 5:  //Score
			{
				float confidence = (float)SIGMOID(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 4]);
				float MaxScore = 0.0;
				int   MaxScoreClsID = 0;
				for (int ClsIdx = 0; ClsIdx < m_ClassNum; ClsIdx++) {
					float class_probs = (float)SIGMOID(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 5 + ClsIdx]);
					float box_scores = confidence * class_probs;
					if (MaxScore <= box_scores) {
						MaxScoreClsID = ClsIdx;
						MaxScore = box_scores;
					}
				}
				if (ItemPos == 4) {
					res = (float)MaxScoreClsID;

				}
				else {
					res = MaxScore;
				}
				break;
			}
			default: //Score
			{
				res = 0.0;
				printf("Error ItemPos :%d\n", ItemPos);
			}
			}
			return res;
		}
	};

	//##### PostProcessの処理はここで切り替える。
	//#define RANDOM
	//#define SEQUENCE
#define THRUST

	TRT_YOLOv3::TRT_YOLOv3()
	{
#ifdef RANDOM
		std::cout << "PostProcess = Random" << std::endl;
#endif
#ifdef SEQUENCE
		std::cout << "PostProcess = Sequence" << std::endl;
#endif
#ifdef THRUST
		std::cout << "PostProcess = Thrust" << std::endl;
#endif
		//	m_outputSize_list.push_back(2048);
	}

	TRT_YOLOv3::~TRT_YOLOv3()
	{
		Release();
	}

	bool TRT_YOLOv3::Init(
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
		const std::vector<std::string>& output_layers)

	{
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

#ifdef THRUST
		m_is_post_process_cuda = true;
#else
		m_is_post_process_cuda = false;
#endif

		if (m_is_pre_process_cuda)
		{
			std::cout << "PreProcess  = CUDA" << std::endl;
		}
		else
		{
			std::cout << "PreProcess  = CPU" << std::endl;
		}
		int nOutLayers = (int)output_layers.size();
		if (nOutLayers != 3) {
			std::cout << "[" << TAG << "] Error : the number of output layers must be 3" << std::endl;
			return false;
		}
		m_class_num = class_num;
		m_score_threshold = score_threshold;
		m_nms_threshold = nms_threshold;
		m_batch = batch;
		m_super_batch = super_batch;
		m_base_grid_w = width / 32;
		m_base_grid_h = height / 32;
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

#ifdef _OPENMP
		if (m_thread_num > omp_get_num_procs()) {
			std::cout << "[" << TAG << "] Error : thread_num must be set within the number of processors" << std::endl;
			return false;
		}
		omp_set_num_threads(m_thread_num);
#endif

		m_grid_width[0] = m_base_grid_w;
		m_grid_width[1] = m_base_grid_w * 2;
		m_grid_width[2] = m_base_grid_w * 4;
		m_grid_height[0] = m_base_grid_h;
		m_grid_height[1] = m_base_grid_h * 2;
		m_grid_height[2] = m_base_grid_h * 4;
		m_GridSize[0] = m_grid_width[0] * m_grid_height[0];
		m_GridSize[1] = m_grid_width[1] * m_grid_height[1];
		m_GridSize[2] = m_grid_width[2] * m_grid_height[2];

		m_UnitLen = m_class_num + 5;

		m_anchor_size[0] = (int)m_yolo_masks[0].size() * m_UnitLen;
		m_anchor_size[1] = (int)m_yolo_masks[1].size() * m_UnitLen;
		m_anchor_size[2] = (int)m_yolo_masks[2].size() * m_UnitLen;

		m_outputSize_list.push_back(m_anchor_size[0] * m_GridSize[0]);
		m_outputSize_list.push_back(m_anchor_size[1] * m_GridSize[1]);
		m_outputSize_list.push_back(m_anchor_size[2] * m_GridSize[2]);

		m_OutUnitLen = 6;  //x,y,w,h, ClassID,score クラス分が最大値１に固定されるので、クラス個数倍する事は無く、アイテム数6固定。(probは中間けいさんでのみ使用なので、引数
		m_OutAnchorSize[0] = m_OutUnitLen * 3;			//Ope2出力側ユニットが3アンカー分
		m_OutAnchorSize[1] = m_OutUnitLen * 3;
		m_OutAnchorSize[2] = m_OutUnitLen * 3;
		//
		m_OutSize_list.push_back(m_OutAnchorSize[0] * m_GridSize[0]);    //Ope2出力側アンカーサイズ*グリッド数=1バッチ分
		m_OutSize_list.push_back(m_OutAnchorSize[1] * m_GridSize[1]);
		m_OutSize_list.push_back(m_OutAnchorSize[2] * m_GridSize[2]);

		//PostProcess Thrust版用引数構造体へのセット。（クラスからC関数を呼ぶ為、C関数内からクラスのメンバ変数を参照出来ない事への措置。
		TRT_YOLOv3_PostProcess_cuda::st_argumets PostProcessThrustArg;

		PostProcessThrustArg.class_num = m_class_num;
		PostProcessThrustArg.num_batch = batch;
		PostProcessThrustArg.num_super_batch = m_super_batch;
		PostProcessThrustArg.width = width;
		PostProcessThrustArg.height = height;
		PostProcessThrustArg.threshold = m_score_threshold;
		PostProcessThrustArg.MidUnitLen = m_UnitLen;
		PostProcessThrustArg.OutUnitLen = m_OutUnitLen;

		for (int i = 0; i < m_thread_num; i++) {
			TRT_YOLOv3_PostProcess_cuda* cTRT_YOLOv3_PostProcess_cuda = new TRT_YOLOv3_PostProcess_cuda;
			cTRT_YOLOv3_PostProcess_cuda->Init(PostProcessThrustArg, m_grid_width, m_grid_height);
			mp_TRT_YOLOv3_PostProcess_cuda.push_back(cTRT_YOLOv3_PostProcess_cuda);
		}
		//クラス数を付与
		std::string model_name2 = model_name + "_c" + std::to_string(class_num);

		return TRT::Init(gpu_id, model_name2, model_dir, input_layers, output_layers, width, height, batch,
			super_batch, thread_num, m_is_post_process_cuda, m_is_pre_process_cuda);

		//		return TRT::Init(model_name2, model_dir, m_input_layers, m_output_layers, width, height, batch,
		//		                 super_batch, thread_num,m_is_post_proces_cuda, m_is_pre_process_cuda);

	}

	void TRT_YOLOv3::Release() {

		int num_buffer = (int)mPreprocess_cuda_in_buffer.size();
		for (int i = 0; i < num_buffer; i++)
		{
			cudaFree(mPreprocess_cuda_in_buffer[i]);
		}
		mPreprocess_cuda_in_buffer.clear();


		int nInstance = (int)mp_TRT_YOLOv3_PostProcess_cuda.size();
		for (int i = 0; i < nInstance; i++) {
			delete mp_TRT_YOLOv3_PostProcess_cuda[i];
		}
		mp_TRT_YOLOv3_PostProcess_cuda.clear();

		TRT::Release();
	}

	bool TRT_YOLOv3::InferParallel(
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
		for (int loop_idx = 0; loop_idx < num_infer; loop_idx++) {
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


	bool TRT_YOLOv3::Infer(
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

	bool TRT_YOLOv3::DoNMS(
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

	void TRT_YOLOv3::DrawResult(
		std::vector<cv::Mat>& images,
		const std::vector<std::vector<std::vector<float>>>& detection_result,
		const std::vector<std::string>& label_list)
	{
		CV_Assert((int)images.size() == (int)detection_result.size());
		for (int i = 0; i < (int)images.size(); i++) {
			TRT_YOLOv3::DrawResult(images[i], detection_result[i], label_list);
		}
	}

	void TRT_YOLOv3::DrawResult(
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

	cv::Scalar TRT_YOLOv3::HSV2RGB(int classID) {
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

	void TRT_YOLOv3::PreProcess(
		const std::vector<cv::Mat>& input_images,
		TRT::CPU_buffer& buffer,
		float* GPU_In_buffers,
		const cudaStream_t stream,
		const int thread_id)
	{
		//    	auto start = cv::getTickCount();
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
			//            std::cout << "[TRT_YOLOv3::PreProcess]PreProcess　cpu版" << std::endl;
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
		//       	auto end = cv::getTickCount();
		//       	double elapsedMsec = (end - start) * 1000 / cv::getTickFrequency();
		//        std::cout << "[TRT_YOLOv4::PreProcess] :" << elapsedMsec << "ms" << std::endl;

	}
	void TRT_YOLOv3::PostProcess(
		const TRT::CPU_buffer& buffer,
		const std::vector<void*>& gpu_buffers,
		const std::vector<cv::Size>& original_image_sizes,
		const int super_batch_id,
		const int thread_id,
		std::vector< std::vector<std::vector<float> > >& output_result) const
	{
		//  ############ 使用したい物を適宜コメントを差し替える ##############
		//  Random版 従来
#ifdef RANDOM
		PostProcess_random(buffer, gpu_buffers, original_image_sizes, super_batch_id, output_result);
#endif

		//	std::transform版
#ifdef SEQUENCE
		PostProcess_sequence(buffer, gpu_buffers, original_image_sizes, super_batch_id, output_result);
#endif
		//	Thrust版（CUDA使用）版
#ifdef THRUST
		std::vector< std::vector<std::vector<float> > > output_result2;
		mp_TRT_YOLOv3_PostProcess_cuda[thread_id]->Exec(gpu_buffers, original_image_sizes, super_batch_id, output_result2);

		for (int i = 0; i < buffer.numInput; i++) {
			output_result.push_back(output_result2[i]);
		}
#endif
	}

	void TRT_YOLOv3::PostProcess_random(
		const TRT::CPU_buffer& buffer,
		const std::vector<void*>& gpu_buffers,
		const std::vector<cv::Size>& original_image_sizes,
		const int super_batch_id,
		std::vector< std::vector<std::vector<float> > >& output_result) const
	{
		//DEBUG 途中結果確認用 出力 Loop=1のみ使用可。 (削除しないで下さい）
		//	char FileName[256];
		//	sprintf(FileName,"YOLOv3_Random_Ope1_b%02d.txt",buffer.numInput);
		//	std::ofstream Debuglog_Ope1(FileName);
		//
		//	sprintf(FileName,"YOLOv3_Random_Ope2_b%02d.txt",buffer.numInput);
		//	std::ofstream Debuglog_Ope2(FileName);
		//DEBUG
			//batchを外側で回す
		for (int batch_idx = 0; batch_idx < buffer.numInput; batch_idx++) {
			//        auto  time_0 = cv::getTickCount();
			cv::Size orig_image_size = original_image_sizes[batch_idx];
			std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> trt_outputs;
			//	  ( 3, 3, 6, 19, 19)  Layer、anchors、class, h ,w
			//1次元のバッファだとアドレッシングが難しので、5次元のvectorを作る。
			//エレメント ループ
			for (int layerIdx = 0; layerIdx < 3; layerIdx++) {
				//        	printf("%d,%d,%d,%d\n", 3,(m_class_num+5),output_h[layerIdx],output_w[layerIdx]);
				std::vector<std::vector<std::vector<std::vector<float>>>> dimlayer;
				float* bufpos = buffer.out_buffer[layerIdx] + (batch_idx * m_outputSize_list[layerIdx]);
				for (int anchorIdx = 0; anchorIdx < 3; anchorIdx++) {
					std::vector<std::vector<std::vector<float>>> dimAanchor;
					for (int dim4idx = 0; dim4idx < (m_class_num + 5); dim4idx++) {
						std::vector<std::vector<float>> dimH;
						for (int outHidx = 0; outHidx < m_grid_height[layerIdx]; outHidx++) {
							std::vector<float> dimW;
							for (int outWidx = 0; outWidx < m_grid_width[layerIdx]; outWidx++) {
								dimW.push_back(*bufpos);
								bufpos++;  //アドレッシングが煩雑になるので、ポインタ加算方式にした。もし、openMP等でこのループをマルチスレッド化する際は要注意。
							}
							dimH.push_back(dimW);
						}
						dimAanchor.push_back(dimH);
					}
					dimlayer.push_back(dimAanchor);
				}
				trt_outputs.push_back(dimlayer);
			}

			//        auto  time_1 = cv::getTickCount();
			//推論結果から検出結果を算出
			std::vector<std::vector<float>> ResBoxes;

			for (int layerIdx = 0; layerIdx < 3; layerIdx++) {
				std::vector<std::vector<std::vector<st_Boxes>>> out_box_elm;

				for (int HeightIdx = 0; HeightIdx < m_grid_height[layerIdx]; HeightIdx++) {
					std::vector<std::vector<st_Boxes>> out_box_h;

					for (int WidthIdx = 0; WidthIdx < m_grid_width[layerIdx]; WidthIdx++) {
						std::vector<st_Boxes> out_box_w;

						for (int anchorIdx = 0; anchorIdx < 3; anchorIdx++) {
							st_Boxes out_box_e;
							int yolo_anchors[2];
							yolo_anchors[0] = m_yolo_anchors[m_yolo_masks[layerIdx][anchorIdx]][0];
							yolo_anchors[1] = m_yolo_anchors[m_yolo_masks[layerIdx][anchorIdx]][1];

							float box_xy[2];
							float box_wh[2];

							box_xy[0] = (float)SIGMOID(trt_outputs[layerIdx][anchorIdx][0][HeightIdx][WidthIdx]);
							box_xy[1] = (float)SIGMOID(trt_outputs[layerIdx][anchorIdx][1][HeightIdx][WidthIdx]);

							box_xy[0] += WidthIdx;
							box_xy[1] += HeightIdx;

							box_xy[0] /= m_grid_width[layerIdx];
							box_xy[1] /= m_grid_height[layerIdx];

							box_wh[0] = EXPONENTIAL(trt_outputs[layerIdx][anchorIdx][2][HeightIdx][WidthIdx]) * yolo_anchors[0];
							box_wh[1] = EXPONENTIAL(trt_outputs[layerIdx][anchorIdx][3][HeightIdx][WidthIdx]) * yolo_anchors[1];

							box_wh[0] /= m_width;
							box_wh[1] /= m_height;

							box_xy[0] -= (float)(box_wh[0] / 2.0);
							box_xy[1] -= (float)(box_wh[1] / 2.0);

							out_box_e.boxes[0] = box_xy[0];
							out_box_e.boxes[1] = box_xy[1];
							out_box_e.boxes[2] = box_wh[0];
							out_box_e.boxes[3] = box_wh[1];

							out_box_e.box_confidence = (float)SIGMOID(trt_outputs[layerIdx][anchorIdx][4][HeightIdx][WidthIdx]);

							//Idx5以降はカテゴリが入る。今回は１なので、5だけだが、初期の80の場合は、5から85まで回す事になる。
							for (int ClsIdx = 0; ClsIdx < m_class_num; ClsIdx++) {
								float class_probs = (float)SIGMOID(trt_outputs[layerIdx][anchorIdx][5 + ClsIdx][HeightIdx][WidthIdx]);
								float box_scores = out_box_e.box_confidence * class_probs;

								out_box_e.box_class_probs.push_back(class_probs);
								out_box_e.box_scores.push_back(box_scores);
							}
							//DEBUG 途中結果確認用 出力 Loop=1のみ使用可。(削除しないで下さい）
							//						Debuglog_Ope1  	<< trt_outputs[layerIdx][anchorIdx][0][HeightIdx][WidthIdx] << ","
							//										<< trt_outputs[layerIdx][anchorIdx][1][HeightIdx][WidthIdx] << ","
							//										<< trt_outputs[layerIdx][anchorIdx][2][HeightIdx][WidthIdx] << ","
							//										<< trt_outputs[layerIdx][anchorIdx][3][HeightIdx][WidthIdx] << ","
							//										<< trt_outputs[layerIdx][anchorIdx][4][HeightIdx][WidthIdx] << ",";
							//						for (int ClsIdx = 0; ClsIdx < m_class_num-1; ClsIdx++) {
							//							Debuglog_Ope1  	<< trt_outputs[layerIdx][anchorIdx][5+ClsIdx][HeightIdx][WidthIdx]  << ",";
							//						}
							//						Debuglog_Ope1  	<< trt_outputs[layerIdx][anchorIdx][5+m_class_num-1][HeightIdx][WidthIdx]  << std::endl;
							//DEBUG
							out_box_w.push_back(out_box_e);
						}
						out_box_h.push_back(out_box_w);
					}
					out_box_elm.push_back(out_box_h);
				}

				for (int h = 0; h < m_grid_height[layerIdx]; h++) {
					for (int w = 0; w < m_grid_width[layerIdx]; w++) {
						float max_score[3];
						float sv_boxes[3][4];
						int max_score_idx[3];
						for (int anchorIdx = 0; anchorIdx < 3; anchorIdx++) {
							max_score[anchorIdx] = 0.0;
							max_score_idx[anchorIdx] = 0;
							//アンカー内最大スコアのクラスIDX,スコア、BOX位置
							for (int ClsIdx = 0; ClsIdx < m_class_num; ClsIdx++) {
								if (max_score[anchorIdx] < out_box_elm[h][w][anchorIdx].box_scores[ClsIdx]) {
									max_score_idx[anchorIdx] = ClsIdx;
									max_score[anchorIdx] = out_box_elm[h][w][anchorIdx].box_scores[ClsIdx];
									sv_boxes[anchorIdx][0] = out_box_elm[h][w][anchorIdx].boxes[0];
									sv_boxes[anchorIdx][1] = out_box_elm[h][w][anchorIdx].boxes[1];
									sv_boxes[anchorIdx][2] = out_box_elm[h][w][anchorIdx].boxes[2];
									sv_boxes[anchorIdx][3] = out_box_elm[h][w][anchorIdx].boxes[3];
								}
							}
							//DEBUG 途中結果確認用 出力 Loop=1のみ使用可。(削除しないで下さい）
							//						Debuglog_Ope2 << sv_boxes[anchorIdx][0] <<"," << sv_boxes[anchorIdx][1] <<"," << sv_boxes[anchorIdx][2] << ",";
							//						Debuglog_Ope2 << sv_boxes[anchorIdx][3] <<"," << max_score_idx[anchorIdx] <<"," << max_score[anchorIdx] << std::endl;
							//DEBUG
						}

						//閾値以上の物に絞る
						for (int anchorIdx = 0; anchorIdx < 3; anchorIdx++) {
							if (max_score[anchorIdx] >= m_score_threshold) {
								std::vector<float> ResBox;
								ResBox.push_back(sv_boxes[anchorIdx][0] * orig_image_size.width);
								ResBox.push_back(sv_boxes[anchorIdx][1] * orig_image_size.height);
								ResBox.push_back(sv_boxes[anchorIdx][2] * orig_image_size.width);
								ResBox.push_back(sv_boxes[anchorIdx][3] * orig_image_size.height);
								ResBox.push_back((float)max_score_idx[anchorIdx]);
								ResBox.push_back(max_score[anchorIdx]);
								ResBoxes.push_back(ResBox);
							}
						}
					}
				}
			}
			output_result.push_back(ResBoxes);
		}
	}

	void TRT_YOLOv3::PostProcess_sequence(
		const TRT::CPU_buffer& buffer,
		const std::vector<void*>& gpu_buffers,
		const std::vector<cv::Size>& original_image_sizes,
		const int super_batch_id,
		std::vector< std::vector<std::vector<float> > >& output_result) const
	{
		std::vector<float> MidBuffer(buffer.numInput * m_outputSize_list[0] +
			buffer.numInput * m_outputSize_list[1] +
			buffer.numInput * m_outputSize_list[2]);

		//	clock_t startClock, endClock;       // clock_t型変数宣言
		//	startClock = clock();               // 開始時刻のcpu時間取得

			//Operater1
		std::transform(MidBuffer.begin(), MidBuffer.end(), MidBuffer.begin(),
			operator1(buffer.out_buffer, MidBuffer, m_UnitLen, buffer.numInput, m_GridSize, m_anchor_size)); //正式にはm_anchor_sizeはanchorで可変出来る様にしないとダメだけど。暫定的に０だけ使用

	//DEBUG 途中結果確認用 出力 Loop=1のみ使用可。(削除しないで下さい）
	//	int gpuMid1Size = MidBuffer.size();
	//	thrust::host_vector<float> HostBufferOpe1(gpuMid1Size);
	//	thrust::copy(MidBuffer.begin(), MidBuffer.end(), HostBufferOpe1.begin());
	//	{
	//		char FileName[256];
	//		sprintf(FileName,"YOLOv3_sequence_Ope1_b%02d.txt",buffer.numInput);
	//		std::ofstream Debuglog_Ope1(FileName);
	//		int cnt = 0;
	//		for(int idx = 0; idx < gpuMid1Size; idx++) {
	//			Debuglog_Ope1 << HostBufferOpe1[idx];
	//			cnt++;
	//			if(cnt%m_UnitLen == 0){
	//				Debuglog_Ope1 << std::endl;
	//			}
	//			else{
	//				Debuglog_Ope1 << ",";
	//			}
	//		}
	//		Debuglog_Ope1.close();
	//	}
	//DEBUG

	//	endClock = clock();                 // 開始時刻のcpu時間取得
	//	double mcpusec = (endClock - startClock)/(double)CLOCKS_PER_SEC*1000; // cpu時間を計算
	//	printf("std::trasform PostProcess Ope1 :%.6lf ms\n",mcpusec);


		std::vector<float> OutBuffer(buffer.numInput * m_OutSize_list[0] +
			buffer.numInput * m_OutSize_list[1] +
			buffer.numInput * m_OutSize_list[2]);


		//Operater2
		std::transform(OutBuffer.begin(), OutBuffer.end(), OutBuffer.begin(),
			operator2(MidBuffer.data(), OutBuffer.data(), m_UnitLen, m_OutUnitLen, m_grid_width, m_grid_height,
				m_GridSize, m_anchor_size, m_OutAnchorSize, original_image_sizes[0].width, original_image_sizes[0].height,
				m_width, m_height, m_class_num));

		//DEBUG 途中結果確認用 出力 Loop=1のみ使用可。(削除しないで下さい）
		//	int gpuMid2Size = OutBuffer.size();
		//	thrust::host_vector<float> HostBufferOpe2(gpuMid2Size);
		//	thrust::copy(OutBuffer.begin(), OutBuffer.end(), HostBufferOpe2.begin());
		//	{
		//		char FileName[256];
		//		sprintf(FileName,"YOLOv3_sequence_Ope2_b%02d.txt",buffer.numInput);
		//		std::ofstream Debuglog_Ope2(FileName);
		//		int cnt = 0;
		//		for(int idx = 0; idx < gpuMid2Size; idx++) {
		//			Debuglog_Ope2 << HostBufferOpe2[idx];
		//			cnt++;
		//			if(cnt%6 == 0){
		//				Debuglog_Ope2 << std::endl;
		//			}
		//			else{
		//				Debuglog_Ope2 << ",";
		//			}
		//		}
		//		Debuglog_Ope2.close();
		//	}
		//DEBUG

				//閾値以上の物に絞る
		int Out1BatchLen = (int)OutBuffer.size() / buffer.numInput;
		int numUnits = Out1BatchLen / m_OutUnitLen;
		for (int batch_idx = 0; batch_idx < buffer.numInput; batch_idx++) {
			cv::Size orig_image_size = original_image_sizes[batch_idx];
			std::vector<std::vector<float>> ResBoxes;
			for (int UnitIdx = 0; UnitIdx < numUnits; UnitIdx++) {
				uint bufidx = Out1BatchLen * batch_idx + UnitIdx * m_OutUnitLen;
				if (OutBuffer[bufidx + 5] >= m_score_threshold) {
					std::vector<float> ResBox;
					ResBox.push_back(OutBuffer[bufidx + 0] * orig_image_size.width);
					ResBox.push_back(OutBuffer[bufidx + 1] * orig_image_size.height);
					ResBox.push_back(OutBuffer[bufidx + 2] * orig_image_size.width);
					ResBox.push_back(OutBuffer[bufidx + 3] * orig_image_size.height);
					ResBox.push_back(OutBuffer[bufidx + 4]);
					ResBox.push_back(OutBuffer[bufidx + 5]);
					ResBoxes.push_back(ResBox);
				}
			}
			output_result.push_back(ResBoxes);
		}
	}
}
