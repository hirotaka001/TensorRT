#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/TRT_YOLOv3/TRT_YOLOv3_PostProcess_cuda.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <time.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	struct operator1 {
		float* m_in_buffer1;
		float* m_in_buffer2;
		float* m_in_buffer3;
		float* m_midTopPtr;
		int m_UnitLen;  //detection結果と求める為の一番小さい単位（x,y,w,h,conf,class1...classn）クラスを１だとすると、x,y,w,h,conf,class1で6となる。
		int m_GridSize[3];
		int m_InBufSize[3];
		int m_numbatch;
		int m_AnchorSize[3];
		int m_1BatchLen;
		int m_LayerBatchSize[3];  //leyer内の１バッチのサイズ。１バッチのサイズはlayerに寄って異なる。

		operator1(void* in_Buffer1, void* in_Buffer2, void* in_Buffer3, float* midTopPtr, const int UnitLen, const int nBatch, const int* GridSize, const int* AnchorSize) :
			m_midTopPtr(midTopPtr), m_UnitLen(UnitLen), m_numbatch(nBatch) {

			m_in_buffer1 = (float*)in_Buffer1;
			m_in_buffer2 = (float*)in_Buffer2;
			m_in_buffer3 = (float*)in_Buffer3;
			m_GridSize[0] = GridSize[0];
			m_GridSize[1] = GridSize[1];
			m_GridSize[2] = GridSize[2];

			m_AnchorSize[0] = AnchorSize[0];
			m_AnchorSize[1] = AnchorSize[1];
			m_AnchorSize[2] = AnchorSize[2];

			m_LayerBatchSize[0] = m_AnchorSize[0] * m_GridSize[0];
			m_LayerBatchSize[1] = m_AnchorSize[1] * m_GridSize[1];
			m_LayerBatchSize[2] = m_AnchorSize[2] * m_GridSize[2];

			m_InBufSize[0] = m_LayerBatchSize[0] * m_numbatch;
			m_InBufSize[1] = m_LayerBatchSize[1] * m_numbatch;
			m_InBufSize[2] = m_LayerBatchSize[2] * m_numbatch;

			m_1BatchLen = m_LayerBatchSize[0] + m_LayerBatchSize[1] + m_LayerBatchSize[2];
		}
		//ValueはInBuffer側で回す。Leyer単位でtranform
		__device__ float operator () (float& Value) const {
			int Cur_pos = &Value - m_midTopPtr;

			int BatchNo = Cur_pos / m_1BatchLen;
			int BatchInterIdx = Cur_pos % m_1BatchLen;
			int BatchOffset;
			float* in_Buffer;
			int layerBatchSize;
			int AnchorSize;
			int GridSize;
			if (BatchInterIdx < m_LayerBatchSize[0]) {
				in_Buffer = m_in_buffer1;
				BatchOffset = BatchInterIdx;
				layerBatchSize = m_LayerBatchSize[0];
				AnchorSize = m_AnchorSize[0];
				GridSize = m_GridSize[0];
			}
			else if (BatchInterIdx < (m_LayerBatchSize[0] + m_LayerBatchSize[1])) {
				in_Buffer = m_in_buffer2;
				BatchOffset = (BatchInterIdx - m_LayerBatchSize[0]);
				layerBatchSize = m_LayerBatchSize[1];
				AnchorSize = m_AnchorSize[1];
				GridSize = m_GridSize[1];
			}
			else {
				in_Buffer = m_in_buffer3;
				BatchOffset = (BatchInterIdx - m_LayerBatchSize[0] - m_LayerBatchSize[1]);
				layerBatchSize = m_LayerBatchSize[2];
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
		int m_Width;
		int m_Height;
		int m_ClassNum;

		operator2(float* midTopPtr, float* outTopPtr, const int MidUnitLen, const int OutUnitLen,
			const int* GridWidth, const int* GridHeight, const int* GridSize, const int* MidAnchorSize, const int* OutAnchorSize,
			const int Width, const int Height, const int ClassNum) :
			m_midTopPtr(midTopPtr), m_outTopPtr(outTopPtr), m_MidUnitLen(MidUnitLen), m_OutUnitLen(OutUnitLen),
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
		}

		//ValueはInBuffer側で回す。Leyer単位でtranform
		__device__ float operator () (float& Value) const {
			int OutLayerOffset;
			int MidLayerOffset;
			int MidAnchorSize;
			int OutAnchorSize;
			int GridWidth;
			int GridHeight;
			int LayerNo;
			//Out側の現在位置
			int Cur_pos = &Value - m_outTopPtr;

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
			case 0: //X
			{
				float val = 1.0 / (1.0 + exp(-(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 0])));
				int WidthPos = OutGridNo % GridWidth;
				val += (float)WidthPos;
				val /= (float)GridWidth;

				//Wの先行読み込み
				int yolo_anchorW = m_AnchorX[LayerNo * 3 + OutAnchorNo];
				float W = exp(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 2]) * yolo_anchorW;
				W /= (float)m_Width;
				val -= (W / 2.0);
				res = val;
				break;
			}
			case 1:  //Y
			{
				float val = 1.0 / (1.0 + exp(-(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 1])));
				int HeightPos = OutGridNo / GridWidth;
				val += (float)HeightPos;
				val /= (float)GridHeight;
				//Hの先行読み込み
				int yolo_anchorH = m_AnchorY[LayerNo * 3 + OutAnchorNo];
				float H = exp(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 3]) * yolo_anchorH;
				H /= (float)m_Height;
				val -= (H / 2.0);
				res = val;
				break;
			}
			case 2:  //W
			{
				int yolo_anchorW = m_AnchorX[LayerNo * 3 + OutAnchorNo];
				float val = exp(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 2]) * yolo_anchorW;
				val /= (float)m_Width;
				res = val;
				break;
			}
			case 3:  //H
			{
				int yolo_anchorH = m_AnchorY[LayerNo * 3 + OutAnchorNo];
				float val = exp(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 3]) * yolo_anchorH;
				val /= (float)m_Height;
				res = val;
				break;
			}
			case 4:  //ClassID
			case 5:  //Score
			{
				float confidence = 1.0 / (1.0 + exp(-(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 4])));
				float MaxScore = 0.0;
				int   MaxScoreClsID = 0;
				for (int ClsIdx = 0; ClsIdx < m_ClassNum; ClsIdx++) {
					float class_probs = 1.0 / (1.0 + exp(-(m_midTopPtr[m_Mid1BatchLen * BatchNo + MidLayerOffset + MidAnchorSize * OutGridNo + m_MidUnitLen * OutAnchorNo + 5 + ClsIdx])));
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

	TRT_YOLOv3_PostProcess_cuda::TRT_YOLOv3_PostProcess_cuda() {
	}

	TRT_YOLOv3_PostProcess_cuda::~TRT_YOLOv3_PostProcess_cuda() {
		m_Mid1Buffer.clear();
		m_Mid2Buffer.clear();
	}


	void TRT_YOLOv3_PostProcess_cuda::Init(const st_argumets& argumets, const int* GridWidth, const int* GridHeight) {
		m_class_num = argumets.class_num;
		m_num_batch = argumets.num_batch;
		m_num_super_batch = argumets.num_super_batch;

		memcpy(m_GridWidth, GridWidth, sizeof(m_GridWidth));
		memcpy(m_GridHeight, GridHeight, sizeof(m_GridHeight));
		m_width = argumets.width;
		m_height = argumets.height;
		m_threshold = argumets.threshold;
		m_MidUnitLen = argumets.MidUnitLen;
		m_OutUnitLen = argumets.OutUnitLen;

		m_GridSize[0] = m_GridWidth[0] * m_GridHeight[0];
		m_GridSize[1] = m_GridWidth[1] * m_GridHeight[1];
		m_GridSize[2] = m_GridWidth[2] * m_GridHeight[2];

		m_MidAnchorSize[0] = m_MidUnitLen * 3;			//ユニットが3アンカー分
		m_MidAnchorSize[1] = m_MidUnitLen * 3;
		m_MidAnchorSize[2] = m_MidUnitLen * 3;

		m_MidSize_list.push_back(m_MidAnchorSize[0] * m_GridSize[0]);    //アンカーサイズ*グリッド数=1バッチ分
		m_MidSize_list.push_back(m_MidAnchorSize[1] * m_GridSize[1]);
		m_MidSize_list.push_back(m_MidAnchorSize[2] * m_GridSize[2]);

		m_OutAnchorSize[0] = m_OutUnitLen * 3;			//Ope2出力側ユニットが3アンカー分
		m_OutAnchorSize[1] = m_OutUnitLen * 3;
		m_OutAnchorSize[2] = m_OutUnitLen * 3;

		m_OutSize_list.push_back(m_OutAnchorSize[0] * m_GridSize[0]);    //Ope2出力側アンカーサイズ*グリッド数=1バッチ分
		m_OutSize_list.push_back(m_OutAnchorSize[1] * m_GridSize[1]);
		m_OutSize_list.push_back(m_OutAnchorSize[2] * m_GridSize[2]);

		m_gpuMid1Size = m_MidSize_list[0] + m_MidSize_list[1] + m_MidSize_list[2];
		m_gpuMid2Size = m_OutSize_list[0] + m_OutSize_list[1] + m_OutSize_list[2];

		//
		for (int i = 0; i < m_num_super_batch; i++) {
			thrust::device_vector<float> Mid1Buffer(m_num_batch * m_gpuMid1Size);
			m_Mid1Buffer.push_back(Mid1Buffer);
		}

		for (int i = 0; i < m_num_super_batch; i++) {
			thrust::device_vector<float> Mid21Buffer(m_num_batch * m_gpuMid2Size);
			m_Mid2Buffer.push_back(Mid21Buffer);
		}

	}


	void TRT_YOLOv3_PostProcess_cuda::Exec(const std::vector<void*>& gpu_buffers,
		const std::vector<cv::Size>& original_image_sizes,
		const int super_batch_id,
		std::vector< std::vector<std::vector<float> > >& output_result)
	{

		//------operator1------
		float* Mid1BufferPtr = thrust::raw_pointer_cast(m_Mid1Buffer[super_batch_id].data());
		thrust::transform(m_Mid1Buffer[super_batch_id].begin(), m_Mid1Buffer[super_batch_id].end(), m_Mid1Buffer[super_batch_id].begin(),
			operator1(gpu_buffers[0], gpu_buffers[1], gpu_buffers[2], Mid1BufferPtr, m_MidUnitLen, m_num_batch, m_GridSize, m_MidAnchorSize));

		//DEBUG  Ope1 途中結果確認用 出力 Loop=1のみ使用可。(削除しないで下さい）
		//	int gpuMid1Size = Mid1Buffer.size();
		//	thrust::host_vector<float> HostBufferOpe1(gpuMid1Size);
		//	thrust::copy(Mid1Buffer.begin(), Mid1Buffer.end(), HostBufferOpe1.begin());
		//	{
		//		char FileName[256];
		//		sprintf(FileName,"YOLOv3_Thrust_Ope1_b%02d.txt",m_num_batch);
		//		std::ofstream Debuglog_Ope1(FileName);
		//		int cnt = 0;
		//		for(int idx = 0; idx < gpuMid1Size; idx++) {
		//			Debuglog_Ope1 << HostBufferOpe1[idx];
		//			cnt++;
		//			if(cnt%m_MidUnitLen == 0){
		//				Debuglog_Ope1 << std::endl;
		//			}
		//			else{
		//				Debuglog_Ope1 << ",";
		//			}
		//		}
		//		Debuglog_Ope1.close();
		//	}
		//DEBUG

				//------operator2------
		float* Mid2BufferPtr = thrust::raw_pointer_cast(m_Mid2Buffer[super_batch_id].data());
		thrust::transform(m_Mid2Buffer[super_batch_id].begin(), m_Mid2Buffer[super_batch_id].end(), m_Mid2Buffer[super_batch_id].begin(),
			operator2(Mid1BufferPtr, Mid2BufferPtr, m_MidUnitLen, m_OutUnitLen, m_GridWidth, m_GridHeight, m_GridSize,
				m_MidAnchorSize, m_OutAnchorSize, m_width, m_height, m_class_num));


		//------閾値で判定------
		int gpuMid2Size = (int)m_Mid2Buffer[super_batch_id].size();
		std::vector<float> HostBufferOpe2(gpuMid2Size);

		cudaMemcpy(HostBufferOpe2.data(), thrust::raw_pointer_cast(m_Mid2Buffer[super_batch_id].data()), gpuMid2Size * sizeof(float), cudaMemcpyDeviceToHost);

		//DEBUG   Ope1 途中結果確認用 出力 Loop=1のみ使用可。(削除しないで下さい）
		//	{
		//		char FileName[256];
		//		sprintf(FileName,"YOLOv3_Thrust_Ope2_b%02d.txt",m_num_batch);
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

		int Out1BatchLen = (int)HostBufferOpe2.size() / m_num_batch;
		int numUnits = Out1BatchLen / m_OutUnitLen;
		for (int batch_idx = 0; batch_idx < m_num_batch; batch_idx++) {
			int org_width = original_image_sizes[batch_idx].width;
			int org_height = original_image_sizes[batch_idx].height;
			std::vector<std::vector<float> > ResBoxes;
			for (int UnitIdx = 0; UnitIdx < numUnits; UnitIdx++) {
				int bufidx = Out1BatchLen * batch_idx + UnitIdx * m_OutUnitLen;
				if (HostBufferOpe2[bufidx + 5] >= m_threshold) {
					std::vector<float> ResBox;
					ResBox.push_back(HostBufferOpe2[bufidx + 0] * (float)org_width);
					ResBox.push_back(HostBufferOpe2[bufidx + 1] * (float)org_height);
					ResBox.push_back(HostBufferOpe2[bufidx + 2] * (float)org_width);
					ResBox.push_back(HostBufferOpe2[bufidx + 3] * (float)org_height);
					ResBox.push_back(HostBufferOpe2[bufidx + 4]);
					ResBox.push_back(HostBufferOpe2[bufidx + 5]);
					ResBoxes.push_back(ResBox);
				}
			}
			output_result.push_back(ResBoxes);
		}
	}
}
