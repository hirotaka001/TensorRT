#define TAG "TRT_CPN"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/TRT_CPN/TRT_CPN.h>
#include <algorithm>
#include <omp.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace kros
{
	TRT_CPN::TRT_CPN()
	{
	}

	TRT_CPN::~TRT_CPN()
	{
		Release();
	}

	bool TRT_CPN::Init(
		const int gpu_id,
		const std::string& model_name,
		const std::string& model_dir,
		const int width,
		const int height,
		const int batch,
		const int super_batch,
		const int thread_num,
		const int resize_margin,
		const bool keep_aspect,
		const std::string& input_layers,
		const std::string& output_layers)
	{
		m_output_c = 17;
		m_output_h = height / 4;
		m_output_w = width / 4;
		m_batch = batch;
		m_super_batch = super_batch;
		m_thread_num = thread_num;

#ifdef _OPENMP
		if (m_thread_num > omp_get_num_procs()) {
			std::cout << "[" << TAG << "] Error : thread_num must be set within the number of processors" << std::endl;
			return false;
		}
		omp_set_num_threads(m_thread_num);
#endif

		std::vector<std::string> output_layers2 = { "refine_out/Conv2D" };

		m_resize_margin = resize_margin;
		m_keep_aspect = keep_aspect;
		m_inside_w = width - (m_resize_margin * 2); //左右のマージンなのでx2
		m_inside_h = height - (m_resize_margin * 2);
		m_aspect = (float)m_inside_w / (float)m_inside_h;

		cv::Scalar val((int)m_means[0], (int)m_means[1], (int)m_means[2]);
		cv::Mat backmat(cv::Size(width, height), CV_8UC3, val);
		m_bachImg = backmat.clone();

		m_method = SECOND_POINT_MODIFICATION;
		m_outputSize_list.push_back(m_output_h * m_output_w * m_output_c);

		return TRT::Init(gpu_id, model_name, model_dir, input_layers, output_layers2, width, height, batch, super_batch, thread_num);
	}

	void TRT_CPN::Release() {
		TRT::Release();
	}

	bool TRT_CPN::InferParallel(
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


	bool TRT_CPN::Infer(
		const std::vector<cv::Mat>& input_images,
		std::vector<std::vector<std::vector<float> > >& detection_result,
		const int thread_id)
	{
		bool ret = TRT::Infer(input_images, detection_result, thread_id);

		if (ret) {
			return true;
		}
		else {
			return false;
		}
		return false;
	}

	void TRT_CPN::PreProcess(
		const std::vector<cv::Mat>& input_images,
		TRT::CPU_buffer& buffer,
		float* GPU_In_buffers,
		const cudaStream_t stream,
		const int thread_id)
	{
		const int NumInputData = (int)input_images.size();
		buffer.numInput = NumInputData;

		cv::Mat resized;
		vector<cv::Mat> planes;
		cv::Mat planeR;
		cv::Mat planeG;
		cv::Mat planeB;

		float* inbufpos = buffer.in_buffer;
		for (int i = 0; i < NumInputData; i++) {

			keep_aspect_resize(input_images[i], resized);

			//      Resize直後のInferに入力する画像をテスト的に出力
			//		cv::imwrite("resized_image.png", resized);

			// 3つのチャネルB, G, Rに分離。
			cv::split(resized, planes);
			//正規化 （8bit実数から32bit整数変換も同時に行う。
			planes[0].convertTo(planeB, CV_32F, 1.0f / 255.0f, -m_means[0] / 255.0f); //B
			planes[1].convertTo(planeG, CV_32F, 1.0f / 255.0f, -m_means[1] / 255.0f); //G
			planes[2].convertTo(planeR, CV_32F, 1.0f / 255.0f, -m_means[2] / 255.0f); //R

			//1次元配列へ変換。Channel firstで、BGRの順番にする。（本来ならRGBの順番のはずだが、作者の作りからに習う）
			memcpy(inbufpos, planeB.data, planeB.total() * planeR.elemSize());
			inbufpos += planeB.total();

			memcpy(inbufpos, planeG.data, planeG.total() * planeR.elemSize());
			inbufpos += planeG.total();

			memcpy(inbufpos, planeR.data, planeR.total() * planeR.elemSize());
			inbufpos += planeR.total();
		}
	}

	void TRT_CPN::PostProcess(
		const TRT::CPU_buffer& buffer,
		const std::vector<void*>& gpu_buffers,
		const std::vector<cv::Size>& original_image_sizes,
		const int super_batch_id,
		const int thread_id,
		std::vector< std::vector<std::vector<float> > >& detection_result) const
	{
		std::vector<std::vector<float>> pose_values_vec;
		std::vector<float> confidence_map(m_outputSize_list[0] * buffer.numInput), origin_confidence_map(m_outputSize_list[0] * buffer.numInput);

		for (int i = 0; i < m_outputSize_list[0] * buffer.numInput; i++) {
			confidence_map[i] = buffer.out_buffer[0][i];
			origin_confidence_map[i] = buffer.out_buffer[0][i];
		}

		for (int n = 0; n < buffer.numInput; n++) {
			int BufferOffset = n * m_outputSize_list[0];
			std::vector<float> pose_values;
			// [y][x][c]
			std::vector<std::vector<std::vector<float>>> cpn_output = std::vector<std::vector<std::vector<float>>>
				(m_output_h, std::vector<std::vector<float>>(m_output_w, std::vector<float>(m_output_c, 0)));

			std::vector<std::vector<std::vector<float>>> origin_cpn_output = std::vector<std::vector<std::vector<float>>>
				(m_output_h, std::vector<std::vector<float>>(m_output_w, std::vector<float>(m_output_c, 0)));

			std::vector<std::vector<std::vector<float>>> chw_origin_cpn_output = std::vector<std::vector<std::vector<float>>>
				(m_output_c, std::vector<std::vector<float>>(m_output_h, std::vector<float>(m_output_w, 0)));

			// vector of (int y, int x)
			std::vector<std::vector<int>> max_point_positons = std::vector<std::vector<int>>(m_output_c, std::vector<int>(2, 0));
			std::vector<std::vector<int>> second_highest_point_positons = std::vector<std::vector<int>>(m_output_c, std::vector<int>(2, 0));

			// set values for parse cpn output
			for (int y = 0; y < m_output_h; y++) {
				int y_start = y * m_output_w * m_output_c;

				for (int x = 0; x < m_output_w; x++) {
					int x_start = y_start + x * m_output_c;

					for (int c = 0; c < m_output_c; c++) {
						cpn_output[y][x][c] = confidence_map[BufferOffset + x_start + c];
						origin_cpn_output[y][x][c] = origin_confidence_map[BufferOffset + x_start + c];
						chw_origin_cpn_output[c][y][x] = origin_confidence_map[BufferOffset + x_start + c];

						if (cpn_output[y][x][c] > cpn_output[second_highest_point_positons[c][0]][second_highest_point_positons[c][1]][c]) {
							if (cpn_output[y][x][c] > cpn_output[max_point_positons[c][0]][max_point_positons[c][1]][c]) {
								second_highest_point_positons[c][0] = max_point_positons[c][0];
								second_highest_point_positons[c][1] = max_point_positons[c][1];
								max_point_positons[c][0] = y;
								max_point_positons[c][1] = x;
							}
							else {
								second_highest_point_positons[c][0] = y;
								second_highest_point_positons[c][1] = x;
							}
						}
					}
				}
			}

			if (m_method == SECOND_POINT_MODIFICATION || m_method == GAUSSIAN_FILTER_AND_SECOND_POINT_MODIFICATION) {
				for (int c = 0; c < m_output_c; c++) {
					float x_diff = (float)(max_point_positons[c][1] - second_highest_point_positons[c][1]),
						y_diff = (float)(max_point_positons[c][0] - second_highest_point_positons[c][0]);
					double dist = std::hypot(x_diff, y_diff);

					if (dist > m_min_value_of_cpn_output_modfiy_by_second_highest) {
						float key_x = (float)(max_point_positons[c][1] + m_delta * x_diff / dist);
						float key_y = (float)(max_point_positons[c][0] + m_delta * y_diff / dist);
						pose_values.push_back(key_x);
						pose_values.push_back(key_y);
					}
					else {
						pose_values.push_back((float)max_point_positons[c][1]);
						pose_values.push_back((float)max_point_positons[c][0]);
					}

					// dummy
					pose_values.push_back(0);
				}
			}
			else {
				for (int c = 0; c < m_output_c; c++) {
					pose_values.push_back((float)max_point_positons[c][1]);
					pose_values.push_back((float)max_point_positons[c][0]);
					// dummy
					pose_values.push_back(0);
				}
			}

			for (int i = 0; i < (int)pose_values.size(); i += 3) {
				int c_ = i / 3;
				int round_x = (int)std::round(pose_values[i]),
					round_y = (int)std::round(pose_values[i + 1]);

				if (round_x > m_output_w) {
					round_x = m_output_w - 1;
				}

				if (round_y > m_output_h) {
					round_y = m_output_h - 1;
				}

				float confidence = (float)(origin_cpn_output[round_y][round_x][c_] / 255 + 0.5);
				pose_values[i + 2] = confidence;
			}

			pose_values_vec.push_back(pose_values);
		}

		// map to cpn input size.
		for (auto pose_values_itr = pose_values_vec.begin(); pose_values_itr != pose_values_vec.end(); pose_values_itr++) {
			for (int i = 0; i < (int)(*pose_values_itr).size(); i += 3) {
				if ((*pose_values_itr)[i] < 0) {
					(*pose_values_itr)[i] = 0;
				}
				else if ((*pose_values_itr)[i] > m_output_w - 1) {
					(*pose_values_itr)[i] = (float)(m_output_w - 1);
				}

				(*pose_values_itr)[i] = (*pose_values_itr)[i] * 4 + 2;

				if ((*pose_values_itr)[i + 1] < 0) {
					(*pose_values_itr)[i + 1] = 0;
				}
				else if ((*pose_values_itr)[i + 1] > m_output_h - 1) {
					(*pose_values_itr)[i + 1] = (float)(m_output_h - 1);
				}

				(*pose_values_itr)[i + 1] = (*pose_values_itr)[i + 1] * 4 + 2;

				if ((*pose_values_itr)[i + 2] < 0) {
					(*pose_values_itr)[i + 2] = 0;
				}
				else if ((*pose_values_itr)[i + 2] > 1) {
					(*pose_values_itr)[i + 2] = 1;
				}
			}
		}

		for (auto pose_values_itr = pose_values_vec.begin(); pose_values_itr != pose_values_vec.end(); pose_values_itr++) {
			std::vector<std::vector<float>> ResConnect2;
			std::vector<float> ResConnect;

			ResConnect2.clear();
			ResConnect.clear();

			for (int i = 0; i < (int)(*pose_values_itr).size(); i++) {
				if ((i % 3) == 0 && i > 0) {
					ResConnect2.push_back(ResConnect);
					ResConnect.clear();
				}
				ResConnect.push_back((*pose_values_itr)[i]);
			}
			ResConnect2.push_back(ResConnect);
			detection_result.push_back(ResConnect2);
		}

		//Resize上のKeyPointを原画上の位置へ戻す
		calc_org_img_pos(original_image_sizes, detection_result);

	}

	void TRT_CPN::calc_org_img_pos(const std::vector<cv::Size>& original_image_sizes,
		std::vector< std::vector<std::vector<float> > >& detection_result) const
	{
		CV_Assert((int)original_image_sizes.size() == (int)detection_result.size());
		int nImage = (int)original_image_sizes.size();
		for (int i = 0; i < nImage; i++) {
			cv::Rect ResizeSize;
			calc_resize_size(original_image_sizes[i], ResizeSize);

			int nKetPosins = (int)detection_result[i].size();
			for (int j = 0; j < nKetPosins; j++) {
				detection_result[i][j][0] = (float)(int)((detection_result[i][j][0] - ResizeSize.x) * (float)original_image_sizes[i].width / (float)ResizeSize.width);
				detection_result[i][j][1] = (float)(int)((detection_result[i][j][1] - ResizeSize.y) * (float)original_image_sizes[i].height / (float)ResizeSize.height);
			}
		}
	}

	void TRT_CPN::DrawResult(
		const cv::Mat& input_images,
		cv::Mat& output_images,
		const std::vector<std::vector<float> >& detection_result,
		const bool print_keypoint_score)
	{
		if (!print_keypoint_score) {
			DrawKeyPoints(input_images, detection_result, output_images);
		}
		else {
			float image_mag = 1.0;
			cv::Mat points_img;
			if (input_images.rows < 512) {
				image_mag = 512.0f / (float)input_images.rows;
				cv::Size ReSize{ (int)(image_mag * (float)input_images.cols) ,512 };
				cv::Mat resized_img;
				cv::resize(input_images, resized_img, ReSize, cv::INTER_AREA);
				std::vector<std::vector<float> > resize_detection_result;
				for (int i = 0; i < (int)detection_result.size(); i++) {
					std::vector<float> pos;
					for (int j = 0; j < 2; j++) {
						pos.push_back((float)(int)((float)detection_result[i][j] * image_mag));
					}
					resize_detection_result.push_back(pos);
				}
				DrawKeyPoints(resized_img, resize_detection_result, points_img);
				DrawPointList(points_img, detection_result, output_images);
			}
			else {
				DrawKeyPoints(input_images, detection_result, points_img);
				DrawPointList(points_img, detection_result, output_images);
			}

		}
	}

	void TRT_CPN::DrawResult(
		const std::vector<cv::Mat>& input_images,
		std::vector<cv::Mat>& output_images,
		const std::vector<std::vector<std::vector<float> > >& detection_result,
		bool print_keypoint_score)
	{
		CV_Assert((int)input_images.size() == (int)detection_result.size());
		for (int i = 0; i < (int)input_images.size(); i++) {
			cv::Mat output_image;
			DrawResult(input_images[i], output_image, detection_result[i], print_keypoint_score);
			output_images.push_back(output_image);
		}

	}

	std::vector<float> TRT_CPN::GetKeyPoint(
		const int keypoint_idx,
		const std::vector<std::vector<float> >& detection_results)
	{
		return detection_results[keypoint_idx];
	}

	void TRT_CPN::DrawKeyPoints(const cv::Mat& input_images, const std::vector<std::vector<float> >& detection_result, cv::Mat& output_images) const
	{
		int nKeyPoints = (int)detection_result.size();

		CV_Assert(nKeyPoints == 17);

		output_images = input_images.clone();

		//画像サイズにより、ポイント及びラインの大きさ（太さ）を選択
		int line_thickness = 2;
		int point_thickness = 5;

		if (input_images.rows <= 256) {
			line_thickness = 1;
			point_thickness = 2;
		}
		else if (input_images.rows <= 512) {
			line_thickness = 2;
			point_thickness = 3;
		}
		else if (input_images.rows <= 1024) {
			line_thickness = 3;
			point_thickness = 6;
		}
		else if (input_images.rows <= 1280) {
			line_thickness = 3;
			point_thickness = 8;
		}
		else {
			line_thickness = 5;
			point_thickness = 10;
		}

		for (int i = 0; i < 18; i++) {
			cv::Scalar line_col(m_linecol[m_link_line[i][2]][0], m_linecol[m_link_line[i][2]][1], m_linecol[m_link_line[i][2]][2]);
			cv::Point pt1((int)detection_result[m_link_line[i][0]][0], (int)(int)detection_result[m_link_line[i][0]][1]);
			cv::Point pt2((int)detection_result[m_link_line[i][1]][0], (int)(int)detection_result[m_link_line[i][1]][1]);
			cv::line(output_images, pt1, pt2, line_col, line_thickness, CV_AVX);
		}
		for (int i = 0; i < nKeyPoints; i++) {
			cv::Scalar col(m_keycol[i][0], m_keycol[i][1], m_keycol[i][2]);
			cv::circle(output_images, cv::Point((int)detection_result[i][0], (int)(int)detection_result[i][1]), point_thickness, col, -1, CV_AVX);
		}
	}

	void TRT_CPN::DrawPointList(const cv::Mat& input_images, const std::vector<std::vector<float> >& detection_result, cv::Mat& output_images) const
	{
		cv::Scalar base_col(0, 0, 0);
		cv::Mat draw_points_list(cv::Size(450, input_images.rows), CV_8UC3, base_col);

		char PointName[16];
		char PosXStr[16];
		char PosYStr[16];
		char ScoreStr[16];
		int baseline = 0;
		double fontSize = 0.6;
		int fontFace = cv::FONT_HERSHEY_TRIPLEX;
		for (int i = 0; i < 17; i++) {
			sprintf(PointName, "%-10s", m_keyname[i].c_str());
			sprintf(PosXStr, "X:%4d", (int)detection_result[i][0]);
			sprintf(PosYStr, "Y:%4d", (int)detection_result[i][1]);
			sprintf(ScoreStr, "score:%6.4f", detection_result[i][2]);
			cv::Size txtSize = cv::getTextSize(PointName, fontFace, fontSize, 1, &baseline);
			cv::Scalar col(m_keycol[i][0], m_keycol[i][1], m_keycol[i][2]);
			cv::putText(draw_points_list, PointName, cv::Point(10, (i * (txtSize.height + 15)) + txtSize.height + 5), fontFace, fontSize, col, 1, cv::LINE_AA);
			cv::putText(draw_points_list, PosXStr, cv::Point(145, (i * (txtSize.height + 15)) + txtSize.height + 5), fontFace, fontSize, col, 1, cv::LINE_AA);
			cv::putText(draw_points_list, PosYStr, cv::Point(220, (i * (txtSize.height + 15)) + txtSize.height + 5), fontFace, fontSize, col, 1, cv::LINE_AA);
			cv::putText(draw_points_list, ScoreStr, cv::Point(300, (i * (txtSize.height + 15)) + txtSize.height + 5), fontFace, fontSize, col, 1, cv::LINE_AA);
		}

		output_images = cv::Mat(cv::Size(input_images.cols + draw_points_list.cols, input_images.rows), CV_8UC3);
		cv::Rect roi_rect;
		roi_rect.width = input_images.cols;
		roi_rect.height = input_images.rows;
		cv::Mat roi1(output_images, roi_rect);
		input_images.copyTo(roi1);
		roi_rect.x += input_images.cols;
		roi_rect.width = draw_points_list.cols;
		roi_rect.height = draw_points_list.rows;
		cv::Mat roi2(output_images, roi_rect);
		draw_points_list.copyTo(roi2);
	}

	void TRT_CPN::keep_aspect_resize(const cv::Mat& inImg, cv::Mat& OutImg)  const {
		cv::Mat resized_img;
		OutImg = m_bachImg.clone(); //上書きするので毎回コピー

		//Resiseをとオフセット値取得
		cv::Size OrgSize{ inImg.cols,inImg.rows };
		cv::Rect ResizeSize;
		calc_resize_size(OrgSize, ResizeSize);

		cv::resize(inImg, resized_img, cv::Size(ResizeSize.width, ResizeSize.height), cv::INTER_AREA);

		cv::Mat roi_dst = OutImg(ResizeSize);
		resized_img.copyTo(roi_dst);

	}

	void TRT_CPN::calc_resize_size(const cv::Size& OrgSize, cv::Rect& ResizeSize) const {

		float Inimg_aspect = (float)OrgSize.width / (float)OrgSize.height;

		int resize_w;
		int resize_h;

		if (m_keep_aspect == false) {
			resize_w = m_inside_w;
			resize_h = m_inside_h;
		}
		else {
			if (m_aspect <= Inimg_aspect) {
				resize_w = m_inside_w;
				resize_h = (int)(((float)m_inside_w / (float)OrgSize.width) * (float)OrgSize.height);
			}
			else {
				resize_w = (int)(((float)m_inside_h / (float)OrgSize.height) * (float)OrgSize.width);
				resize_h = m_inside_h;
			}
		}
		int offset_x = (int)((float)(m_width - resize_w) / 2.0f);
		int offset_y = (int)((float)(m_height - resize_h) / 2.0f);;

		ResizeSize.x = offset_x;
		ResizeSize.y = offset_y;
		ResizeSize.width = resize_w;
		ResizeSize.height = resize_h;
	}
}
