#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <kros/trt/Common/PreProcess_cuda.h>
#include <stdio.h>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _WIN32
#pragma warning(pop)
#endif

__global__ void Bilinear24Kernel(unsigned char* in, float* out, int width_in, int height_in, int channels, int width_out, int height_out) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	// 拡大・縮小率に合わせて入力画像の座標を調整
	float x_in = (float)x * ((float)width_in / width_out);
	float y_in = (float)y * ((float)height_in / height_out);

	// バッファオーバーフローしないように調整
	if (x < width_out && y < height_out) {
		if ((int)x_in >= width_in) {
			x_in = width_in - 1;
		}
		if ((int)y_in >= height_in) {
			y_in = height_in - 1;
		}

		float after_point_x = (x_in - (int)x_in);
		float after_point_y = (y_in - (int)y_in);
		float k1 = (1.0f - after_point_x) * (1.0f - after_point_y);
		float k2 = (after_point_x) * (1.0f - after_point_y);
		float k3 = (1.0f - after_point_x) * (after_point_y);
		float k4 = (after_point_x) * (after_point_y);
		//B
		out[(width_out * height_out * 2) + y * width_out + x] = (((k1 * (float)in[(int)y_in * width_in * 3 + (int)x_in * 3 + 0]
			+ k2 * (float)in[(int)y_in * width_in * 3 + (int)(x_in + 1) * 3 + 0]
			+ k3 * (float)in[(int)(y_in + 1) * width_in * 3 + (int)x_in * 3 + 0]
			+ k4 * (float)in[(int)(y_in + 1) * width_in * 3 + (int)(x_in + 1) * 3 + 0]) / 255.f)
			- g_const_mean[2]) / g_const_std[2];
		//G
		out[(width_out * height_out) + y * width_out + x] = (((k1 * (float)in[(int)y_in * width_in * 3 + (int)x_in * 3 + 1]
			+ k2 * (float)in[(int)y_in * width_in * 3 + (int)(x_in + 1) * 3 + 1]
			+ k3 * (float)in[(int)(y_in + 1) * width_in * 3 + (int)x_in * 3 + 1]
			+ k4 * (float)in[(int)(y_in + 1) * width_in * 3 + (int)(x_in + 1) * 3 + 1]) / 255.f)
			- g_const_mean[1]) / g_const_std[1];
		//R
		out[y * width_out + x] = (((k1 * (float)in[(int)y_in * width_in * 3 + (int)x_in * 3 + 2]
			+ k2 * (float)in[(int)y_in * width_in * 3 + (int)(x_in + 1) * 3 + 2]
			+ k3 * (float)in[(int)(y_in + 1) * width_in * 3 + (int)x_in * 3 + 2]
			+ k4 * (float)in[(int)(y_in + 1) * width_in * 3 + (int)(x_in + 1) * 3 + 2]) / 255.f)
			- g_const_mean[0]) / g_const_std[0];
	}
}

void PreProcess_cuda(const cv::Mat& input_images, void* Preprocess_cuda_in_buffer, float* GPU_In_buffer, int out_width, int out_height, cudaStream_t stream)
{
	int memsize_in;
	int in_width = (int)input_images.cols;
	int in_height = (int)input_images.rows;
	int channels = (int)input_images.channels();

	memsize_in = in_width * in_height * channels;
	//    clock_t start = clock();
		//入力画像をGPUメモリーにコピー
	cudaMemcpyAsync(Preprocess_cuda_in_buffer, input_images.data, memsize_in, cudaMemcpyHostToDevice, stream);
	//    cudaMemcpy(Preprocess_cuda_in_buffer, input_images.data, memsize_in, cudaMemcpyHostToDevice);
	//    clock_t memcpu_end = clock();

	dim3 threads(16, 16);
	// 全画素処理するため、小数点は切り上げる
	dim3 blocks((int)((float)out_width / threads.x + 0.5), (int)((float)out_height / threads.y + 0.5));

	Bilinear24Kernel << < blocks, threads, 0, stream >> > ((unsigned char*)Preprocess_cuda_in_buffer, GPU_In_buffer, in_width, in_height, channels, out_width, out_height);
	cudaStreamSynchronize(stream);

	//    const double time1 = static_cast<double>(memcpu_end - start) / CLOCKS_PER_SEC * 1000.0;
	//    const double time2 = static_cast<double>(resize_end - memcpu_end) / CLOCKS_PER_SEC * 1000.0;
	//    printf("cudaMemcpy time %lf[ms] cudaResize time %lf[ms]\n", time1,time2);
}

void SetConstantValue(const float* const_mean, const float* const_std)
{
	//Preprocess内で使用する固定値をコンスタントメモリにロードする。
	cudaMemcpyToSymbol(g_const_mean, const_mean, sizeof(float) * 3, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(g_const_std, const_std, sizeof(float) * 3, 0, cudaMemcpyHostToDevice);
}


