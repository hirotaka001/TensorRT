# YOLOv3

## How To Use (Ubuntu / Jetson AGX Xavier)

### Download Model and Dataset
```
cd trt/data
wget --header='PRIVATE-TOKEN: '"$GITLAB_ACCESS_TOKEN"'' \
'http://kros.sig.kddilabs.jp/api/v4/projects/4/repository/files/script%2Fdownload_yolov3_model_and_dataset.sh/raw?ref=master' \
-O ./download_yolov3_model_and_dataset.sh
bash download_yolov3_model_and_dataset.sh

```

### Run Inference
```
cd ../linux/TRT_YOLOv3
bash run.sh
```
You can check the detection results at "trt/data/yolov3_outputs/"   

 <img src="http://kros.sig.kddilabs.jp/kros/trt-resources/raw/master/result/yolov3/yolov3_outputs/sample4.jpg" width="600px">

### Run Benchmark
There are three parameters that determine inference time, delay (= batch_size x super_batch_size x thread_num), and GPU memory usage.
1.  Batch Size : Number of images to infer at once
2.  Super Batch Size  (coined word) : Number of parallel transfers (Host-To-Device)
3.  Cpu Thread Num : Number of parallel transfers (Pre Processing and Post Processing)

With run_benchmark.sh, you can easily find parameter combinations.
```
bash run_benchmark.sh
```
The benchmark process takes several hours to complete.
You can check the benchmark results at "trt/data/yolov3_benchmarks.csv".
Inference time means that end-end processing consists of pre processing, host-to-device transfer, gpu inference, device-to-host transfer, post processing, score thresholding, and non-maximum suppression.

|Model Name|Type|OS|CPU|GPU|Input Size|Inference Time Per Frame|
|---|---|---|---|---|---|---|
|YOLOv3|Desktop PC|Ubuntu 16.04|Core i7 7700K| TITAN V  |320 x 320| [1.4 ms - 7.1 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_corei7_titanv_320.csv)  |
|YOLOv3|Desktop PC|Ubuntu 16.04|Core i7 7700K| TITAN V  |512 x 512| [3.5 ms - 9.0 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_corei7_titanv.csv)  |
|YOLOv3|Desktop PC|Ubuntu 16.04|Xeon Gold 5118| TITAN V  |512 x 512| [3.4 ms - 8.9 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_xeon_titanv.csv)  |
|YOLOv3|Desktop PC|Ubuntu 16.04|Core i7 7800X| GeForce RTX 2080 Ti  | 512 x 512|[3.7 ms - 8.5 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_corei7_2080ti.csv)  |
|YOLOv3|Desktop PC|Ubuntu 16.04|Core i7 7820X| GeForce GTX 1080 Ti  | 512 x 512|[10.7 ms - 16.0 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_corei7_1080ti.csv)  |
|YOLOv3|Desktop PC|Windows 10|Core i7 6700K| GeForce GTX 980 Ti | 512 x 512|xx ms -  23.5 ms|
|YOLOv3|AWS p3.2xlarge |Ubuntu 16.04|Xeon E5-2686 v4| Tesla V100 |512 x 512| [3.1 ms - 6.8 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_aws_p3_2xl_v100.csv)  |
|YOLOv3|AWS g4dn.2xlarge|Ubuntu 16.04|Xeon Platinum 8259CL| Tesla T4 |512 x 512| [8.3 ms - 11.6 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_aws_g4_2xl_t4.csv)  |
|YOLOv3|AWS g4dn.xlarge|Ubuntu 16.04|Xeon Platinum 8259CL| Tesla T4 |512 x 512| [8.5 ms - 11.4 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_aws_g4_xl_t4.csv)  |
|YOLOv3|AWS g4dn.xlarge|Ubuntu 16.04|Xeon Platinum 8259CL| Tesla T4 ([No Boost](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/optimize_gpu.html)) |512 x 512| [8.8 ms - 19.2 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_aws_t4.csv)  |
|YOLOv3|AWS g3s.xlarge|Ubuntu 16.04|Xeon E5-2686 v4| Tesla M60 |512 x 512| [32.4 ms - 34.2 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_aws_g3s_xl_m60.csv)  |
|YOLOv3|Jetson AGX Xavier|L4T|8-core ARM v8.2 64-bit CPU, 8MB L2 + 4MB L3|512-core Volta GPU with Tensor Cores|320 x 320| [9.7 ms - 14.9 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_jetson_xavier_320.csv)  |
|YOLOv3|Jetson AGX Xavier|L4T|8-core ARM v8.2 64-bit CPU, 8MB L2 + 4MB L3|512-core Volta GPU with Tensor Cores|512 x 512| [24.7 ms - 27.5 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/blob/master/result/yolov3/yolov3_benchmarks/yolov3_benchmarks_jetson_xavier.csv)  |

## How To Use (Windows)
### Download Model and Dataset
Download the trained models and dataset to "trt/data/" in the same way as [download_yolov3_model_and_dataset.sh](http://kros.sig.kddilabs.jp/kros/trt-resources/raw/master/script/download_yolov3_model_and_dataset.sh)

### Run Inference
Execute "trt/vc16/TRT_YOLOv3_Main/run.bat".
