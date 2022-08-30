# CPN

## How To Use (Ubuntu / Jetson AGX Xavier)

### Download Model and Dataset
```
# Download Trained Model and Dataset
cd trt/data
wget --header='PRIVATE-TOKEN: '"$GITLAB_ACCESS_TOKEN"'' \
'http://kros.sig.kddilabs.jp/api/v4/projects/4/repository/files/script%2Fdownload_cpn_model_and_dataset.sh/raw?ref=master' \
-O ./download_cpn_model_and_dataset.sh
bash download_cpn_model_and_dataset.sh
```

### Run Inference
```
bash run.sh
```
You can check the pose estimation results at "trt/data/cpn_outputs/"   

 <img src="http://kros.sig.kddilabs.jp/kros/trt-resources/raw/master/result/cpn/cpn_outputs/sample9.png" width="600px">
 
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
You can check the benchmark results at "trt/data/cpn_benchmarks.csv".
Inference time means that end-end processing consists of pre processing, host-to-device transfer, gpu inference, device-to-host transfer, and post processing.

|Model Name|Type|OS|CPU|GPU|Input Size|Inference Time Per Frame|
|---|---|---|---|---|---|---|
|CPN|Desktop PC|Ubuntu 16.04|Core i7 7700K| TITAN V  |512 x 512| [1.5 ms - 7.4 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/raw/master/result/cpn/cpn_benchmarks/cpn_benchmarks_corei7_titanv.csv)  |
|CPN|Jetson AGX Xavier|L4T|8-core ARM v8.2 64-bit CPU, 8MB L2 + 4MB L3|512-core Volta GPU with Tensor Cores|512 x 512| [13.7 ms - 19.2 ms](http://kros.sig.kddilabs.jp/kros/trt-resources/raw/master/result/cpn/cpn_benchmarks/cpn_benchmarks_jetson_xavier.csv)  |

## How To Use (Windows)
### Download Model and Dataset
Download the trained models and dataset to "trt/data/" in the same way as [download_cpn_model_and_dataset.sh](http://kros.sig.kddilabs.jp/kros/trt-resources/raw/master/script/download_cpn_model_and_dataset.sh)

### Run Inference
Execute "trt/vc16/TRT_CPN_Main/run.bat".
