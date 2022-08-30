#!/bin/bash
ARCH=`arch`

if [ ! -e ../../bin/gcc_${ARCH}_Release/TRT_YOLOv3_benchmark ]; then
    cd benchmark
    make -j4
    cd ..
fi

mkdir -p ../../data/yolov3_outputs
echo "delay,batch,super batch,thread num,width,height,average_time,gpu_memory_usage" > ../../data/yolov3_outputs/yolov3_benchmark.csv

for batch in 1 5 10 15 20 25 30 35 40 ; do
    for super_batch in 1 2 3 4; do
         for thread_num in 1 2 3 4; do
             ../../bin/gcc_${ARCH}_Release/TRT_YOLOv3_benchmark -gpu=0 -b=$batch -sb=$super_batch -t=$thread_num -l=10 -m=yolov3 -w=512 -h=512 -c=80 -ppc=1 -sc=0.5 -n=0.5 -d=0 -o=1
         done
    done
done
