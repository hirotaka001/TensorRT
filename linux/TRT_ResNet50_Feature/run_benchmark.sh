#!/bin/bash
ARCH=`arch`

if [ ! -e ../../bin/gcc_${ARCH}_Release/TRT_ResNet50_Feature_benchmark ]; then
    cd benchmark
    make -j4
    cd ..
fi

mkdir -p ../../data/resnet50_feature_outputs
echo "delay,batch,super batch,thread num,average_time,gpu_memory_usage" > ../../data/resnet50_feature_outputs/resnet50_feature_benchmark.csv

for batch in 1 5 10 15 20 25 30 35 40 ; do
    for super_batch in 1 2 3 4; do
         for thread_num in 1 2 3 4; do
             ../../bin/gcc_${ARCH}_Release/TRT_ResNet50_Feature_benchmark -gpu=0 -b=$batch -sb=$super_batch -t=$thread_num -l=10 -m=resnet50_feature -w=128 -h=256 -ppc=1 -p=0 -d=1
         done
    done
done
