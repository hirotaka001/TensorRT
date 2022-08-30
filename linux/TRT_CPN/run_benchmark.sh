#!/bin/bash
ARCH=`arch`

if [ ! -e ../../bin/gcc_${ARCH}_Release/TRT_CPN_benchmark ]; then
    cd benchmark
    make -j4
    cd ..
fi

mkdir -p ../../data/cpn_outputs
echo "delay,batch,super batch,thread num,width,height,average_time,gpu_memory_usage" > ../../data/cpn_outputs/cpn_benchmark.csv

for batch in 1 5 10 15 20 25 30 35 40 ; do
    for super_batch in 1 2 3 4; do
         for thread_num in 1 2 3 4; do
             ../../bin/gcc_${ARCH}_Release/TRT_CPN_benchmark -gpu=0 -b=$batch -sb=$super_batch -t=$thread_num -l=10 -m=cpn -w=192 -h=256 -d=0 -o=1
         done
    done
done
