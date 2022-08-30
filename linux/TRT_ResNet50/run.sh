#!/bin/bash
ARCH=`arch`
../../bin/gcc_${ARCH}_Release/TRT_ResNet50 -gpu=0 -b=1 -sb=1 -t=1 -m=resnet50 -w=128 -h=256 -c=3 -ppc=1 -st=0.5 -p=1 -sr=1
