#!/bin/bash
ARCH=`arch`
../../bin/gcc_${ARCH}_Release/TRT_ResNet50_Feature -gpu=0 -b=1 -sb=1 -t=1 -m=resnet50_feature -w=128 -h=256 -ppc=1 -p=1
