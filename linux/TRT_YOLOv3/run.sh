#!/bin/bash
ARCH=`arch`
../../bin/gcc_${ARCH}_Release/TRT_YOLOv3 -gpu=0 -b=1 -sb=1 -t=1 -m=yolov3 -w=512 -h=512 -c=80 -ppc=1 -sc=0.5 -n=0.5 -d=1
